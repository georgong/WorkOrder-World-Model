from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

import wandb

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from src.runner.mlp_baseline import (
    batch_to_tabular_per_seed,
    complement,
    ensure_all_node_types_have_x,
    make_kfold_splits,
    normalize_node_features_inplace,
    parse_seeds_arg,
    pick_device,
    sanitize_for_neighbor_loader,
    split_indices,
)


# -------------------------
# collect tabular from loader (same feature pipeline as mlp_baseline)
# -------------------------
def collect_tabular_from_loader(
    loader: NeighborLoader,
    *,
    target: str,
    neighbor_types: List[str],
    in_dims: Dict[str, int],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect (X, y) from loader into numpy arrays for LightGBM. Same piecewise features as MLP."""
    X_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []
    for batch in tqdm(loader, desc="collect", leave=False):
        batch = batch.to(device)
        X, y = batch_to_tabular_per_seed(
            batch, target=target, neighbor_types=neighbor_types, in_dims=in_dims
        )
        X_list.append(X.cpu())
        y_list.append(y.cpu())
    X_np = torch.cat(X_list, dim=0).numpy()
    y_np = torch.cat(y_list, dim=0).numpy()
    return X_np, y_np


# -------------------------
# metrics (match GNN / MLP)
# -------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 1.0) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    n = yt.size
    if n == 0:
        return {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan"), "smoothl1": float("nan")}
    mse = float(np.mean((yp - yt) ** 2))
    mae = float(np.mean(np.abs(yp - yt)))
    rmse = mse ** 0.5
    diff = np.abs(yp - yt)
    smoothl1 = float(np.mean(np.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)))
    return {"mse": mse, "mae": mae, "rmse": rmse, "smoothl1": smoothl1}


@torch.no_grad()
def baseline_smoothl1_on_loader(
    loader: NeighborLoader,
    *,
    target: str,
    device: torch.device,
    c: float,
    beta: float = 1.0,
) -> float:
    """Constant baseline smooth L1 on loader (align mlp_baseline)."""
    total = 0.0
    n = 0
    for batch in tqdm(loader, desc="baseline", leave=False):
        batch = batch.to(device)
        bs = int(batch[target].batch_size)
        y = batch[target].y[:bs].float()
        pred = torch.full_like(y, float(c))
        diff = (pred - y).abs()
        loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta).sum().item()
        total += loss
        n += bs
    return total / max(n, 1)


# -------------------------
# one-fold training (align mlp_baseline.run_one_fold_mlp)
# -------------------------
def run_one_fold_lightgbm(
    *,
    base_data: HeteroData,
    target: str,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    fold: int,
    group_name: str,
    neighbor_types: List[str],
    use_wandb: bool = True,
    out_dir: Optional[Path] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    if lgb is None:
        raise RuntimeError("lightgbm is not installed. pip install lightgbm")

    data = base_data.clone()
    data[target].y = data[target].y.float()

    normalize_node_features_inplace(data, drop_const=True)
    in_dims = ensure_all_node_types_have_x(data)
    data = sanitize_for_neighbor_loader(data)

    num_neighbors = {et: [args.num_neighbors] * args.layers for et in data.edge_types}
    train_loader = NeighborLoader(
        data,
        input_nodes=(target, train_idx),
        num_neighbors=num_neighbors,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = NeighborLoader(
        data,
        input_nodes=(target, val_idx),
        num_neighbors=num_neighbors,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )

    t_start = time.time()
    X_train, y_train = collect_tabular_from_loader(
        train_loader, target=target, neighbor_types=neighbor_types, in_dims=in_dims, device=device
    )
    X_val, y_val = collect_tabular_from_loader(
        val_loader, target=target, neighbor_types=neighbor_types, in_dims=in_dims, device=device
    )
    d_in = X_train.shape[1]

    c_med = float(np.median(y_train))
    baseline_val = baseline_smoothl1_on_loader(
        val_loader, target=target, device=device, c=c_med, beta=args.beta
    )

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.wandb_run_name or 'kfold'}-lightgbm-fold{fold:02d}",
            group=group_name,
            config={
                **vars(args),
                "fold": fold,
                "fold_train_n": int(train_idx.numel()),
                "fold_val_n": int(val_idx.numel()),
                "d_in": d_in,
                "model_type": "lightgbm",
            },
            reinit=True,
        )
        wandb.log({"baseline/median_c": c_med, "baseline/val_smoothl1": baseline_val}, step=0)

    num_boost_round = (
        min(args.n_estimators, args.max_rounds) if args.max_rounds > 0 else args.n_estimators
    )
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        {
            "objective": args.objective,
            "metric": "mae",
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "min_child_samples": args.min_child_samples,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "verbosity": -1,
            "seed": args.seed,
        },
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[val_data],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    train_time = time.time() - t_start
    y_val_pred = model.predict(X_val)
    va = compute_metrics(y_val, y_val_pred, beta=args.beta)
    best_val_sl1 = float(va["smoothl1"])
    best_val_rmse = float(va["rmse"])

    print(
        f"[fold {fold:02d}][lightgbm] val_smoothl1={best_val_sl1:.6f}  val_rmse={best_val_rmse:.6f}  "
        f"(base {baseline_val:.6f})  time_s={train_time:.1f}"
    )

    if use_wandb:
        wandb.log(
            {
                "train/epoch_smoothl1": float(
                    np.mean(np.abs(model.predict(X_train) - y_train))
                ),
                "val/mse": va["mse"],
                "val/mae": va["mae"],
                "val/rmse": va["rmse"],
                "val/smoothl1": va["smoothl1"],
                "epoch": 1,
                "time/elapsed_s": train_time,
            },
            step=1,
        )
        wandb.log(
            {
                "summary/best_epoch": 1,
                "summary/best_val_smoothl1": best_val_sl1,
                "summary/best_val_rmse": best_val_rmse,
                "summary/baseline_val_smoothl1": baseline_val,
            },
            step=1,
        )
        wandb.finish()

    summary = {
        "fold": fold,
        "model": "lightgbm",
        "best_epoch": 1,
        "best_val_smoothl1": best_val_sl1,
        "best_val_rmse": best_val_rmse,
        "baseline_val_smoothl1": baseline_val,
    }
    if out_dir is not None and seed is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"seed{seed}_fold{fold:02d}"
        (out_dir / f"{prefix}.json").write_text(json.dumps(summary, indent=2))
        model_path = out_dir / f"{prefix}.txt"
        model.save_model(str(model_path))
    return summary


# -------------------------
# main (align mlp_baseline CLI and flow)
# -------------------------
def main() -> None:
    if lgb is None:
        raise SystemExit("lightgbm is not installed. pip install lightgbm")

    ap = argparse.ArgumentParser(
        description="LightGBM baseline aligned with mlp_baseline (same data_path, k-fold, piecewise features, W&B)."
    )
    # data (align mlp_baseline / train_kfold)
    ap.add_argument("--data_path", type=str, default=None, help="Path to HeteroData .pt")
    ap.add_argument("--pt", type=str, default=None, help="Alias for --data_path")
    ap.add_argument("--target", type=str, default="assignments")
    ap.add_argument("--label_key", type=str, default="y")

    # k-fold
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--seeds_file", type=str, default=None)

    # loader (align mlp_baseline)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--eval_batch_size", type=int, default=2048)
    ap.add_argument("--num_neighbors", type=int, default=5)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)

    # LightGBM hyperparams
    ap.add_argument("--num_leaves", type=int, default=31)
    ap.add_argument("--max_depth", type=int, default=-1)
    ap.add_argument("--learning_rate", type=float, default=0.1)
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--max_rounds", type=int, default=2800, help="Cap num_boost_round (0 = use n_estimators only)")
    ap.add_argument("--min_child_samples", type=int, default=20)
    ap.add_argument("--reg_alpha", type=float, default=0.0)
    ap.add_argument("--reg_lambda", type=float, default=0.0)
    ap.add_argument("--early_stopping_rounds", type=int, default=50)
    ap.add_argument("--objective", type=str, default="regression_l1", choices=["regression", "regression_l1"],
                    help="regression_l1 = optimize MAE (default); regression = MSE")
    ap.add_argument("--beta", type=float, default=1.0)

    # wandb (align mlp_baseline)
    ap.add_argument("--wandb_project", type=str, default="kfold")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--no_wandb", action="store_true")

    ap.add_argument("--device", type=str, default="auto")

    args = ap.parse_args()

    pt = args.data_path or args.pt
    if not pt:
        raise SystemExit("Provide --data_path or --pt")
    data_path = Path(pt)
    if not data_path.exists():
        raise FileNotFoundError(f"data not found: {data_path}")

    device = pick_device(args.device)
    print(f"[info] device={device} | model=lightgbm")

    data: HeteroData = torch.load(data_path, map_location="cpu", weights_only=False)
    if not isinstance(data, HeteroData):
        raise TypeError("Loaded object is not HeteroData")

    target = args.target
    if target not in data.node_types:
        raise ValueError(f"target {target!r} not in node_types")

    if args.label_key != "y":
        if not hasattr(data[target], args.label_key):
            raise ValueError(f"target has no attr {args.label_key!r}")
        data[target].y = getattr(data[target], args.label_key)

    y = data[target].y
    if not isinstance(y, torch.Tensor):
        raise TypeError(f"{target}.y is not a Tensor")
    y = y.view(-1).float()
    data[target].y = y

    finite = torch.isfinite(y)
    if finite.all():
        full_idx = torch.arange(y.numel(), dtype=torch.long)
    else:
        full_idx = torch.nonzero(finite, as_tuple=False).view(-1).long()
        print(f"[warn] non-finite y; using subset: {full_idx.numel()}/{y.numel()}")

    neighbor_types = ["engineers", "tasks", "task_types", "districts", "departments"]

    seed_list = parse_seeds_arg(args.seeds, args.seeds_file)
    if seed_list is None:
        seed_list = torch.tensor([args.seed], dtype=torch.long)

    out_dir = Path("checkpoints") / "piecewise_lightgbm"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: List[Dict[str, Any]] = []
    for si, seed in enumerate(seed_list.tolist()):
        seed_int = int(seed)
        torch.manual_seed(seed_int)
        np.random.seed(seed_int)
        print(f"\n[seed {seed}] ({si + 1}/{seed_list.numel()})")
        if args.k >= 2:
            folds = make_kfold_splits(full_idx, k=args.k, seed=seed_int)
            group_name = f"{args.wandb_run_name or 'kfold'}-lightgbm-seed{seed_int}"
            for fold, val_idx in enumerate(folds):
                train_idx = complement(full_idx, val_idx)
                summary = run_one_fold_lightgbm(
                    base_data=data,
                    target=target,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    args=args,
                    device=device,
                    fold=fold,
                    group_name=group_name,
                    neighbor_types=neighbor_types,
                    use_wandb=not args.no_wandb,
                    out_dir=out_dir,
                    seed=seed_int,
                )
                summary["seed"] = seed_int
                all_summaries.append(summary)
        else:
            train_idx, val_idx, _ = split_indices(full_idx, seed_int, 0.8, 0.1)
            group_name = f"{args.wandb_run_name or 'kfold'}-lightgbm-seed{seed_int}"
            summary = run_one_fold_lightgbm(
                base_data=data,
                target=target,
                train_idx=train_idx,
                val_idx=val_idx,
                args=args,
                device=device,
                fold=0,
                group_name=group_name,
                neighbor_types=neighbor_types,
                use_wandb=not args.no_wandb,
                out_dir=out_dir,
                seed=seed_int,
            )
            summary["seed"] = seed_int
            all_summaries.append(summary)

    out_path = out_dir / "kfold_summary.json"
    out_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"\n[ok] wrote summary -> {out_path}")


if __name__ == "__main__":
    main()
