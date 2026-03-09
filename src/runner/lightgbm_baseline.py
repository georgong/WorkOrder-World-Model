# src/runner/lightgbm_baseline.py
# LightGBM baseline using the same tabular feature pipeline as mlp_baseline (1-hop neighbor aggregation).
# Use for comparison with GNN and MLP in W&B (same metrics: metrics/val_rmse, metrics/val_mae, etc.).

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


# -------------------------
# seeds + k-fold (align train_kfold / mlp_baseline)
# -------------------------
def _parse_int_range_token(tok: str) -> List[int]:
    tok = tok.strip()
    if not tok:
        return []
    if "-" in tok:
        a, b = tok.split("-", 1)
        a, b = int(a.strip()), int(b.strip())
        if b < a:
            a, b = b, a
        return list(range(a, b + 1))
    return [int(tok)]


def parse_seeds_arg(seeds: Optional[str], seeds_file: Optional[str]) -> Optional[torch.Tensor]:
    items: List[int] = []
    if seeds_file:
        p = Path(seeds_file)
        if not p.exists():
            raise FileNotFoundError(f"--seeds_file not found: {p}")
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for tok in line.split(","):
                items.extend(_parse_int_range_token(tok))
    if seeds:
        for tok in seeds.split(","):
            items.extend(_parse_int_range_token(tok))
    if not items:
        return None
    return torch.tensor(sorted(set(items)), dtype=torch.long)


def make_kfold_splits(idx: torch.Tensor, *, k: int, seed: int) -> List[torch.Tensor]:
    assert idx.ndim == 1 and k >= 2
    N = idx.numel()
    if N < k:
        raise ValueError(f"Not enough samples for k-fold: N={N}, k={k}")
    g = torch.Generator().manual_seed(seed)
    perm = idx[torch.randperm(N, generator=g)]
    fold_sizes = [N // k] * k
    for i in range(N % k):
        fold_sizes[i] += 1
    folds: List[torch.Tensor] = []
    offset = 0
    for fs in fold_sizes:
        folds.append(perm[offset : offset + fs])
        offset += fs
    return folds


def complement(all_idx: torch.Tensor, holdout: torch.Tensor) -> torch.Tensor:
    hold_set = set(holdout.tolist())
    keep = [int(x) for x in all_idx.tolist() if int(x) not in hold_set]
    return torch.tensor(keep, dtype=torch.long)


# -------------------------
# normalization (align train_kfold)
# -------------------------
@torch.no_grad()
def normalize_node_features_inplace(
    data: HeteroData,
    *,
    eps: float = 1e-6,
    drop_const: bool = True,
    const_std_thr: float = 1e-8,
) -> Dict[str, Dict[str, torch.Tensor]]:
    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    for nt in data.node_types:
        if not hasattr(data[nt], "x"):
            continue
        x = data[nt].x
        if not isinstance(x, torch.Tensor) or x.dim() != 2:
            continue
        if not x.is_floating_point():
            x = x.float()
        x = x.cpu()
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)
        std = torch.sqrt(var + eps)
        keep = torch.ones_like(std, dtype=torch.bool)
        if drop_const:
            keep = std > const_std_thr
            if keep.sum().item() == 0:
                keep[0] = True
        x2 = (x[:, keep] - mean[keep]) / std[keep]
        data[nt].x = x2
        stats[nt] = {"mean": mean[keep], "std": std[keep], "keep_mask": keep}
        if hasattr(data[nt], "attr_name"):
            an = data[nt].attr_name
            if isinstance(an, list) and len(an) == int(keep.numel()):
                data[nt].attr_name = [an[i] for i in range(len(an)) if bool(keep[i].item())]
    return stats


# -------------------------
# utils (same as mlp_baseline)
# -------------------------
def pick_device(s: str) -> torch.device:
    s = s.lower()
    if s == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(s)


def compute_target_degree(data: HeteroData, target: str, degree_mode: str = "in") -> torch.Tensor:
    N = data[target].num_nodes
    deg = torch.zeros(N, dtype=torch.long)
    for (src, rel, dst) in data.edge_types:
        ei = data[(src, rel, dst)].edge_index
        if degree_mode in ("in", "inout") and dst == target:
            deg += torch.bincount(ei[1], minlength=N)
        if degree_mode in ("out", "inout") and src == target:
            deg += torch.bincount(ei[0], minlength=N)
    return deg


def split_indices(idx: torch.Tensor, seed: int, train_ratio: float, val_ratio: float):
    g = torch.Generator().manual_seed(seed)
    perm = idx[torch.randperm(idx.numel(), generator=g)]
    n_train = int(idx.numel() * train_ratio)
    n_val = int(idx.numel() * val_ratio)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx


def ensure_all_node_types_have_x(data: HeteroData) -> Dict[str, int]:
    in_dims = {nt: data[nt].x.size(-1) for nt in data.node_types if hasattr(data[nt], "x")}
    for nt in data.node_types:
        if nt not in in_dims:
            in_dims[nt] = 1
            data[nt].x = torch.zeros((data[nt].num_nodes, 1), dtype=torch.float)
    return in_dims


def sanitize_for_neighbor_loader(data: HeteroData) -> HeteroData:
    for nt in data.node_types:
        store = data[nt]
        for key in list(store.keys()):
            if isinstance(store[key], torch.Tensor):
                continue
            del store[key]
    for et in data.edge_types:
        store = data[et]
        for key in list(store.keys()):
            if isinstance(store[key], torch.Tensor):
                continue
            del store[key]
    return data


# -------------------------
# per-seed neighbor aggregation (same as mlp_baseline)
# -------------------------
def _find_edge_types_between(batch: HeteroData, src_nt: str, dst_nt: str) -> List[Tuple[str, str, str]]:
    out = []
    for et in batch.edge_types:
        if et[0] == src_nt and et[2] == dst_nt:
            out.append(et)
    return out


def _agg_1hop_from_seed_to_type(
    batch: HeteroData,
    *,
    seed_nt: str,
    seed_bs: int,
    neigh_nt: str,
    in_dim_neigh: int,
) -> torch.Tensor:
    device = batch[seed_nt].x.device
    bs = int(seed_bs)
    Fdim = int(in_dim_neigh)

    if neigh_nt not in batch.node_types or not hasattr(batch[neigh_nt], "x"):
        return torch.zeros((bs, 3 * Fdim + 1), device=device)

    x_neigh = batch[neigh_nt].x
    if x_neigh.dim() != 2 or x_neigh.size(1) != Fdim:
        return torch.zeros((bs, 3 * Fdim + 1), device=device)

    neigh_lists: List[List[int]] = [[] for _ in range(bs)]

    for et in _find_edge_types_between(batch, seed_nt, neigh_nt):
        ei = batch[et].edge_index
        if ei.numel() == 0:
            continue
        src, dst = ei[0], ei[1]
        mask = src < bs
        for s, d in zip(src[mask].tolist(), dst[mask].tolist()):
            neigh_lists[int(s)].append(int(d))

    for et in _find_edge_types_between(batch, neigh_nt, seed_nt):
        ei = batch[et].edge_index
        if ei.numel() == 0:
            continue
        src, dst = ei[0], ei[1]
        mask = dst < bs
        for n, s in zip(src[mask].tolist(), dst[mask].tolist()):
            neigh_lists[int(s)].append(int(n))

    out = torch.zeros((bs, 3 * Fdim + 1), device=device)
    for i in range(bs):
        idx = neigh_lists[i]
        if not idx:
            continue
        idx_t = torch.tensor(idx, device=device, dtype=torch.long)
        xn = x_neigh.index_select(0, idx_t)
        mean = xn.mean(dim=0)
        summ = xn.sum(dim=0)
        mx = xn.max(dim=0).values
        cnt = torch.tensor([float(xn.size(0))], device=device)
        out[i] = torch.cat([mean, summ, mx, cnt], dim=0)
    return out


def batch_to_tabular_per_seed(
    batch: HeteroData,
    *,
    target: str,
    neighbor_types: List[str],
    in_dims: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    bs = int(batch[target].batch_size)
    y = batch[target].y[:bs].float()
    feats = [batch[target].x[:bs]]
    for nt in neighbor_types:
        feats.append(
            _agg_1hop_from_seed_to_type(
                batch,
                seed_nt=target,
                seed_bs=bs,
                neigh_nt=nt,
                in_dim_neigh=in_dims.get(nt, 1),
            )
        )
    X = torch.cat(feats, dim=1)
    return X, y


def collect_tabular_from_loader(
    loader: NeighborLoader,
    *,
    target: str,
    neighbor_types: List[str],
    in_dims: Dict[str, int],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect (X, y) from loader into numpy arrays for LightGBM."""
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


# -------------------------
# one-fold training (align train_kfold / mlp_baseline)
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
    in_dims: Dict[str, int],
    num_neighbors: Dict[Tuple[str, str, str], List[int]],
    use_wandb: bool = True,
    out_dir: Optional[Path] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    train_loader = NeighborLoader(
        base_data,
        input_nodes=(target, train_idx),
        num_neighbors=num_neighbors,
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_loader = NeighborLoader(
        base_data,
        input_nodes=(target, val_idx),
        num_neighbors=num_neighbors,
        batch_size=args.eval_batch_size,
        shuffle=False,
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
    diff = np.abs(y_val - c_med)
    baseline_val = float(np.mean(np.where(diff < args.beta, 0.5 * diff ** 2 / args.beta, diff - 0.5 * args.beta)))

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.wandb_run_name or 'kfold'}-lightgbm-fold{fold:02d}",
            group=group_name,
            config={
                **vars(args),
                "fold": fold,
                "fold_train_n": int(X_train.shape[0]),
                "fold_val_n": int(X_val.shape[0]),
                "d_in": d_in,
                "model_type": "lightgbm",
            },
            reinit=True,
        )
        wandb.log({"baseline/median_c": c_med, "baseline/val_smoothl1": baseline_val}, step=0)

    num_boost_round = min(args.n_estimators, args.max_rounds) if args.max_rounds > 0 else args.n_estimators
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        {
            "objective": args.objective,
            "metric": "rmse",
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
                "train/epoch_smoothl1": float(np.mean(np.abs(model.predict(X_train) - y_train))),
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

    if out_dir is not None and seed is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"seed{seed}_fold{fold:02d}"
        model_path = out_dir / f"{prefix}.txt"
        model.save_model(str(model_path))

    return {
        "fold": fold,
        "model": "lightgbm",
        "best_epoch": 1,
        "best_val_smoothl1": best_val_sl1,
        "best_val_rmse": best_val_rmse,
        "baseline_val_smoothl1": baseline_val,
    }


# -------------------------
# main
# -------------------------
def main():
    if lgb is None:
        raise SystemExit("lightgbm is not installed. pip install lightgbm")

    ap = argparse.ArgumentParser(description="LightGBM baseline aligned with train_kfold (k-fold, W&B, max_rounds).")
    ap.add_argument("--data_path", type=str, default=None)
    ap.add_argument("--pt", type=str, default=None)
    ap.add_argument("--target", type=str, default="assignments")
    ap.add_argument("--label_key", type=str, default="y")

    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--seeds_file", type=str, default=None)

    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--num_neighbors", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--eval_batch_size", type=int, default=2048)

    ap.add_argument("--num_leaves", type=int, default=31)
    ap.add_argument("--max_depth", type=int, default=-1)
    ap.add_argument("--learning_rate", type=float, default=0.1)
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--max_rounds", type=int, default=2800, help="Cap num_boost_round (0 = use n_estimators only)")
    ap.add_argument("--min_child_samples", type=int, default=20)
    ap.add_argument("--reg_alpha", type=float, default=0.0)
    ap.add_argument("--reg_lambda", type=float, default=0.0)
    ap.add_argument("--early_stopping_rounds", type=int, default=50)
    ap.add_argument("--objective", type=str, default="regression", choices=["regression", "regression_l1"])
    ap.add_argument("--beta", type=float, default=1.0)

    ap.add_argument("--wandb_project", type=str, default="kfold")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--no_wandb", action="store_true")

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
    normalize_node_features_inplace(data, drop_const=True)
    in_dims = ensure_all_node_types_have_x(data)
    data = sanitize_for_neighbor_loader(data)
    num_neighbors = {et: [args.num_neighbors] * args.layers for et in data.edge_types}

    seed_list = parse_seeds_arg(args.seeds, args.seeds_file)
    if seed_list is None:
        seed_list = torch.tensor([args.seed], dtype=torch.long)

    out_dir = Path("checkpoints") / (args.wandb_run_name or "kfold") / "lightgbm"
    all_summaries: List[Dict[str, Any]] = []
    for si, seed in enumerate(seed_list.tolist()):
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        print(f"\n[seed {seed}] ({si + 1}/{seed_list.numel()})")
        if args.k >= 2:
            folds = make_kfold_splits(full_idx, k=args.k, seed=int(seed))
            group_name = f"{args.wandb_run_name or 'kfold'}-lightgbm-seed{int(seed)}"
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
                    in_dims=in_dims,
                    num_neighbors=num_neighbors,
                    use_wandb=not args.no_wandb,
                    out_dir=out_dir,
                    seed=int(seed),
                )
                summary["seed"] = int(seed)
                all_summaries.append(summary)
        else:
            train_idx, val_idx, _ = split_indices(full_idx, int(seed), 0.8, 0.1)
            group_name = f"{args.wandb_run_name or 'kfold'}-lightgbm-seed{int(seed)}"
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
                in_dims=in_dims,
                num_neighbors=num_neighbors,
                use_wandb=not args.no_wandb,
                out_dir=out_dir,
                seed=int(seed),
            )
            summary["seed"] = int(seed)
            all_summaries.append(summary)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kfold_summary.json"
    out_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"\n[ok] wrote summary -> {out_path}")


if __name__ == "__main__":
    main()
