from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from src.runner.eval import (
    _ckpt_args_to_namespace,
    ensure_all_node_types_have_x,
    load_payload,
    make_loader,
    normalize_node_features_inplace,
    sanitize_for_neighbor_loader,
)
from src.runner.mlp_baseline import (
    MLPRegressor,
    batch_to_tabular_per_seed,
    pick_device,
)
from src.runner.train_kfold import build_model


# -------------------------
# data & indices
# -------------------------
def compute_target_degree(data: HeteroData, target: str, *, degree_mode: str = "in") -> torch.Tensor:
    N = data[target].num_nodes
    deg = torch.zeros(N, dtype=torch.long)
    for (src, rel, dst) in data.edge_types:
        ei = data[(src, rel, dst)].edge_index
        if degree_mode in ("in", "inout") and dst == target:
            deg += torch.bincount(ei[1], minlength=N)
        if degree_mode in ("out", "inout") and src == target:
            deg += torch.bincount(ei[0], minlength=N)
    return deg


def split_indices(
    idx: torch.Tensor,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = idx.numel()
    g = torch.Generator().manual_seed(seed)
    perm = idx[torch.randperm(N, generator=g)]
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


def infer_mlp_config(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    """Infer d_in, hidden, depth from MLP state_dict."""
    keys = [k for k in state_dict if k.startswith("net.") and k.endswith(".weight")]
    if not keys:
        raise ValueError("No net.*.weight in state_dict")

    def idx(s: str) -> int:
        return int(s.split(".")[1])

    keys_sorted = sorted(keys, key=idx)
    first_w = state_dict[keys_sorted[0]]
    hidden, d_in = int(first_w.size(0)), int(first_w.size(1))
    num_linear = len(keys_sorted)
    depth = (num_linear - 1) // 2
    return d_in, hidden, depth


# -------------------------
# collect predictions and errors
# -------------------------
NEIGHBOR_TYPES = ["engineers", "tasks", "task_types", "districts", "departments"]


@torch.no_grad()
def collect_predictions_and_errors(
    sage_model: torch.nn.Module,
    mlp_model: torch.nn.Module,
    loader: NeighborLoader,
    *,
    target: str,
    in_dims: Dict[str, int],
    device: torch.device,
    beta: float = 1.0,
    lgb_booster: Optional[Any] = None,
) -> Tuple[List[Tuple[str, np.ndarray, Dict[str, float]]], List[Dict[str, Any]]]:
    """Run models on loader; return (results list, predictions list). Optional lgb_booster for LightGBM."""
    sage_model.eval()
    mlp_model.eval()

    errors_sage_list: List[float] = []
    errors_mlp_list: List[float] = []
    errors_lgb_list: List[float] = []
    y_list: List[float] = []
    predictions_list: List[Dict[str, Any]] = []

    se_sage = ae_sage = sl1_sage = 0.0
    se_mlp = ae_mlp = sl1_mlp = 0.0
    se_lgb = ae_lgb = sl1_lgb = 0.0
    n = 0

    for batch in tqdm(loader, desc="compare"):
        batch = batch.to(device)
        bs = int(batch[target].batch_size)
        n_id = batch[target].n_id[:bs]
        y_b = batch[target].y[:bs].float()

        pred_sage = sage_model(batch)["pred"][:bs]
        X, _ = batch_to_tabular_per_seed(
            batch, target=target, neighbor_types=NEIGHBOR_TYPES, in_dims=in_dims
        )
        pred_mlp = mlp_model(X[:bs])

        y_b = y_b.cpu()
        pred_sage = pred_sage.cpu()
        pred_mlp = pred_mlp.cpu()

        if lgb_booster is not None:
            X_np = X[:bs].cpu().numpy()
            pred_lgb = lgb_booster.predict(X_np)
            y_np = y_b.numpy()
            se_lgb += float(np.sum((pred_lgb - y_np) ** 2))
            ae_lgb += float(np.sum(np.abs(pred_lgb - y_np)))
            diff = np.abs(pred_lgb - y_np)
            sl1_lgb += float(np.sum(np.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)))

        for i in range(bs):
            node_id = int(n_id[i].item())
            yi = float(y_b[i].item())
            ps = float(pred_sage[i].item())
            pm = float(pred_mlp[i].item())
            e_s = abs(ps - yi)
            e_m = abs(pm - yi)
            errors_sage_list.append(e_s)
            errors_mlp_list.append(e_m)
            y_list.append(yi)

            row: Dict[str, Any] = {"id": node_id, "y": yi, "pred_GraphSAGE": ps, "pred_MLP": pm}
            if lgb_booster is not None:
                pl = float(pred_lgb[i])
                row["pred_LightGBM"] = pl
                errors_lgb_list.append(abs(pl - yi))
            predictions_list.append(row)

        se_sage += F.mse_loss(pred_sage, y_b, reduction="sum").item()
        ae_sage += F.l1_loss(pred_sage, y_b, reduction="sum").item()
        sl1_sage += F.smooth_l1_loss(pred_sage, y_b, beta=beta, reduction="sum").item()
        se_mlp += F.mse_loss(pred_mlp, y_b, reduction="sum").item()
        ae_mlp += F.l1_loss(pred_mlp, y_b, reduction="sum").item()
        sl1_mlp += F.smooth_l1_loss(pred_mlp, y_b, beta=beta, reduction="sum").item()
        n += bs

    errors_sage = np.array(errors_sage_list, dtype=np.float64)
    errors_mlp = np.array(errors_mlp_list, dtype=np.float64)
    mse_sage = se_sage / n
    mse_mlp = se_mlp / n
    results: List[Tuple[str, np.ndarray, Dict[str, float]]] = [
        ("GraphSAGE", errors_sage, {"mae": ae_sage / n, "rmse": np.sqrt(mse_sage), "smoothl1": sl1_sage / n}),
        ("MLP", errors_mlp, {"mae": ae_mlp / n, "rmse": np.sqrt(mse_mlp), "smoothl1": sl1_mlp / n}),
    ]
    if lgb_booster is not None:
        errors_lgb = np.array(errors_lgb_list, dtype=np.float64)
        mse_lgb = se_lgb / n
        results.append(
            ("LightGBM", errors_lgb, {"mae": ae_lgb / n, "rmse": np.sqrt(mse_lgb), "smoothl1": sl1_lgb / n})
        )
    return results, predictions_list


# -------------------------
# plotting
# -------------------------
def plot_comparison(
    results: List[Tuple[str, np.ndarray, Dict[str, float]]],
    out_path: Path,
    *,
    figsize: Tuple[float, float] = (14, 5),
) -> None:
    """Left: bar plot. Middle: cumulative error curve. Right: error distribution. results = [(label, errors, metrics), ...]."""
    fig, (ax_bar, ax_cum, ax_dist) = plt.subplots(1, 3, figsize=figsize)
    num_models = len(results)
    colors = ["C0", "C1", "C2"][:num_models]
    keys = ["mae", "rmse", "smoothl1"]
    metric_names = ["MAE", "RMSE", "SmoothL1"]

    # ---- Left: bar plot ----
    x = np.arange(len(metric_names))
    width = 0.8 / max(num_models, 1)
    for i, (label, errors, metrics) in enumerate(results):
        vals = [metrics[k] for k in keys]
        offset = (i - (num_models - 1) / 2) * width
        ax_bar.bar(x + offset, vals, width, label=label, color=colors[i])
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names)
    ax_bar.set_ylabel("Error")
    ax_bar.set_title("Metric comparison (lower is better)")
    ax_bar.legend()
    ax_bar.grid(axis="y", alpha=0.3)

    # ---- Middle: cumulative error curve ----
    n = len(results[0][1])
    frac = np.arange(1, n + 1, dtype=np.float64) / n
    for i, (label, errors, _) in enumerate(results):
        ax_cum.plot(np.sort(errors), frac, label=label, color=colors[i])
    ax_cum.set_xlabel("Absolute error |pred − y|")
    ax_cum.set_ylabel("Fraction of samples (cumulative)")
    ax_cum.set_title("Cumulative Error Curve")
    ax_cum.legend()
    ax_cum.grid(alpha=0.3)

    # ---- Right: error distribution ----
    all_errors = np.concatenate([r[1] for r in results])
    emax = float(max(all_errors.max(), 1e-8))
    bins = np.linspace(0, emax, 51)
    for i, (label, errors, _) in enumerate(results):
        ax_dist.hist(errors, bins=bins, density=True, alpha=0.6, label=label, color=colors[i])
    ax_dist.set_xlabel("Absolute error |pred − y|")
    ax_dist.set_ylabel("Density")
    ax_dist.set_title("Error Distribution")
    ax_dist.legend()
    ax_dist.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] saved figure -> {out_path}")


# -------------------------
# main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare GraphSAGE, MLP, and optionally LightGBM with pretrained weights; bar + cumulative curve + error distribution."
    )
    ap.add_argument("--pt", type=str, required=True, help="Path to graph .pt (HeteroData)")
    ap.add_argument("--sage_ckpt", type=str, required=True, help="Path to GraphSAGE checkpoint (train_kfold format)")
    ap.add_argument("--mlp_ckpt", type=str, required=True, help="Path to MLP checkpoint (piecewise state_dict .pt)")
    ap.add_argument("--lgb_ckpt", type=str, default=None, help="Path to LightGBM checkpoint .txt (piecewise_lightgbm); if set, compare three models")
    ap.add_argument("--target", type=str, default="assignments")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"],
                    help="Which split to evaluate (default: test)")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_neighbors", type=int, default=5)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--min_degree", type=int, default=1)
    ap.add_argument("--out", type=str, default="runs/compare_model/compare_three.png",
                    help="Output figure path")
    ap.add_argument("--predictions_out", type=str, default=None,
                    help="Write per-sample predictions and target to this JSON file (for further analysis)")
    ap.add_argument("--beta", type=float, default=1.0, help="Smooth L1 beta")
    args = ap.parse_args()

    device = pick_device(args.device)
    pt_path = Path(args.pt)
    sage_ckpt_path = Path(args.sage_ckpt)
    mlp_ckpt_path = Path(args.mlp_ckpt)

    if not pt_path.exists():
        raise FileNotFoundError(f"Graph not found: {pt_path}")
    if not sage_ckpt_path.exists():
        raise FileNotFoundError(f"GraphSAGE checkpoint not found: {sage_ckpt_path}")
    if not mlp_ckpt_path.exists():
        raise FileNotFoundError(f"MLP checkpoint not found: {mlp_ckpt_path}")

    lgb_booster: Optional[Any] = None
    if args.lgb_ckpt:
        if lgb is None:
            raise SystemExit("LightGBM requested (--lgb_ckpt) but lightgbm is not installed. pip install lightgbm")
        lgb_ckpt_path = Path(args.lgb_ckpt)
        if not lgb_ckpt_path.exists():
            raise FileNotFoundError(f"LightGBM checkpoint not found: {lgb_ckpt_path}")
        lgb_booster = lgb.Booster(model_file=str(lgb_ckpt_path))
        print("[info] Loaded LightGBM checkpoint")

    # Load graph
    data: HeteroData = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(data, HeteroData):
        raise TypeError("Loaded object is not HeteroData")
    target = args.target
    if target not in data.node_types:
        raise ValueError(f"target {target!r} not in node_types")
    data[target].y = data[target].y.view(-1).float()
    y = data[target].y
    finite = torch.isfinite(y)
    full_idx = torch.nonzero(finite, as_tuple=False).view(-1).long() if not finite.all() else torch.arange(y.numel(), dtype=torch.long)
    if not finite.all():
        print(f"[warn] Using finite y only: {full_idx.numel()}/{y.numel()}")

    deg = compute_target_degree(data, target, degree_mode="in")
    kept = (deg >= args.min_degree).nonzero(as_tuple=False).view(-1)
    full_idx = torch.tensor(sorted(set(full_idx.tolist()) & set(kept.tolist())), dtype=torch.long)
    if full_idx.numel() == 0:
        raise ValueError("No samples after degree filter")

    normalize_node_features_inplace(data, drop_const=True)
    in_dims = ensure_all_node_types_have_x(data)
    data = sanitize_for_neighbor_loader(data)

    # Splits from GraphSAGE checkpoint config
    payload = load_payload(sage_ckpt_path)
    ckpt_args = payload["args"]
    seed = int(ckpt_args.get("seed", 42))
    train_ratio = float(ckpt_args.get("train_ratio", 0.8))
    val_ratio = float(ckpt_args.get("val_ratio", 0.1))
    layers = int(ckpt_args.get("layers", args.layers))
    num_neighbors = int(ckpt_args.get("num_neighbors", args.num_neighbors))

    train_idx, val_idx, test_idx = split_indices(
        full_idx, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio
    )
    if args.split == "train":
        eval_idx = train_idx
    elif args.split == "val":
        eval_idx = val_idx
    elif args.split == "test":
        eval_idx = test_idx
    else:
        eval_idx = full_idx

    loader = make_loader(
        data,
        target,
        eval_idx,
        layers=layers,
        num_neighbors_per_layer=num_neighbors,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Load GraphSAGE (must be sage model)
    ckpt_ns = _ckpt_args_to_namespace(ckpt_args)
    if getattr(ckpt_ns, "model", "sage").lower() != "sage":
        raise ValueError(
            f"Checkpoint model is {getattr(ckpt_ns, 'model', None)}; use a GraphSAGE (sage) checkpoint for --sage_ckpt"
        )
    sage_model = build_model(ckpt_ns, data, in_dims, target).to(device)
    sage_model.load_state_dict(payload["model_state"], strict=True)
    sage_model.eval()

    # Load MLP
    mlp_state = torch.load(mlp_ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(mlp_state, dict):
        raise ValueError("MLP checkpoint must be a state_dict")
    d_in, hidden, depth = infer_mlp_config(mlp_state)
    mlp_model = MLPRegressor(d_in=d_in, hidden=hidden, depth=depth, dropout=0.1).to(device)
    mlp_model.load_state_dict(mlp_state, strict=True)
    mlp_model.eval()

    print(f"[info] Evaluating on split={args.split}, n={eval_idx.numel()}")

    results, predictions_list = collect_predictions_and_errors(
        sage_model,
        mlp_model,
        loader,
        target=target,
        in_dims=in_dims,
        device=device,
        beta=args.beta,
        lgb_booster=lgb_booster,
    )

    for label, _, metrics in results:
        print(f"{label}: {metrics}")

    out_path = Path(args.out)
    plot_comparison(results, out_path)

    if args.predictions_out:
        pred_path = Path(args.predictions_out)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        # Sort by id for reproducible ordering
        predictions_list.sort(key=lambda r: r["id"])
        pred_path.write_text(json.dumps(predictions_list, indent=2), encoding="utf-8")
        print(f"[ok] wrote {len(predictions_list)} predictions -> {pred_path}")


if __name__ == "__main__":
    main()
