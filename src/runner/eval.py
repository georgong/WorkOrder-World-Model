# eval_checkpoints_dir.py
from __future__ import annotations

import argparse
import json
import re
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from src.runner.train_kfold import build_model


# -------------------------
# device
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


# -------------------------
# split logic (same as training)
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
    assert idx.ndim == 1
    N = idx.numel()
    g = torch.Generator().manual_seed(seed)
    perm = idx[torch.randperm(N, generator=g)]

    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


# -------------------------
# normalization (same as training)
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
# NeighborLoader sanitation (same as training)
# -------------------------
def sanitize_for_neighbor_loader(data: HeteroData) -> HeteroData:
    for nt in data.node_types:
        store = data[nt]
        for key in list(store.keys()):
            v = store[key]
            if isinstance(v, torch.Tensor):
                continue
            del store[key]

    for et in data.edge_types:
        store = data[et]
        for key in list(store.keys()):
            v = store[key]
            if isinstance(v, torch.Tensor):
                continue
            del store[key]

    return data


def ensure_all_node_types_have_x(data: HeteroData) -> Dict[str, int]:
    in_dims = {nt: data[nt].x.size(-1) for nt in data.node_types if hasattr(data[nt], "x")}
    for nt in data.node_types:
        if nt not in in_dims:
            in_dims[nt] = 1
            data[nt].x = torch.zeros((data[nt].num_nodes, 1), dtype=torch.float)
    return in_dims


def _ckpt_args_to_namespace(ckpt_args: dict) -> argparse.Namespace:
    """Build an argparse.Namespace from checkpoint args so build_model() works (sage/hgt/rgcn)."""
    ns = argparse.Namespace()
    for k, v in ckpt_args.items():
        setattr(ns, k, v)
    if not hasattr(ns, "model"):
        setattr(ns, "model", "sage")
    if not hasattr(ns, "dropout"):
        setattr(ns, "dropout", 0.1)
    if not hasattr(ns, "heads"):
        setattr(ns, "heads", 4)
    if not hasattr(ns, "hgt_group"):
        setattr(ns, "hgt_group", "sum")
    return ns


@torch.no_grad()
def eval_constant_baseline_on_loader(
    loader: NeighborLoader,
    target: str,
    device: torch.device,
    c: float,
    *,
    beta: float = 1.0,
) -> Dict[str, float]:
    se_sum = 0.0
    ae_sum = 0.0
    sl1_sum = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"baseline(c={c:.3f})")

    for batch in pbar:
        batch = batch.to(device)
        y = batch[target].y.float()
        bs = int(batch[target].batch_size)
        t = y[:bs]

        p = torch.full_like(t, float(c))

        se = F.mse_loss(p, t, reduction="sum").item()
        ae = F.l1_loss(p, t, reduction="sum").item()
        sl1 = F.smooth_l1_loss(p, t, beta=beta, reduction="sum").item()

        se_sum += se
        ae_sum += ae
        sl1_sum += sl1
        n += bs

        mse = se_sum / max(n, 1)
        mae = ae_sum / max(n, 1)
        rmse = mse ** 0.5
        smoothl1 = sl1_sum / max(n, 1)

        pbar.set_postfix(
            mse=f"{mse:.4f}",
            mae=f"{mae:.4f}",
            rmse=f"{rmse:.4f}",
            sl1=f"{smoothl1:.4f}",
        )

    if n == 0:
        return {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan"), "smoothl1": float("nan")}

    mse = se_sum / n
    mae = ae_sum / n
    rmse = mse ** 0.5
    smoothl1 = sl1_sum / n
    return {"mse": mse, "mae": mae, "rmse": rmse, "smoothl1": smoothl1}

# -------------------------
# eval
# -------------------------
@torch.no_grad()
def eval_loader(model: nn.Module, loader: NeighborLoader, target: str, device: torch.device) -> Dict[str, float]:
    model.eval()
    se_sum = 0.0
    ae_sum = 0.0
    sl1_sum = 0.0
    n = 0

    pbar = tqdm(loader, desc="eval")

    for batch in pbar:
        batch = batch.to(device)
        pred = model(batch)["pred"]
        y = batch[target].y.float()

        bs = int(batch[target].batch_size)
        p = pred[:bs]
        t = y[:bs]

        se = F.mse_loss(p, t, reduction="sum").item()
        ae = F.l1_loss(p, t, reduction="sum").item()
        sl1 = F.smooth_l1_loss(p, t, beta=1.0, reduction="sum").item()

        se_sum += se
        ae_sum += ae
        sl1_sum += sl1
        n += bs

        mse = se_sum / max(n, 1)
        mae = ae_sum / max(n, 1)
        rmse = mse ** 0.5
        smoothl1 = sl1_sum / max(n, 1)

        pbar.set_postfix(
            mse=f"{mse:.4f}",
            mae=f"{mae:.4f}",
            rmse=f"{rmse:.4f}",
            sl1=f"{smoothl1:.4f}",
        )

    if n == 0:
        return {
            "mse": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "smoothl1": float("nan"),
        }

    mse = se_sum / n
    mae = ae_sum / n
    rmse = mse ** 0.5
    smoothl1 = sl1_sum / n

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "smoothl1": smoothl1,
    }

def make_loader(
    data: HeteroData,
    target: str,
    idx: torch.Tensor,
    *,
    layers: int,
    num_neighbors_per_layer: int,
    batch_size: int,
    shuffle: bool = False,
) -> NeighborLoader:
    num_neighbors = {et: [num_neighbors_per_layer] * layers for et in data.edge_types}
    return NeighborLoader(
        data,
        input_nodes=(target, idx),
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
    )


# -------------------------
# checkpoint utilities
# -------------------------
_epoch_re = re.compile(r"_epoch(\d+)\.pt$")


def sort_ckpts(paths: List[Path]) -> List[Path]:
    def key(p: Path):
        m = _epoch_re.search(p.name)
        if m:
            return (0, int(m.group(1)))
        return (1, p.name)
    return sorted(paths, key=key)


def load_payload(ckpt_path: Path) -> dict:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state" not in payload or "args" not in payload:
        raise ValueError(f"Bad checkpoint format: {ckpt_path}")
    return payload


def _get_assignment_to_task_mapping(data: HeteroData, target: str = "assignments") -> Optional[torch.Tensor]:
    """Build target_node_index -> task_node_index from (target, *, tasks) edge. Returns None if no such edge."""
    task_type = "tasks"
    if task_type not in data.node_types:
        return None
    for (src, rel, dst) in data.edge_types:
        if src == target and dst == task_type:
            ei = data[(src, rel, dst)].edge_index
            N_target = data[target].num_nodes
            assign_to_task = torch.full((N_target,), -1, dtype=torch.long)
            assign_to_task[ei[0]] = ei[1]
            return assign_to_task
    return None


@torch.no_grad()
def predict_all_and_save(
    model: nn.Module,
    data: HeteroData,
    target: str,
    kept: torch.Tensor,
    *,
    layers: int,
    num_neighbors_per_layer: int,
    batch_size: int,
    device: torch.device,
    assignment_node_ids: List[Any],
    task_node_ids: List[Any],
    assign_to_task: Optional[torch.Tensor],
    out_path: Path,
    include_ground_truth: bool = True,
) -> None:
    """Run model on all target nodes in `kept`, write CSV: task_id, assignment_id, predicted_completion_time [, y]."""
    loader = make_loader(
        data, target, kept,
        layers=layers,
        num_neighbors_per_layer=num_neighbors_per_layer,
        batch_size=batch_size,
        shuffle=False,
    )
    preds: Dict[int, float] = {}
    labels: Dict[int, float] = {}
    has_y = hasattr(data[target], "y") and data[target].y is not None
    for batch in tqdm(loader, desc="predict_all"):
        batch = batch.to(device)
        pred = model(batch)["pred"]
        bs = int(batch[target].batch_size)
        p = pred[:bs].cpu()
        n_id = batch[target].n_id[:bs]
        for i in range(bs):
            global_idx = int(n_id[i].item())
            preds[global_idx] = float(p[i].item())
        if include_ground_truth and has_y and hasattr(batch[target], "y"):
            y = batch[target].y[:bs].cpu()
            for i in range(bs):
                global_idx = int(n_id[i].item())
                labels[global_idx] = float(y[i].item())
    rows: List[Dict[str, Any]] = []
    for global_idx in sorted(preds.keys()):
        assignment_id = assignment_node_ids[global_idx] if global_idx < len(assignment_node_ids) else global_idx
        task_id = None
        if assign_to_task is not None and task_node_ids and 0 <= assign_to_task[global_idx].item() < len(task_node_ids):
            task_id = task_node_ids[int(assign_to_task[global_idx].item())]
        row = {"task_id": task_id, "assignment_id": assignment_id, "predicted_completion_time": preds[global_idx]}
        if include_ground_truth and global_idx in labels:
            row["ground_truth_completion_time"] = labels[global_idx]
        rows.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task_id", "assignment_id", "predicted_completion_time"]
    if include_ground_truth and has_y:
        fieldnames.append("ground_truth_completion_time")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[predict_all] wrote {len(rows)} rows -> {out_path}")


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=str, required=True, help="Path to graph .pt (HeteroData)")
    ap.add_argument("--ckpt", type=str, default=None, help="Path to a single checkpoint .pt (overrides --ckpt_dir)")
    ap.add_argument("--ckpt_dir", type=str, default=None, help="Directory containing checkpoint .pt files (used if --ckpt not set)")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--splits", type=str, default="val,test", help='Comma list: "train,val,test"')
    ap.add_argument("--eval_batch_size", type=int, default=512)
    ap.add_argument("--num_neighbors", type=int, default=3, help="Neighbors per layer for eval")
    ap.add_argument("--out_dir", type=str, default="runs/eval_results")
    ap.add_argument("--predict_all", action="store_true", help="Iterate all target nodes and write predictions CSV")
    ap.add_argument("--predictions_out", type=str, default=None, help="Output path for predictions CSV (default: out_dir/predictions_all.csv)")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"[info] device={device}")

    pt_path = Path(args.pt)
    assert pt_path.exists(), f"Graph file not found: {pt_path}"

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            raise SystemExit(f"Checkpoint not found: {ckpt_path}")
        ckpts = [ckpt_path]
    else:
        ckpt_dir = Path(args.ckpt_dir or "runs/ckpts")
        if not ckpt_dir.exists():
            raise SystemExit(f"ckpt_dir not found: {ckpt_dir}")
        ckpts = sort_ckpts(list(ckpt_dir.glob("*.pt")))
        if not ckpts:
            raise SystemExit(f"No .pt checkpoints found in {ckpt_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name

    want_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for s in want_splits:
        if s not in {"train", "val", "test", "all"}:
            raise ValueError(f"Unknown split: {s}")

    # Load base graph once (we will clone per ckpt to avoid cross-ckpt mutation)
    base_data: HeteroData = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert isinstance(base_data, HeteroData), "Loaded object is not HeteroData"

    # Evaluate every checkpoint
    with out_path.open("w") as f_out:
        for ckpt_path in ckpts:
            payload = load_payload(ckpt_path)
            ckpt_args = payload["args"]
            epoch = payload.get("epoch", None)

            target = ckpt_args["target"]
            layers = int(ckpt_args["layers"])
            hidden = int(ckpt_args["hidden"])

            # Work on a fresh copy of the graph for this ckpt (normalize is in-place)
            data = base_data.clone()

            assert target in data.node_types, f"target {target} not in data.node_types"
            has_y = hasattr(data[target], "y") and data[target].y is not None
            if not has_y:
                if args.predict_all:
                    want_splits_this_ckpt = []  # skip split eval, only export predictions
                else:
                    raise SystemExit(f"data[{target}].y missing; required for split eval. Use --predict_all to only export predictions.")
            else:
                want_splits_this_ckpt = want_splits
            if has_y:
                data[target].y = data[target].y.float()

            # match training: degree filter + split
            deg_mode = ckpt_args.get("degree_mode", "in")
            min_deg = int(ckpt_args.get("min_degree", 1))
            train_ratio = float(ckpt_args.get("train_ratio", 0.8))
            val_ratio = float(ckpt_args.get("val_ratio", 0.1))
            seed = int(ckpt_args.get("seed", 42))

            deg = compute_target_degree(data, target, degree_mode=deg_mode)
            kept = (deg >= min_deg).nonzero(as_tuple=False).view(-1)
            if kept.numel() == 0:
                print(f"[skip] {ckpt_path.name}: no nodes after degree filter")
                continue

            train_idx, val_idx, test_idx = split_indices(
                kept, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio
            )

            # Save node IDs and assignment->task mapping before sanitize (for --predict_all)
            assignment_node_ids: List[Any] = list(data[target].node_ids) if hasattr(data[target], "node_ids") else list(range(data[target].num_nodes))
            task_node_ids: List[Any] = []
            assign_to_task: Optional[torch.Tensor] = None
            if "tasks" in data.node_types and hasattr(data["tasks"], "node_ids"):
                task_node_ids = list(data["tasks"].node_ids)
                assign_to_task = _get_assignment_to_task_mapping(data, target)

            # match training: normalize features
            normalize_node_features_inplace(data, drop_const=True)

            # match training: ensure x exists for all node types
            in_dims = ensure_all_node_types_have_x(data)

            # match training: sanitize
            data = sanitize_for_neighbor_loader(data)

            # rebuild model (sage / hgt / rgcn from checkpoint) + load weights
            ckpt_ns = _ckpt_args_to_namespace(ckpt_args)
            model = build_model(ckpt_ns, data, in_dims, target).to(device)
            try:
                model.load_state_dict(payload["model_state"], strict=True)
            except RuntimeError as e:
                if "state_dict" in str(e) or "size mismatch" in str(e) or "Unexpected key" in str(e):
                    raise SystemExit(
                        "Checkpoint was trained on a different graph (different node types or feature sizes). "
                        "Use the same graph .pt file that was used when training this checkpoint (e.g. the same --data_path in train_kfold)."
                    ) from e
                raise
            model.eval()

            # loaders per split
            idx_map = {"train": train_idx, "val": val_idx, "test": test_idx, "all": all_idx}
            results = {
                "ckpt": str(ckpt_path),
                "ckpt_name": ckpt_path.name,
                "epoch": epoch,
                "target": target,
                "layers": layers,
                "hidden": hidden,
                "min_degree": min_deg,
                "degree_mode": deg_mode,
                "seed": seed,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
            }

            line = f"[ckpt] {ckpt_path.name} (epoch={epoch})"
            for split in want_splits_this_ckpt:
                loader = make_loader(
                    data,
                    target,
                    split_idx,
                    layers=layers,
                    num_neighbors_per_layer=int(args.num_neighbors),
                    batch_size=int(args.eval_batch_size),
                    shuffle=False,
                )

                # For baseline, always use train_idx median
                train_y = data[target].y[train_idx].float()
                c_med = train_y.median().item()
                b = eval_constant_baseline_on_loader(loader, target, device, c_med, beta=1.0)
                m = eval_loader(model, loader, target, device)

                results["baseline_c_median"] = c_med
                results[f"{split}_n"] = int(split_idx.numel())
                results[f"{split}_mse"] = m["mse"]
                results[f"{split}_mae"] = m["mae"]
                results[f"{split}_rmse"] = m["rmse"]
                results[f"{split}_smoothl1"] = m["smoothl1"]
                results[f"{split}_b_mse"] = b["mse"]
                results[f"{split}_b_mae"] = b["mae"]
                results[f"{split}_b_rmse"] = b["rmse"]
                results[f"{split}_b_smoothl1"] = b["smoothl1"]

                # improvement (baseline - model): positive means model is better
                results[f"{split}_imp_mae"] = b["mae"] - m["mae"]
                results[f"{split}_imp_rmse"] = b["rmse"] - m["rmse"]
                results[f"{split}_imp_smoothl1"] = b["smoothl1"] - m["smoothl1"]

                line += (
                    f" | {split}(n={split_idx.numel()}): "
                    f"rmse={m['rmse']:.4f} (b {b['rmse']:.4f}, +{results[f'{split}_imp_rmse']:.4f}) "
                    f"mae={m['mae']:.4f} (b {b['mae']:.4f}, +{results[f'{split}_imp_mae']:.4f}) "
                    f"sl1={m['smoothl1']:.4f} (b {b['smoothl1']:.4f}, +{results[f'{split}_imp_smoothl1']:.4f})"
                )

            print(line)
            f_out.write(json.dumps(results) + "\n")

            # Optionally iterate all target nodes and write predictions CSV (task_id, assignment_id, pred)
            if args.predict_all:
                pred_out = Path(args.predictions_out) if args.predictions_out else out_dir / "predictions_all.csv"
                num_neighbors = int(ckpt_args.get("num_neighbors", args.num_neighbors))
                predict_all_and_save(
                    model,
                    data,
                    target,
                    kept,
                    layers=layers,
                    num_neighbors_per_layer=num_neighbors,
                    batch_size=args.eval_batch_size,
                    device=device,
                    assignment_node_ids=assignment_node_ids,
                    task_node_ids=task_node_ids,
                    assign_to_task=assign_to_task,
                    out_path=pred_out,
                    include_ground_truth=has_y,
                )

            if device.type == "mps":
                torch.mps.empty_cache()

    print(f"[ok] wrote results -> {out_path}")


if __name__ == "__main__":
    # example usage:
    # python -m src.runner.eval --pt data/graph/sdge.pt --ckpt_dir runs/checkpoints

    # run with all splits instead of val/test
    # python -m src.runner.eval --pt data/graph/sdge.pt --ckpt_dir runs/checkpoints/kfold-sage-fold00_fold00_epoch002.pt --splits all --out_name all_results.json
    main()
