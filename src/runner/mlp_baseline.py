# src/runner/mlp_baseline.py
# MLP baseline aligned with train_kfold: same data_path, full_idx from finite y, normalization, k-fold, W&B keys.

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

import wandb


# -------------------------
# seeds (align train_kfold)
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
# utils
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
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
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


@torch.no_grad()
def batch_stats_1d(v: torch.Tensor) -> Dict[str, float]:
    v = v.detach()
    if v.numel() <= 1:
        m = v.mean().item() if v.numel() else float("nan")
        return {"mean": m, "var": 0.0, "std": 0.0}
    var = v.var(unbiased=False).item()
    return {"mean": v.mean().item(), "var": var, "std": (var ** 0.5)}


# -------------------------
# per-seed neighbor aggregation (1-hop)
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
    """
    Returns per-seed aggregated features for neigh_nt.
    Aggregation: mean, sum, max, count  => [bs, 3F+1]
    """
    device = batch[seed_nt].x.device
    bs = int(seed_bs)
    Fdim = int(in_dim_neigh)

    if neigh_nt not in batch.node_types or not hasattr(batch[neigh_nt], "x"):
        return torch.zeros((bs, 3 * Fdim + 1), device=device)

    x_neigh = batch[neigh_nt].x  # [Nn, F]
    if x_neigh.dim() != 2 or x_neigh.size(1) != Fdim:
        return torch.zeros((bs, 3 * Fdim + 1), device=device)

    neigh_lists: List[List[int]] = [[] for _ in range(bs)]

    # seed -> neigh
    for et in _find_edge_types_between(batch, seed_nt, neigh_nt):
        ei = batch[et].edge_index
        if ei.numel() == 0:
            continue
        src = ei[0]
        dst = ei[1]
        mask = src < bs
        src = src[mask]
        dst = dst[mask]
        for s, d in zip(src.tolist(), dst.tolist()):
            neigh_lists[int(s)].append(int(d))

    # neigh -> seed
    for et in _find_edge_types_between(batch, neigh_nt, seed_nt):
        ei = batch[et].edge_index
        if ei.numel() == 0:
            continue
        src = ei[0]
        dst = ei[1]
        mask = dst < bs
        src = src[mask]
        dst = dst[mask]
        for n, s in zip(src.tolist(), dst.tolist()):
            neigh_lists[int(s)].append(int(n))

    out = torch.zeros((bs, 3 * Fdim + 1), device=device)
    for i in range(bs):
        idx = neigh_lists[i]
        if not idx:
            continue
        idx_t = torch.tensor(idx, device=device, dtype=torch.long)
        xn = x_neigh.index_select(0, idx_t)  # [K, F]
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


# -------------------------
# MLP model
# -------------------------
class MLPRegressor(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        d = d_in
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d = hidden 
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# -------------------------
# metrics: match GNN
# -------------------------
@torch.no_grad()
def eval_loader_smoothl1_mae_rmse(
    model: nn.Module,
    loader: NeighborLoader,
    *,
    target: str,
    neighbor_types: List[str],
    in_dims: Dict[str, int],
    device: torch.device,
    beta: float = 1.0,
) -> Dict[str, float]:
    model.eval()
    se_sum = 0.0
    ae_sum = 0.0
    sl1_sum = 0.0
    n = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = batch.to(device)
        X, y = batch_to_tabular_per_seed(batch, target=target, neighbor_types=neighbor_types, in_dims=in_dims)
        pred = model(X)

        se_sum += F.mse_loss(pred, y, reduction="sum").item()
        ae_sum += F.l1_loss(pred, y, reduction="sum").item()
        sl1_sum += F.smooth_l1_loss(pred, y, beta=beta, reduction="sum").item()
        n += int(y.numel())

    if n == 0:
        return {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan"), "smoothl1": float("nan")}

    mse = se_sum / n
    mae = ae_sum / n
    rmse = mse ** 0.5
    smoothl1 = sl1_sum / n
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
    total = 0.0
    n = 0
    for batch in tqdm(loader, desc="baseline", leave=False):
        batch = batch.to(device)
        bs = int(batch[target].batch_size)
        y = batch[target].y[:bs].float()
        pred = torch.full_like(y, float(c))
        loss = F.smooth_l1_loss(pred, y, beta=beta, reduction="sum").item()
        total += loss
        n += bs
    return total / max(n, 1)


# -------------------------
# one-fold training (align train_kfold.run_one_fold)
# -------------------------
def run_one_fold_mlp(
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
        shuffle=True,
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

    probe_n = min(1000, int(train_idx.numel()))
    probe_bs = min(1024, int(args.batch_size))
    first = next(iter(NeighborLoader(
        data, input_nodes=(target, train_idx[:probe_n]),
        num_neighbors=num_neighbors, batch_size=probe_bs, shuffle=False,
    ))).to(device)
    X0, _ = batch_to_tabular_per_seed(first, target=target, neighbor_types=neighbor_types, in_dims=in_dims)
    d_in = int(X0.size(1))

    model = MLPRegressor(
        d_in=d_in,
        hidden=args.mlp_hidden,
        depth=args.mlp_depth,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def loss_fn(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if args.loss == "smoothl1":
            return F.smooth_l1_loss(pred, y, beta=args.beta)
        if args.loss == "mse":
            return F.mse_loss(pred, y)
        return F.l1_loss(pred, y)

    c_med = data[target].y[train_idx].median().item()
    baseline_val = baseline_smoothl1_on_loader(val_loader, target=target, device=device, c=c_med, beta=args.beta)

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.wandb_run_name or 'kfold'}-mlp-fold{fold:02d}",
            group=group_name,
            config={
                **vars(args),
                "fold": fold,
                "fold_train_n": int(train_idx.numel()),
                "fold_val_n": int(val_idx.numel()),
                "d_in": d_in,
                "model_type": "mlp",
            },
            reinit=True,
        )
        wandb.log({"baseline/median_c": c_med, "baseline/val_smoothl1": baseline_val}, step=0)

    global_step = 0
    best_val_sl1 = float("inf")
    best_val_rmse = float("inf")
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seeds = 0
        pbar = tqdm(train_loader, desc=f"Fold {fold:02d} Epoch {epoch:03d} [mlp/train]")
        for batch in pbar:
            batch = batch.to(device)
            X, y = batch_to_tabular_per_seed(batch, target=target, neighbor_types=neighbor_types, in_dims=in_dims)
            pred = model(X)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            bs = int(y.numel())
            total_loss += loss.item() * bs
            total_seeds += bs
            global_step += 1
            pbar.set_postfix(loss=f"{total_loss / max(total_seeds, 1):.5f}")

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            if use_wandb and args.wandb_log_every > 0 and (global_step % args.wandb_log_every == 0):
                wandb.log(
                    {
                        "train/step_smoothl1": float(loss.item()),
                        "train/step_avg_smoothl1": total_loss / max(total_seeds, 1),
                        "train/global_step": global_step,
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

        train_loss = total_loss / max(total_seeds, 1)
        val_metrics = eval_loader_smoothl1_mae_rmse(
            model, val_loader, target=target, neighbor_types=neighbor_types, in_dims=in_dims, device=device, beta=args.beta
        )

        if use_wandb:
            wandb.log(
                {
                    "train/epoch_smoothl1": train_loss,
                    "val/mse": val_metrics["mse"],
                    "val/mae": val_metrics["mae"],
                    "val/rmse": val_metrics["rmse"],
                    "val/smoothl1": val_metrics["smoothl1"],
                    "train/global_step": global_step,
                    "epoch": epoch,
                },
                step=epoch,
            )

        val_sl1 = float(val_metrics["smoothl1"])
        val_rmse = float(val_metrics["rmse"])
        print(f"[fold {fold:02d}][mlp][epoch {epoch:03d}] train_smoothl1={train_loss:.6f}  val_smoothl1={val_sl1:.6f}  val_rmse={val_rmse:.6f}")

        if val_sl1 < best_val_sl1:
            best_val_sl1 = val_sl1
            best_val_rmse = min(best_val_rmse, val_rmse)
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if device.type == "mps":
            torch.mps.empty_cache()

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    if use_wandb:
        wandb.log(
            {
                "summary/best_epoch": best_epoch,
                "summary/best_val_smoothl1": best_val_sl1,
                "summary/best_val_rmse": best_val_rmse,
                "summary/baseline_val_smoothl1": baseline_val,
            },
            step=epoch,
        )
        wandb.finish()

    summary = {
        "fold": fold,
        "model": "mlp",
        "best_epoch": best_epoch,
        "best_val_smoothl1": best_val_sl1,
        "best_val_rmse": best_val_rmse,
        "baseline_val_smoothl1": baseline_val,
    }
    if out_dir is not None and seed is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"seed{seed}_fold{fold:02d}"
        (out_dir / f"{prefix}.json").write_text(json.dumps(summary, indent=2))
        if best_state is not None:
            torch.save(best_state, out_dir / f"{prefix}.pt")
    return summary


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="MLP baseline aligned with train_kfold (k-fold, normalization, W&B).")
    # data (align train_kfold)
    ap.add_argument("--data_path", type=str, default=None, help="Path to HeteroData .pt (same as train_kfold)")
    ap.add_argument("--pt", type=str, default=None, help="Alias for --data_path")
    ap.add_argument("--target", type=str, default="assignments")
    ap.add_argument("--label_key", type=str, default="y")

    # k-fold
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--seeds_file", type=str, default=None)

    # loader (align train_kfold defaults)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--eval_batch_size", type=int, default=2048)
    ap.add_argument("--num_neighbors", type=int, default=5)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)

    # MLP hyperparams
    ap.add_argument("--mlp_hidden", type=int, default=256)
    ap.add_argument("--mlp_depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)

    # optim
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=2800, help="Stop after this many steps (0 = no limit)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--loss", type=str, default="smoothl1", choices=["smoothl1", "mse", "l1"])
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--grad_clip", type=float, default=0.0)

    # wandb (align train_kfold)
    ap.add_argument("--wandb_project", type=str, default="kfold")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_log_every", type=int, default=10, help="Log train step every N steps (0=off)")
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
    print(f"[info] device={device} | model=mlp")

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

    out_dir = Path("checkpoints") / "piecewise"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: List[Dict[str, Any]] = []
    for si, seed in enumerate(seed_list.tolist()):
        seed_int = int(seed)
        torch.manual_seed(seed_int)
        print(f"\n[seed {seed}] ({si + 1}/{seed_list.numel()})")
        if args.k >= 2:
            folds = make_kfold_splits(full_idx, k=args.k, seed=seed_int)
            group_name = f"{args.wandb_run_name or 'kfold'}-mlp-seed{seed_int}"
            for fold, val_idx in enumerate(folds):
                train_idx = complement(full_idx, val_idx)
                summary = run_one_fold_mlp(
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
            group_name = f"{args.wandb_run_name or 'kfold'}-mlp-seed{seed_int}"
            summary = run_one_fold_mlp(
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
