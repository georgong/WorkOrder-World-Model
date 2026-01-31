# train_kfold.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.loader import NeighborLoader

import wandb


# -------------------------
# seeds parsing
# -------------------------
def _parse_int_range_token(tok: str) -> list[int]:
    tok = tok.strip()
    if not tok:
        return []
    if "-" in tok:
        a, b = tok.split("-", 1)
        a = int(a.strip())
        b = int(b.strip())
        if b < a:
            a, b = b, a
        return list(range(a, b + 1))
    return [int(tok)]


def parse_seeds_arg(seeds: str | None, seeds_file: str | None) -> torch.Tensor | None:
    items: list[int] = []

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

    uniq = sorted(set(items))
    return torch.tensor(uniq, dtype=torch.long)


# -------------------------
# device / utilities
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


def check_tensor(name: str, t: torch.Tensor) -> None:
    if not isinstance(t, torch.Tensor):
        return
    if torch.isnan(t).any():
        raise ValueError(f"[NaN DETECTED] {name} contains NaN")
    if torch.isinf(t).any():
        raise ValueError(f"[INF DETECTED] {name} contains Inf")


@torch.no_grad()
def batch_stats_1d(v: torch.Tensor) -> Dict[str, float]:
    v = v.detach()
    if v.numel() <= 1:
        m = v.mean().item() if v.numel() else float("nan")
        return {"mean": m, "var": 0.0, "std": 0.0}
    var = v.var(unbiased=False).item()
    return {"mean": v.mean().item(), "var": var, "std": (var ** 0.5)}


# -------------------------
# degree + kfold split
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


def make_kfold_splits(idx: torch.Tensor, *, k: int, seed: int) -> List[torch.Tensor]:
    """
    Returns list of k folds, each a 1D LongTensor.
    """
    assert idx.ndim == 1
    assert k >= 2
    N = idx.numel()
    if N < k:
        raise ValueError(f"Not enough samples for k-fold: N={N}, k={k}")

    g = torch.Generator().manual_seed(seed)
    perm = idx[torch.randperm(N, generator=g)]

    # roughly equal folds
    fold_sizes = [N // k] * k
    for i in range(N % k):
        fold_sizes[i] += 1

    folds: List[torch.Tensor] = []
    offset = 0
    for fs in fold_sizes:
        folds.append(perm[offset:offset + fs])
        offset += fs
    return folds


def complement(all_idx: torch.Tensor, holdout: torch.Tensor) -> torch.Tensor:
    """
    all_idx and holdout are 1D unique.
    return all_idx  holdout
    """
    hold_set = set(holdout.tolist())
    keep = [int(x) for x in all_idx.tolist() if int(x) not in hold_set]
    return torch.tensor(keep, dtype=torch.long)


# -------------------------
# normalization (same as your training)
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
# NeighborLoader sanitation
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


# -------------------------
# model (same as yours)
# -------------------------
class HeteroSAGERegressor(nn.Module):
    def __init__(self, metadata, in_dims, hidden_dim=128, num_layers=2, target_node_type="assignments"):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.target_node_type = target_node_type

        self.in_proj = nn.ModuleDict({nt: Linear(in_dims[nt], hidden_dim) for nt in self.node_types})

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {et: SAGEConv((-1, -1), hidden_dim) for et in self.edge_types}
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in self.node_types}))

        self.base = nn.Parameter(torch.tensor(0.0))
        self.out = Linear(hidden_dim, 1)

    def forward(self, data: HeteroData):
        x_dict = {nt: F.relu(self.in_proj[nt](data[nt].x)) for nt in self.node_types}

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(self.norms[i][k](v)) for k, v in x_dict.items()}

        delta = self.out(x_dict[self.target_node_type]).squeeze(-1)
        pred = self.base + delta
        return {"pred": pred}


# -------------------------
# eval / baseline
# -------------------------
@torch.no_grad()
def eval_loader_smoothl1_mae_rmse(
    model: nn.Module,
    loader: NeighborLoader,
    target: str,
    device: torch.device,
    *,
    beta: float = 1.0,
) -> Dict[str, float]:
    model.eval()
    se_sum = 0.0
    ae_sum = 0.0
    sl1_sum = 0.0
    n = 0

    for batch in tqdm(loader, leave=False, desc="eval"):
        batch = batch.to(device)
        pred = model(batch)["pred"]
        y = batch[target].y.float()

        bs = int(batch[target].batch_size)
        p = pred[:bs]
        t = y[:bs]

        se_sum += F.mse_loss(p, t, reduction="sum").item()
        ae_sum += F.l1_loss(p, t, reduction="sum").item()
        sl1_sum += F.smooth_l1_loss(p, t, beta=beta, reduction="sum").item()
        n += bs

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
    target: str,
    device: torch.device,
    c: float,
    *,
    beta: float = 1.0,
) -> float:
    total = 0.0
    n = 0
    for batch in tqdm(loader, leave=False, desc="baseline"):
        batch = batch.to(device)
        y = batch[target].y.float()
        bs = int(batch[target].batch_size)
        y = y[:bs]
        pred = torch.full_like(y, float(c))
        loss = F.smooth_l1_loss(pred, y, beta=beta, reduction="sum").item()
        total += loss
        n += bs
    return total / max(n, 1)


def save_checkpoint(save_dir: Path, run_name: str, fold: int, epoch: int, model: nn.Module, opt: torch.optim.Optimizer, args: argparse.Namespace):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"{run_name}_fold{fold:02d}_epoch{epoch:03d}.pt"
    payload = {
        "fold": fold,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "args": vars(args),
    }
    torch.save(payload, ckpt_path)

    args_path = save_dir / f"{run_name}_fold{fold:02d}_config.json"
    if not args_path.exists():
        args_path.write_text(json.dumps(vars(args), indent=2))

    print(f"[ok] saved checkpoint -> {ckpt_path}")


# -------------------------
# one fold training
# -------------------------
def run_one_fold(
    *,
    base_data: HeteroData,
    target: str,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    fold: int,
    group_name: str,
) -> Dict[str, float]:
    # clone graph per fold (normalize is in-place)
    data = base_data.clone()

    # y float
    data[target].y = data[target].y.float()

    # normalize
    normalize_node_features_inplace(data, drop_const=True)

    # ensure x for all
    in_dims = ensure_all_node_types_have_x(data)

    # sanitize
    data = sanitize_for_neighbor_loader(data)

    # loaders
    num_neighbors = {et: [args.num_neighbors] * args.layers for et in data.edge_types}

    train_loader = NeighborLoader(
        data,
        input_nodes=(target, train_idx),
        num_neighbors=num_neighbors,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        data,
        input_nodes=(target, val_idx),
        num_neighbors=num_neighbors,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )

    # model
    model = HeteroSAGERegressor(
        metadata=data.metadata(),
        in_dims=in_dims,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        target_node_type=target,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # wandb per fold
    run = wandb.init(
        project=args.wandb_project,
        name=f"{args.wandb_run_name or 'kfold'}-fold{fold:02d}",
        group=group_name,
        config={**vars(args), "fold": fold, "fold_train_n": int(train_idx.numel()), "fold_val_n": int(val_idx.numel())},
        reinit=True,
    )
    run_name = run.name or run.id

    # baseline from train median (match your logic)
    c_med = data[target].y[train_idx].median().item()
    baseline_val = baseline_smoothl1_on_loader(val_loader, target, device, c_med, beta=1.0)
    wandb.log({"baseline/median_c": c_med, "baseline/val_smoothl1": baseline_val}, step=0)

    save_dir = Path(args.save_dir) / (args.wandb_run_name or "kfold") / f"fold{fold:02d}"

    # training
    global_step = 0
    best_val_sl1 = float("inf")
    best_val_rmse = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seeds = 0

        pbar = tqdm(train_loader, desc=f"Fold {fold:02d} Epoch {epoch:03d} [train]")
        for batch in pbar:
            batch = batch.to(device)
            pred = model(batch)["pred"]
            y = batch[target].y.float()

            bs = int(batch[target].batch_size)
            p = pred[:bs]
            t = y[:bs]

            p_stats = batch_stats_1d(p)
            t_stats = batch_stats_1d(t)
            var_ratio = p_stats["var"] / (t_stats["var"] + 1e-12)

            loss = F.smooth_l1_loss(p, t, beta=1.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * bs
            total_seeds += bs
            avg_loss = total_loss / max(total_seeds, 1)
            pbar.set_postfix(batch_loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}")

            wandb.log(
                {
                    "train/batch_loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/pred_std": p_stats["std"],
                    "train/y_mean": t_stats["mean"],
                    "train/y_var": t_stats["var"],
                    "train/y_std": t_stats["std"],
                    "train/var_ratio_pred_to_y": var_ratio,
                },
                step=global_step,
            )
            global_step += 1

        train_loss = total_loss / max(total_seeds, 1)

        val_metrics = eval_loader_smoothl1_mae_rmse(model, val_loader, target, device, beta=1.0)
        wandb.log(
            {
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "val/mse": val_metrics["mse"],
                "val/mae": val_metrics["mae"],
                "val/rmse": val_metrics["rmse"],
                "val/smoothl1": val_metrics["smoothl1"],
                "baseline/val_smoothl1": baseline_val,
                "val/improve_over_baseline_sl1": baseline_val - val_metrics["smoothl1"],
            },
            step=global_step,
        )

        print(
            f"Fold {fold:02d} Epoch {epoch:03d} | train_loss {train_loss:.6f} | "
            f"val sl1 {val_metrics['smoothl1']:.4f} (base {baseline_val:.4f}) | "
            f"val rmse {val_metrics['rmse']:.4f} mae {val_metrics['mae']:.4f}"
        )

        # save
        if args.save_every > 0 and (epoch % args.save_every == 0):
            save_checkpoint(save_dir, run_name, fold, epoch, model, opt, args)

        # track best (for summary)
        best_val_sl1 = min(best_val_sl1, val_metrics["smoothl1"])
        best_val_rmse = min(best_val_rmse, val_metrics["rmse"])

        if device.type == "mps":
            torch.mps.empty_cache()

    # final save
    save_checkpoint(save_dir, run_name, fold, args.epochs, model, opt, args)
    wandb.finish()

    return {
        "fold": float(fold),
        "baseline_val_smoothl1": float(baseline_val),
        "best_val_smoothl1": float(best_val_sl1),
        "best_val_rmse": float(best_val_rmse),
    }


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pt", type=str, default="data/graph/sdge.pt")
    ap.add_argument("--target", type=str, default="assignments")

    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")

    ap.add_argument("--min_degree", type=int, default=1)
    ap.add_argument("--degree_mode", type=str, default="in", choices=["in", "out", "inout"])

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--num_neighbors", type=int, default=10)

    # kfold
    ap.add_argument("--k_folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=None, help="If set, only run this fold (0-based). Otherwise run all folds.")

    # seeds restriction (optional)
    ap.add_argument("--seeds", type=str, default=None, help='Comma/range list in target index space, e.g. "1,2,10-20"')
    ap.add_argument("--seeds_file", type=str, default=None, help="Text file of seed indices (one per line, allow commas/ranges)")

    # wandb / saving
    ap.add_argument("--wandb_project", type=str, default="scheduling_world_model")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--save_dir", type=str, default="runs/checkpoints_kfold")
    ap.add_argument("--save_every", type=int, default=1)

    args = ap.parse_args()
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    print(f"[info] device={device}")

    pt_path = Path(args.pt)
    assert pt_path.exists(), f"File not found: {pt_path}"

    base_data: HeteroData = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert isinstance(base_data, HeteroData)

    target = args.target
    assert target in base_data.node_types
    assert hasattr(base_data[target], "x")
    assert hasattr(base_data[target], "y")

    # sanity checks
    for nt in base_data.node_types:
        if hasattr(base_data[nt], "x"):
            check_tensor(f"{nt}.x", base_data[nt].x)
        if hasattr(base_data[nt], "y"):
            check_tensor(f"{nt}.y", base_data[nt].y)

    # seeds: user-provided or degree-filtered
    deg = compute_target_degree(base_data, target, degree_mode=args.degree_mode)
    user_seeds = parse_seeds_arg(args.seeds, args.seeds_file)

    if user_seeds is not None:
        if user_seeds.min().item() < 0 or user_seeds.max().item() >= base_data[target].num_nodes:
            raise ValueError(
                f"--seeds out of range for target '{target}': "
                f"[0, {base_data[target].num_nodes - 1}], got min={user_seeds.min().item()} max={user_seeds.max().item()}"
            )
        kept = user_seeds
        if args.min_degree is not None and args.min_degree > 0:
            kept = kept[deg[kept] >= args.min_degree]
        if kept.numel() == 0:
            raise ValueError("No seeds left after applying --seeds (+ optional degree filter).")
        print(f"[info] using user seeds: kept={kept.numel()}")
    else:
        kept = (deg >= args.min_degree).nonzero(as_tuple=False).view(-1)
        if kept.numel() == 0:
            raise ValueError(f"No target nodes left after filtering: min_degree={args.min_degree}, mode={args.degree_mode}")
        print(f"[info] using degree-filtered seeds: kept={kept.numel()}")

    # build folds
    folds = make_kfold_splits(kept, k=args.k_folds, seed=args.seed)

    if args.fold is not None:
        if args.fold < 0 or args.fold >= args.k_folds:
            raise ValueError(f"--fold out of range: got {args.fold}, k_folds={args.k_folds}")
        fold_list = [args.fold]
    else:
        fold_list = list(range(args.k_folds))

    group_name = args.wandb_run_name or f"kfold_{args.k_folds}fold_seed{args.seed}"

    summaries: List[Dict[str, float]] = []
    for f in fold_list:
        val_idx = folds[f]
        train_idx = complement(kept, val_idx)

        print(f"[info] fold={f} train={train_idx.numel()} val={val_idx.numel()}")

        summ = run_one_fold(
            base_data=base_data,
            target=target,
            train_idx=train_idx,
            val_idx=val_idx,
            args=args,
            device=device,
            fold=f,
            group_name=group_name,
        )
        summaries.append(summ)

    # write a quick summary json
    out_dir = Path(args.save_dir) / (args.wandb_run_name or "kfold")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kfold_summary.json"
    out_path.write_text(json.dumps({"k_folds": args.k_folds, "seed": args.seed, "summaries": summaries}, indent=2))
    print(f"[ok] wrote summary -> {out_path}")

    # print macro avg
    if summaries:
        avg_base = sum(s["baseline_val_smoothl1"] for s in summaries) / len(summaries)
        avg_best = sum(s["best_val_smoothl1"] for s in summaries) / len(summaries)
        print(f"[summary] avg baseline val sl1={avg_base:.4f} | avg best val sl1={avg_best:.4f} | improve={avg_base - avg_best:.4f}")


if __name__ == "__main__":
    main()
