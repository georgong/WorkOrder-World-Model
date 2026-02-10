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
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, HGTConv, RGCNConv
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
# degree + kfold split (kept for your tooling)
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
        folds.append(perm[offset : offset + fs])
        offset += fs
    return folds


def complement(all_idx: torch.Tensor, holdout: torch.Tensor) -> torch.Tensor:
    """
    all_idx and holdout are 1D unique.
    return all_idx % holdout
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
# RGCN helper: flatten hetero batch into homogeneous graph + relation ids
# -------------------------
@torch.no_grad()
def hetero_to_rgcn_inputs(
    data: HeteroData,
    x_dict: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    Build:
      x_all: [N_total, H]
      edge_index_all: [2, E_total]
      edge_type_all: [E_total] relation ids in [0, num_relations-1]
      offsets: nt -> global offset
    Relation id assignment is fixed by data.edge_types order for determinism.
    """
    node_types = list(data.node_types)

    offsets: Dict[str, int] = {}
    total = 0
    for nt in node_types:
        offsets[nt] = total
        total += int(data[nt].num_nodes)

    # concat projected x in deterministic node_types order
    x_all = torch.cat([x_dict[nt] for nt in node_types], dim=0)

    edge_indices = []
    edge_types = []

    # relation id: index in edge_types list
    rel2id = {et: i for i, et in enumerate(list(data.edge_types))}
    for et in data.edge_types:
        ei = data[et].edge_index
        if ei.numel() == 0:
            continue

        src, rel, dst = et
        off_s = offsets[src]
        off_d = offsets[dst]

        ei2 = ei.clone()
        ei2[0] = ei2[0] + off_s
        ei2[1] = ei2[1] + off_d

        edge_indices.append(ei2)
        edge_types.append(torch.full((ei2.size(1),), rel2id[et], dtype=torch.long, device=ei2.device))

    if edge_indices:
        edge_index_all = torch.cat(edge_indices, dim=1)
        edge_type_all = torch.cat(edge_types, dim=0)
    else:
        edge_index_all = torch.empty((2, 0), dtype=torch.long, device=x_all.device)
        edge_type_all = torch.empty((0,), dtype=torch.long, device=x_all.device)

    return x_all, edge_index_all, edge_type_all, offsets


# -------------------------
# models
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


class HGTRegressor(nn.Module):
    def __init__(
        self,
        metadata,
        in_dims,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        hgt_group="sum",  # kept for CLI compatibility (not used due to PyG version differences)
        target_node_type="assignments",
    ):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.target_node_type = target_node_type

        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({num_heads})")

        self.in_proj = nn.ModuleDict({nt: Linear(in_dims[nt], hidden_dim) for nt in self.node_types})

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=num_heads,
                )
            )
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in self.node_types}))

        self.dropout = float(dropout)
        self.base = nn.Parameter(torch.tensor(0.0))
        self.out = Linear(hidden_dim, 1)

    def forward(self, data: HeteroData):
        x_dict = {nt: F.relu(self.in_proj[nt](data[nt].x)) for nt in self.node_types}

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(self.norms[i][k](v)) for k, v in x_dict.items()}
            if self.dropout > 0:
                x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        delta = self.out(x_dict[self.target_node_type]).squeeze(-1)
        pred = self.base + delta
        return {"pred": pred}


class RGCNRegressor(nn.Module):
    """
    RGCN baseline by flattening hetero graph into a single node space with relation ids.
    Uses:
      - type-specific input projection (same as others)
      - RGCNConv layers on flattened edges
      - readout only on target node type slice
    """

    def __init__(
        self,
        metadata,
        in_dims,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        target_node_type="assignments",
    ):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.target_node_type = target_node_type

        self.in_proj = nn.ModuleDict({nt: Linear(in_dims[nt], hidden_dim) for nt in self.node_types})

        # num_relations = number of canonical edge types
        self.num_relations = len(self.edge_types)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=self.num_relations))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = float(dropout)
        self.base = nn.Parameter(torch.tensor(0.0))
        self.out = Linear(hidden_dim, 1)

    def forward(self, data: HeteroData):
        x_dict = {nt: F.relu(self.in_proj[nt](data[nt].x)) for nt in self.node_types}
        x_all, edge_index_all, edge_type_all, offsets = hetero_to_rgcn_inputs(data, x_dict)

        h = x_all
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index_all, edge_type_all)
            h = F.relu(self.norms[i](h))
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        off = offsets[self.target_node_type]
        n = int(data[self.target_node_type].num_nodes)
        h_t = h[off : off + n]

        delta = self.out(h_t).squeeze(-1)
        pred = self.base + delta
        return {"pred": pred}


def build_model(args: argparse.Namespace, data: HeteroData, in_dims: Dict[str, int], target: str) -> nn.Module:
    m = args.model.lower()
    if m == "sage":
        return HeteroSAGERegressor(
            metadata=data.metadata(),
            in_dims=in_dims,
            hidden_dim=args.hidden,
            num_layers=args.layers,
            target_node_type=target,
        )
    if m == "hgt":
        return HGTRegressor(
            metadata=data.metadata(),
            in_dims=in_dims,
            hidden_dim=args.hidden,
            num_layers=args.layers,
            num_heads=args.heads,
            dropout=args.dropout,
            hgt_group=args.hgt_group,
            target_node_type=target,
        )
    if m == "rgcn":
        return RGCNRegressor(
            metadata=data.metadata(),
            in_dims=in_dims,
            hidden_dim=args.hidden,
            num_layers=args.layers,
            dropout=args.dropout,
            target_node_type=target,
        )
    raise ValueError(f"Unknown --model={args.model!r}. Use sage|hgt|rgcn")


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
    rmse = mse**0.5
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


def save_checkpoint(
    save_dir: Path,
    run_name: str,
    fold: int,
    epoch: int,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    args: argparse.Namespace,
):
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

    # model
    model = build_model(args, data, in_dims, target).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # wandb per fold
    run = wandb.init(
        project=args.wandb_project,
        name=f"{args.wandb_run_name or 'kfold'}-{args.model}-fold{fold:02d}",
        group=group_name,
        config={**vars(args), "fold": fold, "fold_train_n": int(train_idx.numel()), "fold_val_n": int(val_idx.numel())},
        reinit=True,
    )
    run_name = run.name or run.id

    # baseline from train median (robust index already ensured upstream)
    c_med = data[target].y[train_idx].median().item()
    baseline_val = baseline_smoothl1_on_loader(val_loader, target, device, c_med, beta=args.beta)
    wandb.log({"baseline/median_c": c_med, "baseline/val_smoothl1": baseline_val}, step=0)

    save_dir = Path(args.save_dir) / (args.wandb_run_name or "kfold") / args.model / f"fold{fold:02d}"

    # training
    global_step = 0
    best_val_sl1 = float("inf")
    best_val_rmse = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seeds = 0

        pbar = tqdm(train_loader, desc=f"Fold {fold:02d} Epoch {epoch:03d} [{args.model}/train]")
        for batch in pbar:
            batch = batch.to(device)
            pred_all = model(batch)["pred"]
            y_all = batch[target].y.float()

            bs = int(batch[target].batch_size)
            p = pred_all[:bs]
            t = y_all[:bs]

            loss = F.smooth_l1_loss(p, t, beta=args.beta, reduction="mean")

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            opt.step()

            total_loss += loss.item() * bs
            total_seeds += bs
            global_step += 1

            # tqdm postfix
            avg_loss = total_loss / max(total_seeds, 1)
            postfix = {"loss": f"{avg_loss:.5f}"}

            if args.show_batch_stats:
                ps = batch_stats_1d(p)
                ts = batch_stats_1d(t)
                postfix.update(
                    {
                        "p_mu": f"{ps['mean']:.2f}",
                        "p_sd": f"{ps['std']:.2f}",
                        "t_mu": f"{ts['mean']:.2f}",
                        "t_sd": f"{ts['std']:.2f}",
                    }
                )

            pbar.set_postfix(postfix)

            if args.nan_check:
                check_tensor("pred", p)
                check_tensor("target", t)

            # wandb step-level (optional)
            if args.wandb_log_every > 0 and (global_step % args.wandb_log_every == 0):
                wandb.log(
                    {
                        "train/step_smoothl1": float(loss.item()),
                        "train/step_avg_smoothl1": float(avg_loss),
                        "train/global_step": global_step,
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

        train_loss = total_loss / max(total_seeds, 1)

        # epoch-level eval
        val_metrics = eval_loader_smoothl1_mae_rmse(model, val_loader, target, device, beta=args.beta)

        # log per epoch
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
            step=global_step,
        )

        val_sl1 = float(val_metrics["smoothl1"])
        val_rmse = float(val_metrics["rmse"])

        print(
            f"[fold {fold:02d}][{args.model}][epoch {epoch:03d}] "
            f"train_smoothl1={train_loss:.6f}  "
            f"val_smoothl1={val_sl1:.6f}  val_rmse={val_rmse:.6f}"
        )

        improved = val_sl1 < best_val_sl1
        if improved:
            best_val_sl1 = val_sl1
            best_val_rmse = min(best_val_rmse, val_rmse)
            best_epoch = epoch
            if args.save_best:
                save_checkpoint(save_dir, run_name, fold, epoch, model, opt, args)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(save_dir, run_name, fold, epoch, model, opt, args)

    wandb.log(
        {
            "summary/best_epoch": best_epoch,
            "summary/best_val_smoothl1": best_val_sl1,
            "summary/best_val_rmse": best_val_rmse,
            "summary/baseline_val_smoothl1": baseline_val,
        },
        step=global_step,
    )
    wandb.finish()

    return {
        "fold": fold,
        "model": args.model,
        "best_epoch": best_epoch,
        "best_val_smoothl1": best_val_sl1,
        "best_val_rmse": best_val_rmse,
        "baseline_val_smoothl1": baseline_val,
    }


# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_path", type=str, required=True, help="Path to torch-saved HeteroData (.pt)")
    parser.add_argument("--target", type=str, default="assignments")
    parser.add_argument("--label_key", type=str, default="y")

    # model selection
    parser.add_argument("--model", type=str, default="hgt", choices=["sage", "hgt", "rgcn"])

    # kfold
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--seeds_file", type=str, default=None)

    # loader
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--num_neighbors", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)

    # model hparams
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)

    # HGT-specific
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hgt_group", type=str, default="sum", choices=["sum", "mean", "min", "max"])

    # train
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--nan_check", action="store_true")
    parser.add_argument("--show_batch_stats", action="store_true")

    # wandb
    parser.add_argument("--wandb_project", type=str, default="kfold")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_log_every", type=int, default=10, help="log train step every N steps (0=off)")

    # checkpoint
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--save_best", action="store_true")

    # device
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"[info] device={device} | model={args.model}")

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"data not found: {data_path}")

    data: HeteroData = torch.load(data_path, map_location="cpu", weights_only=False)
    if not isinstance(data, HeteroData):
        raise TypeError(f"Loaded object is not HeteroData: {type(data)}")

    target = args.target
    if target not in data.node_types:
        raise ValueError(f"target node type {target!r} not in node_types={data.node_types}")

    # map label_key -> y
    if args.label_key != "y":
        if not hasattr(data[target], args.label_key):
            raise ValueError(f"target store has no attribute {args.label_key!r}")
        data[target].y = getattr(data[target], args.label_key)

    # -------------------------
    # IMPORTANT: supervised index must align with y length, not num_nodes
    # This fixes your out-of-bounds issue.
    # -------------------------
    y = data[target].y
    if not isinstance(y, torch.Tensor):
        raise TypeError(f"{target}.y is not a Tensor")
    y = y.view(-1).float()
    data[target].y = y

    finite = torch.isfinite(y)
    if finite.all():
        full_idx = torch.arange(int(y.numel()), dtype=torch.long)
    else:
        full_idx = torch.nonzero(finite, as_tuple=False).view(-1).long()
        print(f"[warn] {target}.y has non-finite values; using labeled subset: {full_idx.numel()}/{y.numel()}")

    # multiple seeds
    seed_list = parse_seeds_arg(args.seeds, args.seeds_file)
    if seed_list is None:
        seed_list = torch.tensor([args.seed], dtype=torch.long)

    all_summaries: list[dict] = []
    for si, seed in enumerate(seed_list.tolist()):
        print(f"\n[seed {seed}] ({si+1}/{seed_list.numel()})")
        folds = make_kfold_splits(full_idx, k=args.k, seed=int(seed))
        group_name = f"{args.wandb_run_name or 'kfold'}-{args.model}-seed{int(seed)}"

        for fold, val_idx in enumerate(folds):
            train_idx = complement(full_idx, val_idx)
            summary = run_one_fold(
                base_data=data,
                target=target,
                train_idx=train_idx,
                val_idx=val_idx,
                args=args,
                device=device,
                fold=fold,
                group_name=group_name,
            )
            summary["seed"] = int(seed)
            all_summaries.append(summary)

    out_dir = Path(args.save_dir) / (args.wandb_run_name or "kfold") / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kfold_summary.json"
    out_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"\n[ok] wrote summary -> {out_path}")


if __name__ == "__main__":
    main()
