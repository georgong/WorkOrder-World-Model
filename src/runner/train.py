# train_sampling_mps_full_wandb.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.loader import NeighborLoader

import wandb

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
    """
    Returns 1D LongTensor of unique seed indices (sorted).
    Supports:
      --seeds "1,2,3"
      --seeds "1-10,20,33-35"
      --seeds_file path.txt  (one int per line, allow comma/range too)
    """
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

    # uniq + sort
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

@torch.no_grad()
def batch_stats_1d(v: torch.Tensor) -> Dict[str, float]:
    # v: 1D tensor
    v = v.detach()
    if v.numel() <= 1:
        return {"mean": v.mean().item(), "var": 0.0, "std": 0.0}
    # unbiased=False 更稳定，跟你 normalize 用法一致
    var = v.var(unbiased=False).item()
    return {"mean": v.mean().item(), "var": var, "std": (var ** 0.5)}

def check_tensor(name: str, t: torch.Tensor) -> None:
    if not isinstance(t, torch.Tensor):
        return
    if torch.isnan(t).any():
        raise ValueError(f"[NaN DETECTED] {name} contains NaN")
    if torch.isinf(t).any():
        raise ValueError(f"[INF DETECTED] {name} contains Inf")


# -------------------------
# target degree filter + split
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


def split_indices(idx: torch.Tensor, *, seed: int, train_ratio: float, val_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
# normalization (per node type)
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
            # NOTE: this only works if attr_name matches original feature dim
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


# -------------------------
# model
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
def eval_loader(model: nn.Module, loader: NeighborLoader, target: str, device: torch.device) -> Dict[str, float]:
    model.eval()
    se_sum = 0.0
    ae_sum = 0.0
    n = 0

    for batch in tqdm(loader):
        batch = batch.to(device)
        pred = model(batch)["pred"]
        y = batch[target].y.float()

        bs = int(batch[target].batch_size)
        p = pred[:bs]
        t = y[:bs]

        se_sum += F.mse_loss(p, t, reduction="sum").item()
        ae_sum += F.l1_loss(p, t, reduction="sum").item()
        n += bs

    if n == 0:
        return {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan")}

    mse = se_sum / n
    mae = ae_sum / n
    rmse = mse ** 0.5
    return {"mse": mse, "mae": mae, "rmse": rmse}


@torch.no_grad()
def baseline_smoothl1_on_loader(loader, target: str, device, c: float, beta: float = 1.0) -> float:
    total = 0.0
    n = 0
    for batch in tqdm(loader, leave=False):
        batch = batch.to(device)
        y = batch[target].y.float()
        bs = int(batch[target].batch_size)
        y = y[:bs]
        pred = torch.full_like(y, float(c))
        loss = F.smooth_l1_loss(pred, y, beta=beta, reduction="sum").item()
        total += loss
        n += bs
    return total / max(n, 1)


def save_checkpoint(save_dir: Path, run_name: str, epoch: int, model: nn.Module, opt: torch.optim.Optimizer, args: argparse.Namespace):
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = save_dir / f"{run_name}_epoch{epoch:03d}.pt"
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "args": vars(args),
    }
    torch.save(payload, ckpt_path)

    # also save args as json once
    args_path = save_dir / f"{run_name}_config.json"
    if not args_path.exists():
        args_path.write_text(json.dumps(vars(args), indent=2))

    print(f"[ok] saved checkpoint -> {ckpt_path}")


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=str, default="data/graph/sdge.pt")
    ap.add_argument("--target", type=str, default="assignments")

    ap.add_argument("--epochs", type=int, default=2)  # <<< only 2 by default
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")

    ap.add_argument("--min_degree", type=int, default=1)
    ap.add_argument("--degree_mode", type=str, default="in", choices=["in", "out", "inout"])
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--num_neighbors", type=int, default=10)

    # wandb / saving
    ap.add_argument("--wandb_project", type=str, default="scheduling_world_model")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--save_dir", type=str, default="runs/checkpoints")
    ap.add_argument("--save_every", type=int, default=1)  # save each epoch
    ap.add_argument("--seeds", type=str, default=None,
                    help='Comma/range list in target index space, e.g. "1,2,10-20"')
    ap.add_argument("--seeds_file", type=str, default=None,
                    help="Text file of seed indices (one per line, allow commas/ranges)")
    ap.add_argument("--seeds_use_splits", action="store_true",
                    help="If set: apply train/val/test split on provided seeds. Otherwise use all as train.")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    print(f"[info] device={device}")

    pt_path = Path(args.pt)
    assert pt_path.exists(), f"File not found: {pt_path}"

    # --- init wandb
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )
    run_name = run.name or run.id

    save_dir = Path(args.save_dir)

    # 1) load CPU
    data: HeteroData = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert isinstance(data, HeteroData)

    target = args.target
    assert target in data.node_types
    assert hasattr(data[target], "x")
    assert hasattr(data[target], "y")

    # 2) sanity check
    for nt in data.node_types:
        if hasattr(data[nt], "x"):
            check_tensor(f"{nt}.x", data[nt].x)
        if hasattr(data[nt], "y"):
            check_tensor(f"{nt}.y", data[nt].y)

    # 3) cast y float
    data[target].y = data[target].y.float()

    # 4) filter seeds by degree
    deg = compute_target_degree(data, target, degree_mode=args.degree_mode)

    user_seeds = parse_seeds_arg(args.seeds, args.seeds_file)  # target index space
    if user_seeds is not None:
        # sanity: bounds
        if user_seeds.min().item() < 0 or user_seeds.max().item() >= data[target].num_nodes:
            raise ValueError(
                f"--seeds out of range for target '{target}': "
                f"[0, {data[target].num_nodes - 1}], got min={user_seeds.min().item()} max={user_seeds.max().item()}"
            )

        kept = user_seeds

        # optionally also apply min_degree filter on top (keeps your original constraint)
        if args.min_degree is not None and args.min_degree > 0:
            kept = kept[deg[kept] >= args.min_degree]

        if kept.numel() == 0:
            raise ValueError("No seeds left after applying --seeds (and optional degree filter).")

        if args.seeds_use_splits:
            train_idx, val_idx, test_idx = split_indices(
                kept, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio
            )
        else:
            train_idx = kept
            val_idx = kept[:0]
            test_idx = kept[:0]

        print(f"[info] using user seeds: kept={kept.numel()} train={train_idx.numel()} val={val_idx.numel()} test={test_idx.numel()}")

    else:
        kept = (deg >= args.min_degree).nonzero(as_tuple=False).view(-1)
        if kept.numel() == 0:
            raise ValueError(f"No target nodes left after filtering: min_degree={args.min_degree}, mode={args.degree_mode}")

        train_idx, val_idx, test_idx = split_indices(
            kept, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio
        )

        print(
            f"[info] target={target} total={data[target].num_nodes} kept={kept.numel()} "
            f"train={train_idx.numel()} val={val_idx.numel()} test={test_idx.numel()}"
        )

    print(f"[info] target={target} total={data[target].num_nodes} kept={kept.numel()} "
          f"train={train_idx.numel()} val={val_idx.numel()} test={test_idx.numel()}")

    # 5) normalize node features
    normalize_node_features_inplace(data, drop_const=True)

    # 6) ensure all node types have x
    in_dims = {nt: data[nt].x.size(-1) for nt in data.node_types if hasattr(data[nt], "x")}
    for nt in data.node_types:
        if nt not in in_dims:
            in_dims[nt] = 1
            data[nt].x = torch.zeros((data[nt].num_nodes, 1), dtype=torch.float)

    # 7) model
    model = HeteroSAGERegressor(
        metadata=data.metadata(),
        in_dims=in_dims,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        target_node_type=target,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # 8) loaders
    num_neighbors = {et: [args.num_neighbors] * args.layers for et in data.edge_types}
    data = sanitize_for_neighbor_loader(data)

    train_loader = NeighborLoader(
        data,
        input_nodes=(target, train_idx),
        num_neighbors=num_neighbors,
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_train_loader = NeighborLoader(
        data,
        input_nodes=(target, train_idx),
        num_neighbors=num_neighbors,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )
    val_loader = NeighborLoader(
        data,
        input_nodes=(target, val_idx),
        num_neighbors=num_neighbors,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )
    test_loader = NeighborLoader(
        data,
        input_nodes=(target, test_idx),
        num_neighbors=num_neighbors,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )

    # 9) smarter static baseline: median (for SmoothL1)
    train_y = data[target].y[train_idx].float()
    c_med = train_y.median().item()
    baseline_med = baseline_smoothl1_on_loader(train_loader, target, device, c_med, beta=1.0)
    print(f"[baseline] median c={c_med:.6f} SmoothL1={baseline_med:.6f}")
    wandb.log({"baseline/median_c": c_med, "baseline/smoothl1": baseline_med}, step=0)

    # 10) training
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seeds = 0

        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [train]")
        for batch in tqdm_bar:
            batch = batch.to(device)
            pred = model(batch)["pred"]
            y = batch[target].y.float()
            bs = int(batch[target].batch_size)
            p = pred[:bs]
            t = y[:bs]

            # variance within this batch
            p_stats = batch_stats_1d(p)
            t_stats = batch_stats_1d(t)

            # ratio is a quick sanity check: pred variance vs label variance
            var_ratio = p_stats["var"] / (t_stats["var"] + 1e-12)

            loss = F.smooth_l1_loss(pred[:bs], y[:bs], beta=1.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * bs
            total_seeds += bs
            avg_loss = total_loss / max(total_seeds, 1)

            tqdm_bar.set_postfix(batch_loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}")

            # log per step (you can downsample if it’s too chatty)
            wandb.log(
                {"train/batch_loss": loss.item(), 
                 "train/avg_loss": avg_loss,
                 "train/pred_std": p_stats["std"],
                 "train/y_mean": t_stats["mean"],
                 "train/y_var": t_stats["var"],
                 "train/y_std": t_stats["std"],
                 "train/var_ratio_pred_to_y": var_ratio
                },
                step=global_step,
            )
            global_step += 1

        train_loss = total_loss / max(total_seeds, 1)
        if args.save_every > 0 and (epoch % args.save_every == 0):
            save_checkpoint(save_dir, run_name, epoch, model, opt, args)

        #tr = eval_loader(model, eval_train_loader, target, device)
        va = eval_loader(model, val_loader, target, device)
        te = eval_loader(model, test_loader, target, device)

        print(
            f"Epoch {epoch:03d} | train_loss {train_loss:.6f} | "
            f"val rmse {va['rmse']:.4f} mae {va['mae']:.4f} | "
            f"test rmse {te['rmse']:.4f} mae {te['mae']:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "metrics/val_rmse": va["rmse"],
                "metrics/val_mae": va["mae"],
                "metrics/test_rmse": te["rmse"],
                "metrics/test_mae": te["mae"],
            },
            step=global_step,
        )

        # save weights


        if device.type == "mps":
            torch.mps.empty_cache()

    # final save
    save_checkpoint(save_dir, run_name, args.epochs, model, opt, args)
    wandb.finish()


if __name__ == "__main__":
    main()