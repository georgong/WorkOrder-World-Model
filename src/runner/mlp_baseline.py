# src/runner/mlp_baseline_subgraph.py
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

import wandb


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
    # NeighborLoader hates non-tensors in storages
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
            layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, 1))
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
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=str, required=True)
    ap.add_argument("--target", type=str, default="assignments")

    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--num_neighbors", type=int, default=3)

    # sane defaults
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--eval_batch_size", type=int, default=8192)

    ap.add_argument("--min_degree", type=int, default=1)
    ap.add_argument("--degree_mode", type=str, default="in")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    # MLP hyperparams
    ap.add_argument("--mlp_hidden", type=int, default=256)
    ap.add_argument("--mlp_depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)

    # optim
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--loss", type=str, default="smoothl1", choices=["smoothl1", "mse", "l1"])
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--grad_clip", type=float, default=5.0)

    # logging
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--wandb_project", type=str, default="scheduling_world_model")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--no_wandb", action="store_true")

    args = ap.parse_args()
    device = pick_device(args.device)
    print(f"[info] device={device}")

    torch.manual_seed(int(args.seed))

    data: HeteroData = torch.load(Path(args.pt), map_location="cpu", weights_only=False)
    target = args.target
    data[target].y = data[target].y.float()

    neighbor_types = ["engineers", "tasks", "task_types", "districts"]

    in_dims = ensure_all_node_types_have_x(data)
    data = sanitize_for_neighbor_loader(data)

    # split by degree
    deg = compute_target_degree(data, target, degree_mode=args.degree_mode)
    kept = (deg >= args.min_degree).nonzero(as_tuple=False).view(-1)
    train_idx, val_idx, test_idx = split_indices(kept, int(args.seed), float(args.train_ratio), float(args.val_ratio))
    print(f"[split] train={train_idx.numel()} val={val_idx.numel()} test={test_idx.numel()}")

    num_neighbors = {et: [int(args.num_neighbors)] * int(args.layers) for et in data.edge_types}

    def make_loader(idx, bs, shuffle):
        return NeighborLoader(
            data,
            input_nodes=(target, idx),
            num_neighbors=num_neighbors,
            batch_size=int(bs),
            shuffle=bool(shuffle),
        )

    train_loader = make_loader(train_idx, args.batch_size, True)
    val_loader = make_loader(val_idx, args.eval_batch_size, False)
    test_loader = make_loader(test_idx, args.eval_batch_size, False)

    # infer feature dim
    probe_n = min(1000, int(train_idx.numel()))
    probe_bs = min(1024, int(args.batch_size))
    first = next(iter(make_loader(train_idx[:probe_n], probe_bs, False))).to(device)
    X0, _ = batch_to_tabular_per_seed(first, target=target, neighbor_types=neighbor_types, in_dims=in_dims)
    d_in = int(X0.size(1))
    print(f"[feat] d_in={d_in} (target_x={int(first[target].x.size(1))}, neighbor_types={neighbor_types})")

    model = MLPRegressor(
        d_in=d_in,
        hidden=int(args.mlp_hidden),
        depth=int(args.mlp_depth),
        dropout=float(args.dropout),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    def loss_fn(pred, y):
        if args.loss == "smoothl1":
            return F.smooth_l1_loss(pred, y, beta=float(args.beta))
        if args.loss == "mse":
            return F.mse_loss(pred, y)
        return F.l1_loss(pred, y)

    # baseline (match your GNN logic: train median as constant)
    c_med = float(data[target].y[train_idx].median().item())
    baseline_val_sl1 = baseline_smoothl1_on_loader(val_loader, target=target, device=device, c=c_med, beta=float(args.beta))
    print(f"[baseline] c_med={c_med:.6f} val_smoothl1={baseline_val_sl1:.6f}")

    run = None
    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                **vars(args),
                "d_in": d_in,
                "neighbor_types": neighbor_types,
                "train_n": int(train_idx.numel()),
                "val_n": int(val_idx.numel()),
                "test_n": int(test_idx.numel()),
                "baseline_c_med": c_med,
                "baseline_val_smoothl1": baseline_val_sl1,
            },
        )
        wandb.log({"baseline/median_c": c_med, "baseline/val_smoothl1": baseline_val_sl1}, step=0)

    global_step = 0
    t_start = time.time()

    for ep in range(1, int(args.epochs) + 1):
        model.train()
        ep_t0 = time.time()

        batch_losses: List[float] = []
        loss_sum_by_samples = 0.0
        n_samples = 0

        pbar = tqdm(train_loader, desc=f"train ep{ep}", leave=True)
        for bi, batch in enumerate(pbar):
            step_t0 = time.time()

            batch = batch.to(device)
            X, y = batch_to_tabular_per_seed(batch, target=target, neighbor_types=neighbor_types, in_dims=in_dims)
            pred = model(X)
            loss = loss_fn(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()

            loss_val = float(loss.detach().cpu().item())
            bs = int(y.numel())
            batch_losses.append(loss_val)
            loss_sum_by_samples += loss_val * bs
            n_samples += bs

            avg_mean_loss_so_far = sum(batch_losses) / max(len(batch_losses), 1)
            avg_loss_by_samples_so_far = loss_sum_by_samples / max(n_samples, 1)

            p_stats = batch_stats_1d(pred)
            y_stats = batch_stats_1d(y)
            var_ratio = p_stats["var"] / (y_stats["var"] + 1e-12)

            step_dt = time.time() - step_t0
            pbar.set_postfix(
                loss=loss_val,
                avg_mean_loss=avg_mean_loss_so_far,
                avg_loss_by_samples=avg_loss_by_samples_so_far,
                step_s=step_dt,
            )

            if run is not None and (global_step % int(args.log_every) == 0):
                wandb.log(
                    {
                        "train/batch_loss": loss_val,
                        "train/avg_mean_loss_running": avg_mean_loss_so_far,
                        "train/avg_loss_by_samples_running": avg_loss_by_samples_so_far,
                        "train/batch_size": bs,
                        "train/pred_std": p_stats["std"],
                        "train/y_mean": y_stats["mean"],
                        "train/y_var": y_stats["var"],
                        "train/y_std": y_stats["std"],
                        "train/var_ratio_pred_to_y": var_ratio,
                        "time/step_s": step_dt,
                        "epoch": ep,
                    },
                    step=global_step,
                )

            global_step += 1

        ep_dt = time.time() - ep_t0
        avg_mean_loss = sum(batch_losses) / max(len(batch_losses), 1)
        avg_loss_by_samples = loss_sum_by_samples / max(n_samples, 1)

        va = eval_loader_smoothl1_mae_rmse(
            model, val_loader, target=target, neighbor_types=neighbor_types, in_dims=in_dims, device=device, beta=float(args.beta)
        )
        te = eval_loader_smoothl1_mae_rmse(
            model, test_loader, target=target, neighbor_types=neighbor_types, in_dims=in_dims, device=device, beta=float(args.beta)
        )

        print(
            f"[ep {ep}] "
            f"train avg_mean_loss={avg_mean_loss:.4f} avg_loss_by_samples={avg_loss_by_samples:.4f} | "
            f"val sl1={va['smoothl1']:.4f} (base {baseline_val_sl1:.4f}) rmse={va['rmse']:.4f} mae={va['mae']:.4f} mse={va['mse']:.4f} | "
            f"test sl1={te['smoothl1']:.4f} rmse={te['rmse']:.4f} mae={te['mae']:.4f} mse={te['mse']:.4f} | "
            f"epoch_s={ep_dt:.1f}"
        )

        if run is not None:
            wandb.log(
                {
                    "train/avg_mean_loss": avg_mean_loss,
                    "train/avg_loss_by_samples": avg_loss_by_samples,
                    "val/mse": va["mse"],
                    "val/mae": va["mae"],
                    "val/rmse": va["rmse"],
                    "val/smoothl1": va["smoothl1"],
                    "val/improve_over_baseline_sl1": baseline_val_sl1 - va["smoothl1"],
                    "test/mse": te["mse"],
                    "test/mae": te["mae"],
                    "test/rmse": te["rmse"],
                    "test/smoothl1": te["smoothl1"],
                    "time/epoch_s": ep_dt,
                    "time/elapsed_s": time.time() - t_start,
                    "epoch": ep,
                },
                step=global_step,
            )

        if device.type == "mps":
            torch.mps.empty_cache()

    if run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
