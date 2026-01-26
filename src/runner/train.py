# baseline_train_sampling_mps.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.loader import NeighborLoader


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
    # CPU 上检查即可，别把整图搬上 MPS 只为查 NaN
    if torch.isnan(t).any():
        raise ValueError(f"[NaN DETECTED] {name} contains NaN")
    if torch.isinf(t).any():
        raise ValueError(f"[INF DETECTED] {name} contains Inf")


def compute_target_degree(
    data: HeteroData,
    target: str,
    *,
    degree_mode: str = "in",
) -> torch.Tensor:
    """
    计算 target 节点的 degree，用于过滤 seed nodes。
    - degree_mode="in": 统计所有 (src, rel, target) 的入度
    - degree_mode="out": 统计所有 (target, rel, dst) 的出度
    - degree_mode="inout": 入度+出度
    """
    N = data[target].num_nodes
    deg = torch.zeros(N, dtype=torch.long)

    for (src, rel, dst) in data.edge_types:
        ei = data[(src, rel, dst)].edge_index
        if degree_mode in ("in", "inout") and dst == target:
            # 入度：dst index 在 ei[1]
            deg += torch.bincount(ei[1], minlength=N)
        if degree_mode in ("out", "inout") and src == target:
            # 出度：src index 在 ei[0]
            deg += torch.bincount(ei[0], minlength=N)

    return deg


def split_indices(
    idx: torch.Tensor,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    在给定候选 idx（已过滤）上随机切分 train/val/test。
    """
    assert idx.ndim == 1
    N = idx.numel()
    g = torch.Generator().manual_seed(seed)
    perm = idx[torch.randperm(N, generator=g)]

    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    n_test = N - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    return train_idx, val_idx, test_idx


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
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))  # <<< sum -> mean
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in self.node_types}))

        self.out = Linear(hidden_dim, 1)

    def forward(self, data: HeteroData):
        x_dict = {nt: F.relu(self.in_proj[nt](data[nt].x)) for nt in self.node_types}

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(self.norms[i][k](v)) for k, v in x_dict.items()}

        pred = self.out(x_dict[self.target_node_type]).squeeze(-1)
        return {"pred": pred}

def sanitize_for_neighbor_loader(data: HeteroData) -> HeteroData:
    # 清理 node stores
    for nt in data.node_types:
        store = data[nt]

        # store.keys() 包含所有挂在这个 node type 上的字段
        for key in list(store.keys()):
            v = store[key]

            # NeighborLoader 只能可靠处理 Tensor（以及少量特殊类型）
            if isinstance(v, torch.Tensor):
                continue

            # 其他一律删掉（比如 list 的 node_ids，字符串 name 等）
            del store[key]

    # 清理 edge stores（以防你也塞了 list 型 edge_ids）
    for et in data.edge_types:
        store = data[et]
        for key in list(store.keys()):
            v = store[key]
            if isinstance(v, torch.Tensor):
                continue
            del store[key]

    return data

@torch.no_grad()
def eval_loader(model: nn.Module, loader: NeighborLoader, target: str, device: torch.device) -> Dict[str, float]:
    model.eval()
    se_sum = 0.0
    ae_sum = 0.0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        pred = out["pred"]
        y = batch[target].y
        if y.dtype != torch.float32:
            y = y.float()

        # seed nodes 数量：PyG 会在 batch[target] 里给 batch_size
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=str, default="data/graph/sdge.pt")
    ap.add_argument("--target", type=str, default="assignments")
    ap.add_argument("--epochs", type=int, default=10)

    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-1)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--device", type=str, default="auto")

    # sampling / filtering
    ap.add_argument("--min_degree", type=int, default=1, help="只采样 degree >= min_degree 的 target seed nodes")
    ap.add_argument("--degree_mode", type=str, default="in", choices=["in", "out", "inout"])
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--eval_batch_size", type=int, default=512)
    ap.add_argument("--num_neighbors", type=int, default=10, help="每层每种 edge type 采样的邻居数（同一个数复制到每层）")

    args = ap.parse_args()
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    if device.type != "mps":
        print(f"[warn] 你说要用 MPS，但现在 device={device}. 继续跑也行。")

    pt_path = Path(args.pt)
    assert pt_path.exists(), f"File not found: {pt_path}"

    # 1) 读数据：先上 CPU
    data: HeteroData = torch.load(pt_path, map_location="cpu", weights_only=False)
    assert isinstance(data, HeteroData), "pt 必须是 torch_geometric.data.HeteroData"

    target = args.target
    assert target in data.node_types, f"target node type {target!r} not in {data.node_types}"
    assert hasattr(data[target], "x"), f"{target}.x missing"
    assert hasattr(data[target], "y"), f"{target}.y missing"

    # 2) CPU 上 sanity check（别把整图搬上 MPS 才查）
    for nt in data.node_types:
        if hasattr(data[nt], "x"):
            check_tensor(f"{nt}.x", data[nt].x)
        if hasattr(data[nt], "y"):
            check_tensor(f"{nt}.y", data[nt].y)

    # label cast
    if data[target].y.dtype != torch.float32:
        data[target].y = data[target].y.float()

    # 3) 只采样“边够多”的 target nodes
    deg = compute_target_degree(data, target, degree_mode=args.degree_mode)
    kept = (deg >= args.min_degree).nonzero(as_tuple=False).view(-1)

    if kept.numel() == 0:
        raise ValueError(
            f"No target nodes left after filtering: degree_mode={args.degree_mode}, min_degree={args.min_degree}"
        )

    print(f"[info] target={target} total={data[target].num_nodes} kept={kept.numel()} "
          f"(min_degree={args.min_degree}, mode={args.degree_mode})")

    # 4) 在 kept 上做 train/val/test split
    train_idx, val_idx, test_idx = split_indices(
        kept, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    print(f"[info] split sizes: train={train_idx.numel()} val={val_idx.numel()} test={test_idx.numel()}")

    # 5) 建模型（仍在 CPU 上初始化，之后上 device）
    in_dims = {nt: data[nt].x.size(-1) for nt in data.node_types}
    
    model = HeteroSAGERegressor(
        metadata=data.metadata(),
        in_dims=in_dims,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        target_node_type=target,
    ).to(device)
    train_y = data[target].y[train_idx]
    y_mean = train_y.mean().item()
    # nn.init.zeros_(model.out.weight)
    # model.out.bias.data.fill_(y_mean)
    # with torch.no_grad():
    #     model.out.bias.fill_(y_mean)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # 6) Neighbor sampling loaders（data 保持在 CPU）
    num_neighbors = {et: [args.num_neighbors] * args.layers for et in data.edge_types}
    data = sanitize_for_neighbor_loader(data) 

    train_loader = NeighborLoader(
        data,
        input_nodes=(target, train_idx),
        num_neighbors=num_neighbors,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # eval loaders 不 shuffle
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

    # 7) Train loop (mini-batch)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seeds = 0
        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [train]") 
        count = 0
        for batch in tqdm_bar:
            batch = batch.to(device)

            out = model(batch)
            pred = out["pred"]
            y = batch[target].y
            if y.dtype != torch.float32:
                y = y.float()

            bs = int(batch[target].batch_size)  # seed nodes count
            loss = F.mse_loss(pred[:bs], y[:bs])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item() * bs
            total_seeds += bs
            count += 1
            tqdm_bar.set_postfix(loss=f"{loss.item():.4e}")

        train_loss = total_loss / max(total_seeds, 1)

        tr = eval_loader(model, eval_train_loader, target, device)
        va = eval_loader(model, val_loader, target, device)
        te = eval_loader(model, test_loader, target, device)

        print(
            f"Epoch {epoch:03d} | train_loss {train_loss:.6f} | "
            f"train rmse {tr['rmse']:.4f} mae {tr['mae']:.4f} | "
            f"val rmse {va['rmse']:.4f} mae {va['mae']:.4f} | "
            f"test rmse {te['rmse']:.4f} mae {te['mae']:.4f}"
        )

        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
