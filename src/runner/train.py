# baseline_train.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear


def ensure_masks(data: HeteroData, node_type: str, seed: int = 42,
                 train_ratio: float = 0.8, val_ratio: float = 0.1) -> None:
    """
    Ensure train/val/test masks exist for a node type.
    If they exist, do nothing. Otherwise create random split.
    """
    nt = data[node_type]
    if hasattr(nt, "train_mask") and hasattr(nt, "val_mask") and hasattr(nt, "test_mask"):
        return

    assert hasattr(nt, "y"), f"{node_type} has no y target."

    N = nt.num_nodes
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)

    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    n_test = N - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    nt.train_mask = train_mask
    nt.val_mask = val_mask
    nt.test_mask = test_mask


class HeteroSAGERegressor(nn.Module):
    """
    Simple baseline:
      - per-node-type input projection -> hidden dim
      - K layers of HeteroConv with SAGEConv (mean aggregation)
      - regression head on target node type
    """
    def __init__(self, metadata: Tuple[list, list], in_dims: Dict[str, int],
                 hidden_dim: int = 128, num_layers: int = 2,
                 target_node_type: str = "assignments"):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.target_node_type = target_node_type

        # project each node type to hidden_dim
        self.in_proj = nn.ModuleDict({
            nt: Linear(in_dims[nt], hidden_dim)
            for nt in self.node_types
        })

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for et in self.edge_types:
                # et = (src, rel, dst)
                conv_dict[et] = SAGEConv((-1, -1), hidden_dim)
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        self.out = Linear(hidden_dim, 1)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = {}
        for nt in self.node_types:
            x = data[nt].x
            x_dict[nt] = F.relu(self.in_proj[nt](x))

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        pred = self.out(x_dict[self.target_node_type]).squeeze(-1)
        return {"pred": pred}


@torch.no_grad()
def eval_split(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    if mask.sum().item() == 0:
        return {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan")}
    p = pred[mask]
    t = y[mask]
    mse = F.mse_loss(p, t).item()
    mae = F.l1_loss(p, t).item()
    rmse = mse ** 0.5
    return {"mse": mse, "mae": mae, "rmse": rmse}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=str, default="data/graph/sdge.pt")
    ap.add_argument("--target", type=str, default="assignments")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    pt_path = Path(args.pt)
    assert pt_path.exists(), f"File not found: {pt_path}"

    data: HeteroData = torch.load(pt_path, map_location="cpu",weights_only=False)
    assert isinstance(data, HeteroData), "sdge.pt must contain a torch_geometric.data.HeteroData"

    target = args.target
    assert target in data.node_types, f"target node type {target!r} not in {data.node_types}"
    assert hasattr(data[target], "x"), f"{target}.x missing"
    assert hasattr(data[target], "y"), f"{target}.y missing (move label out of x first)"

    # masks
    ensure_masks(data, target, seed=args.seed)

    # infer per-node-type in dims
    in_dims = {nt: data[nt].x.size(-1) for nt in data.node_types}

    model = HeteroSAGERegressor(
        metadata=data.metadata(),
        in_dims=in_dims,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        target_node_type=target,
    ).to(args.device)

    data = data.to(args.device)

    # y may be float64; baseline: cast to float32 for stability
    y = data[target].y
    if y.dtype != torch.float32:
        data[target].y = y.float()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    train_mask = data[target].train_mask
    val_mask = data[target].val_mask
    test_mask = data[target].test_mask

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        out = model(data)
        pred = out["pred"]
        y = data[target].y

        loss = F.mse_loss(pred[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out["pred"]
            tr = eval_split(pred, y, train_mask)
            va = eval_split(pred, y, val_mask)
            te = eval_split(pred, y, test_mask)

        print(
            f"Epoch {epoch:03d} | loss {loss.item():.6f} | "
            f"train rmse {tr['rmse']:.4f} mae {tr['mae']:.4f} | "
            f"val rmse {va['rmse']:.4f} mae {va['mae']:.4f} | "
            f"test rmse {te['rmse']:.4f} mae {te['mae']:.4f}"
        )


if __name__ == "__main__":
    main()
