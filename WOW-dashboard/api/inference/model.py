"""
Model builder utility — constructs the HeteroSAGERegressor
with the same architecture used during training.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear


class HeteroSAGERegressor(nn.Module):
    """Heterogeneous GraphSAGE regressor (mirrors src/model/gnn.py)."""

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

    def forward(self, data):
        x_dict = {nt: F.relu(self.in_proj[nt](data[nt].x)) for nt in self.node_types}

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {k: F.relu(self.norms[i][k](v)) for k, v in x_dict.items()}

        delta = self.out(x_dict[self.target_node_type]).squeeze(-1)
        pred = self.base + delta
        return {"pred": pred}


def build_model(
    metadata,
    in_dims: Dict[str, int],
    hidden_dim: int = 64,
    num_layers: int = 2,
    target: str = "assignments",
) -> HeteroSAGERegressor:
    """Build a HeteroSAGERegressor model."""
    return HeteroSAGERegressor(
        metadata=metadata,
        in_dims=in_dims,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        target_node_type=target,
    )
