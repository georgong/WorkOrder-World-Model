from torch_geometric.nn import HeteroConv, GCNConv
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import torch.nn as nn
import torch.nn.functional as F

class HeteroGCN(torch.nn.Module):
    def __init__(self, metadata, hidden, out_dim):
        super().__init__()

        self.conv1 = HeteroConv({
            edge_type: GCNConv(-1, hidden)
            for edge_type in metadata[1]
        }, aggr="sum")

        self.conv2 = HeteroConv({
            edge_type: GCNConv(hidden, out_dim)
            for edge_type in metadata[1]
        }, aggr="sum")

    def forward(self, data):
        x_dict = self.conv1(data.x_dict, data.edge_index_dict)
        x_dict = {k: v.relu() for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        return x_dict

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