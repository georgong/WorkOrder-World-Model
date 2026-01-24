from torch_geometric.nn import HeteroConv, GCNConv
import torch

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