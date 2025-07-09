import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class PatchGraphGATv2(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=256, num_layers=2, heads=4):
        """
        Args:
            in_channels (int): Input feature dimension
            hidden_channels (int): GNN hidden layer size
            out_channels (int): Output node embedding dimension (expected by decoder)
            num_layers (int): Total GNN layers (≥ 2)
            heads (int): Attention heads for intermediate layers
        """
        super().__init__()
        assert num_layers >= 2, "PatchGraphGATv2 requires at least 2 layers"

        self.gnn_layers = nn.ModuleList()

        # First layer: [in_channels → hidden_channels * heads]
        self.gnn_layers.append(
            GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True)
        )

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.gnn_layers.append(
                GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
            )

        # Final layer: [hidden_channels * heads → out_channels]
        self.gnn_layers.append(
            GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False)
        )

        self.output_dim = out_channels  # For downstream use

    def forward(self, x, edge_index):
        """
        Args:
            x (Tensor): [N, in_channels] — node features
            edge_index (Tensor): [2, E] — graph edges

        Returns:
            Tensor: [N, out_channels]
        """
        for layer in self.gnn_layers[:-1]:
            x = F.relu(layer(x, edge_index))
        x = self.gnn_layers[-1](x, edge_index)
        return x