import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class PatchGraphGATv2(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=256, num_layers=2, heads=4):
        super().__init__()
        assert num_layers >= 2, "PatchGraphGATv2 requires at least 2 layers"

        self.gnn_layers = nn.ModuleList()

        # First layer: [in_channels → hidden_channels * heads]
        self.gnn_layers.append(
            GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True)
        )

        # Intermediate layers: [hidden_channels * heads → hidden_channels * heads]
        for _ in range(num_layers - 2):
            self.gnn_layers.append(
                GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
            )

        # Final layer: [hidden_channels * heads → out_channels (e.g., 256)]
        self.gnn_layers.append(
            GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False)
        )

    def forward(self, x, edge_index):
        """
        Args:
            x (Tensor): [N, in_channels]
            edge_index (Tensor): [2, E] graph edges
        Returns:
            Tensor: [N, out_channels] → expected 256-dim
        """
        for layer in self.gnn_layers[:-1]:
            x = F.relu(layer(x, edge_index))
        x = self.gnn_layers[-1](x, edge_index)
        return x