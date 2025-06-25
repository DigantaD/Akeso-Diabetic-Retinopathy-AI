import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3
from torch_geometric.nn import GATv2Conv, global_mean_pool

class PatchToGraph(nn.Module):
    def __init__(self, in_channels, patch_size=(7, 7)):
        super().__init__()
        self.patch_size = patch_size
        self.project = nn.Conv2d(384, 64, kernel_size=1)

    def forward(self, feature_map, batch_size):
        """
        feature_map: B x C x H x W
        returns:
            x: N x F (flattened node features)
            edge_index: 2 x E (adjacency indices)
            batch: N (graph batch vector)
        """
        x = self.project(feature_map)  # B x 64 x H x W
        B, C, H, W = x.shape
        nodes = x.flatten(2).permute(0, 2, 1).reshape(-1, C)  # (B*H*W) x C

        # Create 8-neighbor edges for grid
        edge_index = []
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < H and 0 <= nj < W:
                            nidx = ni * W + nj
                            edge_index.append([idx, nidx])
        edge_index = torch.tensor(edge_index).t().contiguous().to(nodes.device)
        edge_index = edge_index.repeat(1, B)
        offset = torch.arange(B, device=nodes.device) * H * W
        batch = torch.arange(B, device=nodes.device).repeat_interleave(H * W)

        edge_index += offset.repeat_interleave(edge_index.size(1) // B)

        return nodes, edge_index, batch

class LocalizationModel(nn.Module):
    def __init__(self, use_vlm_head=True, gnn_hidden=128, num_heads=4):
        super().__init__()
        self.use_vlm_head = use_vlm_head

        # Visual backbone
        self.backbone = efficientnet_b3(weights="IMAGENET1K_V1")
        self.backbone_features = nn.Sequential(*list(self.backbone.features.children())[:-1])  # B x C x H x W
        self.patch_to_graph = PatchToGraph(in_channels=1536)

        # GNN
        self.gnn1 = GATv2Conv(64, gnn_hidden, heads=num_heads, concat=True)
        self.gnn2 = GATv2Conv(gnn_hidden * num_heads, gnn_hidden, heads=1, concat=True)

        # Regressors
        self.od_head = nn.Sequential(nn.Linear(gnn_hidden, 64), nn.ReLU(), nn.Linear(64, 2))
        self.fovea_head = nn.Sequential(nn.Linear(gnn_hidden, 64), nn.ReLU(), nn.Linear(64, 2))

        # Optional: VLM alignment head
        if use_vlm_head:
            self.vlm_proj = nn.Linear(gnn_hidden, 512)  # Match CLIP-style dim

    def forward(self, images, return_embeddings=False):
        B = images.size(0)
        feats = self.backbone_features(images)  # B x C x H x W
        x, edge_index, batch = self.patch_to_graph(feats, B)

        x = F.relu(self.gnn1(x, edge_index))
        x = F.relu(self.gnn2(x, edge_index))
        pooled = global_mean_pool(x, batch)  # B x gnn_hidden

        od = self.od_head(pooled)      # B x 2
        fovea = self.fovea_head(pooled)  # B x 2

        output = {"od": od, "fovea": fovea}
        if self.use_vlm_head or return_embeddings:
            output["embedding"] = self.vlm_proj(pooled) if self.use_vlm_head else pooled
        return output