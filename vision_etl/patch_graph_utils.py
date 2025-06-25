import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse

def build_patch_graph(feature_map, patch_size=16):
    """
    feature_map: B x C x H x W
    Return: node_features (B x N x C), edge_index (2 x E)
    """
    B, C, H, W = feature_map.shape
    if H % patch_size != 0 or W % patch_size != 0:
        # Automatically downgrade patch_size to smallest divisor
        for ps in [16, 8, 4, 2, 1]:
            if H % ps == 0 and W % ps == 0:
                patch_size = ps
                break
        else:
            raise ValueError(f"No valid patch size found for feature map shape: {H}x{W}")

    ph, pw = patch_size, patch_size
    gh, gw = H // ph, W // pw
    num_nodes = gh * gw

    node_features = []
    edge_indices = []

    for b in range(B):
        fmap = feature_map[b]  # C x H x W
        patches = fmap.unfold(1, ph, ph).unfold(2, pw, pw)  # C x gh x gw x ph x pw
        patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, C, ph * pw)
        pooled = patches.mean(dim=2)  # N x C

        node_features.append(pooled)

        # Build grid adjacency
        adj = torch.zeros((num_nodes, num_nodes))
        for i in range(gh):
            for j in range(gw):
                idx = i * gw + j
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:  # 4-neighbors
                    ni, nj = i+dy, j+dx
                    if 0 <= ni < gh and 0 <= nj < gw:
                        nidx = ni * gw + nj
                        adj[idx, nidx] = 1

        edge_index = dense_to_sparse(adj)[0]
        edge_indices.append(edge_index)

    node_features = torch.stack(node_features)  # B x N x C
    return node_features, edge_indices  # edge_indices is list of len B