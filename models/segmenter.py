import torch
import torch.nn as nn
import torch.nn.functional as F
from models.segmentation_backbone import MedSAMBackbone
from models.seg_gnn import PatchGraphGATv2
from models.decoder import DecoderUNetPlusPlus
from vision_etl.patch_graph_utils import build_patch_graph

class AkesoSegmentationModel(nn.Module):
    def __init__(self, sam_ckpt_path: str, model_type: str = "vit_h"):
        super().__init__()

        self.model_type = model_type
        self.out_channels = 5  # ✅ Define this so it can be used below

        # --- Encoder (MedSAM) ---
        self.encoder = MedSAMBackbone(
            model_type=model_type,
            checkpoint_path=sam_ckpt_path
        )

        if model_type == "vit_h":
            self.feat_keys = ["feat3", "feat6", "feat9"]
            decoder_in_chs = [256, 512, 1024]
            gnn_feat_key = "feat3"
            gnn_in_channels = 256
        elif model_type == "vit_b":
            self.feat_keys = ["feat3", "feat5", "feat7"]
            decoder_in_chs = [768, 768, 768]
            gnn_feat_key = "feat5"
            gnn_in_channels = 768
            gnn_out_channels = 256   # ✅ This is critical
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.gnn_feat_key = gnn_feat_key

        self.gnn = PatchGraphGATv2(
            in_channels=gnn_in_channels,
            hidden_channels=128,
            out_channels=gnn_out_channels,  # ✅ Ensures compatibility with decoder
            num_layers=2,
            heads=4
        )

        self.decoder = DecoderUNetPlusPlus(
            in_chs=decoder_in_chs,
            feat_keys=self.feat_keys,
            gnn_ch=gnn_out_channels,  # ✅ Match expected input
            out_ch=self.out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape [B, 3, 512, 512]
        Returns:
            logits: Segmentation map of shape [B, 5, 512, 512]
        """
        feats = self.encoder(x)  # Extract features

        # Build patch graph on selected feature map
        gnn_feat = feats[self.gnn_feat_key]
        node_feats, edge_indices = build_patch_graph(gnn_feat)  # [B, N, C], list of [2, E]

        B, N, C = node_feats.shape
        H, W = gnn_feat.shape[2:]

        gnn_outputs = []
        for b in range(B):
            gnn_out = self.gnn(node_feats[b], edge_indices[b].to(x.device))  # [N, C]
            gnn_out = gnn_out.view(H, W, -1).permute(2, 0, 1).unsqueeze(0)   # [1, C, H, W]
            gnn_outputs.append(gnn_out)

        gnn_feat_map = torch.cat(gnn_outputs, dim=0)  # [B, 256, H, W]
        logits = self.decoder(feats, gnn_feat_map)    # e.g., [B, 5, H/4, W/4]

        # 🔧 Ensure logits match input resolution
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=True)

        return logits  # [B, 5, 512, 512]
