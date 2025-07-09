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
        self.out_channels = 5  # ✅ Define output classes

        # --- Encoder (MedSAM) ---
        self.encoder = MedSAMBackbone(
            model_type=model_type,
            checkpoint_path=sam_ckpt_path
        )

        if model_type == "vit_h":
            self.feat_keys = ["feat3", "feat6", "feat9"]
            decoder_in_chs = [self.encoder.output_dims[k] for k in self.feat_keys]
            gnn_feat_key = "feat3"
            gnn_in_channels = self.encoder.output_dims[gnn_feat_key]
            gnn_out_channels = 256  # ✅ Explicitly define
        elif model_type == "vit_b":
            self.feat_keys = ["feat3", "feat5", "feat7"]
            decoder_in_chs = [self.encoder.output_dims[k] for k in self.feat_keys]
            gnn_feat_key = "feat5"
            gnn_in_channels = self.encoder.output_dims[gnn_feat_key]
            gnn_out_channels = 256
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.gnn_feat_key = gnn_feat_key

        # --- GNN ---
        self.gnn = PatchGraphGATv2(
            in_channels=gnn_in_channels,
            hidden_channels=128,
            out_channels=gnn_out_channels,
            num_layers=2,
            heads=4
        )

        # --- Decoder ---
        self.decoder = DecoderUNetPlusPlus(
            in_chs=decoder_in_chs,
            decoder_channels=[256, 128, 64],
            gnn_in_ch=64,     # ✅ match actual GNN output channels
            gnn_out_ch=128,   # optional — use whatever your fusion layer expects
            feat_keys=self.feat_keys,
            out_ch=self.out_channels
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)

        # GNN input
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
        logits = self.decoder(feats, gnn_feat_map)    # [B, out_ch, H/4, W/4]

        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=True)
        return logits  # [B, 5, 512, 512]

    @property
    def output_channels(self):
        return self.out_channels