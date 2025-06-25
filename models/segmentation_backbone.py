import torch
import torch.nn as nn
from segment_anything import sam_model_registry

class MedSAMBackbone(nn.Module):
    def __init__(self, model_type="vit_b", checkpoint_path=None):
        super().__init__()
        assert checkpoint_path is not None, "Must provide SAM checkpoint path"

        self.model_type = model_type
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.image_encoder = self.sam.image_encoder

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        if model_type == "vit_b":
            self.return_indices = {
                "feat3": 3,
                "feat5": 5,
                "feat7": 7
            }
        elif model_type == "vit_h":
            self.return_indices = {
                "feat3": 3,
                "feat6": 6,
                "feat9": 9
            }
        else:
            raise ValueError(f"Unsupported SAM model type: {model_type}")

    def forward(self, x: torch.Tensor) -> dict:
        x = self.image_encoder.patch_embed(x)  # [B, H, W, C]
        embed_dim = self.image_encoder.patch_embed.proj.out_channels

        if x.ndim == 4 and x.shape[1] != embed_dim:
            x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        B, C, H, W = x.shape

        x_flat = x.flatten(2).transpose(1, 2).contiguous()  # [B, HW, C]

        pos_embed = self.image_encoder.pos_embed
        if pos_embed.ndim == 4:
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = torch.nn.functional.interpolate(
                pos_embed, size=(H, W), mode="bicubic", align_corners=False
            )
            pos_embed = pos_embed.flatten(2).permute(0, 2, 1).contiguous()  # [1, HW, C]

        if pos_embed.shape[1] != x_flat.shape[1]:
            raise ValueError(f"Positional embedding mismatch: {pos_embed.shape} vs {x_flat.shape}")

        x = x_flat + pos_embed
        x = x.view(B, H, W, C)

        feats = {}
        for i, blk in enumerate(self.image_encoder.blocks):
            x = blk(x)
            for name, idx in self.return_indices.items():
                if i == idx:
                    x_flat = x.view(B, -1, x.shape[-1])  # [B, HW, C]
                    feats[name] = self._to_feature_map(x_flat)

        # Don't apply norm if not present
        # If you want, you can manually normalize: x = F.layer_norm(x, x.shape[-1])
        return feats

    def _to_feature_map(self, x):
        B, HW, C = x.shape
        H = W = int(HW ** 0.5)
        return x.transpose(1, 2).contiguous().view(B, C, H, W)