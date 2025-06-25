import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class DecoderUNetPlusPlus(nn.Module):
    def __init__(self, in_chs=[256, 512, 1024], feat_keys=["feat3", "feat6", "feat9"], gnn_ch=256, out_ch=5):
        super().__init__()
        self.feat_keys = feat_keys

        self.conv_low = ConvBlock(in_chs[0], 256)    # e.g., feat3 or feat1
        self.conv_mid = ConvBlock(in_chs[1], 512)    # e.g., feat6 or feat3
        self.conv_high = ConvBlock(in_chs[2], 1024)  # e.g., feat9 or feat5

        self.up1 = UpBlock(1024 + 512, 512)
        self.up2 = UpBlock(512 + 256, 256)

        self.gnn_fuse = nn.Conv2d(64, 256, kernel_size=1)

        self.final = nn.Sequential(
            nn.Conv2d(256 + 256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, kernel_size=1)
        )

    def forward(self, feats, gnn_feat_map):
        f_low = self.conv_low(feats[self.feat_keys[0]])
        f_mid = self.conv_mid(feats[self.feat_keys[1]])
        f_high = self.conv_high(feats[self.feat_keys[2]])

        u1 = self.up1(f_high, f_mid)
        u2 = self.up2(u1, f_low)

        gnn_fused = self.gnn_fuse(gnn_feat_map)
        if gnn_fused.size()[2:] != u2.size()[2:]:
            gnn_fused = F.interpolate(gnn_fused, size=u2.shape[2:], mode='bilinear', align_corners=True)

        concat = torch.cat([u2, gnn_fused], dim=1)
        out = self.final(concat)
        return out