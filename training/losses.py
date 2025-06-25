import torch
import torch.nn as nn
import torch.nn.functional as F

# Existing segmentation losses (unchanged)
def weighted_dice_loss(logits, targets, weights=None, smooth=1e-6):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    if weights is not None:
        dice = dice * weights.view(1, -1)
    return (1 - dice).mean()

def multilabel_bce_loss(logits, targets):
    bce = nn.BCEWithLogitsLoss()
    return bce(logits, targets)

def focal_loss(pred, target, alpha=0.8, gamma=2.0, reduction="mean"):
    pred = torch.sigmoid(pred)
    pt = pred * target + (1 - pred) * (1 - target)
    weight = alpha * target + (1 - alpha) * (1 - target)
    loss = -weight * (1 - pt) ** gamma * torch.log(pt + 1e-8)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

# 🔧 New: Localization Loss
def localization_loss(
    od_pred, fovea_pred, target_coords,
    gnn_embed=None, clip_img=None, clip_txt=None,
    λ_align=1.0, λ_clip=0.5,
    align_loss_type="cosine"
):
    """
    - od_pred, fovea_pred: B x 2
    - target_coords: B x 2 x 2 (2 points: OD and Fovea), normalized to [0,1]
    - gnn_embed: B x D
    - clip_img: B x D
    - clip_txt: B x D
    """
    # Base Regression Loss (Smooth L1)
    loss_fn = nn.SmoothL1Loss()
    reg_loss = loss_fn(od_pred, target_coords[:, 0]) + loss_fn(fovea_pred, target_coords[:, 1])

    align_loss = 0.0
    if gnn_embed is not None and clip_img is not None:
        if align_loss_type == "cosine":
            align_loss = 1 - F.cosine_similarity(gnn_embed, clip_img, dim=1).mean()
        else:  # fallback to MSE
            align_loss = F.mse_loss(gnn_embed, clip_img)

    clip_loss = 0.0
    if clip_txt is not None:
        # Contrastive loss between image and text (NT-Xent style)
        img_norm = F.normalize(clip_img, dim=1)
        txt_norm = F.normalize(clip_txt, dim=1)
        logits = img_norm @ txt_norm.T  # B x B
        labels = torch.arange(logits.size(0), device=logits.device)
        clip_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    total_loss = reg_loss + λ_align * align_loss + λ_clip * clip_loss
    return total_loss