import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ Dice Loss with optional class weights
def weighted_dice_loss(logits, targets, weights=None, smooth=1e-6):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    if weights is not None:
        dice = dice * weights.view(1, -1)
    return (1 - dice).mean()

# ✅ BCE Loss with optional class balancing
def multilabel_bce_loss(logits, targets, pos_weight=None):
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return bce(logits, targets)

# ✅ Focal Loss with default fallback and optional class weights
def focal_loss(logits, targets, alpha=None, gamma=2.0):
    """
    Multi-label focal loss for segmentation:
    - logits, targets: [B, C, H, W]
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # [B, C, H, W]
    pt = torch.exp(-bce)  # pt = 1 - bce_loss
    focal = (1 - pt) ** gamma * bce

    if alpha is not None:
        alpha = alpha.view(1, -1, 1, 1)  # [1, C, 1, 1]
        focal = alpha * focal

    return focal.mean()

# ✅ Tversky Loss (highly effective for segmentation with class imbalance)
def tversky_loss(y_pred, y_true, alpha=0.7, beta=0.3, smooth=1e-6):
    y_pred = torch.sigmoid(y_pred)
    TP = (y_pred * y_true).sum(dim=(2, 3))
    FP = ((1 - y_true) * y_pred).sum(dim=(2, 3))
    FN = (y_true * (1 - y_pred)).sum(dim=(2, 3))
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return 1 - tversky.mean()

# ✅ Localization Loss with optional alignment and contrastive terms
def localization_loss(
    od_pred, fovea_pred, target_coords,
    gnn_embed=None, clip_img=None, clip_txt=None,
    λ_align=1.0, λ_clip=0.5,
    align_loss_type="cosine"
):
    """
    - od_pred, fovea_pred: B x 2
    - target_coords: B x 2 x 2 (OD, Fovea)
    - gnn_embed: B x D
    - clip_img: B x D
    - clip_txt: B x D
    """
    # Base regression loss
    loss_fn = nn.SmoothL1Loss()
    reg_loss = loss_fn(od_pred, target_coords[:, 0]) + loss_fn(fovea_pred, target_coords[:, 1])

    align_loss = 0.0
    if gnn_embed is not None and clip_img is not None:
        if align_loss_type == "cosine":
            align_loss = 1 - F.cosine_similarity(gnn_embed, clip_img, dim=1, eps=1e-6).mean()
        else:
            align_loss = F.mse_loss(gnn_embed, clip_img)

    clip_loss = 0.0
    if clip_txt is not None and clip_img.norm(dim=1).min() > 1e-6:
        img_norm = F.normalize(clip_img, dim=1)
        txt_norm = F.normalize(clip_txt, dim=1)
        logits = img_norm @ txt_norm.T  # B x B
        labels = torch.arange(logits.size(0), device=logits.device)
        clip_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    total_loss = reg_loss + λ_align * align_loss + λ_clip * clip_loss
    return total_loss

# ✅ Optional Loss Factory
def get_loss(name, **kwargs):
    if name == "focal":
        return lambda outputs, targets: focal_loss(outputs, targets, **kwargs)
    if name == "tversky":
        return lambda outputs, targets: tversky_loss(outputs, targets, **kwargs)
    if name == "dice":
        return lambda outputs, targets: weighted_dice_loss(outputs, targets, **kwargs)
    if name == "bce":
        return lambda outputs, targets: multilabel_bce_loss(outputs, targets, **kwargs)
    raise ValueError(f"Unknown loss: {name}")