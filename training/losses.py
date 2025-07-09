# === losses.py ===

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# === SEGMENTATION LOSSES ===

def weighted_dice_loss(logits, targets, weights=None, smooth=1e-6):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    if weights is not None:
        dice = dice * weights.view(1, -1)
    return (1 - dice).mean()

def multilabel_bce_loss(logits, targets, pos_weight=None):
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return bce(logits, targets)

def focal_loss(logits, targets, alpha=None, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce)
    focal = (1 - pt) ** gamma * bce
    if alpha is not None:
        alpha = alpha.view(1, -1, 1, 1)
        focal = alpha * focal
    return focal.mean()

def tversky_loss(y_pred, y_true, alpha=0.7, beta=0.3, smooth=1e-6):
    y_pred = torch.sigmoid(y_pred)
    TP = (y_pred * y_true).sum(dim=(2, 3))
    FP = ((1 - y_true) * y_pred).sum(dim=(2, 3))
    FN = (y_true * (1 - y_pred)).sum(dim=(2, 3))
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return 1 - tversky.mean()

# === LOCALIZATION LOSS ===

def localization_loss(od_pred, fovea_pred, target_coords,
                      gnn_embed=None, clip_img=None, clip_txt=None,
                      λ_align=1.0, λ_clip=1.0):
    loss_fn = nn.SmoothL1Loss()
    target_od = target_coords[:, :2]
    target_fovea = target_coords[:, 2:]

    loss = loss_fn(od_pred, target_od) + loss_fn(fovea_pred, target_fovea)

    # Optional: GNN vs CLIP image alignment
    if gnn_embed is not None and clip_img is not None:
        gnn_embed = F.normalize(gnn_embed, dim=1)
        clip_img = F.normalize(clip_img, dim=1)
        align_loss = 1 - F.cosine_similarity(gnn_embed, clip_img, dim=1).mean()
        loss += λ_align * align_loss

    # Optional: GNN vs CLIP text alignment
    if clip_txt is not None:
        clip_txt = F.normalize(clip_txt, dim=1)
        txt_align_loss = 1 - F.cosine_similarity(gnn_embed, clip_txt, dim=1).mean()
        loss += λ_clip * txt_align_loss

    return loss

# === DYNAMIC SAFE LOSSES ===

def segmentation_loss_fn(outputs, targets):
    if targets is None or torch.all(targets == 0):
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)
    bce = multilabel_bce_loss(outputs, targets)
    dice = weighted_dice_loss(outputs, targets)
    return 0.5 * bce + 0.5 * dice

def grading_loss_fn(outputs, targets):
    mask = targets != -1
    if not mask.any():
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)
    filtered_outputs = outputs[mask]
    filtered_targets = targets[mask]
    ce = nn.CrossEntropyLoss()
    return ce(filtered_outputs, filtered_targets)

def localization_loss_fn(outputs, targets):
    if targets is None or targets.min() < 0:
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)
    od_pred, fovea_pred = outputs[:, :2], outputs[:, 2:]
    return localization_loss(od_pred, fovea_pred, targets)

# === LOSS FACTORY (Optional) ===

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