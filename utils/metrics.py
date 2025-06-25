from sklearn.metrics import accuracy_score, f1_score
import torch

def compute_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average='macro')
    }

def compute_segmentation_metrics(preds, targets, threshold=0.5):
    preds_bin = (preds > threshold).float()
    intersection = (preds_bin * targets).sum(dim=(2,3))
    union = preds_bin.sum(dim=(2,3)) + targets.sum(dim=(2,3)) - intersection
    dice = (2 * intersection + 1e-6) / (preds_bin.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return {
        "dice": dice.mean(dim=0),
        "iou": iou.mean(dim=0)
    }