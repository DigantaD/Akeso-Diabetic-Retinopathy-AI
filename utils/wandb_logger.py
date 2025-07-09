import wandb
from PIL import Image, ImageDraw
import numpy as np
import torch
import os

def _sanitize_for_wandb(config_dict):
    sanitized = {}
    for k, v in config_dict.items():
        if isinstance(v, dict):
            sanitized[k] = _sanitize_for_wandb(v)
        else:
            sanitized[k] = v
    return sanitized

class WandBLogger:
    def __init__(self, task_name: str, cfg: dict, mode: str = "train"):
        self.task = task_name.lower()
        self.mode = mode
        self.cfg = cfg
        self.enabled = True
        self.artifact_logged = False

        sanitized_cfg = _sanitize_for_wandb(cfg)

        # Force wandb.init — raise error if it fails
        self.run = wandb.init(
            project="akeso-eyecare",
            name=f"{self.task.capitalize()}-{mode}",
            config=sanitized_cfg,
            reinit=True,
            settings=wandb.Settings(start_method="thread", _disable_stats=True)
        )

    def log_metrics(self, metrics: dict, step: int = None):
        if self.run:
            self.run.log(metrics, step=step)

    def log_model_artifact(self, path: str, name: str = None, alias: str = "best"):
        if self.run and not self.artifact_logged and os.path.exists(path):
            name = name or f"{self.task}_model"
            artifact = wandb.Artifact(name, type="model")
            artifact.add_file(path)
            self.run.log_artifact(artifact, aliases=[alias])
            self.artifact_logged = True

    def log_grading_predictions(self, images, preds, labels, class_map=None, key="grading_examples"):
        if not self.run:
            return
        panels = []
        for img, pred, gt in zip(images, preds, labels):
            caption = f"Pred: {class_map.get(pred, pred) if class_map else pred} | GT: {class_map.get(gt, gt) if class_map else gt}"
            panels.append(wandb.Image(self._prepare_image(img), caption=caption))
        self.run.log({key: panels})

    def log_segmentation_masks(self, images, preds, gts, class_labels=None, step=None, key="segmentation_examples"):
        if not self.run:
            return
        panels = []
        for img, pred, gt in zip(images, preds, gts):
            img_np = self._prepare_image(img)
            gt_overlay = self._overlay_mask(img_np, gt, color=(0, 255, 0))
            pred_overlay = self._overlay_mask(img_np, pred, color=(255, 0, 0))
            stacked = np.concatenate([gt_overlay, pred_overlay], axis=1)
            caption = "GT (Green) vs Prediction (Red)"
            if class_labels:
                caption += f" | Labels: {class_labels}"
            panels.append(wandb.Image(stacked, caption=caption))
        self.run.log({key: panels}, step=step)

    def log_localization_points(self, images, gt_points, pred_points, key="localization_examples"):
        if not self.run:
            return
        panels = []
        for img, gt, pred in zip(images, gt_points, pred_points):
            img_np = self._prepare_image(img)
            vis_img = self._draw_points(img_np, gt, pred)
            panels.append(wandb.Image(vis_img, caption="GT: Green, Pred: Red"))
        self.run.log({key: panels})

    def finish(self):
        if self.run:
            wandb.finish()

    # === Utility Functions ===

    def _prepare_image(self, img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1) * 255
        return img.astype(np.uint8)

    def _overlay_mask(self, image, mask, color=(255, 0, 0), alpha=0.5):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        if mask.ndim == 3:
            mask = np.max(mask, axis=0)
        mask = (mask > 0.5).astype(np.uint8)
        h, w, _ = image.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        color_mask[:] = color
        mask_3c = np.stack([mask] * 3, axis=-1)
        blended = np.where(mask_3c, ((1 - alpha) * image + alpha * color_mask).astype(np.uint8), image)
        return blended

    def _draw_points(self, img, gt_points, pred_points):
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        for x, y in gt_points:
            draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], fill=(0, 255, 0))
        for x, y in pred_points:
            draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], fill=(255, 0, 0))
        return np.array(pil_img)