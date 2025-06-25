# train_segmentation.py

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from vision_etl.dataset_factory import get_idrid_dataset
from models.segmenter import AkesoSegmentationModel
from training.losses import weighted_dice_loss, multilabel_bce_loss, focal_loss
from utils.metrics import compute_segmentation_metrics
from training.scheduler import get_scheduler
from utils.postprocess import save_predictions_to_disk
from utils.wandb_logger import WandBLogger
from utils.model_uploader import upload_to_s3

tqdm_pos = int(os.getenv("TQDM_POS", 1))
CLASS_NAMES = ["MA", "HE", "EX", "SE", "OD"]

with open("config/segmentation_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, optimizer, scaler=None):
    model.train()
    epoch_loss = 0
    for batch in tqdm(loader, desc="[SEGMENTATION] Training", position=tqdm_pos):
        images = batch["image"].to(device)
        masks = batch["segmentation_mask"].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            class_weights = torch.tensor([4.5, 4.5, 2.8, 1.8, 1.4], device=logits.device)
            loss_dice = weighted_dice_loss(logits, masks, weights=class_weights)
            loss_focal = focal_loss(logits, masks, gamma=2.0)
            loss_bce = multilabel_bce_loss(logits, masks)
            loss = (
                cfg["loss_weights"]["dice"] * loss_dice +
                cfg["loss_weights"]["bce"] * (0.7 * loss_bce + 0.3 * loss_focal)
            )

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)

def evaluate(model, loader, epoch=0, log_masks=False, logger=None):
    model.eval()
    dice_total = torch.zeros(5).to(device)
    iou_total = torch.zeros(5).to(device)
    images_to_log = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="[SEGMENTATION] Validating", position=tqdm_pos)):
            images = batch["image"].to(device)
            masks = batch["segmentation_mask"].to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            metrics = compute_segmentation_metrics(probs, masks, threshold=0.5)

            dice_total += metrics["dice"]
            iou_total += metrics["iou"]

            if log_masks and batch_idx == 0 and logger:
                pred_bin = (probs > 0.5).float().cpu()
                gt_bin = masks.cpu()
                logger.log_segmentation_masks(
                    images=images.cpu(),
                    preds=pred_bin,
                    gts=gt_bin,
                    class_labels=CLASS_NAMES,
                    step=epoch
                )

    dice_avg = (dice_total / len(loader)).tolist()
    iou_avg = (iou_total / len(loader)).tolist()
    return dice_avg, iou_avg

def run_training_loop(cfg):
    print(f"📦 [SEGMENTATION] Using device: {device}")
    dataset = get_idrid_dataset(tasks=["segmentation"], mode="train")["segmentation"]
    
    val_len = int(len(dataset) * cfg.get("val_split", 0.2))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 4), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 4), pin_memory=True)

    model = AkesoSegmentationModel(
        sam_ckpt_path=cfg["sam_ckpt_path"],
        model_type=cfg["sam_model_type"]
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    scheduler = get_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler() if cfg.get("amp", True) else None

    logger = WandBLogger(task_name="segmentation", cfg=cfg, mode="train")
    best_dice = 0.0
    global_step = 0  # 🔁 Monotonic global step counter

    for epoch in tqdm(range(cfg["epochs"]), position=tqdm_pos, desc="[SEGMENTATION] Epoch", leave=True):
        print(f"\n[SEGMENTATION] 🌱 Epoch {epoch+1}/{cfg['epochs']}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler)
        dice, iou = evaluate(model, val_loader, epoch=global_step, log_masks=True, logger=logger)

        print(f"[SEGMENTATION] 🔹 Train Loss: {train_loss:.4f}")
        print(f"[SEGMENTATION] 🔸 Val Dice: {[round(d, 4) for d in dice]}")
        print(f"[SEGMENTATION] 🔸 Val IoU : {[round(i, 4) for i in iou]}")

        mean_dice = sum(dice) / len(dice)
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), cfg["save_path"])
            print("[SEGMENTATION] ✅ Model checkpoint saved!")
            logger.log_model_artifact(cfg["save_path"], name="segmentation_model")

        logger.log_metrics({
            "train/loss": train_loss,
            "val/dice": mean_dice,
            "val/iou": sum(iou) / len(iou),
            "val/dice_per_class": {CLASS_NAMES[i]: d for i, d in enumerate(dice)},
            "val/iou_per_class": {CLASS_NAMES[i]: iou[i] for i in range(len(CLASS_NAMES))}
        }, step=global_step)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_loss)
        else:
            scheduler.step()

        if (epoch + 1) % cfg.get("save_interval", 10) == 0:
            save_predictions_to_disk(model, val_loader, f"outputs/epoch_{epoch+1}", device)

        global_step += 1  # 🔁 Increment global step at the end of each epoch

    logger.finish()

    upload_to_s3(cfg["save_path"], cfg["save_path"])

if __name__ == "__main__":
    run_training_loop(cfg)