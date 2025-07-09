# training/train_localization.py

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from vision_etl.dataset_factory import get_idrid_dataset
from models.localization_model import LocalizationModel
from models.localization_vlm_module import VLMEmbedder
from training.losses import localization_loss
from training.scheduler import get_scheduler
from utils.wandb_logger import WandBLogger
from utils.model_uploader import upload_to_s3
from dashboard.inference.s3_model_loader import download_model_from_s3
from vision_etl.s3_loader import check_s3_key_exists
from utils.safe_collate import localization_collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tqdm_pos = int(os.getenv("TQDM_POS", 2))


def train_one_epoch(model, vlm, loader, optimizer, cfg, scaler=None):
    model.train()
    epoch_loss = 0
    norm_factor = cfg["image_size"] if isinstance(cfg["image_size"], int) else cfg["image_size"][0]

    for batch in loader:
        images = batch["image"].to(DEVICE)
        coords = batch["localization"]  # [B, 4]
        if not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, dtype=torch.float32)
        coords = coords.to(DEVICE)

        od_coords = coords[:, :2] / norm_factor
        fovea_coords = coords[:, 2:] / norm_factor
        gt = torch.cat([od_coords, fovea_coords], dim=1)  # [B, 4]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(images)
            od_pred, fovea_pred = output["od"], output["fovea"]
            gnn_embed = output["embedding"]

            captions = batch.get("caption", None)
            vlm_out = vlm(images, captions)
            vlm_img, vlm_txt = vlm_out["image_embed"], vlm_out.get("text_embed")

            loss = localization_loss(
                od_pred, fovea_pred, gt,
                gnn_embed=gnn_embed,
                clip_img=vlm_img,
                clip_txt=vlm_txt,
                λ_align=cfg["loss_weights"]["lambda_align"],
                λ_clip=cfg["loss_weights"]["lambda_clip"]
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


def evaluate(model, loader, image_size):
    model.eval()
    all_od_errors, all_fovea_errors = [], []

    norm_factor = image_size if isinstance(image_size, int) else image_size[0]

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE)
            coords = batch["localization"]  # [B, 4]
            if not isinstance(coords, torch.Tensor):
                coords = torch.tensor(coords, dtype=torch.float32)
            coords = coords.to(DEVICE)

            gt_od = coords[:, :2]
            gt_fovea = coords[:, 2:]

            output = model(images)
            pred_od = output["od"] * norm_factor
            pred_fovea = output["fovea"] * norm_factor

            od_error = torch.norm(pred_od - gt_od, dim=-1)
            fovea_error = torch.norm(pred_fovea - gt_fovea, dim=-1)

            all_od_errors.append(od_error)
            all_fovea_errors.append(fovea_error)

    avg_od_error = torch.cat(all_od_errors).mean().item()
    avg_fovea_error = torch.cat(all_fovea_errors).mean().item()
    return {"od_error": avg_od_error, "fovea_error": avg_fovea_error}


def run_training_loop(cfg):
    print(f"📦 [LOCALIZATION] Using device: {DEVICE}")
    logger = WandBLogger("localization", cfg, mode="train")

    dataset = get_idrid_dataset(
        tasks=["localization"],
        mode="train",
        localization_mode=cfg["localization_mode"]
    )["localization"]

    val_len = int(len(dataset) * cfg.get("val_split", 0.2))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=localization_collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=localization_collate_fn
    )

    model = LocalizationModel(use_vlm_head=True).to(DEVICE)
    vlm = VLMEmbedder(
        model_name=cfg["model_config"]["vlm_model"],
        pretrained=cfg["model_config"]["vlm_pretrained"]
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    scheduler = get_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler() if cfg.get("amp", True) else None

    local_path = cfg["save_path"]
    s3_key = cfg.get("model_s3_key")
    s3_bucket = "akeso-eyecare"

    if s3_key and not os.path.exists(local_path):
        if check_s3_key_exists(s3_key):
            print(f"📥 Found checkpoint in S3 ({s3_key}). Downloading...")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            download_model_from_s3(s3_bucket, s3_key, local_path)
        else:
            print("🆕 No previous weights found in S3. Starting from scratch.")

    if os.path.exists(local_path):
        print(f"📦 Loading model weights from: {local_path}")
        state = torch.load(local_path, map_location=DEVICE)
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)

    best_val_error = float("inf")

    for epoch in tqdm(range(cfg["epochs"]), position=tqdm_pos, desc="[LOCALIZATION] Epoch", leave=True):
        print(f"\n[LOCALIZATION] 🌱 Epoch {epoch + 1}/{cfg['epochs']}")

        train_loss = train_one_epoch(model, vlm, train_loader, optimizer, cfg, scaler)
        val_error = evaluate(model, val_loader, cfg["image_size"])

        print(f"[LOCALIZATION] 📉 Train Loss: {train_loss:.4f}")
        print(f"[LOCALIZATION] ✅ Val OD Error: {val_error['od_error']:.2f}px | Val Fovea Error: {val_error['fovea_error']:.2f}px")

        logger.log_metrics({
            "train/loss": train_loss,
            "val/od_error": val_error["od_error"],
            "val/fovea_error": val_error["fovea_error"]
        }, step=epoch)

        if (val_error["od_error"] + val_error["fovea_error"]) < best_val_error:
            best_val_error = val_error["od_error"] + val_error["fovea_error"]
            torch.save({"model_state_dict": model.state_dict()}, local_path)
            print("✅ New best model checkpoint saved.")
            logger.log_model_artifact(local_path, name="localization_model")

        scheduler.step(train_loss if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else None)

    logger.finish()
    upload_to_s3(local_path, s3_key or local_path)