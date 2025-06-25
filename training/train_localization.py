import os
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

# 📍 Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tqdm_pos = int(os.getenv("TQDM_POS", 2))

def train_one_epoch(model, vlm, loader, optimizer, cfg, scaler=None):
    model.train()
    epoch_loss = 0

    for batch in tqdm(loader, desc="[LOCALIZATION] Training", position=tqdm_pos):
        images = batch["image"].to(DEVICE)
        coords = batch["localization"]

        od_coords = torch.stack([coords["OD"][0], coords["OD"][1]], dim=1).float()
        fovea_coords = torch.stack([coords["Fovea"][0], coords["Fovea"][1]], dim=1).float()
        gt = torch.stack([od_coords, fovea_coords], dim=1).to(DEVICE) / cfg["image_size"][0]

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
                λ_align=cfg["lambda_align"],
                λ_clip=cfg["lambda_clip"]
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
    errors = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[LOCALIZATION] Validating", position=tqdm_pos):
            images = batch["image"].to(DEVICE)
            coords = batch["localization"]

            od_coords = torch.stack([coords["OD"][0], coords["OD"][1]], dim=1).float()
            fovea_coords = torch.stack([coords["Fovea"][0], coords["Fovea"][1]], dim=1).float()
            gt = torch.stack([od_coords, fovea_coords], dim=1).to(DEVICE)

            output = model(images)
            od_pred = output["od"] * image_size[0]
            fovea_pred = output["fovea"] * image_size[0]
            pred_coords = torch.stack([od_pred, fovea_pred], dim=1)

            dist = torch.norm(pred_coords - gt, dim=-1).mean().item()
            errors.append(dist)

    return sum(errors) / len(errors)

def run_training_loop(cfg):
    print(f"📦 [LOCALIZATION] Using device: {DEVICE}")
    logger = WandBLogger("localization", cfg, mode="train")

    dataset = get_idrid_dataset(
        tasks=["localization"],
        mode="train",
        localization_mode=cfg["localization_mode"]
    )["localization"]

    val_split = cfg.get("val_split", 0.2)
    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 4))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 4))

    model = LocalizationModel(use_vlm_head=True).to(DEVICE)
    vlm = VLMEmbedder(model_name=cfg["vlm_model"], pretrained=cfg["vlm_pretrained"]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    scheduler = get_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler() if cfg.get("amp", True) else None

    best_val_error = float("inf")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    best_ckpt_path = os.path.join(cfg["output_dir"], cfg["checkpoint_name"])

    for epoch in tqdm(range(cfg["epochs"]), position=tqdm_pos, desc="[LOCALIZATION] Epoch", leave=True):
        print(f"\n[LOCALIZATION] 🌱 Epoch {epoch + 1}/{cfg['epochs']}")

        train_loss = train_one_epoch(model, vlm, train_loader, optimizer, cfg, scaler)
        val_error = evaluate(model, val_loader, cfg["image_size"])

        print(f"[LOCALIZATION] 📉 Train Loss: {train_loss:.4f}")
        print(f"[LOCALIZATION] ✅ Val Avg Error (px): {val_error:.2f}")

        logger.log_metrics({
            "train/loss": train_loss,
            "val/error": val_error
        }, step=epoch)

        if val_error < best_val_error:
            best_val_error = val_error
            torch.save(model.state_dict(), best_ckpt_path)
            print("[LOCALIZATION] 🟢 Best model checkpoint saved!")
            logger.log_model_artifact(best_ckpt_path, name="localization_model")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_loss)
        else:
            scheduler.step()

    logger.finish()

    upload_to_s3(best_ckpt_path, best_ckpt_path)

    # 🧹 Cleanup only temporary epoch_* folders inside outputs/
    output_root = os.path.dirname(cfg["output_dir"])  # typically "outputs"
    for item in os.listdir(output_root):
        item_path = os.path.join(output_root, item)
        if os.path.isdir(item_path) and item.startswith("epoch_"):
            try:
                import shutil
                shutil.rmtree(item_path)
                print(f"[CLEANUP] Removed {item_path}")
            except Exception as e:
                print(f"⚠️ Failed to delete {item_path}: {e}")

if __name__ == "__main__":
    config = {
        "image_size": (224, 224),
        "batch_size": 8,
        "epochs": 100,
        "lr": 1e-4,
        "lambda_align": 1.0,
        "lambda_clip": 0.5,
        "localization_mode": "point",
        "output_dir": "outputs/checkpoints",
        "checkpoint_name": "localizer_od_fovea_best.pt",
        "vlm_model": "ViT-B-32",
        "vlm_pretrained": "openai",
        "amp": True,
        "val_split": 0.2,
        "num_workers": 4
    }
    run_training_loop(config)