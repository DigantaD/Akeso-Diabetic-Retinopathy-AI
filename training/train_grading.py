import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import classification_report

from vision_etl.dataset_factory import get_idrid_dataset
from models.grading_model import GradingModel
from agents.embedding_agent import load_encoder
from utils.metrics import compute_metrics
from utils.wandb_logger import WandBLogger
from utils.model_uploader import upload_to_s3
from dashboard.inference.s3_model_loader import download_model_from_s3
from vision_etl.s3_loader import check_s3_key_exists
from utils.safe_collate import grading_collate_fn

def safe_collate(batch):
    return [b for b in batch if b is not None]

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["grading"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(loader)
    return metrics

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["grading"].to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(loader)
    return metrics, all_preds, all_labels

def run_training_loop(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\U0001F4E6 Using device: {device}")

    logger = WandBLogger("grading", cfg, mode="train")
    datasets = get_idrid_dataset(tasks=["grading"], mode="train", localization_mode="point")
    full_train_ds = datasets["grading"]

    val_split = cfg.get("val_split", 0.2)
    val_len = int(len(full_train_ds) * val_split)
    train_len = len(full_train_ds) - val_len
    train_ds, val_ds = random_split(full_train_ds, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 4), collate_fn=grading_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 4), collate_fn=grading_collate_fn)

    encoder = load_encoder(cfg["model_config"]["encoder"], pretrained=True, freeze=cfg["model_config"]["freeze_encoder"])
    model = GradingModel(encoder).to(device)

    s3_bucket = "akeso-eyecare"
    s3_key = cfg.get("model_s3_key")
    local_weight_path = cfg["save_path"]
    os.makedirs(os.path.dirname(local_weight_path), exist_ok=True)

    if s3_key and check_s3_key_exists(s3_key):
        print(f"\U0001F4E5 Found S3 checkpoint: {s3_key} — downloading...")
        download_model_from_s3(s3_bucket, s3_key, local_weight_path)
        if os.path.exists(local_weight_path):
            state = torch.load(local_weight_path, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
            print(f"✅ Loaded pretrained weights from: {local_weight_path}")
    else:
        print(f"🆕 No checkpoint found at {s3_key}. Training from scratch.")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_f1 = 0
    for epoch in tqdm(range(cfg['epochs']), desc=f"[GRADING] Epoch", leave=True):
        print(f"\n[GRADING] Epoch {epoch+1}/{cfg['epochs']}")
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics, preds, labels = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        logger.log_metrics({
            "train/loss": train_metrics["loss"],
            "train/acc": train_metrics["acc"],
            "train/f1": train_metrics["f1_macro"],
            "val/loss": val_metrics["loss"],
            "val/acc": val_metrics["acc"],
            "val/f1": val_metrics["f1_macro"],
        }, step=epoch)

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            save_dict = {"model_state_dict": model.state_dict()}
            # Only save router state if model has one
            if hasattr(model, "router"):
                save_dict["router_state_dict"] = model.router.state_dict()

            torch.save(save_dict, cfg["save_path"])
            upload_to_s3(local_weight_path, s3_key)
            print(f"[GRADING] ✅ New best model saved and uploaded @ F1={best_f1:.4f}")
            logger.log_model_artifact(local_weight_path, name="grading_model")

    print("\n📊 [GRADING] Final Classification Report:")
    print(classification_report(labels, preds, digits=4))

    logger.finish()