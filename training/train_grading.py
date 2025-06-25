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

# 📍 Dynamic tqdm position for multiprocessing
tqdm_pos = int(os.getenv("TQDM_POS", 0))


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
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
            labels = batch["label"].to(device)
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
    print(f"📦 Using device: {device}")

    logger = WandBLogger("grading", cfg, mode="train")

    datasets = get_idrid_dataset(tasks=["grading"], mode="train", localization_mode="point")
    full_train_ds = datasets["grading"]

    val_split = cfg.get("val_split", 0.2)
    if cfg.get("use_val_split", True):
        val_len = int(len(full_train_ds) * val_split)
        train_len = len(full_train_ds) - val_len
        train_ds, val_ds = random_split(full_train_ds, [train_len, val_len])
    else:
        val_ds = get_idrid_dataset(tasks=["grading"], mode="test")["grading"]
        train_ds = full_train_ds

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 4))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 4))

    encoder = load_encoder(cfg["encoder"], pretrained=True, freeze=cfg["freeze_encoder"])
    model = GradingModel(encoder).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_f1 = 0
    for epoch in tqdm(range(cfg['epochs']), position=tqdm_pos, desc=f"[GRADING] Epoch", leave=True):
        print(f"\n[GRADING] Epoch {epoch+1}/{cfg['epochs']}")
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics, preds, labels = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        print(f"[GRADING] Train → Acc: {train_metrics['acc']:.4f}, F1: {train_metrics['f1_macro']:.4f}")
        print(f"[GRADING] Val   → Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1_macro']:.4f}")

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
            torch.save(model.state_dict(), cfg["save_path"])
            print(f"[GRADING] ✅ Saved new best model at F1={best_f1:.4f}")
            logger.log_model_artifact(cfg["save_path"], name="grading_model")

    print("\n📊 [GRADING] Final Classification Report:")
    print(classification_report(labels, preds, digits=4))

    logger.finish()

    upload_to_s3(cfg["save_path"], cfg["save_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/grading_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_training_loop(cfg)