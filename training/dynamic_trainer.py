import os
import torch
import yaml
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from utils.model_loader import load_pretrained_model
from utils.freeze_utils import freeze_modules, unfreeze_modules
from vision_etl.dataset_factory import get_idrid_dataset
from training.losses import grading_loss_fn, segmentation_loss_fn, localization_loss
from vision_etl.multitask_dataset import MultiTaskDataset
from utils.safe_collate import multitask_collate_fn, EmptyBatchException
from dashboard.inference.s3_model_loader import download_model_from_s3
from utils.model_uploader import upload_to_s3
from training.federated_transfer import federated_weight_transfer

def train_dynamic_model():
    with open("config/dynamic_config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=cfg["wandb_project"], name=cfg["wandb_run_name"], config=cfg)

    print("📥 Downloading latest individual model weights from S3...")
    download_model_from_s3("akeso-eyecare", "outputs/checkpoints/grading.pt", "outputs/checkpoints/grading.pt")
    download_model_from_s3("akeso-eyecare", "outputs/checkpoints/segmentor.pt", "outputs/checkpoints/segmentor.pt")
    download_model_from_s3("akeso-eyecare", "outputs/checkpoints/localizer_od_fovea_best.pt", "outputs/checkpoints/localizer_od_fovea_best.pt")

    model = load_pretrained_model(cfg).to(device)

    datasets = get_idrid_dataset(["grading", "segmentation", "localization"], mode="train")
    multi_ds = MultiTaskDataset(
        grading_ds=datasets["grading"],
        segmentation_ds=datasets["segmentation"],
        localization_ds=datasets["localization"]
    )

    # Split train/val ensuring segmentation examples are balanced
    val_split = cfg.get("val_split", 0.2)
    total_indices = list(range(len(multi_ds)))
    segmentation_indices = [i for i, id in enumerate(multi_ds.ids) if id in multi_ds.segmentation_ids]
    non_segmentation_indices = list(set(total_indices) - set(segmentation_indices))

    val_indices = segmentation_indices[:int(val_split * len(segmentation_indices))] + \
                  non_segmentation_indices[:int(val_split * len(non_segmentation_indices))]
    train_indices = list(set(total_indices) - set(val_indices))

    train_ds = Subset(multi_ds, train_indices)
    val_ds = Subset(multi_ds, val_indices)

    weights = [1.0 if multi_ds.ids[i] in multi_ds.segmentation_ids else 0.05 for i in train_indices]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, collate_fn=multitask_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=multitask_collate_fn)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"])
    )

    best_val_loss = float("inf")

    for epoch in tqdm(range(cfg["epochs"]), desc="Epoch"):
        model.train()
        running_loss = 0.0

        if epoch < cfg["warmup_epochs"]:
            freeze_modules([model.encoder] + list(model.task_heads))
            unfreeze_modules(model.router)
        else:
            unfreeze_modules([model.encoder, model.router] + list(model.task_heads))

        for batch in train_loader:
            if batch is None:
                continue
            try:
                images = batch["image"].to(device)
                targets = {
                    "grading": batch["grading"].to(device),
                    "segmentation": batch["segmentation"].to(device),
                    "localization": batch["localization"].to(device),
                }
                valid_tasks = batch["valid_tasks"]
                grading_mask = torch.tensor(valid_tasks["grading"], device=device)
                segmentation_mask = torch.tensor(valid_tasks["segmentation"], device=device)
                localization_mask = torch.tensor(valid_tasks["localization"], device=device)

                outputs, raw_task_weights = model(images, return_all=True)

                temp = max(cfg.get("min_temperature", 0.5), cfg.get("start_temperature", 5.0) * (1.0 - epoch / max(1, cfg["temperature_anneal_epochs"])))
                task_weights = torch.softmax(torch.log(raw_task_weights + 1e-8) / temp, dim=1)

                loss_g = grading_loss_fn(outputs[0][grading_mask], targets["grading"][grading_mask]) if grading_mask.any() else torch.tensor(0.0, device=device)
                loss_s = segmentation_loss_fn(outputs[1][segmentation_mask], targets["segmentation"][segmentation_mask]) if segmentation_mask.any() else torch.tensor(0.0, device=device)
                loc_out = outputs[2]
                od_pred = loc_out["od"] if isinstance(loc_out, dict) else loc_out[:, :2]
                fovea_pred = loc_out["fovea"] if isinstance(loc_out, dict) else loc_out[:, 2:]
                loss_l = localization_loss(od_pred[localization_mask], fovea_pred[localization_mask], targets["localization"][localization_mask]) if localization_mask.any() else torch.tensor(0.0, device=device)

                entropy = -(task_weights * torch.log(task_weights + 1e-8)).sum(dim=1).mean()
                loss = (
                    cfg["loss_weights"]["grading"] * loss_g +
                    cfg["loss_weights"]["segmentation"] * loss_s +
                    cfg["loss_weights"]["localization"] * loss_l +
                    cfg["loss_weights"]["entropy"] * entropy
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            except EmptyBatchException:
                print("[TRAIN] Skipping empty batch.")
                continue

        wandb.log({
            "epoch": epoch,
            "loss/train": running_loss,
            "loss/grading": loss_g.item(),
            "loss/segmentation": loss_s.item(),
            "loss/localization": loss_l.item(),
            "routing/grading": task_weights[:, 0].mean().item(),
            "routing/segmentation": task_weights[:, 1].mean().item(),
            "routing/localization": task_weights[:, 2].mean().item(),
            "routing/entropy": entropy.item(),
        })

        print(f"[ROUTER]  Epoch {epoch} Summary:")
        print(f"[ROUTER]  Train Loss        : {running_loss:.4f}")
        print(f"[ROUTER]  Grading Loss      : {loss_g.item():.4f}")
        print(f"[ROUTER]  Segmentation Loss : {loss_s.item():.4f}")
        print(f"[ROUTER]  Localization Loss : {loss_l.item():.4f}")
        print(f"[ROUTER]  Routing Weights   : [G: {task_weights[:, 0].mean():.4f}, S: {task_weights[:, 1].mean():.4f}, L: {task_weights[:, 2].mean():.4f}]")
        print(f"[ROUTER]  Entropy           : {entropy.item():.4f}")

        if running_loss < best_val_loss:
            best_val_loss = running_loss
            os.makedirs(os.path.dirname(cfg["checkpoint_path"]), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "router_state_dict": model.router.state_dict()
            }, cfg["checkpoint_path"])
            print(f"[ROUTER] ✅ Best model saved at epoch {epoch} with loss {running_loss:.4f}")

    # === Save final routing stats ===
    with torch.no_grad():
        final_task_weights = task_weights.mean(dim=0).cpu().tolist()
        final_task_entropy = -(task_weights * torch.log(task_weights + 1e-8)).sum(dim=1).mean().item()

    cycle_suffix = f"_cycle_{cfg.get('cycle_number', 'final')}"
    stats_path = os.path.join("outputs", "checkpoints", f"router_stats{cycle_suffix}.pt")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    torch.save({
        "routing_entropy": final_task_entropy,
        "task_weights": final_task_weights,
    }, stats_path)

    print(f"[ROUTER] ✅ Saved routing stats to {stats_path}")

    # Upload router model
    upload_to_s3(cfg["checkpoint_path"], cfg["checkpoint_path"])

    # Federated Transfer
    print("🧠 Running Federated Transfer...")
    federated_weight_transfer(
        grading_ckpt="outputs/checkpoints/grading.pt",
        segmentation_ckpt="outputs/checkpoints/segmentor.pt",
        localization_ckpt="outputs/checkpoints/localizer_od_fovea_best.pt",
        router_ckpt=cfg["checkpoint_path"]
    )

    print("📤 Uploading updated model weights to S3...")
    upload_to_s3("outputs/checkpoints/grading.pt", "outputs/checkpoints/grading.pt")
    upload_to_s3("outputs/checkpoints/segmentor.pt", "outputs/checkpoints/segmentor.pt")
    upload_to_s3("outputs/checkpoints/localizer_od_fovea_best.pt", "outputs/checkpoints/localizer_od_fovea_best.pt")

    wandb.finish()