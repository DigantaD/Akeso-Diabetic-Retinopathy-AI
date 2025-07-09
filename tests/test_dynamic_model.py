# File: tests/test_dynamic_model.py
import os
import torch
import yaml
# import wandb
from torch.utils.data import DataLoader, ConcatDataset
from utils.model_loader import load_pretrained_model
from vision_etl.dataset_factory import get_idrid_dataset
from vision_etl.multitask_dataset import MultiTaskDataset
from training.losses import grading_loss_fn, segmentation_loss_fn, localization_loss_fn
from utils.metrics import (
    compute_grading_accuracy,
    compute_segmentation_metrics,
    compute_localization_error,
)
from dashboard.inference.s3_model_loader import download_model_from_s3
from tqdm import tqdm

def extract_image_ids_from_concat(ds):
    if isinstance(ds, torch.utils.data.ConcatDataset):
        all_ids = []
        for sub_ds in ds.datasets:
            if hasattr(sub_ds, "image_ids"):
                all_ids.extend(sub_ds.image_ids)
        ds.image_ids = all_ids  # inject attribute
    return ds

@torch.no_grad()
def test_dynamic_model():
    with open("config/dynamic_config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download model checkpoints
    download_model_from_s3("akeso-eyecare", cfg["model_s3_key"], cfg["save_path"])
    download_model_from_s3("akeso-eyecare", cfg["grading_ckpt"], cfg["grading_ckpt"])
    download_model_from_s3("akeso-eyecare", cfg["segmentation_ckpt"], cfg["segmentation_ckpt"])
    download_model_from_s3("akeso-eyecare", cfg["localization_ckpt"], cfg["localization_ckpt"])

    # wandb.init(project=cfg["wandb_project"], name=f"{cfg['wandb_run_name']}-eval", config=cfg)

    model = load_pretrained_model(cfg).to(device)
    model.eval()

    # Combine training and test datasets
    # Load train and test splits
    train_datasets = get_idrid_dataset(["grading", "segmentation", "localization"], mode="train")
    test_datasets = get_idrid_dataset(["grading", "segmentation", "localization"], mode="test")

    # Combine each task's dataset
    grading_ds = ConcatDataset([train_datasets["grading"], test_datasets["grading"]])
    segmentation_ds = ConcatDataset([train_datasets["segmentation"], test_datasets["segmentation"]])
    localization_ds = ConcatDataset([train_datasets["localization"], test_datasets["localization"]])

    # Inject `image_ids` attribute into each combined dataset
    grading_ds = extract_image_ids_from_concat(grading_ds)
    segmentation_ds = extract_image_ids_from_concat(segmentation_ds)
    localization_ds = extract_image_ids_from_concat(localization_ds)

    full_dataset = MultiTaskDataset(
        grading_ds=grading_ds,
        segmentation_ds=segmentation_ds,
        localization_ds=localization_ds
    )

    test_loader = DataLoader(full_dataset, batch_size=cfg["batch_size"], shuffle=False)

    grading_correct, grading_total = 0, 0
    dice_sum = torch.zeros(5).to(device)
    iou_sum = torch.zeros(5).to(device)
    localization_errors = []

    for batch in tqdm(test_loader, desc="Evaluating Routing Model"):
        images = batch["image"].to(device)
        targets = {
            "grading": batch["grading"].to(device),
            "segmentation": batch["segmentation"].to(device),
            "localization": batch["localization"].to(device),
        }

        outputs, task_weights = model(images, return_all=True)

        acc, correct, total = compute_grading_accuracy(outputs[0], targets["grading"])
        grading_correct += correct
        grading_total += total

        metrics = compute_segmentation_metrics(outputs[1], targets["segmentation"])
        dice = metrics["dice"]
        iou = metrics["iou"]

        dice_sum += dice
        iou_sum += iou

        # Fix localization output structure
        if isinstance(outputs[2], dict):
            preds = torch.cat([outputs[2]["od"], outputs[2]["fovea"]], dim=1)  # [B, 4]
        else:
            preds = outputs[2]  # Already [B, 4]

        loc_target = targets["localization"].to(device)  # Already [B, 4]
        loc_error = compute_localization_error(preds, loc_target)
        localization_errors.append(loc_error)

    grading_acc = grading_correct / grading_total
    dice_avg = dice_sum.mean().item()
    iou_avg = iou_sum.mean().item()
    loc_avg = torch.tensor(localization_errors).mean().item()

    print("=== Evaluation Report ===")
    print(f"[Grading Accuracy]   : {grading_acc:.4f}")
    print(f"[Segmentation Dice]  : {dice_avg:.4f}")
    print(f"[Segmentation IoU]   : {iou_avg:.4f}")
    print(f"[Localization Error] : {loc_avg:.2f} px")

    class_names = ["Microaneurysms", "Haemorrhages", "Hard Exudates", "Soft Exudates", "Optic Disc"]
    for i in range(5):
        print(f"  - {class_names[i]}: Dice={dice_sum[i].item():.4f}, IoU={iou_sum[i].item():.4f}")

    metrics = {
        "test/grading_accuracy": grading_acc,
        "test/seg_dice_avg": dice_avg,
        "test/seg_iou_avg": iou_avg,
        "test/localization_error_px": loc_avg,
    }

    # wandb.log(metrics)

    # for i in range(5):
    #     wandb.log({
    #         f"test/seg_dice_class_{i}": dice_sum[i].item(),
    #         f"test/seg_iou_class_{i}": iou_sum[i].item(),
    #     })

    # wandb.finish()
    os.remove(cfg["save_path"])

    return metrics

if __name__ == "__main__":
    test_dynamic_model()