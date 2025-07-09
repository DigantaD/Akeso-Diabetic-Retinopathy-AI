# File: tests/test_segmentation.py
import torch
from torch.utils.data import DataLoader, ConcatDataset
from vision_etl.dataset_factory import get_idrid_dataset
from models.segmenter import AkesoSegmentationModel
from utils.metrics import compute_segmentation_metrics
# from utils.wandb_logger import WandBLogger
import yaml, os
from dashboard.inference.s3_model_loader import download_model_from_s3
from tqdm import tqdm

def test_segmentation_model():
    with open("config/segmentation_config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Download model checkpoint from S3
    save_path = cfg["save_path"]
    s3_key = cfg["model_s3_key"]
    download_model_from_s3("akeso-eyecare", s3_key, save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = get_idrid_dataset(["segmentation"], mode="train")["segmentation"]
    test_ds = get_idrid_dataset(["segmentation"], mode="test")["segmentation"]
    full_ds = ConcatDataset([train_ds, test_ds])
    loader = DataLoader(full_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = AkesoSegmentationModel(
        sam_ckpt_path=cfg["model_config"]["sam_ckpt_path"],
        model_type=cfg["model_config"]["sam_model_type"]
    ).to(device)
    state_dict = torch.load(cfg["save_path"])
    model.load_state_dict(state_dict["model_state_dict"])
    model.eval()

    # logger = WandBLogger(task_name="segmentation", cfg=cfg, mode="test")
    dice_total = torch.zeros(5).to(device)
    iou_total = torch.zeros(5).to(device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating Segmentation Model")):
            images = batch["image"].to(device)
            masks = batch["segmentation"].to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)

            metrics = compute_segmentation_metrics(probs, masks, threshold=0.5)
            dice_total += metrics["dice"]
            iou_total += metrics["iou"]

            # if batch_idx % 10 == 0:  # log every 10th batch
            #     logger.log_segmentation_masks(
            #         images=images.cpu(),
            #         preds=(probs > 0.5).float().cpu(),
            #         gts=masks.cpu(),
            #         class_labels=["MA", "HE", "EX", "SE", "OD"],
            #         step=batch_idx
            #     )

    dice_avg = (dice_total / len(loader)).tolist()
    iou_avg = (iou_total / len(loader)).tolist()

    metrics = {
        "test/dice_avg": sum(dice_avg) / len(dice_avg),
        "test/iou_avg": sum(iou_avg) / len(iou_avg),
        **{f"test/dice_class_{i}": v for i, v in enumerate(dice_avg)},
        **{f"test/iou_class_{i}": v for i, v in enumerate(iou_avg)},
    }

    # logger.log_metrics(metrics, step=0)

    # logger.finish()
    os.remove(cfg["save_path"])

    print(f"🎯 Dice: {[round(d, 4) for d in dice_avg]}")
    print(f"🎯 IoU : {[round(i, 4) for i in iou_avg]}")

    return metrics

if __name__ == "__main__":
    test_segmentation_model()