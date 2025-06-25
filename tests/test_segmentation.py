import torch
from torch.utils.data import DataLoader
from vision_etl.dataset_factory import get_idrid_dataset
from models.segmenter import AkesoSegmentationModel
from utils.metrics import compute_segmentation_metrics
from utils.wandb_logger import WandBLogger
import yaml, os

def test_segmentation_model():
    with open("config/segmentation_config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = get_idrid_dataset(["segmentation"], mode="test")["segmentation"]
    loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = AkesoSegmentationModel(
        sam_ckpt_path=cfg["sam_ckpt_path"],
        model_type=cfg["sam_model_type"]
    ).to(device)
    model.load_state_dict(torch.load(cfg["save_path"]))
    model.eval()

    logger = WandBLogger(task_name="segmentation", cfg=cfg, mode="test")

    dice_total = torch.zeros(5).to(device)
    iou_total = torch.zeros(5).to(device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            masks = batch["segmentation_mask"].to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)

            metrics = compute_segmentation_metrics(probs, masks, threshold=0.5)
            dice_total += metrics["dice"]
            iou_total += metrics["iou"]

            if batch_idx == 0:  # log only first batch of masks to avoid clutter
                logger.log_segmentation_masks(
                    images=images.cpu(),
                    preds=(probs > 0.5).float().cpu(),
                    gts=masks.cpu(),
                    class_labels=["MA", "HE", "EX", "SE", "OD"],
                    step=0  # ✅ Corrected
                )

    dice_avg = (dice_total / len(loader)).tolist()
    iou_avg = (iou_total / len(loader)).tolist()

    logger.log_metrics({
        "test/dice_avg": sum(dice_avg) / len(dice_avg),
        "test/iou_avg": sum(iou_avg) / len(iou_avg),
        **{f"test/dice_class_{i}": v for i, v in enumerate(dice_avg)},
        **{f"test/iou_class_{i}": v for i, v in enumerate(iou_avg)},
    }, step=0)  # ✅ avoid step warnings

    logger.finish()
    os.remove(cfg["save_path"])

    print(f"🎯 Dice: {[round(d, 4) for d in dice_avg]}")
    print(f"🎯 IoU : {[round(i, 4) for i in iou_avg]}")

if __name__ == "__main__":
    test_segmentation_model()