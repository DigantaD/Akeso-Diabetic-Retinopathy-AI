import torch
from torch.utils.data import DataLoader
from vision_etl.dataset_factory import get_idrid_dataset
from models.localization_model import LocalizationModel
from utils.wandb_logger import WandBLogger
import os

def test_localization_model():
    cfg = {
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

    best_ckpt_path = os.path.join(cfg["output_dir"], cfg["checkpoint_name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = get_idrid_dataset(["localization"], mode="test")["localization"]
    loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = LocalizationModel(use_vlm_head=True).to(device)
    model.load_state_dict(torch.load(best_ckpt_path))
    model.eval()

    logger = WandBLogger(task_name="localization", cfg=cfg, mode="test")

    errors = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            coords = batch["localization"]

            od_gt = torch.stack([coords["OD"][0], coords["OD"][1]], dim=1).float()
            fovea_gt = torch.stack([coords["Fovea"][0], coords["Fovea"][1]], dim=1).float()
            gt = torch.stack([od_gt, fovea_gt], dim=1).to(device)

            out = model(images)
            od_pred = out["od"] * cfg["image_size"][0]
            fovea_pred = out["fovea"] * cfg["image_size"][0]
            pred = torch.stack([od_pred, fovea_pred], dim=1)

            error = torch.norm(pred - gt, dim=-1).mean().item()
            errors.append(error)

    avg_error = sum(errors) / len(errors)
    logger.log_metrics({"test/avg_pixel_error": avg_error})
    logger.finish()
    os.remove(best_ckpt_path)

    print(f"📍 Avg Pixel Error: {avg_error:.2f}")

if __name__ == "__main__":
    test_localization_model()