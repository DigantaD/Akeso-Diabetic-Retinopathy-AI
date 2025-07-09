# File: tests/test_localization.py

import torch
from torch.utils.data import DataLoader, ConcatDataset
from vision_etl.dataset_factory import get_idrid_dataset
from models.localization_model import LocalizationModel
# from utils.wandb_logger import WandBLogger
import os
from dashboard.inference.s3_model_loader import download_model_from_s3
from tqdm import tqdm

def test_localization_model():
    cfg = {
        "image_size": (224, 224),
        "batch_size": 8,
        "output_dir": "outputs/checkpoints",
        "checkpoint_name": "localizer_od_fovea_best.pt",
        "model_s3_key": "outputs/checkpoints/localizer_od_fovea_best.pt",
        "save_path": "outputs/checkpoints/localizer_od_fovea_best.pt",
        "vlm_model": "ViT-B-32",
        "vlm_pretrained": "openai",
        "amp": True,
        "num_workers": 4,
    }

    # 📥 Download model checkpoint
    save_path = cfg["save_path"]
    s3_key = cfg["model_s3_key"]
    download_model_from_s3("akeso-eyecare", s3_key, save_path)

    best_ckpt_path = os.path.join(cfg["output_dir"], cfg["checkpoint_name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 📦 Dataset and Loader
    train_ds = get_idrid_dataset(["localization"], mode="train")["localization"]
    test_ds = get_idrid_dataset(["localization"], mode="test")["localization"]
    full_ds = ConcatDataset([train_ds, test_ds])
    loader = DataLoader(full_ds, batch_size=cfg["batch_size"], shuffle=False)

    # 🧠 Load model
    model = LocalizationModel(use_vlm_head=True).to(device)
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # logger = WandBLogger(task_name="localization", cfg=cfg, mode="test")
    errors, od_errors, fovea_errors = [], [], []

    norm_factor = cfg["image_size"][0]

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating Localization Model"):
            images = batch["image"].to(device)
            coords = batch["localization"]

            # Ensure coords is tensor on device
            if not isinstance(coords, torch.Tensor):
                coords = torch.tensor(coords, dtype=torch.float32)
            coords = coords.to(device)

            # Ground truth in pixels
            gt_od = coords[:, :2]
            gt_fovea = coords[:, 2:]
            gt = torch.stack([gt_od, gt_fovea], dim=1)

            # Predictions
            out = model(images)
            od_pred = out["od"] * norm_factor
            fovea_pred = out["fovea"] * norm_factor
            pred = torch.stack([od_pred, fovea_pred], dim=1)

            # Pixel-wise error
            pixel_error = torch.norm(pred - gt, dim=-1)  # [B, 2]
            od_errors.extend(pixel_error[:, 0].tolist())
            fovea_errors.extend(pixel_error[:, 1].tolist())
            errors.extend(pixel_error.mean(dim=1).tolist())

    avg_error = sum(errors) / len(errors)
    avg_od_error = sum(od_errors) / len(od_errors)
    avg_fovea_error = sum(fovea_errors) / len(fovea_errors)

    metrics = {
        "test/avg_pixel_error": avg_error,
        "test/od_pixel_error": avg_od_error,
        "test/fovea_pixel_error": avg_fovea_error
    }

    # logger.log_metrics(metrics)
    # logger.finish()

    os.remove(best_ckpt_path)

    print(f"\n📍 Avg Error       : {avg_error:.2f} px")
    print(f"📍 OD Error        : {avg_od_error:.2f} px")
    print(f"📍 Fovea Error     : {avg_fovea_error:.2f} px")

    return metrics

if __name__ == "__main__":
    test_localization_model()