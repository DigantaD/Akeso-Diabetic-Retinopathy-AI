import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import yaml

from utils.model_loader import load_pretrained_model
from dashboard.inference.s3_model_loader import download_model_from_s3

class InferenceRunner:
    def __init__(self, s3_bucket, model_key, cache_dir="~/.akeso_cache", config_path="config/dynamic_config.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        # 🧾 Load config
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        # 📥 Download unified model checkpoint
        self.checkpoint_path = os.path.join(self.cache_dir, "dynamic_router.pt")
        print(f"📦 Model Location: {s3_bucket} / {model_key} → {self.checkpoint_path}")
        download_model_from_s3(s3_bucket, model_key, self.checkpoint_path)
        self.cfg["save_path"] = self.checkpoint_path

        # 🧠 Load unified Akeso model
        self.model = load_pretrained_model(self.cfg).to(self.device)

        state = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state.get("model_state_dict", state))
        self.model.eval()

        # 🔖 Labels
        self.grade_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        self.segmentation_labels = [
            "Soft Exudates", "Microaneurysms", "Hard Exudates", "Hemorrhages", "Optic Disc"
        ]

        # 🌀 Preprocessing
        img_size = 224 if self.cfg.get("encoder_name") == "openclip-vit-b-16" else self.cfg.get("image_size", 512)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def preprocess(self, image: Image.Image):
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def infer_all(self, image: Image.Image):
        x = self.preprocess(image)

        with torch.no_grad():
            predictions, routing_weights = self.model(x)

        result = {}

        # 🎯 Grading Prediction
        if "grading" in predictions:
            logits = predictions["grading"].cpu().numpy().flatten()
            pred_idx = int(np.argmax(logits))
            confidence = float(logits[pred_idx]) * 100
            result["grading"] = {
                "class": self.grade_labels[pred_idx],
                "confidence_percent": round(confidence, 2)
            }

        # 🩸 Segmentation Mask
        if "segmentation" in predictions:
            seg_probs = torch.sigmoid(predictions["segmentation"]).cpu().numpy()[0]
            result["segmentation"] = {
                self.segmentation_labels[i]: (seg_probs[i] > 0.5).astype(np.uint8)
                for i in range(len(self.segmentation_labels))
            }

        # 🧠 Localization Coordinates
        if "localization" in predictions:
            coords_np = (predictions["localization"] * self.cfg.get("image_size", 512)).cpu().numpy()[0]
            result["localization"] = {
                "points": {
                    "optic_disc": tuple(map(int, coords_np[:2])),
                    "fovea": tuple(map(int, coords_np[2:]))
                }
            }

        # 🧭 Routing Weights
        if routing_weights is not None:
            result["routing_weights"] = {
                "grading": float(routing_weights[:, 0]),
                "segmentation": float(routing_weights[:, 1]),
                "localization": float(routing_weights[:, 2])
            }

        return result