import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.grading_model import GradingModel
from models.segmenter import AkesoSegmentationModel
from models.localization_model import LocalizationModel
from models.localization_vlm_module import VLMEmbedder
from dashboard.inference.s3_model_loader import download_model_from_s3
from agents.embedding_agent import load_encoder


class InferenceRunner:
    def __init__(self, s3_bucket, model_keys, cache_dir="~/.akeso_cache"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = os.path.expanduser(cache_dir)

        # 📥 Download model checkpoints
        grading_path = download_model_from_s3(s3_bucket, model_keys["grading"], f"{self.cache_dir}/grading.pth")
        segmentation_path = download_model_from_s3(s3_bucket, model_keys["segmentation"], f"{self.cache_dir}/segmentation.pth")
        localization_path = download_model_from_s3(s3_bucket, model_keys["localization"], f"{self.cache_dir}/localization.pth")

        # 🧠 Grading Model
        encoder = load_encoder("openclip-vit-b-16", pretrained=True, freeze=True)
        self.grading_model = GradingModel(encoder).to(self.device)
        self.grading_model.load_state_dict(torch.load(grading_path, map_location=self.device))
        self.grading_model.eval()

        # 🧠 Segmentation Model
        self.segmentation_model = AkesoSegmentationModel(
            sam_ckpt_path="pretrained/sam_vit_b_01ec64.pth",
            model_type="vit_b"
        ).to(self.device)
        self.segmentation_model.load_state_dict(torch.load(segmentation_path, map_location=self.device))
        self.segmentation_model.eval()

        # 🧠 Localization Model
        self.localization_model = LocalizationModel(use_vlm_head=True).to(self.device)
        self.localization_model.load_state_dict(torch.load(localization_path, map_location=self.device))
        self.localization_model.eval()

        # 📎 VLM Embedder (optional, future)
        self.vlm = VLMEmbedder(model_name="ViT-B-32", pretrained="openai").to(self.device)
        self.vlm.eval()

        # 🧼 Common Image Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.segmentation_labels = [
            "Soft Exudates", "Microaneurysms", "Hard Exudates", "Hemorrhages", "Optic Disc"
        ]

        self.grade_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

    def preprocess(self, image: Image.Image):
        return self.transform(image).unsqueeze(0).to(self.device)

    def infer_all(self, image: Image.Image, caption: str = None):
        x = self.preprocess(image)

        with torch.no_grad():
            # 🎯 Grading
            grading_logits = self.grading_model(x).cpu().numpy().flatten()
            pred_idx = int(np.argmax(grading_logits))
            confidence = float(grading_logits[pred_idx]) * 100
            grading_result = (self.grade_labels[pred_idx], confidence)

            # 🧠 Segmentation
            seg_logits = self.segmentation_model(x)  # shape [B, 5, H, W]
            seg_probs = torch.sigmoid(seg_logits).cpu().numpy()[0]
            segmentation_result = {
                self.segmentation_labels[i]: (seg_probs[i] > 0.5).astype(np.uint8)
                for i in range(len(self.segmentation_labels))
            }

            # 📍 Localization
            loc_out = self.localization_model(x)
            od_coords = (loc_out["od"] * 224).cpu().numpy()[0]
            fovea_coords = (loc_out["fovea"] * 224).cpu().numpy()[0]
            localization_result = {
                "points": {
                    "optic_disc": tuple(od_coords.astype(int)),
                    "fovea": tuple(fovea_coords.astype(int))
                }
            }

        return {
            "grading": grading_result,
            "segmentation": segmentation_result,
            "localization": localization_result
        }