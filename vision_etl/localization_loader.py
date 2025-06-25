# vision_etl/localization_loader.py

import numpy as np
import torch
import cv2
from vision_etl.s3_loader import load_csv_from_s3

def generate_heatmap(h, w, x, y, sigma=10):
    """
    Generate a 2D Gaussian heatmap centered at (x, y).
    """
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    heatmap = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    return heatmap.astype(np.float32)

class LocalizationLoader:
    def __init__(self, od_csv_key, fovea_csv_key, output="point", image_size=(384, 384)):
        """
        Args:
            od_csv_key: S3 key for Optic Disc center CSV
            fovea_csv_key: S3 key for Fovea center CSV
            output: "point" (returns coordinates) or "heatmap" (returns tensor)
            image_size: Desired output size for heatmap
        """
        self.output = output
        self.image_size = image_size
        self.coord_map = {}

        # Load Optic Disc coordinates
        od_df = load_csv_from_s3(od_csv_key)[["Image No", "X- Coordinate", "Y - Coordinate"]]
        od_df["Image No"] = od_df["Image No"].astype(str).str.strip()

        for _, row in od_df.iterrows():
            img_id = self._normalize_image_id(row["Image No"])
            self.coord_map[img_id] = {
                "OD": (row["X- Coordinate"], row["Y - Coordinate"])
            }

        # Load Fovea coordinates
        fovea_df = load_csv_from_s3(fovea_csv_key)[["Image No", "X- Coordinate", "Y - Coordinate"]]
        fovea_df["Image No"] = fovea_df["Image No"].astype(str).str.strip()

        for _, row in fovea_df.iterrows():
            img_id = self._normalize_image_id(row["Image No"])
            if img_id in self.coord_map:
                self.coord_map[img_id]["Fovea"] = (row["X- Coordinate"], row["Y - Coordinate"])
            else:
                self.coord_map[img_id] = {
                    "OD": (0, 0),
                    "Fovea": (row["X- Coordinate"], row["Y - Coordinate"])
                }

    def _normalize_image_id(self, path_or_id):
        """
        Normalize image name to format like 'IDRiD_001'.
        """
        filename = path_or_id.split("/")[-1]
        return filename.replace(".jpg", "").replace(".JPG", "").strip()

    def get(self, image_id):
        image_id = self._normalize_image_id(image_id)
        coords = self.coord_map.get(image_id)

        if coords is None:
            if self.output == "point":
                return {"OD": (0, 0), "Fovea": (0, 0)}
            else:
                return torch.zeros((2, *self.image_size), dtype=torch.float32)

        if self.output == "point":
            return coords
        else:
            h, w = self.image_size
            heatmaps = [
                generate_heatmap(h, w, *coords.get("OD", (0, 0))),
                generate_heatmap(h, w, *coords.get("Fovea", (0, 0))),
            ]
            return torch.tensor(np.stack(heatmaps), dtype=torch.float32)