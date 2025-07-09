import numpy as np
import torch
import os
import pandas as pd
from vision_etl.s3_loader import load_csv_from_s3

def generate_heatmap(h, w, x, y, sigma=10):
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    heatmap = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    return heatmap.astype(np.float32)

class LocalizationLoader:
    def __init__(self, od_csv_key, fovea_csv_key, output="point", image_size=(384, 384)):
        self.output = output
        self.image_size = image_size
        self.coord_map = {}

        # Optic Disc
        od_df = load_csv_from_s3(od_csv_key)[["Image No", "X- Coordinate", "Y - Coordinate"]]
        od_df = od_df.dropna(subset=["Image No"])
        od_df["Image No"] = od_df["Image No"].astype(str).str.strip()

        for _, row in od_df.iterrows():
            try:
                img_id = row["Image No"]
                self.coord_map[img_id] = {
                    "OD": (row["X- Coordinate"], row["Y - Coordinate"])
                }
            except Exception as e:
                print(f"⚠️ Skipping OD row due to error: {e}")

        # Fovea
        fovea_df = load_csv_from_s3(fovea_csv_key)[["Image No", "X- Coordinate", "Y - Coordinate"]]
        fovea_df = fovea_df.dropna(subset=["Image No"])
        fovea_df["Image No"] = fovea_df["Image No"].astype(str).str.strip()

        for _, row in fovea_df.iterrows():
            try:
                img_id = row["Image No"]
                if img_id in self.coord_map:
                    self.coord_map[img_id]["Fovea"] = (row["X- Coordinate"], row["Y - Coordinate"])
                else:
                    self.coord_map[img_id] = {
                        "OD": (0.0, 0.0),
                        "Fovea": (row["X- Coordinate"], row["Y - Coordinate"])
                    }
            except Exception as e:
                print(f"⚠️ Skipping Fovea row due to error: {e}")

        self.image_ids = sorted(self.coord_map.keys())

    def get(self, image_id):
        coords = self.coord_map.get(image_id)

        if coords is None:
            if self.output == "point":
                return torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            else:
                return torch.zeros((2, *self.image_size), dtype=torch.float32)

        if self.output == "point":
            od_x, od_y = coords.get("OD", (0.0, 0.0))
            fovea_x, fovea_y = coords.get("Fovea", (0.0, 0.0))
            return torch.tensor([od_x, od_y, fovea_x, fovea_y], dtype=torch.float32)
        else:
            h, w = self.image_size
            heatmaps = [
                generate_heatmap(h, w, *coords.get("OD", (0.0, 0.0))),
                generate_heatmap(h, w, *coords.get("Fovea", (0.0, 0.0)))
            ]
            return torch.tensor(np.stack(heatmaps), dtype=torch.float32)