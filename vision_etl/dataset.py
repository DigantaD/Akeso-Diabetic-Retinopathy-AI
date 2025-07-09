import os, re
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from vision_etl.grading_loader import GradingLoader
from vision_etl.segmentation_loader import SegmentationLoader
from vision_etl.localization_loader import LocalizationLoader
from vision_etl.s3_loader import load_image_from_s3, list_image_keys_from_s3_prefix
from vision_etl.transforms import get_advanced_transforms

SEG_CLASS_INFO = {
    "MA": ("1. Microaneurysms", "_MA"),
    "HE": ("2. Haemorrhages", "_HE"),
    "EX": ("3. Hard Exudates", "_EX"),
    "SE": ("4. Soft Exudates", "_SE"),
    "OD": ("5. Optic Disc", "_OD")
}

class IDRiDDataset(Dataset):
    def __init__(self, tasks, data_chamber, mode="train", image_ids=None,
                 image_size=(224, 224), localization_mode="point"):
        self.tasks = tasks
        self.mode = mode
        self.image_size = image_size
        self.localization_mode = localization_mode
        self.transforms = get_advanced_transforms(image_size=image_size)

        self.grading_loader = None
        self.seg_loader = None
        self.loc_loader = None

        if "grading" in tasks:
            grading_csv_key = data_chamber["grading"]["groundtruths"][mode]
            self.grading_loader = GradingLoader(grading_csv_key)

        # Determine image source (pick from available task)
        if "grading" in tasks:
            image_source = data_chamber["grading"]["original"][mode]
        elif "segmentation" in tasks:
            image_source = data_chamber["segmentation"]["original"][mode]
        elif "localization" in tasks:
            image_source = data_chamber["localization"]["original"][mode]
        else:
            raise ValueError(f"No valid task found in tasks={tasks} for image source resolution.")

        self.image_keys = (
            image_source if isinstance(image_source, list)
            else list_image_keys_from_s3_prefix(image_source)
        )

        # Get image_ids from each task
        self.grading_ids = self.grading_loader.get_all_image_ids() if self.grading_loader else []

        self.segmentation_ids = []
        if "segmentation" in tasks:
            seg_image_keys = list_image_keys_from_s3_prefix(data_chamber["segmentation"]["original"][mode])
            self.segmentation_ids = [
                os.path.splitext(os.path.basename(k))[0]
                for k in seg_image_keys
            ]

        self.localization_ids = []
        if "localization" in tasks:
            loc_image_keys = list_image_keys_from_s3_prefix(data_chamber["localization"]["original"][mode])
            self.localization_ids = [
                os.path.splitext(os.path.basename(k))[0]
                for k in loc_image_keys
            ]

        # Combine all available unique IDs across tasks
        all_ids = set(self.grading_ids) | set(self.segmentation_ids) | set(self.localization_ids)

        # Ensure the image file actually exists
        valid_image_ids = []
        for image_id in all_ids:
            match = next(
                (k for k in self.image_keys
                 if os.path.splitext(os.path.basename(k))[0].lower() == image_id.lower()),
                None
            )
            if match:
                valid_image_ids.append(image_id)
            else:
                print(f"⚠️ Skipping image ID: {image_id} — not found in image keys.")

        self.image_ids = sorted(valid_image_ids)

        # Loaders for Segmentation and Localization
        if "segmentation" in self.tasks:
            lesion_keys_dict = {
                lesion_code: list_image_keys_from_s3_prefix(data_chamber["segmentation"]["groundtruths"][mode][lesion_code])
                for lesion_code in data_chamber["segmentation"]["groundtruths"][mode]
            }
            self.seg_loader = SegmentationLoader(lesion_keys_dict, img_size=image_size)

        if "localization" in self.tasks:
            od_csv_key, fovea_csv_key = data_chamber["localization"]["groundtruths"][mode]
            self.loc_loader = LocalizationLoader(
                od_csv_key=od_csv_key,
                fovea_csv_key=fovea_csv_key,
                output=localization_mode,
                image_size=image_size
            )

        print(f"\n📦 IDRiDDataset Summary")
        print(f"Mode: {mode} | Tasks: {tasks}")
        print(f"Total Image IDs: {len(self.image_ids)}")
        print(f"Grading IDs: {len(self.grading_ids)} | Segmentation IDs: {len(self.segmentation_ids)} | Localization IDs: {len(self.localization_ids)}")
        print(f"Combined Valid IDs: {len(self.image_ids)}\n")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        image_key = next(
            (k for k in self.image_keys if os.path.splitext(os.path.basename(k))[0].lower() == image_id.lower()),
            None
        )
        if image_key is None:
            print(f"❌ Image key not found for ID: {image_id}")
            return None

        try:
            img_bytes = load_image_from_s3(image_key)
            img = Image.open(img_bytes).convert("RGB")
            img_np = np.array(img)
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).astype(np.uint8)
            transformed = self.transforms(image=img_np)
            img_tensor = transformed["image"]
        except Exception as e:
            print(f"❌ Image load/transform failed for {image_id}: {e}")
            return None

        sample = {
            "image": img_tensor,
            "image_id": image_id
        }

        if "grading" in self.tasks and self.grading_loader and image_id in self.grading_ids:
            try:
                label = self.grading_loader.get_label(image_id)
                sample["grading"] = torch.tensor(label, dtype=torch.long)
            except Exception as e:
                print(f"❌ Grading label fetch failed for {image_id}: {e}")
                sample["grading"] = torch.tensor(-1, dtype=torch.long)
            # print(f"Grading: {image_id}, {sample['grading'].sum()}")
        else:
            sample["grading"] = torch.tensor(-1, dtype=torch.long)

        if "segmentation" in self.tasks and self.seg_loader and image_id in self.segmentation_ids:
            try:
                mask_tensor = self.seg_loader.get_mask(image_id)
                if mask_tensor is not None and mask_tensor.shape[0] == 5:
                    sample["segmentation"] = mask_tensor
                else:
                    sample["segmentation"] = torch.zeros((5, self.image_size[0], self.image_size[1]), dtype=torch.float32)
            except Exception as e:
                print(f"❌ Segmentation mask fetch failed for {image_id}: {e}")
                sample["segmentation"] = torch.zeros((5, self.image_size[0], self.image_size[1]), dtype=torch.float32)
            # print(f"Segmentation: {image_id}, {sample['segmentation'].sum()}")
        else:
            sample["segmentation"] = torch.zeros((5, self.image_size[0], self.image_size[1]), dtype=torch.float32)

        if "localization" in self.tasks and self.loc_loader and image_id in self.localization_ids:
            try:
                coords_data = self.loc_loader.get(image_id)

                if isinstance(coords_data, dict):
                    od = coords_data.get("OD")
                    fovea = coords_data.get("Fovea")
                    if od is None or fovea is None:
                        raise ValueError("Missing OD or Fovea")
                    coords = list(od) + list(fovea)
                    sample["localization"] = torch.tensor(coords, dtype=torch.float32)
                elif isinstance(coords_data, torch.Tensor) and coords_data.shape[-1] == 4:
                    sample["localization"] = coords_data.float()
                else:
                    raise ValueError(f"Unexpected localization format for {image_id}: {type(coords_data)}")
            except Exception as e:
                print(f"❌ Localization fetch failed for {image_id}: {e}")
                sample["localization"] = torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=torch.float32)
            # print(f"Localization: {image_id}, {sample['localization'].sum()}")
        else:
            sample["localization"] = torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=torch.float32)

        valid_tasks = {
            "grading": image_id in self.grading_ids,
            "segmentation": image_id in self.segmentation_ids,
            "localization": image_id in self.localization_ids
        }
        sample["valid_tasks"] = valid_tasks

        return sample

def get_data_chamber():
    segmentation_groundtruths = {
        "train": {
            k: f"IDRiD Dataset/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/{v[0]}/"
            for k, v in SEG_CLASS_INFO.items()
        },
        "test": {
            k: f"IDRiD Dataset/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/{v[0]}/"
            for k, v in SEG_CLASS_INFO.items()
        }
    }

    return {
        "grading": {
            "original": {
                "train": "IDRiD Dataset/B. Disease Grading/1. Original Images/a. Training Set/",
                "test": "IDRiD Dataset/B. Disease Grading/1. Original Images/b. Testing Set/"
            },
            "groundtruths": {
                "train": "IDRiD Dataset/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
                "test": "IDRiD Dataset/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
            }
        },
        "segmentation": {
            "original": {
                "train": "IDRiD Dataset/A. Segmentation/1. Original Images/a. Training Set/",
                "test": "IDRiD Dataset/A. Segmentation/1. Original Images/b. Testing Set/"
            },
            "groundtruths": segmentation_groundtruths
        },
        "localization": {
            "original": {
                "train": "IDRiD Dataset/C. Localization/1. Original Images/a. Training Set/",
                "test": "IDRiD Dataset/C. Localization/1. Original Images/b. Testing Set/"
            },
            "groundtruths": {
                "train": [
                    "IDRiD Dataset/C. Localization/2. Groundtruths/1. Optic Disc Center Location/a. IDRiD_OD_Center_Training Set_Markups.csv",
                    "IDRiD Dataset/C. Localization/2. Groundtruths/2. Fovea Center Location/IDRiD_Fovea_Center_Training Set_Markups.csv"
                ],
                "test": [
                    "IDRiD Dataset/C. Localization/2. Groundtruths/1. Optic Disc Center Location/b. IDRiD_OD_Center_Testing Set_Markups.csv",
                    "IDRiD Dataset/C. Localization/2. Groundtruths/2. Fovea Center Location/IDRiD_Fovea_Center_Testing Set_Markups.csv"
                ]
            }
        }
    }