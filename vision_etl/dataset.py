import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError

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
        self.transforms = get_advanced_transforms(image_size=image_size)

        self.grading_loader = None
        self.seg_loader = None
        self.loc_loader = None

        # Initialize grading loader (may be needed to fetch image_ids)
        if "grading" in tasks:
            grading_csv_key = data_chamber["grading"]["groundtruths"][mode]
            self.grading_loader = GradingLoader(grading_csv_key)

        # Determine image source prefix and resolve image_keys from S3
        if "grading" in tasks:
            image_source = data_chamber["grading"]["original"][mode]
        elif "segmentation" in tasks:
            image_source = data_chamber["segmentation"]["original"][mode]
        elif "localization" in tasks:
            image_source = data_chamber["localization"]["original"][mode]
        else:
            raise ValueError(f"No valid task found in tasks={tasks} for image source resolution.")

        if isinstance(image_source, list):
            self.image_keys = image_source
        elif isinstance(image_source, str):
            self.image_keys = list_image_keys_from_s3_prefix(image_source)
        else:
            raise ValueError(f"Invalid type for image_source: {type(image_source)}")

        # Determine image_ids
        if image_ids is None and self.grading_loader:
            image_ids = self.grading_loader.get_all_image_ids()
        elif isinstance(image_ids, (set, dict)):
            image_ids = list(image_ids)
        elif image_ids is None:
            image_ids = [
                os.path.splitext(os.path.basename(k))[0]
                for k in self.image_keys
            ]

        valid_image_ids = []
        for image_id in image_ids:
            match = next((k for k in self.image_keys
                          if os.path.splitext(os.path.basename(k))[0].lower() == image_id.lower()), None)
            if match:
                valid_image_ids.append(image_id)
            else:
                print(f"⚠️ Skipping image ID: {image_id} — not found in image keys.")

        self.image_ids = sorted(valid_image_ids)

        # Initialize segmentation loader
        if "segmentation" in self.tasks:
            lesion_keys_dict = {
                lesion_code: list_image_keys_from_s3_prefix(data_chamber["segmentation"]["groundtruths"][mode][lesion_code])
                for lesion_code in data_chamber["segmentation"]["groundtruths"][mode]
            }
            self.seg_loader = SegmentationLoader(lesion_keys_dict, img_size=image_size)

        # Initialize localization loader
        if "localization" in self.tasks:
            od_csv_key, fovea_csv_key = data_chamber["localization"]["groundtruths"][mode]
            self.loc_loader = LocalizationLoader(
                od_csv_key=od_csv_key,
                fovea_csv_key=fovea_csv_key,
                output=localization_mode,
                image_size=image_size
            )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        image_key = next(
            (k for k in self.image_keys if os.path.splitext(os.path.basename(k))[0].lower() == image_id.lower()),
            None
        )
        if image_key is None:
            raise ValueError(f"Image not found for ID: {image_id}")

        # Safely attempt to load image
        try:
            img_bytes = load_image_from_s3(image_key)
            img = Image.open(img_bytes).convert("RGB")
            img_np = np.array(img)
        except (UnidentifiedImageError, OSError) as e:
            print(f"❌ Unidentified or corrupted image for ID {image_id}: {e}")
            # Retry next image safely
            return self.__getitem__((idx + 1) % len(self))

        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)

        transformed = self.transforms(image=img_np)
        img_tensor = transformed["image"]

        sample = {
            "image": img_tensor,
            "image_id": image_id
        }

        if "grading" in self.tasks and self.grading_loader:
            sample["label"] = self.grading_loader.get_label(image_id)

        if "segmentation" in self.tasks and self.seg_loader:
            mask_tensor = self.seg_loader.get_mask(image_id)
            if mask_tensor is not None:
                if mask_tensor.shape[0] != 5:
                    padded = torch.zeros((5, *mask_tensor.shape[1:]), dtype=torch.float32)
                    padded[:mask_tensor.shape[0]] = mask_tensor
                    mask_tensor = padded
                sample["segmentation_mask"] = mask_tensor
            else:
                sample["segmentation_mask"] = torch.zeros(
                    (5, self.image_size[0], self.image_size[1]), dtype=torch.float32
                )

        if "localization" in self.tasks and self.loc_loader:
            sample["localization"] = self.loc_loader.get(image_id)

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