import cv2
import numpy as np
import torch
from PIL import Image
from vision_etl.s3_loader import load_image_from_s3
import os

class SegmentationLoader:
    def __init__(self, lesion_keys_dict, img_size=(384, 384)):
        """
        Args:
            lesion_keys_dict (dict): Dict of {lesion_code: [list of .tif keys for that lesion]}
            img_size (tuple): Final (height, width) of the output masks
        """
        self.lesion_keys_dict = lesion_keys_dict
        self.img_size = img_size
        self.image_ids = self._build_image_ids()

    def _build_image_ids(self):
        """
        Collects and normalizes unique image IDs from all lesion mask keys.
        """
        ids = set()
        for lesion_code, keys in self.lesion_keys_dict.items():
            for k in keys:
                try:
                    ids.add(k)
                except ValueError:
                    continue
        return sorted(list(ids))

    def get_mask(self, image_id: str):
        """
        Returns a stacked tensor of all 5 lesion masks [MA, HE, EX, SE, OD] for a given image ID.
        """
        mask_list = []

        for lesion_code in ["MA", "HE", "EX", "SE", "OD"]:
            key_list = self.lesion_keys_dict.get(lesion_code, [])
            match_key = next(
                (k for k in key_list if os.path.basename(k).startswith(image_id)),
                None
            )

            if match_key:
                mask_img = load_image_from_s3(match_key)
                mask = Image.open(mask_img).convert("L").resize(self.img_size)
                mask_tensor = torch.from_numpy(np.array(mask) > 0).float()
            else:
                mask_tensor = torch.zeros(self.img_size, dtype=torch.float32)

            mask_list.append(mask_tensor)

        return torch.stack(mask_list)