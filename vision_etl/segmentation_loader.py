import cv2
import numpy as np
import torch
from vision_etl.s3_loader import load_image_from_s3
import os
from PIL import Image


class SegmentationLoader:
    def __init__(self, lesion_keys_dict, img_size=(384, 384)):
        """
        Args:
            lesion_keys_dict (dict): Dict of {lesion_code: [list of .tif keys for that lesion]}
            img_size (tuple): Final (height, width) of the output masks
        """
        self.lesion_keys_dict = lesion_keys_dict
        self.img_size = img_size

    def get_mask(self, image_id: str):
        mask_list = []
        for lesion_code in ["MA", "HE", "EX", "SE", "OD"]:
            key_list = self.lesion_keys_dict.get(lesion_code, [])
            match_key = next((k for k in key_list if image_id.lower() in os.path.basename(k).lower()), None)
            if match_key:
                mask_img = load_image_from_s3(match_key)
                mask = Image.open(mask_img).convert("L").resize(self.img_size)
                mask_tensor = torch.from_numpy(np.array(mask) > 0).float()
            else:
                # If no mask found, return all zeros
                mask_tensor = torch.zeros(self.img_size, dtype=torch.float32)
            mask_list.append(mask_tensor)

        mask_stack = torch.stack(mask_list)  # Shape: [5, H, W]
        return mask_stack