import os
import torch
import numpy as np
import cv2
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from PIL import Image

LESION_CLASSES = ["MA", "HE", "EX", "SE", "OD"]

def sigmoid_threshold(logits, threshold=0.5):
    probs = torch.sigmoid(logits)
    return (probs > threshold).float()

def overlay_mask_on_image(image, mask, alpha=0.5):
    """
    image: torch.Tensor [3, H, W]
    mask: torch.Tensor [5, H, W] (binary masks)
    """
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)

    mask_sum = mask.sum(dim=0).cpu().numpy()  # collapse across classes
    mask_sum = (mask_sum > 0).astype(np.uint8) * 255
    mask_color = cv2.applyColorMap(mask_sum, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_np, 1 - alpha, mask_color, alpha, 0)

    return overlay

def save_predictions_to_disk(model, dataloader, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            image_ids = batch["image_id"]
            logits = model(images)
            masks = sigmoid_threshold(logits, threshold=0.5)

            for i in range(images.size(0)):
                image_id = image_ids[i]
                mask = masks[i]            # [5, H, W]
                image = images[i]          # [3, H, W]

                # Save binary masks (per class)
                for j, cls in enumerate(LESION_CLASSES):
                    mask_np = mask[j].cpu().numpy() * 255
                    mask_np = mask_np.astype(np.uint8)
                    cv2.imwrite(os.path.join(save_dir, f"{image_id}_{cls}.png"), mask_np)

                # Save overlay image
                overlay = overlay_mask_on_image(image, mask)
                cv2.imwrite(os.path.join(save_dir, f"{image_id}_overlay.jpg"), overlay)