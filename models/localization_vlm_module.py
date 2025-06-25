import torch
import torch.nn as nn
import open_clip
from torchvision import transforms
from PIL import Image

class VLMEmbedder(nn.Module):
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device="cuda"):
        super().__init__()
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.eval()  # inference mode only
        for p in self.model.parameters():
            p.requires_grad = False

    def encode_image(self, images):
        """
        Args:
            images: B x 3 x H x W (torch.Tensor)
        Returns:
            image_features: B x D
        """
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, texts):
        """
        Args:
            texts: List of strings
        Returns:
            text_features: B x D
        """
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, images, texts=None):
        """
        Args:
            images: B x 3 x H x W
            texts: List[str] or None
        Returns:
            dict with 'image_embed' and optionally 'text_embed'
        """
        out = {"image_embed": self.encode_image(images)}
        if texts is not None:
            out["text_embed"] = self.encode_text(texts)
        return out