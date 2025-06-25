import open_clip
import timm

def load_encoder(name="openclip-vit-b-16", pretrained=True, freeze=True):
    if "openclip" in name:
        # Map known names correctly
        model_map = {
            "openclip-vit-b-16": "ViT-B-16",
            "openclip-vit-b-32": "ViT-B-32",
            "openclip-vit-h-14": "ViT-H-14",
            "openclip-vit-l-14": "ViT-L-14",
        }
        model_name = model_map.get(name.lower())
        if model_name is None:
            raise ValueError(f"Unsupported OpenCLIP model name: {name}")

        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained="laion2b_s34b_b88k"
        )
        encoder = model.visual
        if freeze:
            for param in encoder.parameters():
                param.requires_grad = False
        return encoder

    elif "dinov2" in name:
        encoder = timm.create_model(name, pretrained=pretrained, num_classes=0)
        if freeze:
            for param in encoder.parameters():
                param.requires_grad = False
        return encoder

    else:
        raise ValueError(f"Unknown encoder: {name}")