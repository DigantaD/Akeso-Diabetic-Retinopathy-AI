import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_advanced_transforms(image_size=(224, 224), use_clahe=False, normalize=True):
    ops = []

    if use_clahe:
        from vision_etl.transforms import apply_clahe
        ops.append(A.Lambda(image=apply_clahe))

    ops.extend([
        A.Resize(*image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Affine(rotate=(-15, 15), scale=(0.9, 1.1), translate_percent=(0.05, 0.05), shear=(-5, 5), p=0.4),
        A.GridDistortion(p=0.2),
        A.ElasticTransform(alpha=1.0, sigma=50.0, approximate=True, p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, p=0.3),
    ])

    if normalize:
        ops.append(A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        ))

    ops.append(ToTensorV2())

    return A.Compose(ops)