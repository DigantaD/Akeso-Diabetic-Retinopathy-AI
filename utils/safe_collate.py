import torch

class EmptyBatchException(Exception):
    pass

def is_tensor_like(x):
    return isinstance(x, (torch.Tensor, float, int)) or (
        isinstance(x, (list, tuple)) and all(is_tensor_like(e) for e in x)
    )

def grading_collate_fn(batch):
    batch = [b for b in batch if b is not None and "grading" in b]
    if len(batch) == 0:
        raise EmptyBatchException("❌ No grading samples to collate.")

    images = torch.stack([b["image"] for b in batch])
    labels = torch.stack([b["grading"] for b in batch])
    ids = [b["image_id"] for b in batch]

    return {
        "image": images,
        "grading": labels,
        "image_id": ids
    }

def segmentation_collate_fn(batch):
    batch = [b for b in batch if b is not None and "segmentation" in b]
    if len(batch) == 0:
        raise EmptyBatchException("❌ No segmentation samples to collate.")

    images = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["segmentation"] for b in batch])
    ids = [b["image_id"] for b in batch]

    return {
        "image": images,
        "segmentation": masks,
        "image_id": ids
    }

def localization_collate_fn(batch):
    batch = [b for b in batch if b is not None and "localization" in b]
    if len(batch) == 0:
        raise EmptyBatchException("❌ No localization samples to collate.")

    images = torch.stack([b["image"] for b in batch])
    coords = torch.stack([b["localization"] for b in batch])
    ids = [b["image_id"] for b in batch]

    return {
        "image": images,
        "localization": coords,
        "image_id": ids
    }

def multitask_collate_fn(batch):
    batch = [b for b in batch if b is not None and isinstance(b, dict)]
    if len(batch) == 0:
        raise EmptyBatchException("❌ All samples were filtered out before collation.")

    collated = {}
    keys_to_stack = ["image", "grading", "segmentation", "localization"]
    keys_to_list = ["image_id"]

    for key in keys_to_stack:
        tensors = [b[key] for b in batch if key in b and b[key] is not None]
        if len(tensors) > 0:
            try:
                collated[key] = torch.stack(tensors)
            except Exception as e:
                print(f"[COLLATE] ❌ Error stacking key={key}: {e}")

    for key in keys_to_list:
        values = [b[key] for b in batch if key in b]
        if values:
            collated[key] = values

    collated["valid_tasks"] = {
        "grading": [(b.get("grading") is not None and isinstance(b["grading"], torch.Tensor)) for b in batch],
        "segmentation": [(b.get("segmentation") is not None and isinstance(b["segmentation"], torch.Tensor)
                          and b["segmentation"].sum() > 0) for b in batch],
        "localization": [(b.get("localization") is not None and isinstance(b["localization"], torch.Tensor)
                          and not torch.all(b["localization"] == -1.0)) for b in batch]
    }

    return collated