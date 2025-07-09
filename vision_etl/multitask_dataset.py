import torch
from torch.utils.data import Dataset

class MultiTaskDataset(Dataset):
    def __init__(self, grading_ds=None, segmentation_ds=None, localization_ds=None):
        self.grading_ds = grading_ds
        self.segmentation_ds = segmentation_ds
        self.localization_ds = localization_ds

        self.grading_ids = set(grading_ds.image_ids) if grading_ds else set()
        self.segmentation_ids = set(segmentation_ds.image_ids) if segmentation_ds else set()
        self.localization_ids = set(localization_ds.image_ids) if localization_ds else set()

        all_ids = self.grading_ids | self.segmentation_ids | self.localization_ids
        self.ids = sorted(all_ids)

        self.grading_map = {
            image_id: grading_ds[i]
            for i, image_id in enumerate(grading_ds.image_ids)
        } if grading_ds else {}

        self.segmentation_map = {
            image_id: segmentation_ds[i]
            for i, image_id in enumerate(segmentation_ds.image_ids)
        } if segmentation_ds else {}

        self.localization_map = {
            image_id: localization_ds[i]
            for i, image_id in enumerate(localization_ds.image_ids)
        } if localization_ds else {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        sample = {}

        sample["grading"] = torch.tensor(-1, dtype=torch.long)
        sample["segmentation"] = torch.zeros((5, 224, 224), dtype=torch.float32)
        sample["localization"] = torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=torch.float32)
        sample["image"] = None
        sample["image_id"] = image_id

        # Grading
        if image_id in self.grading_map:
            grading_sample = self.grading_map[image_id]
            # print("Grading: ", image_id, grading_sample["grading"])
            sample["grading"] = grading_sample["grading"]
            sample["image"] = grading_sample["image"]

        # Segmentation
        if image_id in self.segmentation_map:
            segmentation_sample = self.segmentation_map[image_id]
            # print("Segmentation: ", image_id, segmentation_sample['segmentation'])
            sample["segmentation"] = segmentation_sample["segmentation"]
            # print(sample["segmentation"].sum())
            if sample["image"] is None:
                sample["image"] = segmentation_sample["image"]

        # Localization
        if image_id in self.localization_map:
            localization_sample = self.localization_map[image_id]
            # print("Localization: ", image_id, localization_sample["localization"])
            if isinstance(localization_sample, dict):
                coords = localization_sample.get("OD", (0.0, 0.0)) + localization_sample.get("Fovea", (0.0, 0.0))
                sample["localization"] = torch.tensor(coords, dtype=torch.float32)
                if sample["image"] is None:
                    sample["image"] = localization_sample.get("image", None)
            elif isinstance(localization_sample, torch.Tensor):
                sample["localization"] = localization_sample.float()

        # Final sanity check
        if sample["image"] is None or not isinstance(sample["image"], torch.Tensor):
            raise ValueError(f"❌ No image found for ID: {image_id}")

        sample["valid_tasks"] = {
            "grading": image_id in self.grading_map,
            "segmentation": image_id in self.segmentation_map,
            "localization": image_id in self.localization_map,
        }

        # if sample["valid_tasks"]["segmentation"] is True:
        #     print(sample['segmentation'].sum())
 
        return sample