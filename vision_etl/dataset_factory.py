from vision_etl.grading_loader import GradingLoader
from vision_etl.dataset import IDRiDDataset, get_data_chamber
from vision_etl.s3_loader import list_objects


def expand_folders_in_chamber(data_chamber):
    def expand(obj):
        if isinstance(obj, dict):
            return {k: expand(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand(v) for v in obj]
        elif isinstance(obj, str):
            if obj.endswith("/"):
                return list_objects(obj)
            return obj
        return obj

    return expand(data_chamber)


def get_idrid_dataset(tasks=[], mode="train", localization_mode="point"):
    """
    Factory function to create IDRiDDataset with appropriate loaders and file paths.
    Supports 'grading', 'segmentation', and 'localization' tasks.
    """
    raw_data_chamber = get_data_chamber()
    data_chamber = expand_folders_in_chamber(raw_data_chamber)

    datasets = {}
    for task in tasks:
        datasets[task] = IDRiDDataset(
            tasks=[task],  # only include one task per dataset instance
            data_chamber=data_chamber,
            mode=mode,
            localization_mode=localization_mode
        )

    return datasets