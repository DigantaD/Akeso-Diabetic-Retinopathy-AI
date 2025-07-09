# File: tests/test_suite.py

from concurrent.futures import ThreadPoolExecutor
from tests.test_grading import test_grading_model
from tests.test_segmentation import test_segmentation_model
from tests.test_localization import test_localization_model
from tests.test_dynamic_model import test_dynamic_model
import csv
import os
# import wandb

CSV_PATH = "outputs/eval_results.csv"

# === Human-readable labels ===

GRADING_LABELS = {
    "0": "No DR",
    "1": "Mild",
    "2": "Moderate",
    "3": "Severe",
    "4": "Proliferative DR"
}

SEGMENTATION_LABELS = {
    "0": "Microaneurysms",
    "1": "Haemorrhages",
    "2": "Hard Exudates",
    "3": "Soft Exudates",
    "4": "Optic Disc"
}

# === Metric descriptions ===

METRIC_DESCRIPTIONS = {
    "test/accuracy": "Grading model classification accuracy",
    "test/precision": "Weighted average precision",
    "test/recall": "Weighted average recall",
    "test/f1_score": "Weighted average F1 score",
    "test/dice_avg": "Mean Dice score across all segmentation classes",
    "test/iou_avg": "Mean IoU across all segmentation classes",
    "test/avg_pixel_error": "Average Euclidean pixel error (localization)",
    "test/od_pixel_error": "Optic disc pixel localization error",
    "test/fovea_pixel_error": "Fovea pixel localization error",
    "test/grading_accuracy": "Accuracy from dynamic model",
    "test/seg_dice_avg": "Dice score from dynamic model",
    "test/seg_iou_avg": "IoU from dynamic model",
    "test/localization_error_px": "Localization error from dynamic model"
}


def get_class_label(model_name: str, class_index: str) -> str:
    if model_name == "Grading":
        return GRADING_LABELS.get(str(class_index), f"Class {class_index}")
    elif model_name == "Segmentation":
        return SEGMENTATION_LABELS.get(str(class_index), f"Class {class_index}")
    else:
        return f"Class {class_index}"


def write_metrics_to_csv(metrics_dict, model_name):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    file_exists = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Model", "Metric", "Value", "Description"])

        for metric, value in metrics_dict.items():
            if isinstance(value, dict):
                # Expand each key-value pair from dict only if value ≠ 0
                for sub_key, sub_value in value.items():
                    if sub_value == 0.0:
                        continue  # Skip zero values
                    label = get_class_label(model_name, sub_key)
                    flat_key = f"test/{metric.split('/')[-1]}/{label}"
                    full_key = f"{model_name}/{flat_key}"

                    # Add proper description
                    if "precision" in metric:
                        description = f"{label} precision"
                    elif "recall" in metric:
                        description = f"{label} recall"
                    elif "f1" in metric:
                        description = f"{label} F1 score"
                    elif "support" in metric:
                        description = f"Support count for {label}"
                    else:
                        description = f"{metric} for {label}"

                    writer.writerow([model_name, full_key, sub_value, description])
            else:
                # Skip logging if value is exactly 0.0
                if value == 0.0:
                    continue
                full_key = f"{model_name}/{metric}"
                description = METRIC_DESCRIPTIONS.get(metric, f"{metric} from {model_name} model")
                writer.writerow([model_name, full_key, value, description])

def run_all_tests():
    print("\n🔍 Starting Parallel Evaluations...")

    results = {}  # Collect all metrics here first

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            "Grading": executor.submit(test_grading_model),
            "Segmentation": executor.submit(test_segmentation_model),
            "Localization": executor.submit(test_localization_model),
            "Dynamic": executor.submit(test_dynamic_model),
        }

        for model_name, future in futures.items():
            print(f"🟡 Waiting for {model_name} test to complete...")
            try:
                metrics = future.result()
                if metrics:
                    print(f"✅ {model_name} test completed.")
                    results[model_name] = metrics
                else:
                    print(f"⚠️  {model_name} returned no metrics.")
            except Exception as e:
                print(f"❌ Error in {model_name} test: {e}")

    print("\n📝 Logging all metrics to CSV...")
    for model_name, metrics in results.items():
        write_metrics_to_csv(metrics, model_name)
        print(f"📁 Logged: {model_name}")

    print("\n✅ All Evaluations Complete and Logged.")
    # wandb.finish()

if __name__ == "__main__":
    run_all_tests()