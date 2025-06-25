import os
import sys
import yaml
from multiprocessing import Process, set_start_method

from training.train_grading import run_training_loop as training_loop_grading
from training.train_segmentation import run_training_loop as training_loop_seg
from training.train_localization import run_training_loop as training_loop_loc

# --- Load Configs ---
with open("/workspace/akeso/Akeso-Diabetic-Retinopathy-AI/config/grading_config.yaml") as f:
    grad_cfg = yaml.safe_load(f)

with open("/workspace/akeso/Akeso-Diabetic-Retinopathy-AI/config/segmentation_config.yaml") as f:
    seg_cfg = yaml.safe_load(f)

loc_config = {
    "image_size": (224, 224),
    "batch_size": 8,
    "epochs": 100,
    "lr": 1e-4,
    "lambda_align": 1.0,
    "lambda_clip": 0.5,
    "localization_mode": "point",
    "output_dir": "outputs/checkpoints",
    "checkpoint_name": "localizer_od_fovea_best.pt",
    "vlm_model": "ViT-B-32",
    "vlm_pretrained": "openai",
    "amp": True,
    "val_split": 0.2,
    "num_workers": 4
}

# --- Logger wrapper for stdout ---
class LoggerWrapper:
    def __init__(self, prefix):
        self.prefix = prefix

    def write(self, message):
        if message.strip():
            sys.__stdout__.write(f"{self.prefix} {message}")

    def flush(self):
        pass

# --- Worker functions ---
def run_grading():
    sys.stdout = LoggerWrapper("[GRADING]")
    os.environ["TQDM_POS"] = "0"
    training_loop_grading(grad_cfg)

def run_segmentation():
    sys.stdout = LoggerWrapper("[SEGMENTATION]")
    os.environ["TQDM_POS"] = "1"
    training_loop_seg(seg_cfg)

def run_localization():
    sys.stdout = LoggerWrapper("[LOCALIZATION]")
    os.environ["TQDM_POS"] = "2"
    training_loop_loc(loc_config)

# --- Main execution ---
if __name__ == "__main__":
    set_start_method("spawn", force=True)

    p1 = Process(target=run_grading)
    p2 = Process(target=run_segmentation)
    p3 = Process(target=run_localization)

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()