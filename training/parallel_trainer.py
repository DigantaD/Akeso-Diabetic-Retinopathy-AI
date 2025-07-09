import os
import sys
from multiprocessing import Process
from training.train_grading import run_training_loop as grading_loop
from training.train_segmentation import run_training_loop as segmentation_loop
from training.train_localization import run_training_loop as localization_loop

class LoggerWrapper:
    def __init__(self, prefix):
        self.prefix = prefix

    def write(self, message):
        if message.strip():
            for line in message.rstrip().splitlines():
                sys.__stdout__.write(f"{self.prefix} {line}\n")
        sys.__stdout__.flush()

    def flush(self):
        sys.__stdout__.flush()

# ✅ TOP-LEVEL PICKLABLE FUNCTION
def run_training_process(name, loop_fn, cfg, pos):
    os.environ["TQDM_POS"] = str(pos)
    sys.stdout = LoggerWrapper(f"[{name.upper()}]")
    try:
        loop_fn(cfg)
    except Exception as e:
        print(f"[{name.upper()}] ❌ Training failed: {e}", file=sys.stderr)
        raise e

def train_all_tasks_parallel(grad_cfg, seg_cfg, loc_cfg):
    grading_proc = Process(target=run_training_process, args=("grading", grading_loop, grad_cfg, 0))
    seg_proc = Process(target=run_training_process, args=("segmentation", segmentation_loop, seg_cfg, 1))
    loc_proc = Process(target=run_training_process, args=("localization", localization_loop, loc_cfg, 2))

    grading_proc.start()
    seg_proc.start()
    loc_proc.start()

    grading_proc.join()
    seg_proc.join()
    loc_proc.join()

    # Exit with appropriate error code if any process failed
    for proc, name in zip([grading_proc, seg_proc, loc_proc], ["Grading", "Segmentation", "Localization"]):
        if proc.exitcode != 0:
            print(f"[{name}] ❌ Process exited with code {proc.exitcode}")