import os
import sys
import traceback
from training.dynamic_trainer import train_dynamic_model

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

def train_router():
    sys.stdout = LoggerWrapper("[ROUTER]")
    os.environ["WANDB_MODE"] = "online"

    print("🚀 Starting Router Training...")
    try:
        train_dynamic_model()
        print("✅ Router Training Complete.")
    except Exception as e:
        print("❌ Router training failed:")
        traceback.print_exc()
        sys.exit(1)