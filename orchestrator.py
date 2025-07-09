import os
import time
import yaml
import argparse
from training.parallel_trainer import train_all_tasks_parallel
from training.router_trainer import train_router
import torch.multiprocessing as mp

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def save_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)

def update_config_epochs(cfg_path, epochs=None, warmup=None, alphas=None, cycle_number=None):
    cfg = load_yaml(cfg_path)
    if epochs is not None:
        cfg["epochs"] = epochs
    if warmup is not None:
        cfg["warmup_epochs"] = warmup
    if alphas is not None and "loss_weights" in cfg:
        cfg["loss_weights"]["grading"] = alphas[0]
        cfg["loss_weights"]["segmentation"] = alphas[1]
        cfg["loss_weights"]["localization"] = alphas[2]
    if cycle_number is not None:
        cfg["cycle_number"] = cycle_number
    save_yaml(cfg_path, cfg)

def orchestrate_training_from_schedule(
    schedule_path: str,
    grading_cfg_path: str,
    segmentation_cfg_path: str,
    localization_cfg_path: str,
    router_cfg_path: str,
    default_sleep: int = 10,
):
    schedule = load_yaml(schedule_path)
    sleep_between_stages = schedule.get("sleep_between_stages", default_sleep)

    for entry in schedule["schedule"]:
        cycle = entry["cycle"]
        print(f"\n🌀 ==== Starting Training Cycle {cycle} ====\n")

        # === Phase A: Task-Specific Training ===
        print("🎯 [PHASE A] Setting configs for Task-Specific Training...")
        update_config_epochs(grading_cfg_path, epochs=entry["phase_a"]["epochs"], cycle_number=cycle)
        update_config_epochs(segmentation_cfg_path, epochs=entry["phase_a"]["epochs"], cycle_number=cycle)
        update_config_epochs(localization_cfg_path, epochs=entry["phase_a"]["epochs"], cycle_number=cycle)

        print("🎯 [PHASE A] Starting Task-Specific Parallel Training...")
        train_all_tasks_parallel(
            load_yaml(grading_cfg_path),
            load_yaml(segmentation_cfg_path),
            load_yaml(localization_cfg_path)
        )

        time.sleep(sleep_between_stages)

        # === Phase B: Router Training ===
        print("🧠 [PHASE B] Setting Router Configs...")
        update_config_epochs(
            router_cfg_path,
            epochs=entry["phase_b"]["epochs"],
            warmup=entry["phase_b"].get("warmup", 0),
            cycle_number=cycle
        )

        # === Phase C: Federated Transfer Alphas ===
        print("🤝 [PHASE C] Injecting Routing Alphas into Router Config...")
        update_config_epochs(
            router_cfg_path,
            alphas=entry["phase_c"]["alphas"]
        )

        print("🧠 [PHASE B+C] Starting Router Training + Federated Transfer...")
        train_router()

        print(f"\n✅ Completed Cycle {cycle} — Model Weights Updated\n")
        time.sleep(sleep_between_stages)

    print("🏁 All Training Cycles Completed Successfully.")

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=str, default="config/training_schedule.yaml")
    parser.add_argument("--grading", type=str, default="config/grading_config.yaml")
    parser.add_argument("--segmentation", type=str, default="config/segmentation_config.yaml")
    parser.add_argument("--localization", type=str, default="config/localization_config.yaml")
    parser.add_argument("--router", type=str, default="config/dynamic_config.yaml")
    args = parser.parse_args()

    orchestrate_training_from_schedule(
        schedule_path=args.schedule,
        grading_cfg_path=args.grading,
        segmentation_cfg_path=args.segmentation,
        localization_cfg_path=args.localization,
        router_cfg_path=args.router
    )