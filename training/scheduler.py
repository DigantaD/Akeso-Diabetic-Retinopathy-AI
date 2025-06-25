import torch.optim as optim
import torch

def get_scheduler(optimizer, cfg):
    sched_cfg = cfg.get("lr_scheduler", {})
    sched_type = sched_cfg.get("type", "cosine")

    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["epochs"],
            eta_min=1e-6
        )
    elif sched_type == "plateau":
        plateau_cfg = sched_cfg.get("plateau", {})
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=plateau_cfg.get("mode", "min"),
            factor=plateau_cfg.get("factor", 0.5),
            patience=plateau_cfg.get("patience", 5),
            threshold=plateau_cfg.get("threshold", 1e-4),
            cooldown=plateau_cfg.get("cooldown", 0),
            min_lr=plateau_cfg.get("min_lr", 1e-6),
            verbose=plateau_cfg.get("verbose", True)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")