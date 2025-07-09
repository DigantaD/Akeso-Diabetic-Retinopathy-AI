# utils/dynamic_alpha.py
import torch
import numpy as np

def compute_dynamic_alphas(stats_path="outputs/checkpoints/router_stats.pt", temperature=1.5):
    """
    Computes task-wise alphas from saved router entropy values.
    
    Lower entropy -> higher confidence -> higher alpha
    """
    data = torch.load(stats_path)
    entropies = data.get("task_entropy", None)

    if entropies is None:
        raise ValueError("task_entropy not found in router_stats.pt")

    entropies = np.array(entropies)
    inverted = 1.0 / (entropies + 1e-8)  # prevent div by zero
    exp_scaled = np.exp(inverted / temperature)
    alphas = exp_scaled / np.sum(exp_scaled)

    return alphas.tolist()