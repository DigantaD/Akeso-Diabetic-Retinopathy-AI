import torch
import torch.nn as nn
from models.router import TaskRouter

class AkesoMultiTaskModel(nn.Module):
    def __init__(self, encoder, task_heads):
        super().__init__()
        self.encoder = encoder  # Expects input: [B, 3, H, W]
        self.task_heads = nn.ModuleList(task_heads)

        if not hasattr(encoder, "output_dim"):
            raise AttributeError("Encoder must define `output_dim` to initialize TaskRouter")

        self.router = TaskRouter(in_dim=encoder.output_dim, num_tasks=len(task_heads))
        self.task_names = ["grading", "segmentation", "localization"]

    def forward(self, x, return_all=False, task=None, threshold=0.3, top_k=None):
        """
        x: input tensor [B, 3, H, W]
        return_all: if True, return all task outputs for training
        task: manually select a task to run (e.g., "grading")
        threshold: routing weight threshold for auto-routing
        top_k: only return top-k routed task outputs

        Returns:
        - if return_all=True: [grading_out, seg_out, loc_out], task_weights
        - else: {task_name: output, ...}, task_weights
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input shape [B, 3, H, W], got {x.shape}")

        B = x.shape[0]

        # 🧠 Run encoder
        encoder_out = self.encoder(x)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]  # [B, D]

        task_weights = self.router(encoder_out)  # [B, num_tasks]

        # 🔧 Compute task outputs
        outputs = {
            "grading": self.task_heads[0](encoder_out),  # classification logits
            "segmentation": self.task_heads[1](x),        # segmentation mask
            "localization": self.task_heads[2](x),        # OD + fovea coordinates
        }

        if return_all:
            return [outputs["grading"], outputs["segmentation"], outputs["localization"]], task_weights

        # 🔀 Auto-routing mode
        routed_output = {}

        for i, task_name in enumerate(self.task_names):
            weight = task_weights[:, i]  # [B]

            include_task = False

            if task is not None:
                include_task = (task == task_name)
            elif top_k is not None:
                topk = torch.topk(task_weights, top_k, dim=1).indices  # [B, k]
                include_task = torch.any(topk == i, dim=1).item()
            else:
                include_task = (weight > threshold).item()

            if include_task:
                routed_output[task_name] = outputs[task_name]

        return routed_output, task_weights