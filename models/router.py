import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskRouter(nn.Module):
    """
    Learns a distribution over task heads (grading, segmentation, localization)
    based on the encoder feature vector.
    """

    def __init__(self, in_dim: int, num_tasks: int = 3):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_tasks)
        self.output_dim = num_tasks  # Optional but useful for consistency

    def forward(self, x, temperature: float = 1.0):
        """
        Input:
            x: [B, in_dim] — encoded feature vector
            temperature: float — scaling for softmax sharpness
        Output:
            task_weights: [B, num_tasks] — softmax probabilities over tasks
        """
        logits = self.fc(x)
        task_weights = F.softmax(logits / temperature, dim=-1)
        return task_weights