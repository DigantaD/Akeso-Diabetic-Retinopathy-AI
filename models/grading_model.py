import torch.nn as nn

class GradingModel(nn.Module):
    def __init__(self, encoder, encoder_dim=512, num_classes=5):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)