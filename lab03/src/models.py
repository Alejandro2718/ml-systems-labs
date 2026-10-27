import torch
from torch import nn


class AttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(49, 64)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(64, 4, 128, batch_first=True)
            for _ in range(2)
        ])
        self.classifier = nn.Linear(64, 10)

    def forward(self, images):
        patches = images.unfold(2, 7, 7).unfold(3, 7, 7)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 16, 49)
        tokens = self.projection(patches)
        for layer in self.layers:
            tokens = layer(tokens)
        return self.classifier(tokens.mean(dim=1))


def make_model(name):
    if name == "MLP":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                             nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
    if name == "CNN":
        return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                             nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1),
                             nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                             nn.Linear(32 * 7 * 7, 10))
    if name == "Attention":
        return AttentionClassifier()
    raise ValueError(name)


MODEL_NAMES = ("MLP", "CNN", "Attention")
