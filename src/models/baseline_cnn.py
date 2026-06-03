"""Baseline convolutional network trained from scratch for EuroSAT_RGB."""

from __future__ import annotations

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Small CNN used as the first reference model for 64x64 RGB images."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        dropout_p: float = 0.25,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be greater than or equal to 2.")

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a PyTorch module."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = ["BaselineCNN", "count_parameters"]
