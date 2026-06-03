"""Improved convolutional network trained from scratch for EuroSAT_RGB."""

from __future__ import annotations

import torch
from torch import nn


class ConvBNReLU(nn.Module):
    """Convolution block with batch normalization and ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ImprovedCNN(nn.Module):
    """Deeper CNN with normalization and regularization for better generalization."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        dropout_p: float = 0.35,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be greater than or equal to 2.")

        self.features = nn.Sequential(
            self._make_stage(input_channels, 32, dropout_p=0.05),
            self._make_stage(32, 64, dropout_p=0.10),
            self._make_stage(64, 128, dropout_p=0.15),
            self._make_stage(128, 256, dropout_p=0.20),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        dropout_p: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            ConvBNReLU(in_channels, out_channels),
            ConvBNReLU(out_channels, out_channels),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a PyTorch module."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = ["ConvBNReLU", "ImprovedCNN", "count_parameters"]
