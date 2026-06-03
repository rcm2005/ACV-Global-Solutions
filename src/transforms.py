"""Factories de transforms para EuroSAT_RGB."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


DEFAULT_INPUT_SIZE = (64, 64)
DEFAULT_STATS_PATH = Path("data/processed/train_stats.json")


def load_train_stats(stats_path: str | Path = DEFAULT_STATS_PATH) -> dict:
    """Load normalization statistics generated from the train split."""

    path = Path(stats_path)
    with path.open("r", encoding="utf-8") as file:
        stats = json.load(file)

    mean = stats.get("mean")
    std = stats.get("std")
    if not _is_rgb_sequence(mean) or not _is_rgb_sequence(std):
        raise ValueError(f"Invalid normalization stats in {path}")
    if any(value <= 0 for value in std):
        raise ValueError(f"Invalid std values in {path}: {std}")

    return stats


def build_train_transforms(
    mean: Sequence[float],
    std: Sequence[float],
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
):
    """Build train transforms with moderate remote-sensing augmentations."""

    transforms = _import_torchvision_transforms()
    return transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.15,
                        contrast=0.15,
                        saturation=0.10,
                        hue=0.02,
                    )
                ],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_eval_transforms(
    mean: Sequence[float],
    std: Sequence[float],
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
):
    """Build deterministic transforms for validation and test."""

    transforms = _import_torchvision_transforms()
    return transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_inference_transforms(
    mean: Sequence[float],
    std: Sequence[float],
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
):
    """Build deterministic transforms for new images in demos/inference."""

    return build_eval_transforms(mean=mean, std=std, input_size=input_size)


def build_transforms_from_stats(
    stats_path: str | Path = DEFAULT_STATS_PATH,
    input_size: tuple[int, int] | None = None,
) -> dict:
    """Return train, val, test and inference transforms from saved stats."""

    stats = load_train_stats(stats_path)
    size = tuple(input_size or stats.get("input_size", DEFAULT_INPUT_SIZE))
    mean = stats["mean"]
    std = stats["std"]
    eval_transform = build_eval_transforms(mean=mean, std=std, input_size=size)

    return {
        "train": build_train_transforms(mean=mean, std=std, input_size=size),
        "val": eval_transform,
        "test": eval_transform,
        "inference": build_inference_transforms(
            mean=mean, std=std, input_size=size
        ),
    }


def _is_rgb_sequence(value: object) -> bool:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return False
    return all(isinstance(item, int | float) for item in value)


def _import_torchvision_transforms():
    try:
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required for image transforms. "
            "Install project dependencies before building DataLoaders."
        ) from exc
    return transforms
