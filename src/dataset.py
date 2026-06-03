"""PyTorch dataset and DataLoader helpers for EuroSAT_RGB."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from src.transforms import DEFAULT_INPUT_SIZE, build_transforms_from_stats


DEFAULT_DATASET_ROOT = Path("EuroSAT_RGB")
DEFAULT_PROCESSED_DIR = Path("data/processed")
SPLITS = ("train", "val", "test")


def load_class_to_idx(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> dict[str, int]:
    path = Path(processed_dir) / "class_to_idx.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_train_stats(processed_dir: str | Path = DEFAULT_PROCESSED_DIR) -> dict:
    path = Path(processed_dir) / "train_stats.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_split_rows(
    split: str,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> list[dict[str, str]]:
    _validate_split(split)
    path = Path(processed_dir) / f"{split}.csv"
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def build_imagefolder_subset(
    split: str,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    transform: Any = None,
):
    """Build an ImageFolder subset using indices saved in split CSVs."""

    _validate_split(split)
    ImageFolder, Subset = _import_torch_dataset_classes()

    dataset_root = Path(dataset_root)
    base_dataset = ImageFolder(root=str(dataset_root), transform=transform)
    sample_index = _index_imagefolder_samples(base_dataset, dataset_root)
    split_rows = load_split_rows(split, processed_dir)

    indices: list[int] = []
    missing: list[str] = []
    for row in split_rows:
        relative_path = _normalize_key(row["relative_path"])
        index = sample_index.get(relative_path)
        if index is None:
            relative_path = _strip_dataset_prefix(relative_path, dataset_root)
            index = sample_index.get(relative_path)
        if index is None:
            missing.append(row["relative_path"])
        else:
            indices.append(index)

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} paths from {split}.csv were not found in "
            f"ImageFolder samples. Examples: {preview}"
        )

    subset = Subset(base_dataset, indices)
    subset.split = split
    subset.relative_paths = [row["relative_path"] for row in split_rows]
    subset.classes = base_dataset.classes
    subset.class_to_idx = base_dataset.class_to_idx
    return subset


def build_datasets(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
) -> dict[str, Any]:
    """Build train, validation and test ImageFolder subsets."""

    transforms_by_split = build_transforms_from_stats(
        Path(processed_dir) / "train_stats.json", input_size=input_size
    )
    return {
        split: build_imagefolder_subset(
            split=split,
            dataset_root=dataset_root,
            processed_dir=processed_dir,
            transform=transforms_by_split[split],
        )
        for split in SPLITS
    }


def build_dataloaders(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    pin_memory: bool | None = None,
) -> dict[str, Any]:
    """Build DataLoaders for train, validation and test splits."""

    DataLoader, torch = _import_dataloader_and_torch()
    datasets = build_datasets(
        dataset_root=dataset_root,
        processed_dir=processed_dir,
        input_size=input_size,
    )
    device = get_device()
    use_pin_memory = device.type == "cuda" if pin_memory is None else pin_memory

    generator = torch.Generator()
    generator.manual_seed(seed)

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            worker_init_fn=seed_worker,
            generator=generator,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        ),
    }


def get_device():
    """Return cuda, mps or cpu according to the local PyTorch runtime."""

    torch = _import_torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_dataloader_batch(loader: Any, batch_size: int = 64) -> dict[str, Any]:
    """Validate one batch shape and label range from a DataLoader."""

    images, labels = next(iter(loader))
    if images.ndim != 4:
        raise ValueError(f"Expected image tensor with 4 dims, got {images.shape}")
    if tuple(images.shape[1:]) != (3, 64, 64):
        raise ValueError(f"Expected [batch, 3, 64, 64], got {images.shape}")
    if images.shape[0] > batch_size:
        raise ValueError(f"Batch is larger than expected: {images.shape[0]}")
    if labels.min().item() < 0 or labels.max().item() > 9:
        raise ValueError("Labels are outside the expected [0, 9] range")

    return {
        "image_shape": list(images.shape),
        "label_shape": list(labels.shape),
        "label_min": int(labels.min().item()),
        "label_max": int(labels.max().item()),
        "image_dtype": str(images.dtype),
        "label_dtype": str(labels.dtype),
    }


def summarize_splits(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> dict[str, Any]:
    """Summarize split counts from saved CSVs without importing PyTorch."""

    summary: dict[str, Any] = {}
    for split in SPLITS:
        rows = load_split_rows(split, processed_dir)
        class_counts: dict[str, int] = {}
        for row in rows:
            class_counts[row["class_name"]] = class_counts.get(row["class_name"], 0) + 1
        summary[split] = {
            "total": len(rows),
            "class_counts": dict(sorted(class_counts.items())),
        }
    return summary


def seed_worker(worker_id: int) -> None:
    """Seed Python random in DataLoader workers from PyTorch's worker seed."""

    torch = _import_torch()
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)


def _validate_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"Invalid split '{split}'. Expected one of {SPLITS}.")


def _index_imagefolder_samples(base_dataset: Any, dataset_root: Path) -> dict[str, int]:
    sample_index: dict[str, int] = {}
    root = dataset_root.resolve()
    cwd = Path.cwd().resolve()

    for index, (sample_path, _) in enumerate(base_dataset.samples):
        path = Path(sample_path).resolve()
        for key in _sample_keys(path, root, cwd):
            sample_index.setdefault(key, index)

    return sample_index


def _sample_keys(path: Path, dataset_root: Path, cwd: Path) -> set[str]:
    keys = {_normalize_key(path.as_posix())}
    for base in (dataset_root, cwd):
        try:
            keys.add(_normalize_key(path.relative_to(base).as_posix()))
        except ValueError:
            pass
    return keys


def _strip_dataset_prefix(relative_path: str, dataset_root: Path) -> str:
    root_parts = [_normalize_key(dataset_root.as_posix()), dataset_root.name]
    for prefix in root_parts:
        prefix = prefix.strip("/")
        if prefix and relative_path.startswith(prefix + "/"):
            return relative_path[len(prefix) + 1 :]
    return relative_path


def _normalize_key(value: str) -> str:
    return value.replace("\\", "/").strip()


def _import_torch_dataset_classes():
    try:
        from torch.utils.data import Subset
        from torchvision.datasets import ImageFolder
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch and torchvision are required to build PyTorch datasets. "
            "Install project dependencies before using src.dataset."
        ) from exc
    return ImageFolder, Subset


def _import_dataloader_and_torch():
    torch = _import_torch()
    try:
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch is required to build DataLoaders."
        ) from exc
    return DataLoader, torch


def _import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch is required for device detection and DataLoaders. "
            "Install project dependencies before using src.dataset."
        ) from exc
    return torch
