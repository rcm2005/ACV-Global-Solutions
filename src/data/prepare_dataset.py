"""Prepare split metadata and train normalization stats for EuroSAT_RGB."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageStat


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MANIFEST_FIELDS = [
    "relative_path",
    "class_name",
    "class_idx",
    "filename",
    "width",
    "height",
    "mode",
]
SPLIT_FIELDS = ["relative_path", "class_name", "class_idx", "split"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic EuroSAT_RGB metadata and splits."
    )
    parser.add_argument("--dataset-root", default="EuroSAT_RGB")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", default="0.70")
    parser.add_argument("--val-ratio", default="0.15")
    parser.add_argument("--test-ratio", default="0.15")
    parser.add_argument("--input-size", type=int, nargs=2, default=(64, 64))
    return parser.parse_args()


def resolve_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def path_relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def find_classes(dataset_root: Path) -> tuple[list[str], dict[str, int]]:
    classes = sorted(
        entry.name for entry in dataset_root.iterdir() if entry.is_dir()
    )
    if not classes:
        raise ValueError(f"No class folders found in {dataset_root}")
    return classes, {class_name: idx for idx, class_name in enumerate(classes)}


def iter_image_paths(class_dir: Path) -> Iterable[Path]:
    for path in sorted(class_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def build_manifest(dataset_root: Path, project_root: Path) -> tuple[list[dict], dict]:
    classes, class_to_idx = find_classes(dataset_root)
    rows: list[dict] = []

    for class_name in classes:
        class_dir = dataset_root / class_name
        for image_path in iter_image_paths(class_dir):
            with Image.open(image_path) as image:
                row = {
                    "relative_path": path_relative_to_project(
                        image_path, project_root
                    ),
                    "class_name": class_name,
                    "class_idx": class_to_idx[class_name],
                    "filename": image_path.name,
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                }
            rows.append(row)

    rows.sort(key=lambda row: row["relative_path"])
    return rows, class_to_idx


def parse_ratios(args: argparse.Namespace) -> dict[str, Fraction]:
    ratios = {
        "train": Fraction(str(args.train_ratio)),
        "val": Fraction(str(args.val_ratio)),
        "test": Fraction(str(args.test_ratio)),
    }
    if any(value <= 0 for value in ratios.values()):
        raise ValueError(f"Ratios must be positive: {ratios}")
    if sum(ratios.values(), start=Fraction(0, 1)) != Fraction(1, 1):
        raise ValueError(f"Ratios must sum to 1.0: {ratios}")
    return ratios


def exact_count(total: int, ratio: Fraction, label: str, class_name: str) -> int:
    value = total * ratio
    if value.denominator != 1:
        raise ValueError(
            "Split count is not an integer for "
            f"{class_name}/{label}: total={total}, ratio={ratio}"
        )
    return value.numerator


def split_manifest(
    manifest_rows: list[dict], ratios: dict[str, Fraction], seed: int
) -> dict[str, list[dict]]:
    rows_by_class: dict[str, list[dict]] = defaultdict(list)
    for row in manifest_rows:
        rows_by_class[row["class_name"]].append(row)

    splits = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)

    for class_name in sorted(rows_by_class):
        class_rows = sorted(
            rows_by_class[class_name], key=lambda row: row["relative_path"]
        )
        rng.shuffle(class_rows)

        total = len(class_rows)
        n_train = exact_count(total, ratios["train"], "train", class_name)
        n_val = exact_count(total, ratios["val"], "val", class_name)

        train_rows = class_rows[:n_train]
        val_rows = class_rows[n_train : n_train + n_val]
        test_rows = class_rows[n_train + n_val :]

        for split_name, selected_rows in (
            ("train", train_rows),
            ("val", val_rows),
            ("test", test_rows),
        ):
            for row in selected_rows:
                splits[split_name].append(
                    {
                        "relative_path": row["relative_path"],
                        "class_name": row["class_name"],
                        "class_idx": row["class_idx"],
                        "split": split_name,
                    }
                )

    for split_name in splits:
        splits[split_name].sort(key=lambda row: row["relative_path"])

    return splits


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=True, indent=2)
        file.write("\n")


def count_by_class(rows: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(row["class_name"] for row in rows).items()))


def compute_train_stats(
    train_rows: list[dict], project_root: Path, input_size: tuple[int, int]
) -> dict:
    channel_sum = [0.0, 0.0, 0.0]
    channel_sum_sq = [0.0, 0.0, 0.0]
    total_pixels = 0
    resized_images = 0

    for row in train_rows:
        image_path = project_root / row["relative_path"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if image.size != input_size:
                image = image.resize(input_size, Image.Resampling.BILINEAR)
                resized_images += 1

            stats = ImageStat.Stat(image)
            pixels = image.width * image.height
            total_pixels += pixels
            for channel in range(3):
                channel_sum[channel] += stats.sum[channel]
                channel_sum_sq[channel] += stats.sum2[channel]

    if total_pixels == 0:
        raise ValueError("Cannot compute stats from an empty train split")

    means_255 = [value / total_pixels for value in channel_sum]
    stds_255 = []
    for channel in range(3):
        mean_value = means_255[channel]
        variance = channel_sum_sq[channel] / total_pixels - mean_value**2
        stds_255.append(math.sqrt(max(variance, 0.0)))

    return {
        "mean": [round(value / 255.0, 10) for value in means_255],
        "std": [round(value / 255.0, 10) for value in stds_255],
        "source_split": "train",
        "num_images": len(train_rows),
        "num_pixels": total_pixels,
        "input_size": list(input_size),
        "resized_images_for_stats": resized_images,
    }


def validate_outputs(
    manifest_rows: list[dict],
    splits: dict[str, list[dict]],
    dataset_root: Path,
    project_root: Path,
) -> dict:
    manifest_paths = [row["relative_path"] for row in manifest_rows]
    if len(manifest_paths) != len(set(manifest_paths)):
        raise ValueError("Duplicate relative_path values found in manifest")

    for relative_path in manifest_paths:
        if not (project_root / relative_path).exists():
            raise FileNotFoundError(relative_path)

    all_split_paths: list[str] = []
    split_counts = {}
    split_class_counts = {}
    for split_name, rows in splits.items():
        split_paths = [row["relative_path"] for row in rows]
        if len(split_paths) != len(set(split_paths)):
            raise ValueError(f"Duplicate paths found in {split_name}.csv")
        split_counts[split_name] = len(rows)
        split_class_counts[split_name] = count_by_class(rows)
        all_split_paths.extend(split_paths)

    if set(all_split_paths) != set(manifest_paths):
        raise ValueError("Split files do not match the manifest paths")
    if len(all_split_paths) != len(set(all_split_paths)):
        raise ValueError("Overlap detected between train, val and test")

    modes = sorted({row["mode"] for row in manifest_rows})
    sizes = sorted({(int(row["width"]), int(row["height"])) for row in manifest_rows})

    return {
        "total_images": len(manifest_rows),
        "split_counts": split_counts,
        "class_counts": count_by_class(manifest_rows),
        "split_class_counts": split_class_counts,
        "modes": modes,
        "sizes": [list(size) for size in sizes],
        "dataset_root": path_relative_to_project(dataset_root, project_root),
    }


def main() -> None:
    args = parse_args()
    project_root = Path.cwd().resolve()
    dataset_root = resolve_path(args.dataset_root, project_root)
    output_dir = resolve_path(args.output_dir, project_root)
    input_size = tuple(args.input_size)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    ratios = parse_ratios(args)
    manifest_rows, class_to_idx = build_manifest(dataset_root, project_root)
    splits = split_manifest(manifest_rows, ratios, args.seed)
    validation = validate_outputs(manifest_rows, splits, dataset_root, project_root)
    train_stats = compute_train_stats(splits["train"], project_root, input_size)

    write_csv(output_dir / "dataset_manifest.csv", manifest_rows, MANIFEST_FIELDS)
    for split_name in ("train", "val", "test"):
        write_csv(output_dir / f"{split_name}.csv", splits[split_name], SPLIT_FIELDS)

    write_json(output_dir / "class_to_idx.json", class_to_idx)
    write_json(output_dir / "train_stats.json", train_stats)

    split_config = {
        "dataset_root": validation["dataset_root"],
        "seed": args.seed,
        "ratios": {name: float(value) for name, value in ratios.items()},
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "num_classes": len(class_to_idx),
        "total_images": validation["total_images"],
        "class_counts": validation["class_counts"],
        "split_counts": validation["split_counts"],
        "split_class_counts": validation["split_class_counts"],
        "image_modes": validation["modes"],
        "image_sizes": validation["sizes"],
        "input_size": list(input_size),
        "manifest_file": "dataset_manifest.csv",
        "split_files": {
            "train": "train.csv",
            "val": "val.csv",
            "test": "test.csv",
        },
        "normalization_file": "train_stats.json",
        "image_copy_strategy": "no_image_copy_metadata_only",
    }
    write_json(output_dir / "split_config.json", split_config)

    print(json.dumps(split_config, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
