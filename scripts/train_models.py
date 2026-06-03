"""Train and evaluate the two CNNs implemented from scratch for EuroSAT_RGB."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gs-2026")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.optim import AdamW

from src.dataset import build_dataloaders, load_class_to_idx, load_train_stats, summarize_splits
from src.models.baseline_cnn import BaselineCNN
from src.models.baseline_cnn import count_parameters as count_baseline_parameters
from src.models.improved_cnn import ImprovedCNN
from src.models.improved_cnn import count_parameters as count_improved_parameters


TARGET_TEST_ACCURACY = 0.88
MODEL_SPECS = {
    "baseline": {
        "class": BaselineCNN,
        "display_name": "BaselineCNN",
        "counter": count_baseline_parameters,
        "checkpoint": "baseline_best.pt",
        "curve": "baseline_accuracy_loss.png",
        "confusion_matrix": "baseline_confusion_matrix.png",
        "regularization": "Dropout no classificador; AdamW com weight decay.",
    },
    "improved": {
        "class": ImprovedCNN,
        "display_name": "ImprovedCNN",
        "counter": count_improved_parameters,
        "checkpoint": "improved_best.pt",
        "curve": "improved_accuracy_loss.png",
        "confusion_matrix": "improved_confusion_matrix.png",
        "regularization": "BatchNorm2d, Dropout2d, dropout no classificador e AdamW com weight decay.",
    },
}


@dataclass(frozen=True)
class TrainConfig:
    dataset_root: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path
    training_report_path: Path
    seed: int
    batch_size: int
    num_workers: int
    input_size: tuple[int, int]
    epochs: dict[str, int]
    learning_rates: dict[str, float]
    weight_decays: dict[str, float]
    patience: int
    min_delta: float
    selected_models: list[str]
    device: torch.device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BaselineCNN and ImprovedCNN from scratch on EuroSAT_RGB."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("EuroSAT_RGB"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--training-report-path",
        type=Path,
        default=Path("reports/training_evaluation_report.md"),
    )
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_SPECS), default=["baseline", "improved"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--epochs-baseline", type=int, default=12)
    parser.add_argument("--epochs-improved", type=int, default=18)
    parser.add_argument("--lr-baseline", type=float, default=1e-3)
    parser.add_argument("--lr-improved", type=float, default=1e-3)
    parser.add_argument("--weight-decay-baseline", type=float, default=1e-4)
    parser.add_argument("--weight-decay-improved", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--force-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(force_cpu=args.force_cpu)
    config = TrainConfig(
        dataset_root=args.dataset_root,
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        training_report_path=args.training_report_path,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=(args.input_size, args.input_size),
        epochs={"baseline": args.epochs_baseline, "improved": args.epochs_improved},
        learning_rates={"baseline": args.lr_baseline, "improved": args.lr_improved},
        weight_decays={
            "baseline": args.weight_decay_baseline,
            "improved": args.weight_decay_improved,
        },
        patience=args.patience,
        min_delta=args.min_delta,
        selected_models=list(args.models),
        device=device,
    )

    create_output_dirs(config)
    set_seed(config.seed)
    torch.set_num_threads(max(1, min(os.cpu_count() or 1, 8)))

    class_to_idx = load_class_to_idx(config.processed_dir)
    class_names = class_names_from_mapping(class_to_idx)
    train_stats = load_train_stats(config.processed_dir)
    split_summary = summarize_splits(config.processed_dir)
    write_class_names(config.models_dir / "class_names.json", class_names, class_to_idx)

    run_started_at = datetime.now().isoformat(timespec="seconds")
    metrics: dict[str, Any] = {
        "run_started_at": run_started_at,
        "seed": config.seed,
        "device": str(config.device),
        "target_test_accuracy": TARGET_TEST_ACCURACY,
        "dataset_root": config.dataset_root.as_posix(),
        "processed_dir": config.processed_dir.as_posix(),
        "input_size": list(config.input_size),
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "class_names": class_names,
        "split_summary": split_summary,
        "normalization": {
            "mean": train_stats["mean"],
            "std": train_stats["std"],
            "source_split": train_stats.get("source_split", "train"),
        },
        "models": {},
    }

    for model_key in config.selected_models:
        result = run_model_experiment(model_key, config, class_names, train_stats)
        metrics["models"][model_key] = result
        write_json(config.models_dir / "metrics.json", metrics)

    best_model_key = choose_best_model(metrics["models"])
    best_checkpoint = config.models_dir / MODEL_SPECS[best_model_key]["checkpoint"]
    best_model_path = config.models_dir / "best_model.pt"
    shutil.copyfile(best_checkpoint, best_model_path)
    metrics["best_model"] = {
        "model_key": best_model_key,
        "display_name": MODEL_SPECS[best_model_key]["display_name"],
        "checkpoint_path": best_checkpoint.as_posix(),
        "best_model_path": best_model_path.as_posix(),
        "test_accuracy": metrics["models"][best_model_key]["test"]["accuracy"],
        "target_reached": metrics["models"][best_model_key]["test"]["accuracy"] >= TARGET_TEST_ACCURACY,
    }
    best_examples = metrics["models"][best_model_key]["prediction_examples"]
    plot_prediction_grid(
        examples=restore_example_paths(best_examples["correct"], config.dataset_root),
        output_path=config.reports_dir / "predictions" / "correct_predictions.png",
        title=f"{metrics['best_model']['display_name']} - acertos no teste",
    )
    plot_prediction_grid(
        examples=restore_example_paths(best_examples["wrong"], config.dataset_root),
        output_path=config.reports_dir / "predictions" / "wrong_predictions.png",
        title=f"{metrics['best_model']['display_name']} - erros no teste",
    )
    metrics["run_finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(config.models_dir / "metrics.json", metrics)
    write_reports(metrics, config)

    print(f"Best model: {metrics['best_model']['display_name']}")
    print(f"Best test accuracy: {metrics['best_model']['test_accuracy']:.4f}")
    print(f"Metrics saved to: {config.models_dir / 'metrics.json'}")
    print(f"Experiment report saved to: {config.reports_dir / 'experiment_report.md'}")


def choose_device(force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    if not force_cpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_output_dirs(config: TrainConfig) -> None:
    for path in (
        config.models_dir,
        config.reports_dir,
        config.reports_dir / "figures",
        config.reports_dir / "confusion_matrices",
        config.reports_dir / "predictions",
        config.training_report_path.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def class_names_from_mapping(class_to_idx: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


def write_class_names(path: Path, class_names: list[str], class_to_idx: dict[str, int]) -> None:
    payload = {
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(index): name for index, name in enumerate(class_names)},
    }
    write_json(path, payload)


def run_model_experiment(
    model_key: str,
    config: TrainConfig,
    class_names: list[str],
    train_stats: dict[str, Any],
) -> dict[str, Any]:
    spec = MODEL_SPECS[model_key]
    display_name = spec["display_name"]
    set_seed(config.seed)

    dataloaders = build_dataloaders(
        dataset_root=config.dataset_root,
        processed_dir=config.processed_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        input_size=config.input_size,
    )

    model = spec["class"](num_classes=len(class_names)).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rates[model_key],
        weight_decay=config.weight_decays[model_key],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    parameter_count = spec["counter"](model)
    checkpoint_path = config.models_dir / spec["checkpoint"]
    history: list[dict[str, Any]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    train_started = time.perf_counter()

    print(f"\nTraining {display_name} on {config.device}")
    print(
        f"epochs={config.epochs[model_key]} batch_size={config.batch_size} "
        f"lr={config.learning_rates[model_key]} weight_decay={config.weight_decays[model_key]}"
    )

    for epoch in range(1, config.epochs[model_key] + 1):
        epoch_started = time.perf_counter()
        train_metrics = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=config.device,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=config.device,
            collect_outputs=False,
        )
        scheduler.step(val_metrics["accuracy"])
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.perf_counter() - epoch_started

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": current_lr,
            "epoch_seconds": epoch_seconds,
        }
        history.append(epoch_result)

        improved = val_metrics["accuracy"] > best_val_accuracy + config.min_delta
        if improved:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                model_key=model_key,
                class_names=class_names,
                train_stats=train_stats,
                config=config,
                parameter_count=parameter_count,
                best_epoch=best_epoch,
                best_val_accuracy=best_val_accuracy,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"{display_name} epoch {epoch:02d}/{config.epochs[model_key]} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"lr={current_lr:.6f} "
            f"time={epoch_seconds:.1f}s"
        )

        if epochs_without_improvement >= config.patience:
            print(
                f"Early stopping {display_name}: no validation improvement for "
                f"{config.patience} epochs."
            )
            break

    train_seconds = time.perf_counter() - train_started
    load_model_state(model, checkpoint_path, config.device)
    test_metrics = evaluate(
        model=model,
        dataloader=dataloaders["test"],
        criterion=criterion,
        device=config.device,
        collect_outputs=True,
    )

    y_true = test_metrics["y_true"]
    y_pred = test_metrics["y_pred"]
    probabilities = test_metrics["probabilities"]
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    curve_path = config.reports_dir / "figures" / spec["curve"]
    matrix_path = config.reports_dir / "confusion_matrices" / spec["confusion_matrix"]
    plot_history(history, curve_path, display_name)
    plot_confusion_matrix(matrix, class_names, matrix_path, display_name)

    test_paths = getattr(dataloaders["test"].dataset, "relative_paths")
    example_payload = build_prediction_examples(
        dataset_root=config.dataset_root,
        relative_paths=test_paths,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        class_names=class_names,
        max_examples=12,
    )

    if model_key == "improved" or len(config.selected_models) == 1:
        plot_prediction_grid(
            examples=example_payload["correct"],
            output_path=config.reports_dir / "predictions" / "correct_predictions.png",
            title=f"{display_name} - acertos no teste",
        )
        plot_prediction_grid(
            examples=example_payload["wrong"],
            output_path=config.reports_dir / "predictions" / "wrong_predictions.png",
            title=f"{display_name} - erros no teste",
        )

    result = {
        "display_name": display_name,
        "parameters": parameter_count,
        "epochs_requested": config.epochs[model_key],
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "checkpoint_path": checkpoint_path.as_posix(),
        "curve_path": curve_path.as_posix(),
        "confusion_matrix_path": matrix_path.as_posix(),
        "training_seconds": train_seconds,
        "optimizer": "AdamW",
        "learning_rate": config.learning_rates[model_key],
        "weight_decay": config.weight_decays[model_key],
        "loss_function": "CrossEntropyLoss",
        "regularization": spec["regularization"],
        "history": history,
        "test": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
            "top_confusions": top_confusions(matrix, class_names),
            "target_reached": test_metrics["accuracy"] >= TARGET_TEST_ACCURACY,
        },
        "prediction_examples": {
            "correct": strip_image_objects(example_payload["correct"]),
            "wrong": strip_image_objects(example_payload["wrong"]),
        },
    }
    print(
        f"{display_name} test_loss={result['test']['loss']:.4f} "
        f"test_acc={result['test']['accuracy']:.4f} best_epoch={best_epoch}"
    )
    return result


def train_one_epoch(
    model: nn.Module,
    dataloader: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: Any,
    criterion: nn.Module,
    device: torch.device,
    collect_outputs: bool,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[list[float]] = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (preds == labels).sum().item()
        total += batch_size

        if collect_outputs:
            probs = torch.softmax(outputs, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())

    result: dict[str, Any] = {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }
    if collect_outputs:
        result.update(
            {
                "y_true": y_true,
                "y_pred": y_pred,
                "probabilities": probabilities,
            }
        )
    return result


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_key: str,
    class_names: list[str],
    train_stats: dict[str, Any],
    config: TrainConfig,
    parameter_count: int,
    best_epoch: int,
    best_val_accuracy: float,
) -> None:
    checkpoint = {
        "model_key": model_key,
        "model_name": MODEL_SPECS[model_key]["display_name"],
        "model_state_dict": model.state_dict(),
        "num_classes": len(class_names),
        "class_names": class_names,
        "input_size": list(config.input_size),
        "normalization": {
            "mean": train_stats["mean"],
            "std": train_stats["std"],
        },
        "seed": config.seed,
        "parameters": parameter_count,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(checkpoint, path)


def load_model_state(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def plot_history(history: list[dict[str, Any]], output_path: Path, title: str) -> None:
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    train_acc = [item["train_accuracy"] for item in history]
    val_acc = [item["val_accuracy"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="Treino")
    axes[0].plot(epochs, val_loss, label="Validacao")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Treino")
    axes[1].plot(epochs, val_acc, label="Validacao")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{title} - matriz de confusao")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def build_prediction_examples(
    dataset_root: Path,
    relative_paths: list[str],
    y_true: list[int],
    y_pred: list[int],
    probabilities: list[list[float]],
    class_names: list[str],
    max_examples: int,
) -> dict[str, list[dict[str, Any]]]:
    correct: list[dict[str, Any]] = []
    wrong: list[dict[str, Any]] = []

    for relative_path, true_idx, pred_idx, probs in zip(relative_paths, y_true, y_pred, probabilities, strict=True):
        confidence = float(probs[pred_idx])
        sample_path = resolve_sample_path(dataset_root, relative_path)
        item = {
            "relative_path": relative_path,
            "path": sample_path.as_posix(),
            "true_class": class_names[true_idx],
            "predicted_class": class_names[pred_idx],
            "confidence": confidence,
        }
        if true_idx == pred_idx and len(correct) < max_examples:
            correct.append(item)
        elif true_idx != pred_idx and len(wrong) < max_examples:
            wrong.append(item)
        if len(correct) >= max_examples and len(wrong) >= max_examples:
            break

    return {"correct": correct, "wrong": wrong}


def resolve_sample_path(dataset_root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.exists():
        return candidate
    dataset_prefix = dataset_root.name + "/"
    if relative_path.startswith(dataset_prefix):
        return dataset_root / relative_path[len(dataset_prefix) :]
    return dataset_root / relative_path


def plot_prediction_grid(examples: list[dict[str, Any]], output_path: Path, title: str) -> None:
    columns = 4
    rows = max(1, int(np.ceil(max(1, len(examples)) / columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.0, rows * 3.2))
    axes_array = np.atleast_1d(axes).reshape(rows, columns)

    for ax in axes_array.ravel():
        ax.axis("off")

    if not examples:
        axes_array[0, 0].text(0.5, 0.5, "Nenhum exemplo disponivel", ha="center", va="center")
    else:
        for ax, example in zip(axes_array.ravel(), examples, strict=False):
            image = Image.open(example["path"]).convert("RGB")
            ax.imshow(image)
            ax.set_title(
                f"Real: {example['true_class']}\nPred: {example['predicted_class']} ({example['confidence']:.2f})",
                fontsize=8,
            )
            ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def strip_image_objects(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "relative_path": example["relative_path"],
            "true_class": example["true_class"],
            "predicted_class": example["predicted_class"],
            "confidence": example["confidence"],
        }
        for example in examples
    ]


def restore_example_paths(examples: list[dict[str, Any]], dataset_root: Path) -> list[dict[str, Any]]:
    restored: list[dict[str, Any]] = []
    for example in examples:
        item = dict(example)
        item["path"] = resolve_sample_path(dataset_root, item["relative_path"]).as_posix()
        restored.append(item)
    return restored


def top_confusions(matrix: np.ndarray, class_names: list[str], limit: int = 8) -> list[dict[str, Any]]:
    confusions: list[dict[str, Any]] = []
    for true_idx, true_name in enumerate(class_names):
        for pred_idx, pred_name in enumerate(class_names):
            if true_idx == pred_idx:
                continue
            count = int(matrix[true_idx, pred_idx])
            if count:
                confusions.append(
                    {
                        "true_class": true_name,
                        "predicted_class": pred_name,
                        "count": count,
                    }
                )
    return sorted(confusions, key=lambda item: item["count"], reverse=True)[:limit]


def choose_best_model(model_metrics: dict[str, Any]) -> str:
    return max(
        model_metrics,
        key=lambda key: (
            model_metrics[key]["test"]["accuracy"],
            model_metrics[key]["best_val_accuracy"],
        ),
    )


def write_reports(metrics: dict[str, Any], config: TrainConfig) -> None:
    report = build_markdown_report(metrics)
    experiment_report_path = config.reports_dir / "experiment_report.md"
    experiment_report_path.write_text(report, encoding="utf-8")
    try:
        config.training_report_path.write_text(report, encoding="utf-8")
    except OSError as exc:
        print(
            f"Warning: could not write {config.training_report_path}: {exc}. "
            f"Primary report remains available at {experiment_report_path}."
        )


def build_markdown_report(metrics: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Relatorio de treinamento e avaliacao das CNNs")
    lines.append("")
    lines.append(f"Data de execucao: {metrics.get('run_finished_at', metrics['run_started_at'])}")
    lines.append("")
    lines.append("## Configuracao")
    lines.append("")
    lines.append(f"- Dataset: `{metrics['dataset_root']}`")
    lines.append(f"- Split processado: `{metrics['processed_dir']}`")
    lines.append(f"- Seed: {metrics['seed']}")
    lines.append(f"- Device: `{metrics['device']}`")
    lines.append(f"- Entrada: `{metrics['input_size'][0]}x{metrics['input_size'][1]}` RGB")
    lines.append(f"- Batch size: {metrics['batch_size']}")
    lines.append(f"- Num workers: {metrics['num_workers']}")
    lines.append(f"- Funcao de loss: `CrossEntropyLoss`")
    lines.append(f"- Otimizador: `AdamW`")
    lines.append(f"- Meta de referencia no teste: {TARGET_TEST_ACCURACY:.0%}")
    lines.append("")
    lines.append("## Classes e splits")
    lines.append("")
    lines.append("| Split | Imagens |")
    lines.append("|---|---:|")
    for split, summary in metrics["split_summary"].items():
        lines.append(f"| {split} | {summary['total']} |")
    lines.append("")
    lines.append("Classes: " + ", ".join(f"`{name}`" for name in metrics["class_names"]))
    lines.append("")
    lines.append("## Resultados finais")
    lines.append("")
    lines.append("| Modelo | Parametros | Melhor epoca | Val acc | Test loss | Test acc | Meta 88% |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for model_key, result in metrics["models"].items():
        target = "atingida" if result["test"]["target_reached"] else "nao atingida"
        lines.append(
            f"| {result['display_name']} | {result['parameters']:,} | {result['best_epoch']} | "
            f"{result['best_val_accuracy']:.4f} | {result['test']['loss']:.4f} | "
            f"{result['test']['accuracy']:.4f} | {target} |"
        )
    lines.append("")

    best = metrics["best_model"]
    lines.append("## Melhor modelo")
    lines.append("")
    lines.append(
        f"O melhor modelo foi `{best['display_name']}`, com accuracy de teste "
        f"{best['test_accuracy']:.4f}."
    )
    if best["target_reached"]:
        lines.append("A meta de referencia de 88% foi atingida no conjunto de teste.")
    else:
        lines.append(
            "A meta de referencia de 88% nao foi atingida. A justificativa tecnica "
            "esta descrita na secao de analise."
        )
    lines.append("")

    lines.append("## Historico por epoca")
    lines.append("")
    for _model_key, result in metrics["models"].items():
        lines.append(f"### {result['display_name']}")
        lines.append("")
        lines.append("| Epoca | Train loss | Train acc | Val loss | Val acc | LR |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for item in result["history"]:
            lines.append(
                f"| {item['epoch']} | {item['train_loss']:.4f} | {item['train_accuracy']:.4f} | "
                f"{item['val_loss']:.4f} | {item['val_accuracy']:.4f} | {item['learning_rate']:.6f} |"
            )
        lines.append("")

    lines.append("## Relatorio de classificacao")
    lines.append("")
    for _model_key, result in metrics["models"].items():
        lines.append(f"### {result['display_name']}")
        lines.append("")
        lines.append("| Classe | Precision | Recall | F1-score | Support |")
        lines.append("|---|---:|---:|---:|---:|")
        report = result["test"]["classification_report"]
        for class_name in metrics["class_names"]:
            item = report[class_name]
            lines.append(
                f"| {class_name} | {item['precision']:.4f} | {item['recall']:.4f} | "
                f"{item['f1-score']:.4f} | {int(item['support'])} |"
            )
        lines.append("")

    lines.append("## Comparacao tecnica")
    lines.append("")
    lines.append(
        "`BaselineCNN` usa tres blocos convolucionais simples com ReLU e MaxPool, "
        "seguido por classificador denso com dropout. Ela funciona como referencia "
        "mais compacta e menos regularizada."
    )
    lines.append("")
    lines.append(
        "`ImprovedCNN` aprofunda a extracao de caracteristicas com quatro estagios, "
        "duas convolucoes por estagio, BatchNorm2d, Dropout2d e pooling adaptativo. "
        "A maior capacidade e a normalizacao tendem a melhorar a generalizacao em "
        "classes visualmente parecidas, mas com maior custo computacional."
    )
    lines.append("")

    lines.append("## Analise de erros")
    lines.append("")
    for _model_key, result in metrics["models"].items():
        lines.append(f"### {result['display_name']}")
        lines.append("")
        confusions = result["test"]["top_confusions"]
        if confusions:
            lines.append("| Classe real | Classe predita | Ocorrencias |")
            lines.append("|---|---|---:|")
            for item in confusions:
                lines.append(
                    f"| {item['true_class']} | {item['predicted_class']} | {item['count']} |"
                )
        else:
            lines.append("Nao houve erros registrados no conjunto de teste.")
        lines.append("")

    best_result = metrics["models"][best["model_key"]]
    lines.append("Exemplos de acertos do melhor modelo:")
    lines.append("")
    for example in best_result["prediction_examples"]["correct"][:5]:
        lines.append(
            f"- `{example['relative_path']}`: real `{example['true_class']}`, "
            f"predito `{example['predicted_class']}`, confianca {example['confidence']:.4f}."
        )
    lines.append("")
    lines.append("Exemplos de erros do melhor modelo:")
    lines.append("")
    if best_result["prediction_examples"]["wrong"]:
        for example in best_result["prediction_examples"]["wrong"][:5]:
            lines.append(
                f"- `{example['relative_path']}`: real `{example['true_class']}`, "
                f"predito `{example['predicted_class']}`, confianca {example['confidence']:.4f}."
            )
    else:
        lines.append("- Nenhum erro encontrado nos exemplos coletados.")
    lines.append("")

    if not best["target_reached"]:
        lines.append("## Justificativa tecnica para accuracy abaixo de 88%")
        lines.append("")
        lines.append(
            "A accuracy ficou abaixo da referencia de 88% neste experimento. Os fatores "
            "tecnicos mais provaveis sao a baixa resolucao das imagens (64x64), a "
            "similaridade visual entre classes agricolas e vegetacao, e a limitacao "
            "de capacidade das CNNs treinadas do zero sem uso de modelos pre-treinados. "
            "A matriz de confusao deve ser usada para orientar novos ajustes de "
            "arquitetura, augmentations e numero de epocas."
        )
        lines.append("")

    lines.append("## Artefatos gerados")
    lines.append("")
    lines.append("- `models/baseline_best.pt`")
    lines.append("- `models/improved_best.pt`")
    lines.append("- `models/best_model.pt`")
    lines.append("- `models/class_names.json`")
    lines.append("- `models/metrics.json`")
    lines.append("- `reports/figures/baseline_accuracy_loss.png`")
    lines.append("- `reports/figures/improved_accuracy_loss.png`")
    lines.append("- `reports/confusion_matrices/baseline_confusion_matrix.png`")
    lines.append("- `reports/confusion_matrices/improved_confusion_matrix.png`")
    lines.append("- `reports/predictions/correct_predictions.png`")
    lines.append("- `reports/predictions/wrong_predictions.png`")
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
