"""Train the mask-guided six-unit defect head on the Quxian dataset.

The script deliberately freezes C3Net M6.  Every original image and its
horizontal flip share one source group, so they can never appear in opposite
train/test partitions.  The test split contains original images only.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import yaml

# Allow running this script from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.c3net import C3Net
from models.stitch_defect_head import DEFECT_CLASSES, StitchDefectHead, multilabel_defect_loss
from utils.checkpoint_utils import load_model_checkpoint, normalize_state_dict_keys


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    source_image: str
    augmentation: str
    image_path: Path
    targets: tuple[tuple[int, int, int, int], ...]
    valid: tuple[int, ...]


def read_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_samples(dataset_root: Path, labels_csv: Path) -> List[Sample]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            grouped[row["sample_id"]].append(row)

    samples: List[Sample] = []
    for sample_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["unit_id"]))
        if len(rows) != 6 or [int(row["unit_id"]) for row in rows] != list(range(6)):
            raise ValueError(f"{sample_id}: exactly six ordered unit labels are required")
        if any(row["review_status"] != "completed" for row in rows):
            raise ValueError(f"{sample_id}: labels are not finalized")
        source_images = {row["source_image"] for row in rows}
        augmentations = {row["augmentation"] for row in rows}
        image_files = {row["image_file"] for row in rows}
        if len(source_images) != 1 or len(augmentations) != 1 or len(image_files) != 1:
            raise ValueError(f"{sample_id}: inconsistent image metadata across units")
        targets = tuple(tuple(int(row[name]) for name in DEFECT_CLASSES) for row in rows)
        valid = tuple(int(row["valid"]) for row in rows)
        samples.append(Sample(
            sample_id=sample_id,
            source_image=next(iter(source_images)),
            augmentation=next(iter(augmentations)),
            image_path=dataset_root / next(iter(image_files)),
            targets=targets,
            valid=valid,
        ))
    return samples


def make_group_split(
    samples: Sequence[Sample], num_folds: int, fold: int, seed: int, val_fraction: float,
    include_flips_in_train: bool, test_original_only: bool,
) -> tuple[List[Sample], List[Sample], List[Sample], Dict]:
    if not 0 <= fold < num_folds:
        raise ValueError(f"fold must be in [0, {num_folds - 1}]")
    groups = sorted({sample.source_image for sample in samples})
    if len(groups) < num_folds:
        raise ValueError("fewer source groups than folds")
    rng = random.Random(seed)
    rng.shuffle(groups)
    fold_groups = [set(part.tolist()) for part in np.array_split(np.asarray(groups, dtype=object), num_folds)]
    test_groups = fold_groups[fold]
    remaining = sorted(set(groups) - test_groups)
    rng = random.Random(seed + 1000 + fold)
    rng.shuffle(remaining)
    val_count = max(1, round(len(remaining) * val_fraction))
    val_groups = set(remaining[:val_count])
    train_groups = set(remaining[val_count:])

    def select(group_set: set[str], mode: str) -> List[Sample]:
        selected = []
        for sample in samples:
            if sample.source_image not in group_set:
                continue
            if mode == "train" and not include_flips_in_train and sample.augmentation != "original":
                continue
            if mode in {"val", "test"} and test_original_only and sample.augmentation != "original":
                continue
            selected.append(sample)
        return sorted(selected, key=lambda item: int(item.sample_id))

    train, val, test = select(train_groups, "train"), select(val_groups, "val"), select(test_groups, "test")
    if not train or not val or not test:
        raise ValueError("empty split; adjust folds or validation fraction")
    manifest = {
        "fold": fold,
        "num_folds": num_folds,
        "train_source_images": sorted(train_groups),
        "validation_source_images": sorted(val_groups),
        "test_source_images": sorted(test_groups),
        "train_sample_ids": [sample.sample_id for sample in train],
        "validation_sample_ids": [sample.sample_id for sample in val],
        "test_sample_ids": [sample.sample_id for sample in test],
    }
    return train, val, test, manifest


class DefectDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], target_size: int) -> None:
        self.samples = list(samples)
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Tensor | str]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if image.size != (self.target_size, self.target_size):
            image = image.resize((self.target_size, self.target_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        rgb = torch.from_numpy(pixels.copy())
        normalised = (rgb - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
        return {
            "input": normalised,
            "rgb": rgb,
            "target": torch.tensor(sample.targets, dtype=torch.float32),
            "valid": torch.tensor(sample.valid, dtype=torch.float32),
            "sample_id": sample.sample_id,
            "source_image": sample.source_image,
        }


def load_segmenter(checkpoint_path: Path, device: torch.device) -> tuple[C3Net, Dict]:
    checkpoint = load_model_checkpoint(checkpoint_path, map_location=device)
    if "config" not in checkpoint or "model" not in checkpoint["config"]:
        raise RuntimeError("M6 checkpoint is missing embedded model configuration")
    model_config = checkpoint["config"]["model"]
    model = C3Net(model_config)
    missing, unexpected = model.load_state_dict(
        normalize_state_dict_keys(checkpoint["model_state_dict"]), strict=False
    )
    if missing or unexpected:
        raise RuntimeError(f"M6 checkpoint/model mismatch. Missing={missing}; unexpected={unexpected}")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, model_config


def class_positive_weights(samples: Iterable[Sample], max_weight: float) -> List[float]:
    total = np.zeros(4, dtype=np.float64)
    positive = np.zeros(4, dtype=np.float64)
    for sample in samples:
        targets = np.asarray(sample.targets, dtype=np.float64)
        valid = np.asarray(sample.valid, dtype=bool)
        total += valid.sum()
        positive += targets[valid].sum(axis=0)
    weights = []
    for count, positives in zip(total, positive):
        weights.append(1.0 if positives == 0 else min(float((count - positives) / positives), max_weight))
    return weights


def metrics_from_rows(rows: List[Dict], thresholds: float | Sequence[float] = 0.5) -> Dict:
    if isinstance(thresholds, (float, int)):
        thresholds = [float(thresholds)] * len(DEFECT_CLASSES)
    if len(thresholds) != len(DEFECT_CLASSES):
        raise ValueError("thresholds must provide one value per defect class")
    result: Dict[str, Dict[str, float]] = {}
    f1_values = []
    for class_index, class_name in enumerate(DEFECT_CLASSES):
        y_true = np.asarray([row["target"][class_index] for row in rows if row["valid"]], dtype=np.int32)
        y_prob = np.asarray([row["probability"][class_index] for row in rows if row["valid"]], dtype=np.float32)
        if not len(y_true):
            continue
        threshold = float(thresholds[class_index])
        y_pred = y_prob >= threshold
        tp = int(np.logical_and(y_pred, y_true == 1).sum())
        fp = int(np.logical_and(y_pred, y_true == 0).sum())
        fn = int(np.logical_and(~y_pred, y_true == 1).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        positive_count = int(y_true.sum())
        # Non-interpolated AP: mean precision at each positive-ranked item.
        # It remains well-defined for the very small per-class sample counts.
        if positive_count:
            order = np.argsort(-y_prob, kind="stable")
            ranked_target = y_true[order]
            ranked_precision = np.cumsum(ranked_target) / np.arange(1, len(ranked_target) + 1)
            average_precision = float(ranked_precision[ranked_target == 1].sum() / positive_count)
        else:
            average_precision = None
        result[class_name] = {
            "positive": positive_count,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "average_precision": average_precision,
        }
        if positive_count > 0:
            f1_values.append(f1)
    result["macro_f1_active_classes"] = float(np.mean(f1_values)) if f1_values else 0.0
    return result


def calibrate_thresholds(
    validation_rows: List[Dict],
    min_positive_source_images: int = 4,
    min_negative_source_images: int = 4,
) -> List[float]:
    """Choose one F1-maximising threshold per class using validation only.

    Ties use the value closest to 0.5.  A class with fewer than the configured
    positive or negative *source images* retains 0.5; counting six units from
    one photo as six independent threshold-calibration examples would leak
    image-specific appearance into the operating point.  Test predictions are
    never inspected by this function.
    """
    thresholds: List[float] = []
    for class_index in range(len(DEFECT_CLASSES)):
        rows = [row for row in validation_rows if row["valid"]]
        target = np.asarray([row["target"][class_index] for row in rows], dtype=np.int32)
        probability = np.asarray([row["probability"][class_index] for row in rows], dtype=np.float32)
        positive_sources = {row["source_image"] for row in rows if row["target"][class_index] == 1}
        negative_sources = {row["source_image"] for row in rows if row["target"][class_index] == 0}
        if (
            not len(target)
            or target.sum() == 0
            or target.sum() == len(target)
            or len(positive_sources) < min_positive_source_images
            or len(negative_sources) < min_negative_source_images
        ):
            thresholds.append(0.5)
            continue
        candidates = np.unique(np.clip(np.r_[0.01, probability, 0.99], 0.01, 0.99))
        best = (float("-inf"), float("-inf"), 0.5)
        for candidate in candidates:
            predicted = probability >= candidate
            tp = int(np.logical_and(predicted, target == 1).sum())
            fp = int(np.logical_and(predicted, target == 0).sum())
            fn = int(np.logical_and(~predicted, target == 1).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            # Prefer high F1, then a threshold nearer 0.5.
            score = (f1, -abs(float(candidate) - 0.5), float(candidate))
            if score > best:
                best = score
        thresholds.append(float(best[2]))
    return thresholds


@torch.inference_mode()
def evaluate(head: StitchDefectHead, segmenter: C3Net, loader: DataLoader, device: torch.device, use_centerline: bool) -> tuple[Dict, List[Dict]]:
    head.eval()
    rows: List[Dict] = []
    for batch in loader:
        image = batch["input"].to(device)
        rgb = batch["rgb"].to(device)
        output = segmenter(image)
        centerline = output.get("centerline_mask") if use_centerline else None
        prediction = head(output["final_mask"], centerline, rgb)["unit_probabilities"].cpu().numpy()
        targets = batch["target"].numpy()
        valid = batch["valid"].numpy()
        for image_index, sample_id in enumerate(batch["sample_id"]):
            for unit_id in range(6):
                rows.append({
                    "sample_id": sample_id,
                    "source_image": batch["source_image"][image_index],
                    "unit_id": unit_id,
                    "valid": bool(valid[image_index, unit_id]),
                    "target": targets[image_index, unit_id].astype(int).tolist(),
                    "probability": prediction[image_index, unit_id].astype(float).tolist(),
                })
    return metrics_from_rows(rows), rows


def save_json(path: Path, content: Dict | List) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen-M6 six-unit defect head")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=None, help="Override config data.fold")
    parser.add_argument("--dry-run", action="store_true", help="Validate labels and print split without loading M6")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.fold is not None:
        config["data"]["fold"] = args.fold
    seed_everything(int(config["seed"]))

    data_cfg = config["data"]
    samples = load_samples(Path(data_cfg["dataset_root"]), Path(data_cfg["labels_csv"]))
    train_samples, val_samples, test_samples, manifest = make_group_split(
        samples, int(data_cfg["num_folds"]), int(data_cfg["fold"]), int(config["seed"]),
        float(data_cfg["validation_group_fraction"]), bool(data_cfg["include_flips_in_train"]),
        bool(data_cfg["test_original_only"]),
    )
    output_dir = Path(config["output"]["root"]) / f"fold_{data_cfg['fold']}"
    save_json(output_dir / "split_manifest.json", manifest)
    weights = class_positive_weights(train_samples, float(config["training"]["max_positive_weight"]))
    print(f"Fold {data_cfg['fold']}: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    print("Training pos_weight:", dict(zip(DEFECT_CLASSES, weights)))
    print(f"Split manifest: {output_dir / 'split_manifest.json'}")
    if args.dry_run:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmenter, model_config = load_segmenter(Path(config["segmenter"]["checkpoint"]), device)
    target_size = int(model_config["image_processing"]["target_size"])
    model_cfg = config["model"]
    head = StitchDefectHead(**model_cfg).to(device)
    train_cfg = config["training"]
    loader_args = {"batch_size": int(train_cfg["batch_size"]), "num_workers": int(train_cfg["num_workers"]), "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(DefectDataset(train_samples, target_size), shuffle=True, **loader_args)
    val_loader = DataLoader(DefectDataset(val_samples, target_size), shuffle=False, **loader_args)
    test_loader = DataLoader(DefectDataset(test_samples, target_size), shuffle=False, **loader_args)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg["epochs"]), eta_min=float(train_cfg["learning_rate"]) * 0.05)
    amp_enabled = bool(train_cfg["use_amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_score, stale = -1.0, 0
    history = []
    best_path = output_dir / "checkpoints" / "defect_head_best.pt"

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        head.train()
        loss_values = []
        for batch in train_loader:
            image, rgb = batch["input"].to(device), batch["rgb"].to(device)
            target, valid = batch["target"].to(device), batch["valid"].to(device)
            with torch.inference_mode():
                segmenter_output = segmenter(image)
                centerline = segmenter_output.get("centerline_mask") if config["segmenter"]["use_centerline"] else None
                final_mask = segmenter_output["final_mask"]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = head(final_mask, centerline, rgb)["unit_logits"]
                loss = multilabel_defect_loss(logits, target, valid, weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_values.append(float(loss.detach().cpu()))
        scheduler.step()
        val_metrics, _ = evaluate(head, segmenter, val_loader, device, bool(config["segmenter"]["use_centerline"]))
        score = float(val_metrics["macro_f1_active_classes"])
        record = {"epoch": epoch, "train_loss": float(np.mean(loss_values)), "validation": val_metrics}
        history.append(record)
        print(f"Epoch {epoch:03d} | loss={record['train_loss']:.4f} | val_macro_f1={score:.4f}")
        if score > best_score:
            best_score, stale = score, 0
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"head_state_dict": head.state_dict(), "config": config, "fold": data_cfg["fold"], "best_validation_metrics": val_metrics}, best_path)
        else:
            stale += 1
            if stale >= int(train_cfg["early_stop_patience"]):
                print(f"Early stopped after {epoch} epochs.")
                break

    best = torch.load(best_path, map_location=device)
    head.load_state_dict(best["head_state_dict"])
    val_metrics, val_rows = evaluate(head, segmenter, val_loader, device, bool(config["segmenter"]["use_centerline"]))
    threshold_cfg = config.get("threshold_calibration", {})
    thresholds = calibrate_thresholds(
        val_rows,
        min_positive_source_images=int(threshold_cfg.get("min_positive_source_images", 4)),
        min_negative_source_images=int(threshold_cfg.get("min_negative_source_images", 4)),
    )
    val_metrics_calibrated = metrics_from_rows(val_rows, thresholds)
    _, test_rows = evaluate(head, segmenter, test_loader, device, bool(config["segmenter"]["use_centerline"]))
    test_metrics_fixed_0_5 = metrics_from_rows(test_rows, 0.5)
    test_metrics = metrics_from_rows(test_rows, thresholds)
    save_json(output_dir / "history.json", history)
    save_json(output_dir / "validation_metrics.json", val_metrics)
    save_json(output_dir / "validation_metrics_calibrated.json", val_metrics_calibrated)
    save_json(output_dir / "calibrated_thresholds.json", dict(zip(DEFECT_CLASSES, thresholds)))
    save_json(output_dir / "test_metrics_fixed_0_5.json", test_metrics_fixed_0_5)
    save_json(output_dir / "test_metrics.json", test_metrics)
    save_json(output_dir / "validation_predictions.json", val_rows)
    save_json(output_dir / "test_predictions.json", test_rows)
    print("Best checkpoint:", best_path)
    print("Held-out original-image test metrics:", json.dumps(test_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
