"""Four-fold, source-grouped validation for TSR-Defect rules.

The detector itself has no trainable weights.  In each outer fold, class-wise
score thresholds are selected only from the other 36 original images, then
frozen for the held-out 12 originals.  It runs both an oracle mask condition
(binary GT mask) and the deployment condition (M6 predicted mask).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.trajectory_self_reference_detector import DEFECT_CLASSES, TSRConfig, TrajectorySelfReferenceDetector, decide_defect_classes


def save_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def load_labels(path: Path) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["augmentation"] != "original":
                continue
            grouped.setdefault(row["sample_id"], []).append(row)
    output = {}
    for sample_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["unit_id"]))
        if len(rows) != 6 or any(row["review_status"] != "completed" for row in rows):
            raise ValueError(f"{sample_id}: expected six completed original-image labels")
        output[sample_id] = rows
    return output


def fold_groups(sample_ids: Sequence[str], num_folds: int, seed: int) -> List[List[str]]:
    groups = list(sorted(sample_ids))
    random.Random(seed).shuffle(groups)
    return [list(part) for part in np.array_split(np.asarray(groups, dtype=object), num_folds)]


def choose_threshold(rows: Iterable[Dict], class_index: int) -> float:
    rows = list(rows)
    target = np.asarray([row["target"][class_index] for row in rows if row["valid"]], dtype=np.int32)
    scores = np.asarray([row["scores"][DEFECT_CLASSES[class_index]] for row in rows if row["valid"]], dtype=np.float32)
    if not len(target) or target.sum() == 0 or target.sum() == len(target):
        return 0.50
    candidates = np.unique(np.clip(np.r_[0.0, scores, 1.0], 0.0, 1.0))
    # Maximise F1; break ties toward 0.5, avoiding arbitrary extreme rules.
    best = (-1.0, -1.0, 0.5)
    for threshold in candidates:
        pred = scores >= threshold
        tp = int(np.logical_and(pred, target == 1).sum())
        fp = int(np.logical_and(pred, target == 0).sum())
        fn = int(np.logical_and(~pred, target == 1).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = (f1, -abs(float(threshold) - 0.5), float(threshold))
        if candidate > best:
            best = candidate
    return float(best[2])


def evaluate_rows(rows: Sequence[Dict]) -> Dict:
    result: Dict[str, object] = {}
    f1s = []
    for index, name in enumerate(DEFECT_CLASSES):
        selected = [row for row in rows if row["valid"]]
        target = np.asarray([row["target"][index] for row in selected], dtype=np.int32)
        score = np.asarray([row["scores"][name] for row in selected], dtype=np.float32)
        pred = np.asarray([row["prediction"][name] for row in selected], dtype=bool)
        tp = int(np.logical_and(pred, target == 1).sum())
        fp = int(np.logical_and(pred, target == 0).sum())
        tn = int(np.logical_and(~pred, target == 0).sum())
        fn = int(np.logical_and(~pred, target == 1).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / len(target) if len(target) else 0.0
        if target.sum():
            order = np.argsort(-score, kind="stable")
            ranked = target[order]
            ap = float((np.cumsum(ranked) / np.arange(1, len(ranked) + 1))[ranked == 1].sum() / target.sum())
            f1s.append(f1)
        else:
            ap = None
        result[name] = {
            "positive": int(target.sum()), "total": int(len(target)),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "average_precision": ap,
        }
    result["macro_f1_active_classes"] = float(np.mean(f1s)) if f1s else 0.0
    result["mean_class_accuracy"] = float(np.mean([
        result[name]["accuracy"] for name in DEFECT_CLASSES
    ])) if DEFECT_CLASSES else 0.0
    target_matrix = np.asarray([row["target"] for row in selected], dtype=np.int32)
    pred_matrix = np.asarray([
        [int(row["prediction"][name]) for name in DEFECT_CLASSES] for row in selected
    ], dtype=np.int32)
    result["exact_unit_accuracy"] = float(np.all(target_matrix == pred_matrix, axis=1).mean()) if len(selected) else 0.0
    return result


def run_condition(
    condition: str, mask_dir: Path, labels: Dict[str, List[Dict]], detector: TrajectorySelfReferenceDetector,
    folds: Sequence[Sequence[str]], output_dir: Path,
) -> Dict:
    evidence: Dict[str, List[Dict]] = {}
    for sample_id in labels:
        probability_path = mask_dir / f"{sample_id}.npy"
        png_path = mask_dir / f"{sample_id}.png"
        if probability_path.exists():
            mask = np.load(probability_path).astype(np.float32)
        elif png_path.exists():
            mask = np.asarray(Image.open(png_path).convert("L"), dtype=np.float32) / 255.0
        else:
            raise FileNotFoundError(f"{condition}: missing probability/mask {probability_path} or {png_path}")
        units = detector.detect(mask)["units"]
        evidence[sample_id] = units

    all_oof: List[Dict] = []
    fold_reports = []
    all_ids = set(labels)
    for fold_index, test_ids in enumerate(folds):
        test_ids = set(test_ids)
        calibration_ids = all_ids - test_ids
        calibration_rows, test_rows = [], []
        for sample_id in sorted(labels):
            for label, unit in zip(labels[sample_id], evidence[sample_id]):
                row = {
                    "sample_id": sample_id,
                    "unit_id": int(label["unit_id"]),
                    "valid": bool(int(label["valid"])),
                    "target": [int(label[name]) for name in DEFECT_CLASSES],
                    "scores": unit["scores"],
                    "evidence": unit["evidence"],
                }
                (calibration_rows if sample_id in calibration_ids else test_rows).append(row)
        thresholds = [choose_threshold(calibration_rows, index) for index in range(len(DEFECT_CLASSES))]
        for row in test_rows:
            row["fold"] = fold_index
            row["thresholds"] = dict(zip(DEFECT_CLASSES, thresholds))
            threshold_map = dict(zip(DEFECT_CLASSES, thresholds))
            row["prediction"] = decide_defect_classes(
                row["scores"], threshold_map,
                local_defects_mutually_exclusive=detector.config.local_defects_mutually_exclusive,
            )
        fold_metrics = evaluate_rows(test_rows)
        fold_reports.append({
            "fold": fold_index,
            "calibration_sample_count": len(calibration_ids),
            "test_sample_ids": sorted(test_ids),
            "thresholds": dict(zip(DEFECT_CLASSES, thresholds)),
            "test_metrics": fold_metrics,
        })
        all_oof.extend(test_rows)
    summary = {
        "condition": condition,
        "detector": "TSR-Defect",
        "folds": fold_reports,
        "oof_metrics": evaluate_rows(all_oof),
        "oof_sample_count": len({row["sample_id"] for row in all_oof}),
        "oof_unit_count": len(all_oof),
    }
    save_json(output_dir / f"{condition}_summary.json", summary)
    save_json(output_dir / f"{condition}_oof_predictions.json", all_oof)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TSR-Defect on original images only")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--gt-mask-dir", type=Path, required=True)
    parser.add_argument("--m6-mask-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    permitted = {item.name for item in fields(TSRConfig)}
    detector = TrajectorySelfReferenceDetector(TSRConfig(**{key: value for key, value in raw_config.items() if key in permitted}))
    labels = load_labels(args.labels)
    if len(labels) != 48:
        raise ValueError(f"expected 48 original images, found {len(labels)}")
    splits = fold_groups(list(labels), args.folds, args.seed)
    save_json(args.output / "validation_protocol.json", {
        "detector_config": asdict(detector.config), "folds": splits,
        "protocol": "Thresholds are selected from 36 original calibration images per fold; flipped images are excluded."
    })
    gt = run_condition("gt_mask", args.gt_mask_dir, labels, detector, splits, args.output)
    m6 = run_condition("m6_mask", args.m6_mask_dir, labels, detector, splits, args.output)
    save_json(args.output / "summary.json", {"gt_mask": gt["oof_metrics"], "m6_mask": m6["oof_metrics"]})
    for name, result in (("GT mask", gt), ("M6 mask", m6)):
        metrics = result["oof_metrics"]
        print(f"{name}: macro F1={metrics['macro_f1_active_classes']:.4f}, mean class accuracy={metrics['mean_class_accuracy']:.4f}, exact unit accuracy={metrics['exact_unit_accuracy']:.4f}")
        for defect in DEFECT_CLASSES:
            item = metrics[defect]
            print(f"  {defect}: Acc={item['accuracy']:.4f}, F1={item['f1']:.4f}, P={item['precision']:.4f}, R={item['recall']:.4f}, AP={item['average_precision']:.4f}")


if __name__ == "__main__":
    main()
