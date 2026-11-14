"""Normal-reference calibration and OOF evaluation for TSR-Defect.

TSR-NR is a lightweight two-stage defect unit designed for the available data:

1. M6 masks are converted into the existing self-reference trajectory scores.
2. For every source-grouped fold, the *normal* score envelope is learned from
   DB/SD/XJ/RX normal images not assigned to that fold.  Each class threshold
   is then calibrated with the fold's normal images and only the 36 available
   defect images.  The held-out 12 defects and held-out normal images are used
   once each for final evaluation.

It deliberately does not use the manuscript's microscope calibration, pixel to
millimetre conversion, contour-centroid pitch, or area rules.  It is a
trajectory self-reference classifier with a real normal reference population.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.trajectory_self_reference_detector import TSRConfig, TrajectorySelfReferenceDetector  # noqa: E402


CLASSES = ("normal", "skip", "break", "skew", "loose")
DEFECTS = CLASSES[1:]
LOCAL = ("skip", "break", "loose")


def parse_source(value: str) -> tuple[str, Path]:
    name, separator, raw = value.partition("=")
    if not separator or not name or not raw:
        raise argparse.ArgumentTypeError("--normal-source requires NAME=MASK_DIRECTORY")
    return name, Path(raw)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_labels(path: Path) -> Dict[str, str]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["augmentation"] == "original" and row["review_status"] == "completed":
                grouped[row["sample_id"]].append(row)
    output = {}
    for sample_id, rows in grouped.items():
        target = [name for name in DEFECTS if any(int(row[name]) for row in rows)]
        if len(target) != 1:
            raise ValueError(f"{sample_id}: expected one defect type, got {target}")
        output[sample_id] = target[0]
    if len(output) != 48:
        raise ValueError(f"Expected 48 labelled defect images, got {len(output)}")
    return output


def load_mask(path: Path) -> np.ndarray:
    probability = path.with_suffix(".npy")
    if probability.exists():
        return np.load(probability).astype(np.float32)
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def image_scores(mask: np.ndarray, detector: TrajectorySelfReferenceDetector) -> Dict[str, float]:
    units = detector.detect(mask)["units"]
    # The first/last segment can be naturally cropped by the camera.  They
    # remain for global skew, but cannot dominate local skip/break/loose scores.
    usable = [unit for unit in units if not unit["evidence"].get("edge_truncated", False)] or units
    values = {name: max(float(unit["scores"][name]) for unit in usable) for name in LOCAL}
    values["skew"] = max(float(unit["scores"]["skew"]) for unit in units)
    return values


def load_defects(mask_dir: Path, labels: Dict[str, str], detector: TrajectorySelfReferenceDetector) -> List[Dict]:
    rows = []
    for sample_id, target in sorted(labels.items()):
        base = mask_dir / f"{sample_id}.png"
        if not base.exists() and not base.with_suffix(".npy").exists():
            raise FileNotFoundError(f"Missing M6 output for defect {sample_id} in {mask_dir}")
        rows.append({"id": sample_id, "domain": "quexian", "target": target, "scores": image_scores(load_mask(base), detector)})
    return rows


def load_normals(sources: Sequence[tuple[str, Path]], detector: TrajectorySelfReferenceDetector) -> List[Dict]:
    rows = []
    for domain, folder in sources:
        paths = sorted(folder.glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No M6 masks in {folder}")
        for path in paths:
            rows.append({"id": f"{domain}:{path.stem}", "domain": domain, "target": "normal", "scores": image_scores(load_mask(path), detector)})
    return rows


def grouped_folds(values: Sequence[Dict], count: int, seed: int, key: str = "id") -> Dict[str, int]:
    by_domain: Dict[str, List[Dict]] = defaultdict(list)
    for value in values:
        by_domain[value["domain"]].append(value)
    fold_by_id = {}
    for domain, rows in by_domain.items():
        rows = list(rows)
        random.Random(f"{seed}:{domain}").shuffle(rows)
        for index, row in enumerate(rows):
            fold_by_id[row[key]] = index % count
    return fold_by_id


def binary_f1(target: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.logical_and(target, predicted).sum())
    fp = int(np.logical_and(~target, predicted).sum())
    fn = int(np.logical_and(target, ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall


def choose_threshold(calibration: Sequence[Dict], defect: str) -> float:
    target = np.asarray([row["target"] == defect for row in calibration], dtype=bool)
    scores = np.asarray([row["scores"][defect] for row in calibration], dtype=np.float64)
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    best = (-1.0, -1.0, -1.0, 0.5)
    for threshold in candidates:
        f1, precision, recall = binary_f1(target, scores >= threshold)
        # Prefer fewer false alarms whenever F1 is tied, then higher recall.
        candidate = (f1, precision, recall, -float(threshold))
        if candidate > best:
            best = candidate
    return float(-best[3])


def predict(row: Dict, thresholds: Dict[str, float]) -> Dict:
    ratios = {name: row["scores"][name] / max(thresholds[name], 1e-6) for name in DEFECTS}
    winner = max(DEFECTS, key=lambda name: (ratios[name], row["scores"][name]))
    label = winner if ratios[winner] >= 1.0 else "normal"
    return {"prediction": label, "ratios": ratios}


def class_metrics(target: Sequence[str], prediction: Sequence[str]) -> Dict:
    result = {"image_count": len(target), "accuracy": float(np.mean(np.asarray(target) == np.asarray(prediction)))}
    f1s = []
    for label in CLASSES:
        tp = sum(t == label and p == label for t, p in zip(target, prediction))
        fp = sum(t != label and p == label for t, p in zip(target, prediction))
        fn = sum(t == label and p != label for t, p in zip(target, prediction))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label] = {"support": sum(t == label for t in target), "precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
    result["macro_f1_5class"] = float(np.mean(f1s))
    result["defect_macro_f1_4class"] = float(np.mean(f1s[1:]))
    normal = result["normal"]
    result["normal_false_alarm_rate"] = float(1.0 - normal["recall"])
    positives = [i for i, t in enumerate(target) if t != "normal"]
    result["defect_alarm_recall"] = float(np.mean([prediction[i] != "normal" for i in positives])) if positives else 0.0
    result["defect_type_accuracy"] = float(np.mean([prediction[i] == target[i] for i in positives])) if positives else 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Normal-reference OOF evaluation of TSR-Defect")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--defect-mask-dir", type=Path, required=True)
    parser.add_argument("--normal-source", type=parse_source, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    allowed = {field.name for field in fields(TSRConfig)}
    detector = TrajectorySelfReferenceDetector(TSRConfig(**{key: value for key, value in raw.items() if key in allowed}))
    labels = read_labels(args.labels)
    defects = load_defects(args.defect_mask_dir, labels, detector)
    normals = load_normals(args.normal_source, detector)
    defect_folds = grouped_folds(defects, args.folds, args.seed)
    normal_folds = grouped_folds(normals, args.folds, args.seed)

    all_predictions = []
    fold_reports = []
    for fold in range(args.folds):
        defect_cal = [row for row in defects if defect_folds[row["id"]] != fold]
        normal_cal = [row for row in normals if normal_folds[row["id"]] != fold]
        thresholds = {name: choose_threshold([*defect_cal, *normal_cal], name) for name in DEFECTS}
        test = [row for row in defects if defect_folds[row["id"]] == fold] + [row for row in normals if normal_folds[row["id"]] == fold]
        fold_rows = []
        for row in test:
            outcome = predict(row, thresholds)
            record = {"fold": fold, **row, **outcome, "thresholds": thresholds, "correct": outcome["prediction"] == row["target"]}
            fold_rows.append(record)
        all_predictions.extend(fold_rows)
        fold_reports.append({"fold": fold, "thresholds": thresholds, "calibration": {"defect_images": len(defect_cal), "normal_images": len(normal_cal)}, "test": class_metrics([row["target"] for row in fold_rows], [row["prediction"] for row in fold_rows])})

    metric = class_metrics([row["target"] for row in all_predictions], [row["prediction"] for row in all_predictions])
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "tsr_nr_oof_predictions.json", all_predictions)
    save_json(args.output / "tsr_nr_fold_reports.json", fold_reports)
    save_json(args.output / "tsr_nr_summary.json", {
        "detector": "TSR-Defect-NR (normal-reference two-stage trajectory classifier)",
        "detector_config": asdict(detector.config),
        "protocol": "Four-fold source-grouped OOF. In every fold, class thresholds are calibrated only with 36 labelled defect images and normal images outside that fold; the held-out 12 defect images and held-out normal images are each predicted once.",
        "normal_sources": {name: str(path) for name, path in args.normal_source},
        "metrics": metric,
    })
    with (args.output / "tsr_nr_metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for name in ("accuracy", "macro_f1_5class", "defect_macro_f1_4class", "normal_false_alarm_rate", "defect_alarm_recall", "defect_type_accuracy"):
            writer.writerow([name, metric[name]])
        for name in CLASSES:
            for key in ("precision", "recall", "f1", "support"):
                writer.writerow([f"{name}_{key}", metric[name][key]])
    print(json.dumps(metric, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
