"""Fair end-to-end deblurring ablation for TSR-Defect.

For every source-grouped outer fold, TSR-Defect operating thresholds are
selected from *only the 36 clear M6 masks* in the calibration split.  The
identical thresholds are then applied to held-out clear, direct-blur and
deblurred masks.  Consequently any change is attributable to the image
condition (and upstream restoration), rather than condition-specific tuning.

Local defects (skip, break, loose) are evaluated per valid stitch unit.
Because the available skew label is image-level (all six units share it),
skew is evaluated once per image.  The mixed macro F1 averages those four
properly defined class F1 values.
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

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.trajectory_self_reference_detector import (  # noqa: E402
    DEFECT_CLASSES,
    TSRConfig,
    TrajectorySelfReferenceDetector,
    decide_defect_classes,
)

LOCAL_CLASSES = ("skip", "break", "loose")


def save_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def load_labels(path: Path) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["augmentation"] == "original":
                grouped.setdefault(row["sample_id"], []).append(row)
    result: Dict[str, List[Dict[str, str]]] = {}
    for sample_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["unit_id"]))
        if len(rows) != 6 or any(row["review_status"] != "completed" for row in rows):
            raise ValueError(f"{sample_id}: expected six completed original labels")
        result[sample_id] = rows
    if len(result) != 48:
        raise ValueError(f"Expected 48 original images, found {len(result)}")
    return result


def fold_groups(sample_ids: Sequence[str], num_folds: int, seed: int) -> List[List[str]]:
    values = sorted(sample_ids)
    random.Random(seed).shuffle(values)
    return [list(part) for part in np.array_split(np.asarray(values, dtype=object), num_folds)]


def metric_from_arrays(target: np.ndarray, prediction: np.ndarray, score: np.ndarray) -> Dict[str, float | int | None]:
    target = target.astype(bool)
    prediction = prediction.astype(bool)
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ap: float | None = None
    if int(target.sum()):
        order = np.argsort(-score, kind="stable")
        ranked = target[order]
        ap = float((np.cumsum(ranked) / np.arange(1, len(ranked) + 1))[ranked].sum() / target.sum())
    return {
        "positive": int(target.sum()), "total": int(target.size), "tp": tp, "fp": fp,
        "tn": tn, "fn": fn, "accuracy": (tp + tn) / target.size if target.size else 0.0,
        "precision": precision, "recall": recall, "f1": f1, "average_precision": ap,
    }


def best_f1_threshold(rows: Iterable[Dict], defect: str) -> float:
    values = list(rows)
    target = np.asarray([row["target"][defect] for row in values], dtype=np.int32)
    score = np.asarray([row["scores"][defect] for row in values], dtype=np.float32)
    if not len(target) or target.sum() in (0, len(target)):
        return 0.5
    candidates = np.unique(np.clip(np.r_[0.0, score, 1.0], 0.0, 1.0))
    best = (-1.0, -1.0, 0.5)
    for threshold in candidates:
        metric = metric_from_arrays(target, score >= threshold, score)
        candidate = (float(metric["f1"]), -abs(float(threshold) - 0.5), float(threshold))
        if candidate > best:
            best = candidate
    return float(best[2])


def make_rows(labels: Dict[str, List[Dict[str, str]]], evidence: Dict[str, List[Dict]]) -> List[Dict]:
    """Build local unit rows, then one image row for global skew."""
    rows: List[Dict] = []
    for sample_id in sorted(labels):
        label_rows, units = labels[sample_id], evidence[sample_id]
        if len(units) != len(label_rows):
            raise ValueError(f"{sample_id}: detector returned {len(units)} rather than six units")
        for label, unit in zip(label_rows, units):
            if bool(int(label["valid"])):
                rows.append({
                    "sample_id": sample_id, "scope": "unit", "unit_id": int(label["unit_id"]),
                    "target": {name: int(label[name]) for name in LOCAL_CLASSES},
                    "scores": {name: float(unit["scores"][name]) for name in LOCAL_CLASSES},
                    "evidence": unit["evidence"],
                })
        # The dataset labels skew per image, so use exactly one global score.
        rows.append({
            "sample_id": sample_id, "scope": "image", "unit_id": None,
            "target": {"skew": max(int(label["skew"]) for label in label_rows)},
            "scores": {"skew": max(float(unit["scores"]["skew"]) for unit in units)},
            "evidence": {"aggregation": "image_max_global_skew"},
        })
    return rows


def rows_for_defect(rows: Sequence[Dict], defect: str) -> List[Dict]:
    scope = "image" if defect == "skew" else "unit"
    return [row for row in rows if row["scope"] == scope]


def evaluate(rows: Sequence[Dict]) -> Dict:
    result: Dict[str, object] = {}
    for defect in DEFECT_CLASSES:
        selected = rows_for_defect(rows, defect)
        target = np.asarray([row["target"][defect] for row in selected], dtype=np.int32)
        score = np.asarray([row["scores"][defect] for row in selected], dtype=np.float32)
        prediction = np.asarray([row["prediction"][defect] for row in selected], dtype=bool)
        result[defect] = metric_from_arrays(target, prediction, score)
    result["macro_f1_mixed_scope"] = float(np.mean([result[name]["f1"] for name in DEFECT_CLASSES]))
    result["mean_class_accuracy_mixed_scope"] = float(np.mean([result[name]["accuracy"] for name in DEFECT_CLASSES]))
    # Class accuracy is a conventional detection-correctness statistic, but
    # with many normal units it can look high even while positives are missed.
    # Report a stricter complement: a unit is correct only when all three
    # local labels agree, and an image is correct only when all valid local
    # units plus its global-skew decision agree.
    local_rows = [row for row in rows if row["scope"] == "unit"]
    if local_rows:
        local_exact = [
            all(row["target"][name] == int(row["prediction"][name]) for name in LOCAL_CLASSES)
            for row in local_rows
        ]
        result["local_unit_exact_decision_accuracy"] = float(np.mean(local_exact))
    else:
        result["local_unit_exact_decision_accuracy"] = 0.0
    image_rows = {row["sample_id"]: row for row in rows if row["scope"] == "image"}
    by_image: Dict[str, List[Dict]] = {}
    for row in local_rows:
        by_image.setdefault(row["sample_id"], []).append(row)
    complete = []
    for sample_id, local_values in by_image.items():
        skew_row = image_rows.get(sample_id)
        local_ok = all(
            row["target"][name] == int(row["prediction"][name])
            for row in local_values for name in LOCAL_CLASSES
        )
        skew_ok = bool(skew_row) and skew_row["target"]["skew"] == int(skew_row["prediction"]["skew"])
        complete.append(local_ok and skew_ok)
    result["image_complete_decision_accuracy"] = float(np.mean(complete)) if complete else 0.0
    return result


def load_evidence(mask_dir: Path, labels: Dict[str, List[Dict[str, str]]], detector: TrajectorySelfReferenceDetector) -> Dict[str, List[Dict]]:
    evidence: Dict[str, List[Dict]] = {}
    for sample_id in labels:
        probability_path = mask_dir / f"{sample_id}.npy"
        png_path = mask_dir / f"{sample_id}.png"
        if probability_path.exists():
            mask = np.load(probability_path).astype(np.float32)
        elif png_path.exists():
            mask = np.asarray(Image.open(png_path).convert("L"), dtype=np.float32) / 255.0
        else:
            raise FileNotFoundError(f"Missing segmentation probability/mask: {probability_path} or {png_path}")
        evidence[sample_id] = detector.detect(mask)["units"]
    return evidence


def run_ablation(
    labels: Dict[str, List[Dict[str, str]]], evidence_by_condition: Dict[str, Dict[str, List[Dict]]],
    folds: Sequence[Sequence[str]], output_dir: Path, local_defects_mutually_exclusive: bool,
    threshold_calibration_condition: str,
) -> Dict[str, Dict]:
    calibration_rows_by_condition = make_rows(labels, evidence_by_condition[threshold_calibration_condition])
    all_oof: Dict[str, List[Dict]] = {name: [] for name in evidence_by_condition}
    reports = []
    all_ids = set(labels)
    for fold_index, test_values in enumerate(folds):
        test_ids = set(test_values)
        calibration_rows = [row for row in calibration_rows_by_condition if row["sample_id"] not in test_ids]
        thresholds = {defect: best_f1_threshold(rows_for_defect(calibration_rows, defect), defect) for defect in DEFECT_CLASSES}
        fold_record = {
            "fold": fold_index, "test_sample_ids": sorted(test_ids),
            "calibration_condition": threshold_calibration_condition,
            "calibration_sample_count": len(all_ids - test_ids), "thresholds": thresholds,
            "test_metrics": {},
        }
        for condition, evidence in evidence_by_condition.items():
            test_rows = [row.copy() for row in make_rows(labels, evidence) if row["sample_id"] in test_ids]
            for row in test_rows:
                row["fold"] = fold_index
                row["thresholds"] = thresholds
                row["prediction"] = decide_defect_classes(
                    row["scores"], thresholds,
                    local_defects_mutually_exclusive=local_defects_mutually_exclusive,
                )
            all_oof[condition].extend(test_rows)
            fold_record["test_metrics"][condition] = evaluate(test_rows)
        reports.append(fold_record)

    summaries = {condition: evaluate(rows) for condition, rows in all_oof.items()}
    for condition, rows in all_oof.items():
        save_json(output_dir / f"{condition}_oof_predictions.json", rows)
    save_json(output_dir / "fold_reports.json", reports)
    return summaries


def write_comparison(output_dir: Path, summaries: Dict[str, Dict]) -> None:
    order = ("clear", "blur_direct", "deblur_then_m6")
    with (output_dir / "end_to_end_comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "condition", "macro_f1_mixed_scope", "mean_class_accuracy_mixed_scope",
            "local_unit_exact_decision_accuracy", "image_complete_decision_accuracy",
            *[f"{name}_f1" for name in DEFECT_CLASSES],
        ])
        for condition in order:
            metric = summaries[condition]
            writer.writerow([
                condition, metric["macro_f1_mixed_scope"], metric["mean_class_accuracy_mixed_scope"],
                metric["local_unit_exact_decision_accuracy"], metric["image_complete_decision_accuracy"],
                *[metric[name]["f1"] for name in DEFECT_CLASSES],
            ])

    # English labels avoid platform-specific CJK font dependencies in the
    # paper-ready figure generated on headless training environments.
    labels = ["Clear", "Direct blur", "Deblur + M6"]
    values = [summaries[name]["macro_f1_mixed_scope"] * 100 for name in order]
    fig, axis = plt.subplots(figsize=(6.6, 4.1), dpi=180)
    bars = axis.bar(labels, values, color=["#4C78A8", "#E45756", "#54A24B"], width=0.58)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Mixed-scope macro F1 (%)")
    axis.set_title("End-to-end defect detection: fixed clear-calibrated thresholds")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "end_to_end_macro_f1_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-threshold deblur -> segmentation -> TSR end-to-end ablation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--clear-mask-dir", type=Path, required=True)
    parser.add_argument("--blur-mask-dir", type=Path, required=True)
    parser.add_argument("--deblur-mask-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-calibration-condition", choices=("clear", "blur_direct", "deblur_then_m6"), default="clear")
    args = parser.parse_args()

    raw_config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    permitted = {item.name for item in fields(TSRConfig)}
    detector = TrajectorySelfReferenceDetector(TSRConfig(**{key: value for key, value in raw_config.items() if key in permitted}))
    labels = load_labels(args.labels)
    conditions = {
        "clear": args.clear_mask_dir,
        "blur_direct": args.blur_mask_dir,
        "deblur_then_m6": args.deblur_mask_dir,
    }
    evidence = {name: load_evidence(path, labels, detector) for name, path in conditions.items()}
    folds = fold_groups(list(labels), args.folds, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "protocol.json", {
        "detector": "TSR-Defect", "detector_config": asdict(detector.config), "folds": folds,
        "conditions": {name: str(path) for name, path in conditions.items()},
        "threshold_rule": "For every outer fold and every defect class, choose the threshold by F1 on only the 36 calibration images from the stated calibration condition; freeze it for all three held-out conditions.",
        "threshold_calibration_condition": args.threshold_calibration_condition,
        "evaluation_scope": {"skip": "unit", "break": "unit", "loose": "unit", "skew": "image"},
        "caveat": "Blur is synthetic motion blur (one deterministic degradation per clear image); this is a controlled robustness ablation, not a replacement for real blurred-data validation.",
    })
    summaries = run_ablation(
        labels, evidence, folds, args.output,
        local_defects_mutually_exclusive=detector.config.local_defects_mutually_exclusive,
        threshold_calibration_condition=args.threshold_calibration_condition,
    )
    save_json(args.output / "summary.json", summaries)
    write_comparison(args.output, summaries)
    for condition, metrics in summaries.items():
        print(
            f"{condition}: macro F1={metrics['macro_f1_mixed_scope']:.4f}; "
            f"mean class accuracy={metrics['mean_class_accuracy_mixed_scope']:.4f}; "
            f"local exact={metrics['local_unit_exact_decision_accuracy']:.4f}; "
            f"image complete={metrics['image_complete_decision_accuracy']:.4f}"
        )
        for defect in DEFECT_CLASSES:
            item = metrics[defect]
            print(f"  {defect}: F1={item['f1']:.4f}, P={item['precision']:.4f}, R={item['recall']:.4f}, Acc={item['accuracy']:.4f}")


if __name__ == "__main__":
    main()
