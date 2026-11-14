"""External-normal evaluation for the lightweight TSR-Defect+ classifier.

The supplied ExtraTrees classifier is fitted only on controlled masks and 36
development defect images.  This evaluator never refits it: DB/SD/XJ/RX are
therefore a genuine external normal-reference test for the false-alarm rate.
It also combines the withheld 12 defect sources with all normal images to
produce the final image-level one-type decision used in an inspection line.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from defect_detection.train_trajectory_feature_classifier import (  # noqa: E402
    LOCAL_CLASSES,
    load_detector,
    normal_rows_from_masks,
    predict,
    read_grouped_labels,
    rows_from_masks,
    split_normal_sources,
)


CLASSES = ("normal", "skip", "break", "skew", "loose")
DEFECTS = CLASSES[1:]


def parse_source(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("--normal-source requires NAME=MASK_DIRECTORY")
    return name, Path(raw_path)


def metric(target: List[str], prediction: List[str]) -> Dict:
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
    result["normal_false_alarm_rate"] = float(1.0 - result["normal"]["recall"])
    defect_indices = [index for index, label in enumerate(target) if label != "normal"]
    result["defect_alarm_recall"] = float(np.mean([prediction[index] != "normal" for index in defect_indices]))
    result["defect_type_accuracy"] = float(np.mean([prediction[index] == target[index] for index in defect_indices]))
    return result


def labelled_target(rows: List[Dict]) -> str:
    present = [name for name in DEFECTS if any(row["target"].get(name, 0) for row in rows)]
    if len(present) != 1:
        raise ValueError(f"Expected one defect class, got {present}")
    return present[0]


def image_decisions(rows: List[Dict], probability: np.ndarray, thresholds: Dict[str, float], skew_threshold: float, skew_scale: float) -> List[Dict]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["source_id"]].append(index)
    output = []
    for source_id, indices in sorted(grouped.items()):
        local_ratio = {
            name: max(float(probability[index, col]) / max(float(thresholds[name]), 1e-6) for index in indices)
            for col, name in enumerate(LOCAL_CLASSES)
        }
        # Match the training script's image-level skew score exactly.
        skew_score = float(np.clip(np.mean([rows[index]["feature"][8] for index in indices]) / max(skew_scale, 1e-6), 0.0, 1.0))
        ratios = {**local_ratio, "skew": skew_score / max(float(skew_threshold), 1e-6)}
        winner = max(DEFECTS, key=lambda name: ratios[name])
        prediction = winner if ratios[winner] >= 1.0 else "normal"
        target = "normal" if rows[indices[0]]["kind"] == "normal_reference" else labelled_target([rows[index] for index in indices])
        output.append({"source_id": source_id, "target": target, "prediction": prediction, "score_over_threshold": ratios, "correct": prediction == target})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TSR-Defect+ with external normal images")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--m6-mask-dir", type=Path, required=True)
    parser.add_argument("--normal-source", type=parse_source, action="append", required=True)
    parser.add_argument("--normal-test-fraction", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    heldout_ids = report["internal_holdout"]["source_ids"]
    package = joblib.load(args.model)
    for model in package["models"].values():
        if "extra_trees" in model.named_steps:
            model.named_steps["extra_trees"].n_jobs = 1
    detector = load_detector(args.config)
    labels = read_grouped_labels(args.labels, originals_only=True)
    defect_rows = rows_from_masks(labels, args.m6_mask_dir, detector, False, "m6_holdout", heldout_ids)
    normal_rows = normal_rows_from_masks(args.normal_source, detector)
    if args.normal_test_fraction < 1.0:
        _, normal_rows = split_normal_sources(normal_rows, args.normal_test_fraction, 42)
    rows = defect_rows + normal_rows
    probabilities = predict(package["models"], rows)
    skew_threshold = float(report["hybrid_four_class_internal_holdout"]["skew_threshold"])
    decisions = image_decisions(rows, probabilities, package["thresholds"], skew_threshold, detector.config.global_slope_score_scale)
    values = metric([row["target"] for row in decisions], [row["prediction"] for row in decisions])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "external_normal_image_predictions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "detector": "TSR-Defect+ (trajectory features + ExtraTrees; global skew rule)",
        "protocol": "The classifier is frozen after fitting controlled masks and 36 development defect sources. The 12 predefined defect hold-out sources and the stated external normal images are evaluated without fitting or threshold adjustment.",
        "metrics": values,
        "normal_sources": {name: str(path) for name, path in args.normal_source},
    }
    (args.output / "external_normal_image_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "external_normal_image_metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for name in ("accuracy", "macro_f1_5class", "defect_macro_f1_4class", "normal_false_alarm_rate", "defect_alarm_recall", "defect_type_accuracy"):
            writer.writerow([name, values[name]])
        for label in CLASSES:
            for name in ("precision", "recall", "f1", "support"):
                writer.writerow([f"{label}_{name}", values[label][name]])
    print(json.dumps(values, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
