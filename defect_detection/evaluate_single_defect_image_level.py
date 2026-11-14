"""Image-level single-defect evaluation for the Quxian labels.

Each original image in this dataset has exactly one labelled defect type.
Local TSR scores are max-pooled across six units; global skew already has one
image score.  Scores are normalised by each fold's frozen operating threshold,
then the strongest class is selected.  When no class crosses its threshold the
output is ``normal``.  This provides the industrially meaningful final-image
defect-type accuracy, distinct from strict six-unit exact agreement.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

DEFECTS = ("skip", "break", "skew", "loose")
LOCAL = ("skip", "break", "loose")


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def gt_types(labels: Path) -> Dict[str, str]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with labels.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["augmentation"] == "original" and row["review_status"] == "completed":
                grouped[row["sample_id"]].append(row)
    result = {}
    for sample_id, rows in grouped.items():
        classes = [name for name in DEFECTS if any(int(row[name]) for row in rows)]
        if len(classes) != 1:
            raise ValueError(f"{sample_id}: expected exactly one image-level defect type, got {classes}")
        result[sample_id] = classes[0]
    return result


def multiclass_metrics(target: List[str], prediction: List[str]) -> Dict:
    result = {"image_count": len(target), "type_top1_accuracy": float(np.mean(np.asarray(target) == np.asarray(prediction)))}
    f1s = []
    for defect in DEFECTS:
        tp = sum(t == defect and p == defect for t, p in zip(target, prediction))
        fp = sum(t != defect and p == defect for t, p in zip(target, prediction))
        fn = sum(t == defect and p != defect for t, p in zip(target, prediction))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[defect] = {"support": sum(t == defect for t in target), "precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
    result["type_macro_f1"] = float(np.mean(f1s))
    return result


def evaluate(path: Path, truth: Dict[str, str]) -> tuple[Dict, List[Dict]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row["sample_id"]].append(row)
    output, targets, predictions = [], [], []
    for sample_id in sorted(truth):
        values = grouped[sample_id]
        local_rows = [row for row in values if row["scope"] == "unit"]
        skew_rows = [row for row in values if row["scope"] == "image"]
        thresholds = local_rows[0].get("thresholds") or local_rows[0].get("thresholds_from_clear_only")
        scores = {name: max(float(row["scores"][name]) for row in local_rows) for name in LOCAL}
        scores["skew"] = max(float(row["scores"]["skew"]) for row in skew_rows)
        normalised = {name: scores[name] / max(float(thresholds[name]), 1e-6) for name in DEFECTS}
        best = max(DEFECTS, key=lambda name: (normalised[name], scores[name]))
        predicted = best if normalised[best] >= 1.0 else "normal"
        targets.append(truth[sample_id])
        predictions.append(predicted)
        output.append({"sample_id": sample_id, "target_type": truth[sample_id], "predicted_type": predicted, "scores": scores, "score_over_threshold": normalised, "correct": predicted == truth[sample_id]})
    metric = multiclass_metrics(targets, predictions)
    metric["defect_alarm_recall"] = float(np.mean(np.asarray(predictions) != "normal"))
    return metric, output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate final image-level type classification on one-defect-per-image labels")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--blur-predictions", type=Path, required=True)
    parser.add_argument("--deblur-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    truth = gt_types(args.labels)
    blur_metric, blur_rows = evaluate(args.blur_predictions, truth)
    deblur_metric, deblur_rows = evaluate(args.deblur_predictions, truth)
    summary = {
        "task": "One labelled defect type per original image; the final prediction is skip, break, skew, loose, or normal.",
        "protocol": "Scores are max-pooled across local units and divided by each held-out fold's frozen class threshold. All 48 original sources are evaluated once out of fold.",
        "blur_direct": blur_metric,
        "deblur_then_m6": deblur_metric,
    }
    save_json(args.output / "single_defect_image_level_summary.json", summary)
    save_json(args.output / "single_defect_image_level_blur_predictions.json", blur_rows)
    save_json(args.output / "single_defect_image_level_deblur_predictions.json", deblur_rows)
    with (args.output / "single_defect_image_level_comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["condition", "image_count", "type_top1_accuracy", "type_macro_f1", "defect_alarm_recall", *[f"{name}_f1" for name in DEFECTS]])
        for name, metric in (("blur_direct", blur_metric), ("deblur_then_m6", deblur_metric)):
            writer.writerow([name, metric["image_count"], metric["type_top1_accuracy"], metric["type_macro_f1"], metric["defect_alarm_recall"], *[metric[x]["f1"] for x in DEFECTS]])
    for name, metric in (("blur_direct", blur_metric), ("deblur_then_m6", deblur_metric)):
        print(f"{name}: type top-1 accuracy={metric['type_top1_accuracy']:.4f}, type macro F1={metric['type_macro_f1']:.4f}, defect alarm recall={metric['defect_alarm_recall']:.4f}")


if __name__ == "__main__":
    main()
