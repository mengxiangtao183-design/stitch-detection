"""Compare nested four-fold TSR results from probability and binary masks.

The two result directories must contain predictions for identical sources in
identical outer folds.  The script reports paired image-level metrics without
retraining, recalibrating, or selecting thresholds on the test predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CONDITIONS = ("clear", "blur_direct", "deblur_then_aphnet")
CLASSES = ("skip", "break", "skew", "loose")


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prob_protocol = read_json(args.probability / "protocol.json")
    binary_protocol = read_json(args.binary / "protocol.json")
    for field in ("seed", "outer_folds", "inner_folds", "outer_test_folds", "training_conditions", "calibration_conditions"):
        if prob_protocol[field] != binary_protocol[field]:
            raise ValueError(f"Experimental protocol differs in {field}")

    prob = read_json(args.probability / "summary.json")["conditions"]
    binary = read_json(args.binary / "summary.json")["conditions"]
    args.output.mkdir(parents=True, exist_ok=True)
    table = []
    paired = []
    for condition in CONDITIONS:
        pred_a = {row["source_id"]: row for row in read_json(args.probability / f"{condition}_oof_image_predictions.json")}
        pred_b = {row["source_id"]: row for row in read_json(args.binary / f"{condition}_oof_image_predictions.json")}
        if set(pred_a) != set(pred_b):
            raise ValueError(f"Source list differs in {condition}")
        for source_id in sorted(pred_a):
            a, b = pred_a[source_id], pred_b[source_id]
            if a["fold"] != b["fold"] or a["target_class"] != b["target_class"]:
                raise ValueError(f"Fold or target differs in {condition}, {source_id}")
            paired.append({
                "condition": condition,
                "source_id": source_id,
                "fold": a["fold"],
                "target_class": a["target_class"],
                "probability_prediction": a["predicted_class"],
                "binary_prediction": b["predicted_class"],
                "probability_correct": int(a["predicted_class"] == a["target_class"]),
                "binary_correct": int(b["predicted_class"] == b["target_class"]),
            })
        for name, stats in (("probability", prob[condition]), ("binary", binary[condition])):
            image = stats["defect_image_classification"]
            row = {
                "condition": condition,
                "mask_input": name,
                "images": image["count"],
                "macro_f1_percent": round(image["macro_f1"] * 100, 2),
                "image_accuracy_percent": round(image["accuracy"] * 100, 2),
                "alarm_recall_percent": round(stats["defect_alarm_recall"] * 100, 2),
            }
            for cls in CLASSES:
                row[f"{cls}_f1_percent"] = round(image["per_class"][cls]["f1"] * 100, 2)
            table.append(row)

    with (args.output / "probability_vs_binary_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with (args.output / "probability_vs_binary_paired_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    print(json.dumps(table, ensure_ascii=False, indent=2))
    print(f"paired source predictions: {args.output / 'probability_vs_binary_paired_predictions.csv'}")


if __name__ == "__main__":
    main()
