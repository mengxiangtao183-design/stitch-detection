"""Write a paired three-way comparison of TSR mask-input variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CONDITIONS = ("clear", "blur_direct", "deblur_then_aphnet")
CLASSES = ("skip", "break", "skew", "loose")


def read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--binary-topology", required=True, type=Path)
    parser.add_argument("--additional", action="append", nargs=2, metavar=("NAME", "RESULT_DIR"), default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    variants = {
        "probability_trajectory": args.probability,
        "binary_trajectory": args.binary,
        "binary_topology": args.binary_topology,
    }
    for name, raw_path in args.additional:
        if name in variants:
            raise ValueError(f"Duplicate variant name: {name}")
        variants[name] = Path(raw_path)
    protocols = {name: read(path / "protocol.json") for name, path in variants.items()}
    for field in ("seed", "outer_folds", "inner_folds", "outer_test_folds", "training_conditions", "calibration_conditions"):
        if len({json.dumps(value[field], sort_keys=True) for value in protocols.values()}) != 1:
            raise ValueError(f"Variants differ in protocol field: {field}")

    summaries = {name: read(path / "summary.json")["conditions"] for name, path in variants.items()}
    metrics = []
    paired = []
    for condition in CONDITIONS:
        predictions = {
            name: {item["source_id"]: item for item in read(path / f"{condition}_oof_image_predictions.json")}
            for name, path in variants.items()
        }
        source_lists = [set(data) for data in predictions.values()]
        if any(source_lists[0] != values for values in source_lists[1:]):
            raise ValueError(f"Source lists differ in {condition}")
        for source_id in sorted(source_lists[0]):
            values = [data[source_id] for data in predictions.values()]
            if len({(v["fold"], v["target_class"]) for v in values}) != 1:
                raise ValueError(f"Fold or target differs in {condition}, {source_id}")
            paired.append({
                "condition": condition,
                "source_id": source_id,
                "fold": values[0]["fold"],
                "target_class": values[0]["target_class"],
                **{name: data[source_id]["predicted_class"] for name, data in predictions.items()},
            })
        for name, summary in summaries.items():
            current = summary[condition]
            image = current["defect_image_classification"]
            metrics.append({
                "condition": condition,
                "variant": name,
                "n_images": image["count"],
                "macro_f1_percent": round(image["macro_f1"] * 100, 2),
                "image_accuracy_percent": round(image["accuracy"] * 100, 2),
                "alarm_recall_percent": round(current["defect_alarm_recall"] * 100, 2),
                **{f"{cls}_f1_percent": round(image["per_class"][cls]["f1"] * 100, 2) for cls in CLASSES},
            })

    args.output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("multi_variant_metrics.csv", metrics), ("multi_variant_paired_predictions.csv", paired)):
        with (args.output / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
