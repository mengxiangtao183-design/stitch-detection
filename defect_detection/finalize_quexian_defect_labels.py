"""Finalize original defect labels and mirror them to horizontal-flip samples.

For the numbered 96-sample dataset, originals are 001--048 and horizontal
flips are 049--096.  A flipped ROI numbered ``u`` receives labels from
original ROI ``5-u``.  Only blank defect cells of valid original units are
filled with zero; existing reviewer-entered labels are never replaced.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


CLASSES = ("skip", "break", "skew", "loose")


def _sample_id(value: int) -> str:
    return f"{value:03d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="defect_labels_template.csv")
    parser.add_argument("--backup", type=Path, required=True, help="Safe copy written before update")
    args = parser.parse_args()

    with args.csv.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 576:
        raise ValueError(f"Expected 576 rows (96 images x 6 units), found {len(rows)}")
    expected_columns = {"sample_id", "augmentation", "unit_id", "valid", "review_status", *CLASSES}
    if not rows or not expected_columns.issubset(rows[0]):
        raise ValueError("CSV does not have the expected defect-label columns")

    by_key = {(row["sample_id"], row["unit_id"]): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("Duplicate sample_id/unit_id pairs found")

    original_positive = {name: 0 for name in CLASSES}
    for sample_number in range(1, 49):
        sample = _sample_id(sample_number)
        for unit_number in range(6):
            row = by_key[(sample, str(unit_number))]
            if row["augmentation"] != "original":
                raise ValueError(f"{sample} is not marked as original")
            if row["valid"] == "1":
                for name in CLASSES:
                    if row[name] == "":
                        row[name] = "0"
                    if row[name] not in {"0", "1"}:
                        raise ValueError(f"Unexpected {name} value in {sample}/unit{unit_number}: {row[name]!r}")
                    original_positive[name] += row[name] == "1"
                row["review_status"] = "completed"
            elif row["valid"] == "0":
                # An ignored ROI is deliberately unlabeled and should stay so.
                for name in CLASSES:
                    row[name] = ""
                row["review_status"] = "excluded_frame_boundary"
            else:
                raise ValueError(f"Unexpected valid value in {sample}/unit{unit_number}: {row['valid']!r}")

    for flipped_number in range(49, 97):
        flipped = _sample_id(flipped_number)
        original = _sample_id(flipped_number - 48)
        for flipped_unit in range(6):
            target = by_key[(flipped, str(flipped_unit))]
            source = by_key[(original, str(5 - flipped_unit))]
            if target["augmentation"] != "horizontal_flip":
                raise ValueError(f"{flipped} is not marked as horizontal_flip")
            target["valid"] = source["valid"]
            for name in CLASSES:
                target[name] = source[name]
            target["review_status"] = source["review_status"]

    args.backup.parent.mkdir(parents=True, exist_ok=True)
    if args.backup.exists():
        raise FileExistsError(f"Backup already exists: {args.backup}")
    shutil.copy2(args.csv, args.backup)
    with args.csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print({"rows": len(rows), "original_positive_units": original_positive, "backup": str(args.backup)})


if __name__ == "__main__":
    main()
