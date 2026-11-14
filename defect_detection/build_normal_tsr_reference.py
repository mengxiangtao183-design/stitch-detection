"""Extract TSR-Defect score distributions from labelled normal stitch images.

This is deliberately a small, deterministic calibration utility.  It does not
create synthetic defects or train a black-box image classifier: its sole role
is to establish which trajectory-score range is normal across DB, SD, XJ and
RX material domains after the same M6 segmentation stage used at inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.trajectory_self_reference_detector import TSRConfig, TrajectorySelfReferenceDetector  # noqa: E402


def parse_source(value: str) -> tuple[str, Path]:
    name, delimiter, raw_path = value.partition("=")
    if not delimiter or not name or not raw_path:
        raise argparse.ArgumentTypeError("Each --normal-source must be NAME=MASK_DIRECTORY")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normal TSR score reference from M6 masks")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--normal-source", action="append", type=parse_source, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    allowed = {item.name for item in fields(TSRConfig)}
    detector = TrajectorySelfReferenceDetector(TSRConfig(**{key: value for key, value in raw.items() if key in allowed}))
    rows = []
    for domain, mask_dir in args.normal_source:
        paths = sorted(mask_dir.glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No PNG M6 masks in {mask_dir}")
        for path in paths:
            mask = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
            units = detector.detect(mask)["units"]
            local = [unit for unit in units if not unit["evidence"].get("edge_truncated", False)] or units
            rows.append({
                "domain": domain,
                "image_id": path.stem,
                "skip_score": max(float(unit["scores"]["skip"]) for unit in local),
                "break_score": max(float(unit["scores"]["break"]) for unit in local),
                "loose_score": max(float(unit["scores"]["loose"]) for unit in local),
                "skew_score": max(float(unit["scores"]["skew"]) for unit in units),
                "valid_unit_count": len(local),
            })
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "normal_tsr_scores.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"image_count": len(rows), "domains": {}}
    for domain in sorted({row["domain"] for row in rows}):
        values = [row for row in rows if row["domain"] == domain]
        summary["domains"][domain] = {"image_count": len(values)}
        for defect in ("skip", "break", "skew", "loose"):
            scores = np.asarray([row[f"{defect}_score"] for row in values], dtype=np.float64)
            summary["domains"][domain][defect] = {"p95": float(np.quantile(scores, .95)), "p99": float(np.quantile(scores, .99)), "max": float(scores.max())}
    (args.output / "normal_tsr_score_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(rows)} normal images -> {csv_path}")


if __name__ == "__main__":
    main()
