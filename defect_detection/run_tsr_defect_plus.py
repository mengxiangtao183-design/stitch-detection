"""Frozen inference entry point for TSR-Defect+.

Input is the final M6 segmentation mask directory.  The module extracts the
same mask-derived trajectory features used during fitting, applies the frozen
ExtraTrees local classifier, combines it with the image-level skew rule, and
writes one final industrial decision per image: normal/skip/break/skew/loose.
"""

from __future__ import annotations

import argparse
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
    feature_vector,
    load_detector,
    predict,
    read_mask,
)


DEFECTS = ("skip", "break", "skew", "loose")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen TSR-Defect+ on M6 mask outputs")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    package = joblib.load(args.model)
    # Per-image online inference is faster without repeatedly starting a
    # parallel worker pool for only six feature vectors.
    for model in package["models"].values():
        if "extra_trees" in model.named_steps:
            model.named_steps["extra_trees"].n_jobs = 1
    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    detector = load_detector(args.config)
    rows: List[Dict] = []
    for path in sorted(args.input_dir.glob("*.png")):
        mask = read_mask(path)
        detected = detector.detect(mask)
        for unit_id, unit in enumerate(detected["units"]):
            rows.append({
                "image_id": path.stem, "unit_id": unit_id,
                "feature": feature_vector(detected, unit, mask.shape[0], mask.shape[1]),
                "evidence": unit["evidence"], "rule_scores": unit["scores"],
            })
    if not rows:
        raise FileNotFoundError(f"No PNG segmentation masks in {args.input_dir}")
    probability = predict(package["models"], rows)
    skew_threshold = float(report["hybrid_four_class_internal_holdout"]["skew_threshold"])
    by_image: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_image[row["image_id"]].append(index)
    output = {"detector": "TSR-Defect+", "images": {}}
    for image_id, indices in sorted(by_image.items()):
        local_ratio = {
            name: max(float(probability[index, column]) / max(float(package["thresholds"][name]), 1e-6) for index in indices)
            for column, name in enumerate(LOCAL_CLASSES)
        }
        skew_score = float(np.clip(np.mean([rows[index]["feature"][8] for index in indices]) / max(detector.config.global_slope_score_scale, 1e-6), 0.0, 1.0))
        ratios = {**local_ratio, "skew": skew_score / max(skew_threshold, 1e-6)}
        winner = max(DEFECTS, key=lambda name: ratios[name])
        final = winner if ratios[winner] >= 1.0 else "normal"
        output["images"][image_id] = {
            "final_type": final,
            "score_over_threshold": ratios,
            "unit_probabilities": [
                {name: float(probability[index, column]) for column, name in enumerate(LOCAL_CLASSES)}
                for index in indices
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = defaultdict(int)
    for value in output["images"].values():
        counts[value["final_type"]] += 1
    print(f"Processed {len(output['images'])} images -> {args.output}")
    print(json.dumps(dict(sorted(counts.items())), ensure_ascii=False))


if __name__ == "__main__":
    main()
