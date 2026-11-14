"""Warm CPU compute-only timing for the deployed BTSR binary-mask unit."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from defect_detection.binary_topology_features import binary_topology_v2_feature_vector
from defect_detection.train_trajectory_feature_classifier import load_detector, predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--detector-config", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    package = joblib.load(args.model)
    for model in package["models"].values():
        model.named_steps["extra_trees"].n_jobs = 1
    detector = load_detector(args.detector_config)
    masks = [
        np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        for path in sorted(args.mask_dir.glob("*.png"))
    ]
    if not masks:
        raise FileNotFoundError(args.mask_dir)

    def infer(mask):
        result = detector.detect(mask)
        rows = [{"feature": binary_topology_v2_feature_vector(result, unit, mask)} for unit in result["units"]]
        return predict(package["models"], rows)

    for mask in masks[:10]:
        infer(mask)
    durations = []
    for _ in range(args.repeats):
        for mask in masks:
            start = time.perf_counter()
            infer(mask)
            durations.append((time.perf_counter() - start) * 1000)
    report = {
        "unit": "BTSR, binary_topology_v2, 200-tree ExtraTrees",
        "hardware": platform.processor(),
        "input_images": len(masks),
        "repeats": args.repeats,
        "batch_size": 1,
        "warmup_images": min(10, len(masks)),
        "exclusions": ["PNG read", "model load", "ACHNet inference", "MBDNet-P+ inference"],
        "mean_ms": float(np.mean(durations)),
        "median_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
