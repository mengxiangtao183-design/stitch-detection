"""Benchmark the frozen trajectory self-reference defect detection unit."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from defect_detection.train_trajectory_feature_classifier import (  # noqa: E402
    LOCAL_CLASSES, feature_vector, load_detector, predict, read_mask,
)


def infer_mask(mask, detector, package, skew_threshold):
    detected = detector.detect(mask)
    rows = [{"feature": feature_vector(detected, unit, mask.shape[0], mask.shape[1])} for unit in detected["units"]]
    probability = predict(package["models"], rows)
    ratios = {
        name: float(probability[:, index].max()) / max(float(package["thresholds"][name]), 1e-6)
        for index, name in enumerate(LOCAL_CLASSES)
    }
    skew_score = float(np.clip(
        np.mean([row["feature"][8] for row in rows]) / max(detector.config.global_slope_score_scale, 1e-6),
        0.0, 1.0,
    ))
    ratios["skew"] = skew_score / max(float(skew_threshold), 1e-6)
    winner = max(ratios, key=ratios.get)
    return winner if ratios[winner] >= 1.0 else "normal"


def statistics(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    detector = load_detector(args.config)
    package = joblib.load(args.model)
    for model in package["models"].values():
        if "extra_trees" in model.named_steps:
            model.named_steps["extra_trees"].n_jobs = 1
    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    skew_threshold = report["hybrid_four_class_internal_holdout"]["skew_threshold"]
    paths = sorted(args.input_dir.glob("*.png"))
    masks = [read_mask(path) for path in paths]
    if not masks:
        raise FileNotFoundError(args.input_dir)
    for mask in masks[:3]:
        infer_mask(mask, detector, package, skew_threshold)

    algorithm = []
    for _ in range(args.repeat):
        for mask in masks:
            start = time.perf_counter()
            infer_mask(mask, detector, package, skew_threshold)
            algorithm.append((time.perf_counter() - start) * 1000.0)

    with_io = []
    for path in paths:
        start = time.perf_counter()
        infer_mask(read_mask(path), detector, package, skew_threshold)
        with_io.append((time.perf_counter() - start) * 1000.0)

    result = {
        "images": len(paths),
        "repeat": args.repeat,
        "device": "CPU",
        "algorithm_only": statistics(algorithm),
        "including_png_read": statistics(with_io),
        "definition": "algorithm_only excludes model loading and PNG reading; including_png_read excludes one-time model loading.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
