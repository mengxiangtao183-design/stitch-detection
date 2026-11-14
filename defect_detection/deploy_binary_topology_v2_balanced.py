"""Fit or run the selected binary-topology TSR unit.

The four-fold results remain the performance estimate.  The final model
uses all 112 defect sources, with its thresholds calibrated by source-grouped
inner OOF predictions; the 340 normal holdout sources remain excluded.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from defect_detection.binary_topology_features import (  # noqa: E402
    FEATURE_NAMES_V2,
    binary_topology_v2_feature_vector,
)
from defect_detection.run_nested_fourfold_tsr_defect import (  # noqa: E402
    DEFECT_CLASSES,
    calibrate_inner_oof,
    controlled_rows,
    fit_weighted,
    group_ids,
    normal_rows,
    rows_from_condition,
    skew_score,
    split_normal_sources,
)
from defect_detection.train_trajectory_feature_classifier import (  # noqa: E402
    LOCAL_CLASSES,
    load_detector,
    predict,
    read_grouped_labels,
)


def train(config_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("feature_set") != "binary_topology_v2":
        raise ValueError("Expected feature_set: binary_topology_v2")
    protocol = cfg["protocol"]
    paths = {key: Path(value) for key, value in cfg["paths"].items()}
    detector = load_detector(paths["tsr_config"])
    labels = read_grouped_labels(paths["labels"], originals_only=True)
    source_ids = sorted(labels)
    conditions = {
        "clear": rows_from_condition(
            labels, paths["clear_aphnet_masks"], detector, "clear_real", binary_topology_v2_feature_vector
        ),
        "deblur_then_aphnet": rows_from_condition(
            labels, paths["deblur_then_aphnet_masks"], detector, "deblur_then_aphnet_real",
            binary_topology_v2_feature_vector,
        ),
    }
    controlled, _ = controlled_rows(
        labels, paths["gt_mask_dir"], detector, int(cfg["augmentation"]["variants_per_local_class"]),
        int(protocol["seed"]), binary_topology_v2_feature_vector,
    )
    normal_dirs = {
        key.removeprefix("normal_"): value for key, value in paths.items() if key.startswith("normal_")
    }
    normal_train, normal_holdout, normal_counts = split_normal_sources(
        normal_rows(normal_dirs, detector, binary_topology_v2_feature_vector),
        float(protocol["normal_test_fraction"]), int(protocol["seed"]),
    )
    thresholds, skew_threshold, calibration = calibrate_inner_oof(
        source_ids, labels, conditions, protocol["training_conditions"], protocol["calibration_conditions"],
        controlled, normal_train, detector, int(protocol["inner_folds"]), int(protocol["seed"]),
        cfg["model"], cfg["weights"], float(protocol["min_defect_alarm_recall"]),
        int(protocol["threshold_coordinate_passes"]), str(protocol["threshold_objective"]),
    )
    fitting = []
    for condition in protocol["training_conditions"]:
        fitting.extend(conditions[condition])
    fitting.extend(controlled)
    fitting.extend(normal_train)
    models = fit_weighted(fitting, cfg["model"], cfg["weights"])
    output_dir.mkdir(parents=True)
    package = {
        "models": models,
        "thresholds": thresholds,
        "skew_threshold": skew_threshold,
        "feature_set": "binary_topology_v2",
        "feature_names": FEATURE_NAMES_V2,
        "train_defect_source_ids": source_ids,
        "normal_training_source_count": len(group_ids(normal_train)),
        "normal_holdout_source_count": len(group_ids(normal_holdout)),
    }
    joblib.dump(package, output_dir / "model_final.joblib")
    report = {
        "role": "Final deployment fit; not an independent evaluation",
        "config": str(config_path.resolve()),
        "input": "ACHNet binary segmentation PNG (320x320; not probability NPY)",
        "training_defect_source_count": len(source_ids),
        "normal_split_counts": normal_counts,
        "normal_holdout_excluded": True,
        "training_unit_rows": len(fitting),
        "calibration": calibration,
        "four_fold_evaluation": str(paths["output"]),
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Final binary-topology model: {output_dir / 'model_final.joblib'}")


def infer(model_path: Path, detector_config: Path, input_dir: Path, output: Path) -> None:
    package = joblib.load(model_path)
    if package.get("feature_set") != "binary_topology_v2":
        raise ValueError("The model is not binary_topology_v2")
    detector = load_detector(detector_config)
    rows = []
    masks = sorted(input_dir.glob("*.png"))
    if not masks:
        raise FileNotFoundError(f"No PNG masks found in {input_dir}")
    for path in masks:
        pixels = np.asarray(Image.open(path).convert("L"))
        if pixels.shape != (320, 320):
            raise ValueError(f"{path}: expected 320x320, got {pixels.shape}")
        if not np.isin(pixels, (0, 255)).all():
            raise ValueError(f"{path}: expected a binary 0/255 mask, not a probability image")
        mask = pixels.astype(np.float32) / 255.0
        result = detector.detect(mask)
        for unit_id, unit in enumerate(result["units"]):
            rows.append({
                "source_id": path.stem,
                "unit_id": unit_id,
                "feature": binary_topology_v2_feature_vector(result, unit, mask),
            })
    for model in package["models"].values():
        if "extra_trees" in model.named_steps:
            model.named_steps["extra_trees"].n_jobs = 1
    probabilities = predict(package["models"], rows)
    by_image = defaultdict(list)
    for index, row in enumerate(rows):
        by_image[row["source_id"]].append(index)
    images = {}
    for source_id, indices in sorted(by_image.items()):
        ratios = {
            name: max(
                float(probabilities[index, column]) / max(float(package["thresholds"][name]), 1e-8)
                for index in indices
            )
            for column, name in enumerate(LOCAL_CLASSES)
        }
        skew = float(np.mean([skew_score(rows[index], detector) for index in indices]))
        ratios["skew"] = skew / max(float(package["skew_threshold"]), 1e-8)
        winner = max(DEFECT_CLASSES, key=lambda name: ratios[name])
        images[source_id] = {
            "predicted_class": winner if ratios[winner] >= 1.0 else "normal",
            "defect_alarm": ratios[winner] >= 1.0,
            "score_over_threshold": ratios,
            "unit_probabilities": [
                {name: float(probabilities[index, column]) for column, name in enumerate(LOCAL_CLASSES)}
                for index in indices
            ],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"model": str(model_path), "images": images}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Predicted {len(images)} binary masks -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--output-dir", type=Path, required=True)
    prediction = sub.add_parser("predict")
    prediction.add_argument("--model", type=Path, required=True)
    prediction.add_argument("--detector-config", type=Path, required=True)
    prediction.add_argument("--input-dir", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.config, args.output_dir)
    else:
        infer(args.model, args.detector_config, args.input_dir, args.output)


if __name__ == "__main__":
    main()
