"""Train a lightweight, source-grouped classifier on TSR trajectory features.

It learns only the local classes (skip, break and loose).  Global skew remains
with the TSR global-slope rule because its annotation is image-level.  The
script keeps controlled masks and their source IDs together during splitting,
then calibrates thresholds from out-of-fold predictions on the 36 development
sources only.  The 12 held-out sources are evaluated exactly once afterwards.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import joblib
import numpy as np
import yaml
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.trajectory_self_reference_detector import TSRConfig, TrajectorySelfReferenceDetector


LOCAL_CLASSES = ("skip", "break", "loose")
FEATURE_NAMES = (
    "coverage", "gap_ratio", "span_ratio", "arc_ratio", "deviation_norm", "abs_sag_norm",
    "bend_energy", "local_slope_norm", "global_slope_norm", "rule_skip", "rule_break", "rule_loose",
    "internal_gap_ratio", "terminal_missing_ratio", "continuity",
    "neighbor_span_difference", "neighbor_coverage_difference", "neighbor_gap_difference",
    "short_residual_break_rule",
    "center_y_deviation_norm", "center_global_slope_norm",
)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_normal_source(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("--normal-source requires NAME=MASK_DIRECTORY")
    return name, Path(raw_path)


def read_grouped_labels(path: Path, originals_only: bool = False) -> Dict[str, List[Dict[str, str]]]:
    groups: Dict[str, List[Dict[str, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if originals_only and row["augmentation"] != "original":
                continue
            groups.setdefault(row["sample_id"], []).append(dict(row))
    for sample_id, rows in groups.items():
        rows.sort(key=lambda row: int(row["unit_id"]))
        if len(rows) != 6:
            raise ValueError(f"{sample_id}: expected six unit labels")
    return groups


def load_detector(path: Path) -> TrajectorySelfReferenceDetector:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    valid = {item.name for item in fields(TSRConfig)}
    return TrajectorySelfReferenceDetector(TSRConfig(**{name: value for name, value in raw.items() if name in valid}))


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def feature_vector(result: Dict, unit: Dict, height: int, width: int) -> List[float]:
    evidence = unit["evidence"]
    scores = unit["scores"]
    all_units = result["units"]
    index = int(unit["unit_id"])
    neighbor_evidence = [all_units[item]["evidence"] for item in (index - 1, index + 1) if 0 <= item < len(all_units)]
    def neighbor_median(name: str, fallback: float) -> float:
        values = [float(item[name]) for item in neighbor_evidence]
        return float(np.median(values)) if values else fallback
    span = float(evidence["span_ratio"])
    coverage = float(evidence["coverage"])
    gap = float(evidence["gap_ratio"])
    continuity = coverage * max(0.0, 1.0 - float(evidence["internal_gap_ratio"]))
    return [
        coverage, gap, span,
        float(evidence["arc_ratio"]), float(evidence["deviation_px"]) / max(height, 1),
        abs(float(evidence["sag_px"])) / max(height, 1), float(evidence["bend_energy"]),
        float(evidence["local_slope"]) * width / max(height, 1),
        float(result["global_trajectory"]["global_slope_normalized"]),
        float(scores["skip"]), float(scores["break"]), float(scores["loose"]),
        float(evidence["internal_gap_ratio"]), float(evidence["terminal_missing_ratio"]), continuity,
        abs(span - neighbor_median("span_ratio", span)),
        abs(coverage - neighbor_median("coverage", coverage)),
        abs(gap - neighbor_median("gap_ratio", gap)),
        float(bool(evidence.get("short_residual_break", False))),
        float(evidence.get("center_y_deviation_norm", 0.0)),
        float(result["global_trajectory"].get("center_slope_normalized", result["global_trajectory"]["global_slope_normalized"])),
    ]


def rows_from_masks(
    labels: Dict[str, List[Dict[str, str]]], mask_dir: Path, detector: TrajectorySelfReferenceDetector,
    source_from_label: bool, kind: str, include_ids: Iterable[str] | None = None,
) -> List[Dict]:
    allowed = set(include_ids) if include_ids is not None else None
    records: List[Dict] = []
    for sample_id, label_rows in labels.items():
        if allowed is not None and sample_id not in allowed:
            continue
        if source_from_label:
            path = mask_dir / label_rows[0]["mask_file"]
            source_id = label_rows[0]["source_sample_id"]
        else:
            path = mask_dir / f"{sample_id}.png"
            source_id = sample_id
        if not path.exists():
            raise FileNotFoundError(path)
        mask = read_mask(path)
        result = detector.detect(mask)
        units = result["units"]
        if len(units) != len(label_rows):
            raise RuntimeError(f"{sample_id}: detector returned {len(units)} units")
        for label, unit in zip(label_rows, units):
            records.append({
                "sample_id": sample_id, "source_id": source_id, "unit_id": int(label["unit_id"]),
                "kind": kind, "feature": feature_vector(result, unit, mask.shape[0], mask.shape[1]),
                "target": {name: int(label[name]) for name in (*LOCAL_CLASSES, "skew")},
            })
    return records


def normal_rows_from_masks(sources: Sequence[tuple[str, Path]], detector: TrajectorySelfReferenceDetector) -> List[Dict]:
    """Represent external normal M6 trajectories as all-zero local labels."""
    records: List[Dict] = []
    for domain, directory in sources:
        paths = sorted(directory.glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No M6 masks in {directory}")
        for path in paths:
            mask = read_mask(path)
            result = detector.detect(mask)
            source_id = f"{domain}:{path.stem}"
            for unit_id, unit in enumerate(result["units"]):
                records.append({
                    "sample_id": source_id, "source_id": source_id, "unit_id": unit_id,
                    "kind": "normal_reference", "feature": feature_vector(result, unit, mask.shape[0], mask.shape[1]),
                    "target": {name: 0 for name in (*LOCAL_CLASSES, "skew")},
                })
    return records


def split_normal_sources(records: Sequence[Dict], test_fraction: float, seed: int) -> tuple[List[Dict], List[Dict]]:
    """Stratified by material domain, keeping all six units of an image together."""
    by_domain: Dict[str, List[str]] = {}
    for row in records:
        domain = row["source_id"].split(":", 1)[0]
        by_domain.setdefault(domain, [])
        if row["source_id"] not in by_domain[domain]:
            by_domain[domain].append(row["source_id"])
    test_ids = set()
    for domain, source_ids in by_domain.items():
        values = sorted(source_ids)
        random.Random(f"{seed}:{domain}").shuffle(values)
        test_count = max(1, int(round(len(values) * test_fraction)))
        test_ids.update(values[:test_count])
    return [row for row in records if row["source_id"] not in test_ids], [row for row in records if row["source_id"] in test_ids]


def metric(target: np.ndarray, prediction: np.ndarray) -> Dict[str, float | int]:
    tp = int(np.logical_and(target == 1, prediction == 1).sum())
    fp = int(np.logical_and(target == 0, prediction == 1).sum())
    tn = int(np.logical_and(target == 0, prediction == 0).sum())
    fn = int(np.logical_and(target == 1, prediction == 0).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"positive": int(target.sum()), "total": int(len(target)), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / len(target) if len(target) else 0.0,
            "precision": precision, "recall": recall, "f1": f1}


def evaluate(records: Sequence[Dict], probabilities: np.ndarray, thresholds: Dict[str, float]) -> Dict:
    output: Dict[str, object] = {}
    f1s = []
    predictions = np.zeros_like(probabilities, dtype=np.int32)
    target = np.asarray([[row["target"][name] for name in LOCAL_CLASSES] for row in records], dtype=np.int32)
    for index, name in enumerate(LOCAL_CLASSES):
        predictions[:, index] = probabilities[:, index] >= thresholds[name]
        output[name] = metric(target[:, index], predictions[:, index])
        f1s.append(output[name]["f1"])
    normal_target = (target.sum(axis=1) == 0).astype(np.int32)
    normal_prediction = (predictions.sum(axis=1) == 0).astype(np.int32)
    output["normal"] = metric(normal_target, normal_prediction)
    output["macro_f1_local_defects"] = float(np.mean(f1s))
    output["exact_local_label_accuracy"] = float(np.all(target == predictions, axis=1).mean())
    return output


def skew_image_scores(records: Sequence[Dict], probability: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate skew once per source image, not six duplicated unit labels."""
    grouped: Dict[str, List[int]] = {}
    for index, row in enumerate(records):
        grouped.setdefault(row["source_id"], []).append(index)
    target, score = [], []
    for indices in grouped.values():
        target.append(int(any(records[index]["target"]["skew"] for index in indices)))
        score.append(float(np.mean(probability[indices])))
    return np.asarray(target, dtype=np.int32), np.asarray(score, dtype=np.float32)


def best_f1_threshold(target: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, probability, 1.0])
    best = (-1.0, -1.0, 0.5)
    for threshold in candidates:
        score = metric(target, probability >= threshold)["f1"]
        candidate = (float(score), -abs(float(threshold) - 0.5), float(threshold))
        if candidate > best:
            best = candidate
    return float(best[2])


def fit_models(
    records: Sequence[Dict], sample_weight: np.ndarray | None = None,
    classifier: str = "logistic", n_estimators: int = 300, min_samples_leaf: int = 2,
) -> Dict[str, Pipeline]:
    features = np.asarray([row["feature"] for row in records], dtype=np.float32)
    models: Dict[str, Pipeline] = {}
    for name in LOCAL_CLASSES:
        target = np.asarray([row["target"][name] for row in records], dtype=np.int32)
        if classifier == "extra_trees":
            model = Pipeline([("extra_trees", ExtraTreesClassifier(
                n_estimators=n_estimators, class_weight="balanced", min_samples_leaf=min_samples_leaf,
                max_features="sqrt", n_jobs=-1, random_state=42,
            ))])
            kwargs = {"extra_trees__sample_weight": sample_weight} if sample_weight is not None else {}
        else:
            model = Pipeline([
                ("scale", StandardScaler()),
                ("logistic", LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, random_state=42)),
            ])
            kwargs = {"logistic__sample_weight": sample_weight} if sample_weight is not None else {}
        model.fit(features, target, **kwargs)
        models[name] = model
    return models


def predict(models: Dict[str, Pipeline], records: Sequence[Dict]) -> np.ndarray:
    features = np.asarray([row["feature"] for row in records], dtype=np.float32)
    return np.column_stack([models[name].predict_proba(features)[:, 1] for name in LOCAL_CLASSES])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight classifier on controlled mask trajectory features")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--augmented-labels", type=Path, required=True)
    parser.add_argument("--augmented-mask-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--original-labels", type=Path, required=True)
    parser.add_argument("--m6-mask-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m6-real-weight", type=float, default=4.0)
    parser.add_argument("--normal-source", type=parse_normal_source, action="append", default=[])
    parser.add_argument("--normal-test-fraction", type=float, default=0.20)
    parser.add_argument("--normal-weight", type=float, default=0.25)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--classifier", choices=("logistic", "extra_trees"), default="logistic")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    detector = load_detector(args.config)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    dev_ids = split["development_source_ids"]
    holdout_ids = split["internal_holdout_source_ids"]
    augmented_labels = read_grouped_labels(args.augmented_labels)
    original_labels = read_grouped_labels(args.original_labels, originals_only=True)

    augmented = rows_from_masks(augmented_labels, args.augmented_mask_dir, detector, True, "controlled_mask")
    dev_real = rows_from_masks(original_labels, args.m6_mask_dir, detector, False, "m6_real", dev_ids)
    holdout_real = rows_from_masks(original_labels, args.m6_mask_dir, detector, False, "m6_holdout", holdout_ids)
    normal_all = normal_rows_from_masks(args.normal_source, detector) if args.normal_source else []
    normal_train, normal_test = split_normal_sources(normal_all, args.normal_test_fraction, 42) if normal_all else ([], [])
    training = augmented + dev_real + normal_train
    weights = np.asarray([
        args.m6_real_weight if row["kind"] == "m6_real" else args.normal_weight if row["kind"] == "normal_reference" else 1.0
        for row in training
    ], dtype=np.float32)

    # Out-of-fold predictions on real M6 masks only.  They calibrate operating
    # thresholds without looking at the hold-out images.
    oof_prob = np.full((len(dev_real), len(LOCAL_CLASSES)), np.nan, dtype=np.float32)
    groups = np.asarray([row["source_id"] for row in training])
    dev_lookup = {(row["source_id"], row["unit_id"]): index for index, row in enumerate(dev_real)}
    for train_index, validation_index in GroupKFold(n_splits=args.folds).split(training, groups=groups):
        fold_train = [training[index] for index in train_index]
        fold_weights = weights[train_index]
        models = fit_models(fold_train, fold_weights, args.classifier, args.n_estimators, args.min_samples_leaf)
        fold_validation = [training[index] for index in validation_index if training[index]["kind"] == "m6_real"]
        probabilities = predict(models, fold_validation)
        for row, probability in zip(fold_validation, probabilities):
            oof_prob[dev_lookup[(row["source_id"], row["unit_id"])]] = probability
    if np.isnan(oof_prob).any():
        raise RuntimeError("incomplete out-of-fold prediction")
    thresholds = {
        name: best_f1_threshold(
            np.asarray([row["target"][name] for row in dev_real], dtype=np.int32), oof_prob[:, index]
        ) for index, name in enumerate(LOCAL_CLASSES)
    }
    oof_metrics = evaluate(dev_real, oof_prob, thresholds)

    # Reference implementation: the unlearned TSR local scores, calibrated
    # only on the same 36 development sources.  This keeps the comparison to
    # the learned lightweight classifier transparent.
    rule_dev_prob = np.asarray([row["feature"][-3:] for row in dev_real], dtype=np.float32)
    rule_thresholds = {
        name: best_f1_threshold(
            np.asarray([row["target"][name] for row in dev_real], dtype=np.int32), rule_dev_prob[:, index]
        ) for index, name in enumerate(LOCAL_CLASSES)
    }
    skew_dev_probability = np.clip(
        np.asarray([row["feature"][8] for row in dev_real], dtype=np.float32)
        / max(detector.config.global_slope_score_scale, 1e-6), 0.0, 1.0
    )
    skew_dev_target, skew_dev_score = skew_image_scores(dev_real, skew_dev_probability, 0.5)
    skew_threshold = best_f1_threshold(skew_dev_target, skew_dev_score)

    models = fit_models(training, weights, args.classifier, args.n_estimators, args.min_samples_leaf)
    holdout_eval = holdout_real + normal_test
    holdout_prob = predict(models, holdout_eval)
    holdout_metrics = evaluate(holdout_eval, holdout_prob, thresholds)
    rule_holdout_metrics = evaluate(
        holdout_eval, np.asarray([row["feature"][-3:] for row in holdout_eval], dtype=np.float32), rule_thresholds
    )
    skew_holdout_probability = np.clip(
        np.asarray([row["feature"][8] for row in holdout_eval], dtype=np.float32)
        / max(detector.config.global_slope_score_scale, 1e-6), 0.0, 1.0
    )
    skew_holdout_target, skew_holdout_score = skew_image_scores(holdout_eval, skew_holdout_probability, skew_threshold)
    skew_holdout_metrics = metric(skew_holdout_target, skew_holdout_score >= skew_threshold)
    four_class_macro = float(np.mean([
        holdout_metrics[name]["f1"] for name in LOCAL_CLASSES
    ] + [skew_holdout_metrics["f1"]]))
    model_path = args.output / "trajectory_feature_classifier.joblib"
    joblib.dump({"models": models, "thresholds": thresholds, "features": FEATURE_NAMES, "local_classes": LOCAL_CLASSES}, model_path)

    predictions = []
    for row, probability in zip(holdout_eval, holdout_prob):
        predictions.append({
            "sample_id": row["sample_id"], "unit_id": row["unit_id"], "target": row["target"],
            "probability": dict(zip(LOCAL_CLASSES, map(float, probability))),
            "prediction": {name: bool(probability[index] >= thresholds[name]) for index, name in enumerate(LOCAL_CLASSES)},
        })
    report = {
        "model": f"balanced {args.classifier} classifier on TSR trajectory features",
        "local_classes": LOCAL_CLASSES, "features": FEATURE_NAMES, "thresholds": thresholds,
        "training": {"controlled_mask_unit_rows": len(augmented), "real_m6_development_unit_rows": len(dev_real),
                     "normal_reference_unit_rows": len(normal_train), "m6_real_weight": args.m6_real_weight,
                     "normal_weight": args.normal_weight, "development_source_count": len(dev_ids)},
        "internal_holdout": {"source_ids": holdout_ids, "source_count": len(holdout_ids),
                             "defect_unit_count": len(holdout_real), "normal_unit_count": len(normal_test),
                             "metrics": holdout_metrics},
        "rule_baseline_internal_holdout": {"thresholds": rule_thresholds, "metrics": rule_holdout_metrics},
        "hybrid_four_class_internal_holdout": {
            "local_classifier": "unit-level skip/break/loose", "skew": "image-level TSR global-slope rule",
            "skew_threshold": skew_threshold, "skew_image_metrics": skew_holdout_metrics,
            "mixed_granularity_macro_f1": four_class_macro,
        },
        "development_oof_m6": {"source_count": len(dev_ids), "unit_count": len(dev_real), "metrics": oof_metrics},
        "caveat": "The 12 defect images are an internal hold-out only. They cannot be presented as an independent final defect test because all 48 sources were inspected during earlier rule development. External normal images are split by source and held out from fitting.",
    }
    save_json(args.output / "report.json", report)
    save_json(args.output / "internal_holdout_predictions.json", predictions)
    print("thresholds:", json.dumps(thresholds, ensure_ascii=False))
    print("development OOF macro F1:", f"{oof_metrics['macro_f1_local_defects']:.4f}")
    print("internal hold-out macro F1:", f"{holdout_metrics['macro_f1_local_defects']:.4f}")
    print("rule baseline internal hold-out macro F1:", f"{rule_holdout_metrics['macro_f1_local_defects']:.4f}")
    print("hybrid four-class macro F1 (skew at image level):", f"{four_class_macro:.4f}")
    for name in (*LOCAL_CLASSES, "normal"):
        item = holdout_metrics[name]
        print(f"  {name}: Acc={item['accuracy']:.4f}, F1={item['f1']:.4f}, P={item['precision']:.4f}, R={item['recall']:.4f}")


if __name__ == "__main__":
    main()
