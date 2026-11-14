"""Leakage-free nested four-fold evaluation for TSR-Defect.

The outer split estimates generalisation on the configured real defect sources.  Within
each outer training split, three-fold source-grouped OOF predictions calibrate
the operating thresholds.  Clear, direct-blur and MBDNet-restored APHNet masks
share the same frozen fold model and clear-calibrated thresholds.

Controlled mask variants are generated deterministically, but a variant is
eligible for fitting only when its original source belongs to the current
training split.  Normal images are split once by material domain into 1,360
training sources and 340 untouched test sources; the latter are assigned to
the four outer folds and evaluated exactly once.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from defect_detection.build_controlled_mask_defect_augmentation import (  # noqa: E402
    choose_normal_unit,
    synthesize_break,
    synthesize_loose,
    synthesize_skip,
)
from defect_detection.train_trajectory_feature_classifier import (  # noqa: E402
    FEATURE_NAMES,
    LOCAL_CLASSES,
    best_f1_threshold,
    feature_vector,
    fit_models,
    load_detector,
    metric,
    predict,
    read_grouped_labels,
)


DEFECT_CLASSES = ("skip", "break", "skew", "loose")
ALL_IMAGE_CLASSES = (*DEFECT_CLASSES, "normal")


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_mask(directory: Path, sample_id: str) -> np.ndarray:
    npy = directory / f"{sample_id}.npy"
    png = directory / f"{sample_id}.png"
    if npy.exists():
        return np.load(npy).astype(np.float32)
    if png.exists():
        return np.asarray(Image.open(png).convert("L"), dtype=np.float32) / 255.0
    raise FileNotFoundError(f"Missing mask for {sample_id}: {npy} or {png}")


def image_class(rows: Sequence[Dict[str, str]]) -> str:
    active = [name for name in DEFECT_CLASSES if any(int(row[name]) for row in rows)]
    if len(active) != 1:
        raise ValueError(f"Expected one image-level defect class, got {active}")
    return active[0]


def stratified_folds(
    source_ids: Sequence[str], labels: Dict[str, List[Dict[str, str]]], n_splits: int, seed: int,
) -> List[List[str]]:
    ids = np.asarray(sorted(source_ids), dtype=object)
    target = np.asarray([image_class(labels[str(value)]) for value in ids], dtype=object)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [[str(ids[index]) for index in validation] for _, validation in splitter.split(ids, target)]


def rows_from_condition(
    labels: Dict[str, List[Dict[str, str]]], directory: Path, detector, kind: str, feature_fn,
) -> List[Dict]:
    output: List[Dict] = []
    for source_id, label_rows in sorted(labels.items()):
        mask = load_mask(directory, source_id)
        result = detector.detect(mask)
        if len(result["units"]) != len(label_rows):
            raise RuntimeError(f"{source_id}: detector did not return six units")
        for label, unit in zip(label_rows, result["units"]):
            if not int(label.get("valid", "1")):
                continue
            output.append({
                "sample_id": source_id,
                "source_id": source_id,
                "unit_id": int(label["unit_id"]),
                "kind": kind,
                "feature": feature_fn(result, unit, mask),
                "target": {name: int(label[name]) for name in DEFECT_CLASSES},
            })
    return output


def list_mask_ids(directory: Path) -> List[str]:
    ids = {path.stem for suffix in ("*.npy", "*.png") for path in directory.glob(suffix)}
    if not ids:
        raise FileNotFoundError(f"No masks found in {directory}")
    return sorted(ids)


def normal_rows(normal_sources: Dict[str, Path], detector, feature_fn) -> List[Dict]:
    output: List[Dict] = []
    for domain, directory in sorted(normal_sources.items()):
        for sample_id in list_mask_ids(directory):
            mask = load_mask(directory, sample_id)
            result = detector.detect(mask)
            source_id = f"{domain}:{sample_id}"
            for unit_id, unit in enumerate(result["units"]):
                output.append({
                    "sample_id": source_id,
                    "source_id": source_id,
                    "unit_id": unit_id,
                    "kind": "normal_reference",
                    "feature": feature_fn(result, unit, mask),
                    "target": {name: 0 for name in DEFECT_CLASSES},
                })
    return output


def group_ids(records: Sequence[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in records:
        grouped[row["source_id"]].append(row)
    return dict(grouped)


def split_normal_sources(
    records: Sequence[Dict], test_fraction: float, seed: int,
) -> tuple[List[Dict], List[Dict], Dict[str, Dict[str, int]]]:
    grouped = group_ids(records)
    by_domain: Dict[str, List[str]] = defaultdict(list)
    for source_id in grouped:
        by_domain[source_id.split(":", 1)[0]].append(source_id)
    test_ids: set[str] = set()
    counts: Dict[str, Dict[str, int]] = {}
    for domain, values in sorted(by_domain.items()):
        ids = sorted(values)
        random.Random(f"{seed}:{domain}").shuffle(ids)
        count = max(1, int(round(len(ids) * test_fraction)))
        test_ids.update(ids[:count])
        counts[domain] = {"all": len(ids), "train": len(ids) - count, "test": count}
    train = [row for row in records if row["source_id"] not in test_ids]
    test = [row for row in records if row["source_id"] in test_ids]
    return train, test, counts


def domain_stratified_partitions(source_ids: Sequence[str], n_splits: int, seed: int) -> List[List[str]]:
    by_domain: Dict[str, List[str]] = defaultdict(list)
    for source_id in source_ids:
        by_domain[source_id.split(":", 1)[0]].append(source_id)
    folds: List[List[str]] = [[] for _ in range(n_splits)]
    for domain, values in sorted(by_domain.items()):
        ids = sorted(values)
        random.Random(f"{seed}:{domain}:folds").shuffle(ids)
        for index, source_id in enumerate(ids):
            folds[index % n_splits].append(source_id)
    return [sorted(values) for values in folds]


def controlled_rows(
    labels: Dict[str, List[Dict[str, str]]], gt_mask_dir: Path, detector,
    variants_per_class: int, seed: int, feature_fn,
) -> tuple[List[Dict], List[Dict]]:
    transforms = {"skip": synthesize_skip, "break": synthesize_break, "loose": synthesize_loose}
    output: List[Dict] = []
    manifest: List[Dict] = []
    for source_id, source_rows in sorted(labels.items()):
        base = (load_mask(gt_mask_dir, source_id) > 0.5).astype(np.uint8) * 255
        rng = random.Random(f"{seed}:controlled:{source_id}")
        variants = [("base", None, None, base, {})]
        for defect, transform in transforms.items():
            for variant_index in range(variants_per_class):
                unit = choose_normal_unit(source_rows, rng)
                altered, parameters = transform(base, int(unit["x0"]), int(unit["x1"]), rng)
                variants.append((defect, variant_index + 1, int(unit["unit_id"]), altered, parameters))
        for defect, variant_index, target_unit, mask, parameters in variants:
            result = detector.detect(mask.astype(np.float32) / 255.0)
            variant_id = f"aug_{source_id}_{defect}" if variant_index is None else f"aug_{source_id}_{defect}_{variant_index:02d}"
            for label, unit in zip(source_rows, result["units"]):
                if not int(label.get("valid", "1")):
                    continue
                target = {name: int(label[name]) for name in DEFECT_CLASSES}
                if target_unit is not None and int(label["unit_id"]) == target_unit:
                    target[defect] = 1
                output.append({
                    "sample_id": variant_id,
                    "source_id": source_id,
                    "unit_id": int(label["unit_id"]),
                    "kind": "controlled_mask",
                    "feature": feature_fn(result, unit, mask.astype(np.float32) / 255.0),
                    "target": target,
                })
            manifest.append({
                "sample_id": variant_id,
                "source_id": source_id,
                "synthetic_class": "normal" if defect == "base" else defect,
                "target_unit": target_unit,
                "parameters": parameters,
            })
    return output, manifest


def select_sources(records: Sequence[Dict], source_ids: Iterable[str]) -> List[Dict]:
    allowed = set(source_ids)
    return [row for row in records if row["source_id"] in allowed]


def record_weights(records: Sequence[Dict], real_weight: float, normal_weight: float) -> np.ndarray:
    return np.asarray([
        real_weight if row["kind"].endswith("_real") else normal_weight if row["kind"] == "normal_reference" else 1.0
        for row in records
    ], dtype=np.float32)


def fit_weighted(records: Sequence[Dict], model_cfg: Dict, weights_cfg: Dict):
    real_weight = float(weights_cfg.get("real_condition", weights_cfg.get("real_clear", 4.0)))
    weights = record_weights(records, real_weight, float(weights_cfg["normal"]))
    return fit_models(
        records,
        weights,
        classifier=str(model_cfg["classifier"]),
        n_estimators=int(model_cfg["n_estimators"]),
        min_samples_leaf=int(model_cfg["min_samples_leaf"]),
    )


def skew_score(row: Dict, detector) -> float:
    return float(np.clip(
        float(row["feature"][8]) / max(float(detector.config.global_slope_score_scale), 1e-6), 0.0, 1.0
    ))


def image_skew_arrays(records: Sequence[Dict], detector) -> tuple[np.ndarray, np.ndarray]:
    grouped = group_ids(records)
    target, score = [], []
    for rows in grouped.values():
        target.append(int(any(row["target"]["skew"] for row in rows)))
        score.append(float(np.mean([skew_score(row, detector) for row in rows])))
    return np.asarray(target, dtype=np.int32), np.asarray(score, dtype=np.float32)


def calibration_observations(
    records: Sequence[Dict], probabilities: np.ndarray, detector,
) -> List[Dict]:
    """Aggregate OOF unit probabilities into independent image-condition observations."""
    grouped: Dict[tuple[str, str], List[int]] = defaultdict(list)
    for index, row in enumerate(records):
        grouped[(row["kind"], row["source_id"])].append(index)
    observations: List[Dict] = []
    for (kind, source_id), indices in sorted(grouped.items()):
        active = [
            name for name in DEFECT_CLASSES
            if any(records[index]["target"][name] for index in indices)
        ]
        if len(active) > 1:
            raise ValueError(f"{kind}/{source_id}: multiple image-level targets {active}")
        observations.append({
            "kind": kind,
            "source_id": source_id,
            "target_class": active[0] if active else "normal",
            "scores": {
                **{
                    name: max(float(probabilities[index, column]) for index in indices)
                    for column, name in enumerate(LOCAL_CLASSES)
                },
                "skew": float(np.mean([skew_score(records[index], detector) for index in indices])),
            },
        })
    return observations


def threshold_objective(observations: Sequence[Dict], thresholds: Dict[str, float]) -> Dict[str, float]:
    entries = []
    for row in observations:
        ratios = {
            name: float(row["scores"][name]) / max(float(thresholds[name]), 1e-8)
            for name in DEFECT_CLASSES
        }
        winner = max(DEFECT_CLASSES, key=lambda name: ratios[name])
        predicted = winner if ratios[winner] >= 1.0 else "normal"
        entries.append({"target_class": row["target_class"], "predicted_class": predicted})
    defect_entries = [row for row in entries if row["target_class"] != "normal"]
    normal_entries = [row for row in entries if row["target_class"] == "normal"]
    image_metric = multiclass_metrics(entries, DEFECT_CLASSES)
    return {
        "macro_f1": image_metric["macro_f1"],
        "accuracy": image_metric["accuracy"],
        "defect_alarm_recall": float(np.mean([
            row["predicted_class"] != "normal" for row in defect_entries
        ])) if defect_entries else 0.0,
        "normal_false_alarm_rate": float(np.mean([
            row["predicted_class"] != "normal" for row in normal_entries
        ])) if normal_entries else 0.0,
    }


def optimize_thresholds_with_recall_floor(
    observations: Sequence[Dict], initial: Dict[str, float], recall_floor: float, passes: int,
    objective_name: str = "macro_f1",
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Coordinate-search one shared threshold set under an image alarm-recall constraint."""
    thresholds = dict(initial)
    candidate_values = {
        name: sorted(set(
            [float(initial[name]), 0.0, 1.0]
            + [float(row["scores"][name]) for row in observations]
        ))
        for name in DEFECT_CLASSES
    }

    def rank(metrics: Dict[str, float], threshold: float) -> tuple:
        feasible = metrics["defect_alarm_recall"] + 1e-12 >= recall_floor
        primary = (
            0.5 * (metrics["macro_f1"] + metrics["defect_alarm_recall"])
            if objective_name == "balanced_f1_alarm" else metrics[objective_name]
        )
        if feasible:
            return (
                1,
                primary,
                -metrics["normal_false_alarm_rate"],
                metrics["defect_alarm_recall"],
                -abs(threshold - 0.5),
            )
        return (
            0,
            metrics["defect_alarm_recall"],
            primary,
            -metrics["normal_false_alarm_rate"],
            -abs(threshold - 0.5),
        )

    for _ in range(max(1, passes)):
        changed = False
        for name in DEFECT_CLASSES:
            best_threshold = thresholds[name]
            best_metrics = threshold_objective(observations, thresholds)
            best_rank = rank(best_metrics, best_threshold)
            for candidate in candidate_values[name]:
                trial = dict(thresholds)
                trial[name] = candidate
                metrics = threshold_objective(observations, trial)
                current_rank = rank(metrics, candidate)
                if current_rank > best_rank:
                    best_threshold, best_metrics, best_rank = candidate, metrics, current_rank
            if abs(best_threshold - thresholds[name]) > 1e-12:
                changed = True
                thresholds[name] = best_threshold
        if not changed:
            break
    return thresholds, threshold_objective(observations, thresholds)


def calibrate_inner_oof(
    outer_train_ids: Sequence[str], labels: Dict[str, List[Dict[str, str]]],
    condition_rows: Dict[str, Sequence[Dict]], training_conditions: Sequence[str],
    calibration_conditions: Sequence[str],
    augmented_rows: Sequence[Dict], normal_train: Sequence[Dict], detector, inner_folds: int,
    seed: int, model_cfg: Dict, weights_cfg: Dict, recall_floor: float, coordinate_passes: int,
    threshold_objective_name: str,
) -> tuple[Dict[str, float], float, Dict]:
    defect_folds = stratified_folds(outer_train_ids, labels, inner_folds, seed)
    normal_ids = sorted(group_ids(normal_train))
    normal_folds = domain_stratified_partitions(normal_ids, inner_folds, seed)
    oof_records: List[Dict] = []
    oof_probabilities: List[List[float]] = []
    fold_audit = []
    outer_set = set(outer_train_ids)
    for fold_index in range(inner_folds):
        defect_validation = set(defect_folds[fold_index])
        defect_training = outer_set - defect_validation
        normal_validation = set(normal_folds[fold_index])
        normal_training = set(normal_ids) - normal_validation
        fitting = []
        for condition in training_conditions:
            fitting.extend(select_sources(condition_rows[condition], defect_training))
        fitting.extend(select_sources(augmented_rows, defect_training))
        fitting.extend(select_sources(normal_train, normal_training))
        validation = []
        for condition in calibration_conditions:
            validation.extend(select_sources(condition_rows[condition], defect_validation))
        validation.extend(select_sources(normal_train, normal_validation))
        models = fit_weighted(fitting, model_cfg, weights_cfg)
        probabilities = predict(models, validation)
        oof_records.extend(validation)
        oof_probabilities.extend(probabilities.tolist())
        fit_sources = {row["source_id"] for row in fitting}
        validation_sources = {row["source_id"] for row in validation}
        overlap = sorted(fit_sources & validation_sources)
        if overlap:
            raise RuntimeError(f"Inner-fold leakage: {overlap[:5]}")
        fold_audit.append({
            "inner_fold": fold_index + 1,
            "defect_training_sources": sorted(defect_training),
            "defect_validation_sources": sorted(defect_validation),
            "normal_training_source_count": len(normal_training),
            "normal_validation_source_count": len(normal_validation),
            "fit_unit_rows": len(fitting),
            "validation_unit_rows": len(validation),
            "source_overlap": overlap,
        })
    probabilities = np.asarray(oof_probabilities, dtype=np.float32)
    expected_sources = set(outer_train_ids) | set(normal_ids)
    observed_sources = {row["source_id"] for row in oof_records}
    if observed_sources != expected_sources:
        raise RuntimeError("Incomplete inner OOF coverage")
    initial_thresholds = {
        name: best_f1_threshold(
            np.asarray([row["target"][name] for row in oof_records], dtype=np.int32), probabilities[:, index]
        )
        for index, name in enumerate(LOCAL_CLASSES)
    }
    skew_target, skew_probability = image_skew_arrays(oof_records, detector)
    initial_thresholds["skew"] = best_f1_threshold(skew_target, skew_probability)
    observations = calibration_observations(oof_records, probabilities, detector)
    optimized_thresholds, optimized_objective = optimize_thresholds_with_recall_floor(
        observations, initial_thresholds, recall_floor, coordinate_passes, threshold_objective_name
    )
    thresholds = {name: optimized_thresholds[name] for name in LOCAL_CLASSES}
    skew_threshold = optimized_thresholds["skew"]
    calibration_metrics = {
        name: metric(
            np.asarray([row["target"][name] for row in oof_records], dtype=np.int32),
            probabilities[:, index] >= thresholds[name],
        )
        for index, name in enumerate(LOCAL_CLASSES)
    }
    calibration_metrics["skew"] = metric(skew_target, skew_probability >= skew_threshold)
    return thresholds, skew_threshold, {
        "inner_folds": fold_audit,
        "oof_source_count": len(observed_sources),
        "oof_unit_rows": len(oof_records),
        "oof_image_condition_observations": len(observations),
        "training_conditions": list(training_conditions),
        "calibration_conditions": list(calibration_conditions),
        "recall_floor": recall_floor,
        "initial_thresholds": initial_thresholds,
        "thresholds": {**thresholds, "skew": skew_threshold},
        "optimized_image_objective": optimized_objective,
        "threshold_objective": threshold_objective_name,
        "metrics": calibration_metrics,
    }


def prediction_entries(
    records: Sequence[Dict], probabilities: np.ndarray, thresholds: Dict[str, float],
    skew_threshold: float, detector, target_class_by_source: Dict[str, str], fold_index: int,
) -> tuple[List[Dict], List[Dict]]:
    unit_entries: List[Dict] = []
    grouped_indices: Dict[str, List[int]] = defaultdict(list)
    for index, (row, probability) in enumerate(zip(records, probabilities)):
        grouped_indices[row["source_id"]].append(index)
        unit_entries.append({
            "fold": fold_index,
            "source_id": row["source_id"],
            "unit_id": row["unit_id"],
            "target": {name: int(row["target"][name]) for name in LOCAL_CLASSES},
            "probability": {name: float(probability[column]) for column, name in enumerate(LOCAL_CLASSES)},
            "prediction": {
                name: bool(probability[column] >= thresholds[name]) for column, name in enumerate(LOCAL_CLASSES)
            },
        })
    image_entries: List[Dict] = []
    for source_id, indices in sorted(grouped_indices.items()):
        local_ratio = {
            name: max(float(probabilities[index, column]) / max(float(thresholds[name]), 1e-8) for index in indices)
            for column, name in enumerate(LOCAL_CLASSES)
        }
        current_skew_score = float(np.mean([skew_score(records[index], detector) for index in indices]))
        ratios = {**local_ratio, "skew": current_skew_score / max(float(skew_threshold), 1e-8)}
        winner = max(DEFECT_CLASSES, key=lambda name: ratios[name])
        final_class = winner if ratios[winner] >= 1.0 else "normal"
        image_entries.append({
            "fold": fold_index,
            "source_id": source_id,
            "target_class": target_class_by_source[source_id],
            "predicted_class": final_class,
            "score_over_threshold": ratios,
            "skew_score": current_skew_score,
            "skew_prediction": bool(current_skew_score >= skew_threshold),
        })
    return unit_entries, image_entries


def multiclass_metrics(entries: Sequence[Dict], classes: Sequence[str]) -> Dict:
    true = [row["target_class"] for row in entries]
    predicted = [row["predicted_class"] for row in entries]
    per_class = {}
    precisions, recalls, f1s = [], [], []
    for name in classes:
        y = np.asarray([value == name for value in true], dtype=np.int32)
        p = np.asarray([value == name for value in predicted], dtype=np.int32)
        current = metric(y, p)
        per_class[name] = current
        precisions.append(current["precision"])
        recalls.append(current["recall"])
        f1s.append(current["f1"])
    confusion_classes = list(classes)
    for value in predicted:
        if value not in confusion_classes:
            confusion_classes.append(value)
    return {
        "count": len(entries),
        "accuracy": float(np.mean([a == b for a, b in zip(true, predicted)])) if entries else 0.0,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_labels": confusion_classes,
        "confusion_matrix": confusion_matrix(true, predicted, labels=confusion_classes).tolist(),
    }


def evaluate_outputs(unit_entries: Sequence[Dict], image_entries: Sequence[Dict]) -> Dict:
    class_metrics = {}
    for name in LOCAL_CLASSES:
        target = np.asarray([row["target"][name] for row in unit_entries], dtype=np.int32)
        prediction_values = np.asarray([row["prediction"][name] for row in unit_entries], dtype=np.int32)
        class_metrics[name] = metric(target, prediction_values)
    skew_target = np.asarray([row["target_class"] == "skew" for row in image_entries], dtype=np.int32)
    skew_prediction = np.asarray([row["skew_prediction"] for row in image_entries], dtype=np.int32)
    class_metrics["skew"] = metric(skew_target, skew_prediction)
    image_metric = multiclass_metrics(image_entries, DEFECT_CLASSES)
    local_exact = [
        all(int(row["prediction"][name]) == row["target"][name] for name in LOCAL_CLASSES)
        for row in unit_entries
    ]
    return {
        "mixed_scope_macro_f1": float(np.mean([class_metrics[name]["f1"] for name in DEFECT_CLASSES])),
        "class_metrics": class_metrics,
        "defect_image_classification": image_metric,
        "defect_alarm_recall": float(np.mean([row["predicted_class"] != "normal" for row in image_entries])),
        "local_unit_exact_accuracy": float(np.mean(local_exact)),
    }


def render_figures(output: Path, summaries: Dict[str, Dict], image_entries: Dict[str, List[Dict]]) -> None:
    order = tuple(summaries)
    display = {"clear": "Clear", "blur_direct": "Direct blur", "deblur_then_aphnet": "MBDNet + ACHNet"}
    labels = tuple(display.get(name, name.replace("_", " ")) for name in order)
    colors = ("#4472C4", "#C55A5A", "#70AD47", "#8064A2")
    fig, axis = plt.subplots(figsize=(7.2, 4.2), dpi=220)
    x = np.arange(len(order))
    f1 = [summaries[name]["defect_image_classification"]["macro_f1"] * 100 for name in order]
    accuracy = [summaries[name]["defect_image_classification"]["accuracy"] * 100 for name in order]
    width = 0.34
    left = axis.bar(x - width / 2, f1, width, label="Image macro F1", color="#5B9BD5")
    right = axis.bar(x + width / 2, accuracy, width, label="Image accuracy", color="#A5A5A5")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Score (%)")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=2, loc="lower center")
    for bars in (left, right):
        for bar in bars:
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                      f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "nested_oof_condition_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(order), figsize=(3.9 * len(order), 3.6), dpi=220, squeeze=False)
    axes = axes.ravel()
    for axis, condition, title, color in zip(axes, order, labels, colors):
        current_confusion = summaries[condition]["defect_image_classification"]
        column_labels = current_confusion["confusion_labels"]
        matrix = np.asarray(current_confusion["confusion_matrix"], dtype=np.int32)[:4, :]
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
        axis.set_title(title, color=color, fontweight="bold")
        display_labels = {"skip": "Skip", "break": "Break", "skew": "Skew", "loose": "Loose", "normal": "Normal"}
        axis.set_xticks(
            range(len(column_labels)),
            [display_labels[value] for value in column_labels],
            rotation=35,
            ha="right",
        )
        axis.set_yticks(range(4), ("Skip", "Break", "Skew", "Loose"))
        axis.set_xlabel("Predicted")
        if axis is axes[0]:
            axis.set_ylabel("Ground truth")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                          color="white" if matrix[i, j] > matrix.max() / 2 else "#1F1F1F")
    fig.tight_layout()
    fig.savefig(output / "nested_oof_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


def write_csv(output: Path, summaries: Dict[str, Dict], deployment: Dict) -> None:
    with (output / "nested_oof_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "condition", "defect_image_macro_f1", "defect_image_accuracy", "defect_alarm_recall",
            "mixed_scope_macro_f1", "local_unit_exact_accuracy", *[f"{name}_f1" for name in DEFECT_CLASSES],
        ])
        for condition in summaries:
            current = summaries[condition]
            writer.writerow([
                condition,
                current["defect_image_classification"]["macro_f1"],
                current["defect_image_classification"]["accuracy"],
                current["defect_alarm_recall"],
                current["mixed_scope_macro_f1"],
                current["local_unit_exact_accuracy"],
                *[current["class_metrics"][name]["f1"] for name in DEFECT_CLASSES],
            ])
    with (output / "deployment_clear_with_normal.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key in ("five_class_macro_f1", "five_class_accuracy", "defect_alarm_recall", "normal_false_alarm_rate"):
            writer.writerow([key, deployment[key]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-free nested four-fold TSR-Defect evaluation")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = {name: Path(value) for name, value in config["paths"].items()}
    output = paths["output"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {output}")
    output.mkdir(parents=True)

    protocol = config["protocol"]
    seed = int(protocol["seed"])
    outer_count = int(protocol["outer_folds"])
    inner_count = int(protocol["inner_folds"])
    detector = load_detector(paths["tsr_config"])
    feature_set = str(config.get("feature_set", "trajectory"))
    if feature_set == "trajectory":
        feature_fn = lambda result, unit, mask: feature_vector(result, unit, mask.shape[0], mask.shape[1])
        feature_names = FEATURE_NAMES
    elif feature_set in ("binary_topology", "binary_topology_v2"):
        from defect_detection.binary_topology_features import (
            FEATURE_NAMES as BINARY_FEATURE_NAMES,
            FEATURE_NAMES_V2 as BINARY_FEATURE_NAMES_V2,
            binary_topology_feature_vector,
            binary_topology_v2_feature_vector,
        )
        if feature_set == "binary_topology_v2":
            feature_fn = binary_topology_v2_feature_vector
            feature_names = BINARY_FEATURE_NAMES_V2
        else:
            feature_fn = binary_topology_feature_vector
            feature_names = BINARY_FEATURE_NAMES
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    labels = read_grouped_labels(paths["labels"], originals_only=True)
    if not labels:
        raise ValueError("No original defect sources were found in the label file")
    class_counts = {name: sum(image_class(rows) == name for rows in labels.values()) for name in DEFECT_CLASSES}
    if min(class_counts.values()) < outer_count:
        raise ValueError(f"Each defect class needs at least {outer_count} sources: {class_counts}")

    condition_dirs = {}
    for path_name, value in paths.items():
        if not path_name.endswith("_aphnet_masks"):
            continue
        stem = path_name.removesuffix("_aphnet_masks")
        # Preserve the historical, explicit condition name used for the
        # MBDNet-restored path while accepting both old and new path keys.
        condition = "deblur_then_aphnet" if stem == "deblur_then" else stem
        condition_dirs[condition] = value
    if "clear" not in condition_dirs:
        raise ValueError("paths.clear_aphnet_masks is required")
    condition_rows = {
        name: rows_from_condition(labels, directory, detector, f"{name}_real", feature_fn)
        for name, directory in condition_dirs.items()
    }
    training_conditions = list(protocol.get("training_conditions", ["clear"]))
    calibration_conditions = list(protocol.get("calibration_conditions", training_conditions))
    unknown_conditions = (set(training_conditions) | set(calibration_conditions)) - set(condition_rows)
    if unknown_conditions:
        raise ValueError(f"Unknown train/calibration conditions: {sorted(unknown_conditions)}")
    controlled, controlled_manifest = controlled_rows(
        labels,
        paths["gt_mask_dir"],
        detector,
        int(config["augmentation"]["variants_per_local_class"]),
        seed,
        feature_fn,
    )
    normal_directories = {
        name.removeprefix("normal_"): value for name, value in paths.items() if name.startswith("normal_")
    }
    all_normal = normal_rows(normal_directories, detector, feature_fn)
    normal_train, normal_test, normal_counts = split_normal_sources(
        all_normal, float(protocol["normal_test_fraction"]), seed
    )
    normal_test_groups = group_ids(normal_test)
    normal_test_folds = domain_stratified_partitions(sorted(normal_test_groups), outer_count, seed)
    source_ids = sorted(labels)
    outer_folds = stratified_folds(source_ids, labels, outer_count, seed)

    save_json(output / "controlled_augmentation_manifest.json", controlled_manifest)
    all_unit_entries: Dict[str, List[Dict]] = {name: [] for name in condition_dirs}
    all_image_entries: Dict[str, List[Dict]] = {name: [] for name in condition_dirs}
    all_normal_image_entries: List[Dict] = []
    fold_reports = []
    all_sources = set(source_ids)
    target_class_by_source = {source_id: image_class(rows) for source_id, rows in labels.items()}
    normal_target_class = {source_id: "normal" for source_id in normal_test_groups}

    for outer_index, test_values in enumerate(outer_folds):
        test_ids = set(test_values)
        train_ids = all_sources - test_ids
        thresholds, skew_threshold, calibration = calibrate_inner_oof(
            sorted(train_ids),
            labels,
            condition_rows,
            training_conditions,
            calibration_conditions,
            controlled,
            normal_train,
            detector,
            inner_count,
            seed + outer_index,
            config["model"],
            config["weights"],
            float(protocol.get("min_defect_alarm_recall", 0.0)),
            int(protocol.get("threshold_coordinate_passes", 2)),
            str(protocol.get("threshold_objective", "macro_f1")),
        )
        fitting = []
        for condition in training_conditions:
            fitting.extend(select_sources(condition_rows[condition], train_ids))
        fitting.extend(select_sources(controlled, train_ids))
        fitting.extend(normal_train)
        fit_sources = {row["source_id"] for row in fitting}
        leak = sorted(fit_sources & test_ids)
        if leak:
            raise RuntimeError(f"Outer-fold leakage: {leak}")
        models = fit_weighted(fitting, config["model"], config["weights"])
        joblib.dump({
            "models": models,
            "thresholds": thresholds,
            "skew_threshold": skew_threshold,
            "feature_names": feature_names,
            "train_source_ids": sorted(train_ids),
            "test_source_ids": sorted(test_ids),
        }, output / f"fold_{outer_index + 1}_model.joblib")

        condition_metrics = {}
        for condition, records in condition_rows.items():
            test_records = select_sources(records, test_ids)
            probabilities = predict(models, test_records)
            units, images = prediction_entries(
                test_records, probabilities, thresholds, skew_threshold, detector,
                target_class_by_source, outer_index + 1,
            )
            all_unit_entries[condition].extend(units)
            all_image_entries[condition].extend(images)
            condition_metrics[condition] = evaluate_outputs(units, images)

        normal_fold_ids = set(normal_test_folds[outer_index])
        normal_fold_records = select_sources(normal_test, normal_fold_ids)
        normal_probabilities = predict(models, normal_fold_records)
        _, normal_images = prediction_entries(
            normal_fold_records, normal_probabilities, thresholds, skew_threshold, detector,
            normal_target_class, outer_index + 1,
        )
        all_normal_image_entries.extend(normal_images)
        fold_reports.append({
            "outer_fold": outer_index + 1,
            "train_source_ids": sorted(train_ids),
            "test_source_ids": sorted(test_ids),
            "test_class_counts": dict(sorted(
                (name, sum(target_class_by_source[value] == name for value in test_ids)) for name in DEFECT_CLASSES
            )),
            "normal_test_source_count": len(normal_fold_ids),
            "training_unit_rows": {
                **{
                    f"real_{condition}": len(select_sources(condition_rows[condition], train_ids))
                    for condition in training_conditions
                },
                "controlled": len(select_sources(controlled, train_ids)),
                "normal": len(normal_train),
                "total": len(fitting),
            },
            "outer_source_overlap": leak,
            "thresholds": {**thresholds, "skew": skew_threshold},
            "inner_calibration": calibration,
            "test_metrics": condition_metrics,
        })
        print(
            f"outer fold {outer_index + 1}/{outer_count}: "
            + "; ".join(
                f"{name} image F1={condition_metrics[name]['defect_image_classification']['macro_f1']:.4f}"
                for name in condition_dirs
            )
        )

    defect_test_occurrences = defaultdict(int)
    for values in outer_folds:
        for source_id in values:
            defect_test_occurrences[source_id] += 1
    normal_test_occurrences = defaultdict(int)
    for values in normal_test_folds:
        for source_id in values:
            normal_test_occurrences[source_id] += 1
    if set(defect_test_occurrences.values()) != {1} or set(normal_test_occurrences.values()) != {1}:
        raise RuntimeError("A source was not evaluated exactly once")

    summaries = {
        condition: evaluate_outputs(all_unit_entries[condition], all_image_entries[condition])
        for condition in condition_dirs
    }
    clear_and_normal = all_image_entries["clear"] + all_normal_image_entries
    deployment_metric = multiclass_metrics(clear_and_normal, ALL_IMAGE_CLASSES)
    deployment = {
        "five_class_macro_f1": deployment_metric["macro_f1"],
        "five_class_accuracy": deployment_metric["accuracy"],
        "defect_alarm_recall": float(np.mean([
            row["predicted_class"] != "normal" for row in all_image_entries["clear"]
        ])),
        "normal_false_alarm_rate": float(np.mean([
            row["predicted_class"] != "normal" for row in all_normal_image_entries
        ])),
        "five_class_metrics": deployment_metric,
    }
    protocol_report = {
        "name": "source-grouped nested four-fold evaluation",
        "seed": seed,
        "outer_folds": outer_count,
        "inner_folds": inner_count,
        "training_conditions": training_conditions,
        "calibration_conditions": calibration_conditions,
        "minimum_inner_oof_defect_alarm_recall": float(protocol.get("min_defect_alarm_recall", 0.0)),
        "threshold_calibration": "one threshold set is selected from source-grouped inner OOF predictions of the configured conditions and normal-training sources only",
        "threshold_objective": str(protocol.get("threshold_objective", "macro_f1")),
        "condition_policy": "one outer-fold model and one jointly calibrated threshold set are frozen and reused for clear, direct blur, and MBDNet+APHNet test paths",
        "source_group_policy": "original, controlled variants, horizontal flips and all image-condition derivatives follow the original source; no outer-test derivative is fitted or calibrated",
        "outer_test_folds": outer_folds,
        "normal_split_counts": normal_counts,
        "normal_training_source_count": len(group_ids(normal_train)),
        "normal_test_source_count": len(group_ids(normal_test)),
        "normal_test_fold_counts": [len(values) for values in normal_test_folds],
        "controlled_variants_per_source": 1 + 3 * int(config["augmentation"]["variants_per_local_class"]),
        "controlled_generation_source_count": len(labels),
        "controlled_training_policy": "generated with fixed parameters; filtered by source in every inner and outer training split",
        "input_paths": {name: str(value) for name, value in paths.items()},
        "model": config["model"],
        "feature_set": feature_set,
        "feature_names": feature_names,
        "weights": config["weights"],
        "leakage_audit": {
            "all_outer_source_overlaps_empty": all(not report["outer_source_overlap"] for report in fold_reports),
            "all_inner_source_overlaps_empty": all(
                not inner["source_overlap"]
                for report in fold_reports for inner in report["inner_calibration"]["inner_folds"]
            ),
            "each_defect_source_tested_once": set(defect_test_occurrences.values()) == {1},
            "each_normal_holdout_source_tested_once": set(normal_test_occurrences.values()) == {1},
        },
        "limitations": [
            f"The {len(labels)} defect sources were previously inspected during method development; nested CV prevents computational leakage but does not turn them into an external dataset.",
            "When blur conditions are enabled, each one is a synthetic counterpart of its clear source.",
            "Normal false-alarm rate is measured on clear normal images and is not a paired deblurring-effect estimate.",
        ],
    }
    save_json(output / "protocol.json", protocol_report)
    save_json(output / "fold_reports.json", fold_reports)
    save_json(output / "summary.json", {
        "conditions": summaries,
        "clear_with_normal_holdout": deployment,
    })
    for condition in condition_dirs:
        save_json(output / f"{condition}_oof_image_predictions.json", all_image_entries[condition])
        save_json(output / f"{condition}_oof_unit_predictions.json", all_unit_entries[condition])
    save_json(output / "normal_holdout_image_predictions.json", all_normal_image_entries)
    write_csv(output, summaries, deployment)
    render_figures(output, summaries, all_image_entries)

    print("\nPooled outer-fold results")
    for condition in condition_dirs:
        current = summaries[condition]
        print(
            f"{condition}: image macro F1={current['defect_image_classification']['macro_f1']:.4f}; "
            f"image accuracy={current['defect_image_classification']['accuracy']:.4f}; "
            f"alarm recall={current['defect_alarm_recall']:.4f}; "
            f"mixed-scope macro F1={current['mixed_scope_macro_f1']:.4f}"
        )
    print(
        f"clear + normal holdout: five-class macro F1={deployment['five_class_macro_f1']:.4f}; "
        f"accuracy={deployment['five_class_accuracy']:.4f}; FAR={deployment['normal_false_alarm_rate']:.4f}"
    )
    print(f"results: {output}")


if __name__ == "__main__":
    main()
