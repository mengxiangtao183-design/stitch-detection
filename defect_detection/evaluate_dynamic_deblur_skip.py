"""Leakage-free threshold experiment for dynamic MBDNet-P+ bypass.

The clear-probability model is fitted only on the controlled SynBlur-v2
validation pairs.  Its threshold is then frozen and evaluated on the separate
48-image Quxian clear/blur pairs.  For blurred Quxian inputs, the downstream
dynamic result is assembled from already frozen APHNet predictions: a bypass
uses the direct-blur mask and a deblur route uses the MBDNet-P+ mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = (
    "log_laplacian_variance",
    "log_tenengrad_energy",
    "log_highpass_energy",
    "log_horizontal_gradient",
    "log_vertical_gradient",
    "log_horizontal_vertical_ratio",
    "log_luminance_std",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate threshold-based dynamic deblurring bypass")
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=Path("H:/C3Netqing/MISCFilter/manifests/c3net_controlled_linetrace_synblur_v2/val.tsv"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("H:/DATASET"))
    parser.add_argument(
        "--clear-dir", type=Path,
        default=Path("H:/DATASET/quexian/dataset_96_320/inference_original48"),
    )
    parser.add_argument(
        "--blur-dir", type=Path,
        default=Path("H:/DATASET/quexian/dataset_96_320/synblur_original48_k3to15"),
    )
    parser.add_argument(
        "--gt-mask-dir", type=Path,
        default=Path("H:/DATASET/quexian/dataset_96_320/masks"),
    )
    parser.add_argument(
        "--direct-mask-dir", type=Path,
        default=Path("H:/C3Netqing/results/prediction/runs/m6-afmguided-highres-mobilenetv4-fpn-full-150-seed42_20260821_235214/results/segmentation/binary"),
    )
    parser.add_argument(
        "--deblur-mask-dir", type=Path,
        default=Path("H:/C3Netqing/results/prediction/runs/m6-afmguided-highres-mobilenetv4-fpn-full-150-seed42_20260822_000545/results/segmentation/binary"),
    )
    parser.add_argument(
        "--restoration-summary", type=Path,
        default=Path("H:/C3Netqing/results/deblurring_ablation/quexian_synblur48_mbdnetpplus_restored/restoration_summary.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("H:/C3Netqing/results/deblurring_ablation/quexian_synblur48_dynamic_skip"),
    )
    parser.add_argument("--minimum-blur-recall", type=float, default=0.98)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_dataset_path(root: Path, relative: str) -> Path:
    return root / Path(relative.replace("/", "\\"))


def read_pairs(manifest: Path, root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            if len(row) < 2:
                raise ValueError(f"Invalid row in {manifest}: {row}")
            clear_path = resolve_dataset_path(root, row[0])
            blur_path = resolve_dataset_path(root, row[1])
            if not clear_path.is_file() or not blur_path.is_file():
                raise FileNotFoundError(f"Missing calibration pair: {clear_path} | {blur_path}")
            pairs.append((clear_path, blur_path))
    if not pairs:
        raise ValueError(f"No calibration pairs in {manifest}")
    return pairs


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def quality_features(path: Path, max_side: int = 320) -> np.ndarray:
    return quality_features_rgb(load_rgb(path), max_side=max_side)


def quality_features_rgb(rgb: np.ndarray, max_side: int = 320) -> np.ndarray:
    """Extract the frozen router features from an already decoded RGB image."""
    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    highpass = gray - cv2.GaussianBlur(gray, (0, 0), 1.2)
    eps = 1e-10
    gx_abs = float(np.mean(np.abs(gx)))
    gy_abs = float(np.mean(np.abs(gy)))
    raw = np.array(
        [
            float(np.var(laplacian)),
            float(np.mean(gx * gx + gy * gy)),
            float(np.mean(np.abs(highpass))),
            gx_abs,
            gy_abs,
            (gx_abs + eps) / (gy_abs + eps),
            float(np.std(gray)),
        ],
        dtype=np.float64,
    )
    return np.log(raw + eps)


def select_threshold(clear_scores: np.ndarray, blur_scores: np.ndarray, minimum_blur_recall: float) -> dict:
    all_scores = np.unique(np.concatenate((clear_scores, blur_scores)))
    candidates = np.concatenate(
        ([0.0], (all_scores[:-1] + all_scores[1:]) / 2.0, [1.0])
    )
    rows = []
    for threshold in candidates:
        clear_bypass = float(np.mean(clear_scores >= threshold))
        blur_recall = float(np.mean(blur_scores < threshold))
        balanced = 0.5 * (clear_bypass + blur_recall)
        rows.append((threshold, clear_bypass, blur_recall, balanced))
    feasible = [row for row in rows if row[2] >= minimum_blur_recall]
    pool = feasible if feasible else rows
    threshold, clear_bypass, blur_recall, balanced = max(
        pool, key=lambda row: (row[3], row[1], -row[0])
    )
    return {
        "threshold": float(threshold),
        "clear_bypass_rate": clear_bypass,
        "blur_recall": blur_recall,
        "balanced_accuracy": balanced,
        "minimum_blur_recall_constraint": minimum_blur_recall,
        "constraint_satisfied": bool(blur_recall >= minimum_blur_recall),
    }


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_sum, target_sum = int(prediction.sum()), int(target.sum())
    if pred_sum == 0 or target_sum == 0:
        return {"precision": 0.0, "recall": 0.0, "binary_f1": 0.0, "binary_iou": 0.0}
    tp = int(np.logical_and(prediction, target).sum())
    precision = tp / pred_sum
    recall = tp / target_sum
    union = int(np.logical_or(prediction, target).sum())
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "binary_f1": f1, "binary_iou": tp / union}


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def aggregate_segmentation(
    filenames: list[str], gt_dir: Path, direct_dir: Path, deblur_dir: Path,
    blur_scores: dict[str, float], threshold: float,
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, float]]] = {"direct": [], "always_deblur": [], "dynamic_skip": []}
    for filename in filenames:
        target = load_mask(gt_dir / filename)
        direct = load_mask(direct_dir / filename)
        deblur = load_mask(deblur_dir / filename)
        dynamic = direct if blur_scores[filename] >= threshold else deblur
        groups["direct"].append(binary_metrics(direct, target))
        groups["always_deblur"].append(binary_metrics(deblur, target))
        groups["dynamic_skip"].append(binary_metrics(dynamic, target))
    return {
        name: {
            metric: float(np.mean([row[metric] for row in values]))
            for metric in values[0]
        } | {"image_count": len(values)}
        for name, values in groups.items()
    }


def save_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_image_panel(ax, image: np.ndarray, title: str, color: str, panel_label: str | None = None) -> None:
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    # 移除所有子图边框
    for spine in ax.spines.values():
        spine.set_visible(False)
    # 子图标题 8pt 不加粗
    ax.set_title(title, fontsize=8, fontweight="normal", pad=3.0, linespacing=1.2)
    if panel_label is not None:
        # (a)(b)标签 8pt 不加粗
        ax.text(
            0.015, 0.985, panel_label, transform=ax.transAxes,
            va="top", ha="left", fontsize=8, fontweight="normal", color="#202124",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
        )


def render_figure(
    output_prefix: Path, filenames: list[str], clear_dir: Path, blur_dir: Path,
    clear_scores: dict[str, float], blur_scores: dict[str, float], threshold: float,
) -> None:
    mpl.rcParams.update(
        {
            # 全局字体 Times New Roman
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 8,
            # 关键：SVG文字转矢量路径，避免Word字体渲染错位
            "svg.fonttype": "path",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(2, 5, figsize=(7.15, 4.0), constrained_layout=False)
    # 布局：标题与第一排间距适中，两排中间间距收紧
    fig.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.88,
        bottom=0.06,
        wspace=0.085,
        hspace=0.18
    )

    bypass_color = "#3C78A8"
    deblur_color = "#D66A3A"
    letters = ("(a)", "(b)", "(c)", "(d)", "(e)")

    for column, filename in enumerate(filenames):
        b_score = blur_scores[filename]
        c_score = clear_scores[filename]
        b_route = "Bypass" if b_score >= threshold else "Deblur"
        c_route = "Bypass" if c_score >= threshold else "Deblur"

        add_image_panel(
            axes[0, column], load_rgb(blur_dir / filename),
            f"Blurred · {b_route}\nq = {b_score:.3f}",
            bypass_color if b_route == "Bypass" else deblur_color,
            panel_label=letters[column],
        )
        add_image_panel(
            axes[1, column], load_rgb(clear_dir / filename),
            f"Clear · {c_route}\nq = {c_score:.3f}",
            bypass_color if c_route == "Bypass" else deblur_color,
        )

    # 主标题 9pt 加粗，仅此处加粗
    fig.suptitle(
        "Image sharpness scoring for dynamic deblurring",
        y=0.97,
        fontsize=9,
        fontweight="bold"
    )

    fig.canvas.draw()
    skill_scripts = Path("C:/Users/user/.codex/skills/nature-figure/scripts")
    sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(output_prefix.with_suffix(".alignment.json")),
        overlay_svg=str(output_prefix.with_suffix(".alignment.svg")),
        axes=list(axes.flat),
        panel_ids=[f"top-{i + 1}" for i in range(5)] + [f"bottom-{i + 1}" for i in range(5)],
        row_groups=[
            [f"top-{i + 1}" for i in range(5)],
            [f"bottom-{i + 1}" for i in range(5)],
        ],
        column_groups=[[f"top-{i + 1}", f"bottom-{i + 1}"] for i in range(5)],
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
        strict=True,
    )

    # 仅输出 SVG 和 PDF，加内边距避免边缘被裁
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)



def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; pass --overwrite to replace it")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    calibration_pairs = read_pairs(args.calibration_manifest, args.dataset_root)
    x_values: list[np.ndarray] = []
    y_values: list[int] = []
    calibration_rows: list[dict] = []
    for pair_index, (clear_path, blur_path) in enumerate(calibration_pairs):
        for condition, path, label in (("clear", clear_path, 1), ("blurred", blur_path, 0)):
            features = quality_features(path)
            x_values.append(features)
            y_values.append(label)
            calibration_rows.append(
                {"pair_index": pair_index, "condition": condition, "path": str(path), "label_clear": label}
                | {name: float(value) for name, value in zip(FEATURE_NAMES, features)}
            )
    x_array = np.vstack(x_values)
    y_array = np.asarray(y_values, dtype=np.int64)
    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)),
        ]
    )
    classifier.fit(x_array, y_array)
    calibration_scores = classifier.predict_proba(x_array)[:, 1]
    for row, score in zip(calibration_rows, calibration_scores):
        row["clear_probability"] = float(score)
    clear_calibration = calibration_scores[y_array == 1]
    blur_calibration = calibration_scores[y_array == 0]
    threshold_result = select_threshold(clear_calibration, blur_calibration, args.minimum_blur_recall)
    threshold = threshold_result["threshold"]

    clear_files = sorted(path.name for path in args.clear_dir.glob("*.png"))
    blur_files = sorted(path.name for path in args.blur_dir.glob("*.png"))
    if clear_files != blur_files or len(clear_files) != 48:
        raise ValueError("Expected 48 one-to-one clear/blur PNG pairs")
    test_rows: list[dict] = []
    test_scores: dict[tuple[str, str], float] = {}
    feature_times = []
    for filename in clear_files:
        for condition, directory, label in (("clear", args.clear_dir, 1), ("blurred", args.blur_dir, 0)):
            started = time.perf_counter()
            features = quality_features(directory / filename)
            score = float(classifier.predict_proba(features[None, :])[0, 1])
            feature_times.append(1000.0 * (time.perf_counter() - started))
            route = "bypass" if score >= threshold else "deblur"
            test_scores[(condition, filename)] = score
            test_rows.append(
                {
                    "filename": filename,
                    "condition": condition,
                    "label_clear": label,
                    "clear_probability": score,
                    "threshold": threshold,
                    "route": route,
                    "correct_route": int((label == 1 and route == "bypass") or (label == 0 and route == "deblur")),
                }
                | {name: float(value) for name, value in zip(FEATURE_NAMES, features)}
            )

    clear_scores = np.array([test_scores[("clear", name)] for name in clear_files])
    blur_scores = np.array([test_scores[("blurred", name)] for name in clear_files])
    clear_bypass = float(np.mean(clear_scores >= threshold))
    blur_recall = float(np.mean(blur_scores < threshold))
    all_correct = [int(row["correct_route"]) for row in test_rows]
    blur_score_map = {name: test_scores[("blurred", name)] for name in clear_files}
    clear_score_map = {name: test_scores[("clear", name)] for name in clear_files}

    segmentation = aggregate_segmentation(
        clear_files, args.gt_mask_dir, args.direct_mask_dir, args.deblur_mask_dir,
        blur_score_map, threshold,
    )
    restoration_summary = json.loads(args.restoration_summary.read_text(encoding="utf-8"))
    mbd_ms = float(restoration_summary["metrics_mean"]["inference_ms"])
    mixed_bypass_count = int(sum(row["route"] == "bypass" for row in test_rows))
    blur_bypass_count = int(np.sum(blur_scores >= threshold))
    estimated_mixed_mbd_ms = mbd_ms * (1.0 - mixed_bypass_count / len(test_rows))

    scaler = classifier.named_steps["scaler"]
    logistic = classifier.named_steps["logistic"]
    summary = {
        "method": "No-reference clear-probability logistic router on seven normalized sharpness features",
        "score_definition": "q = sigmoid(intercept + sum_j coefficient_j * standardized_feature_j)",
        "decision_rule": "q >= threshold: bypass MBDNet-P+; q < threshold: run MBDNet-P+",
        "calibration": {
            "source": "Controlled-LineTrace-SynBlur-v2 validation pairs only",
            "pair_count": len(calibration_pairs),
            "image_count": len(calibration_rows),
            **threshold_result,
            "feature_names": FEATURE_NAMES,
            "feature_mean": scaler.mean_.tolist(),
            "feature_scale": scaler.scale_.tolist(),
            "logistic_coefficients": logistic.coef_[0].tolist(),
            "logistic_intercept": float(logistic.intercept_[0]),
        },
        "frozen_test": {
            "dataset": "Quxian 48 paired clear / synthetic-blur images",
            "pair_count": 48,
            "routing_accuracy": float(np.mean(all_correct)),
            "balanced_accuracy": 0.5 * (clear_bypass + blur_recall),
            "clear_bypass_rate": clear_bypass,
            "blur_recall": blur_recall,
            "mixed_workload_bypass_count": mixed_bypass_count,
            "mixed_workload_image_count": len(test_rows),
            "mixed_workload_bypass_rate": mixed_bypass_count / len(test_rows),
            "blurred_input_bypass_count": blur_bypass_count,
            "mean_router_cpu_ms": float(np.mean(feature_times)),
            "reference_mbdnet_gpu_ms": mbd_ms,
            "estimated_mixed_workload_mbd_ms_excluding_router": estimated_mixed_mbd_ms,
        },
        "blurred_input_downstream_segmentation": segmentation,
        "leakage_control": "Threshold and score coefficients are frozen before Quxian evaluation; RX is not used.",
    }

    save_rows(args.output_dir / "calibration_scores.csv", calibration_rows)
    save_rows(args.output_dir / "frozen_test_routes.csv", test_rows)
    (args.output_dir / "dynamic_skip_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    deblur_indices = np.flatnonzero(blur_scores < threshold)
    bypass_indices = np.flatnonzero(blur_scores >= threshold)
    if len(deblur_indices) < 3 or len(bypass_indices) < 2:
        raise RuntimeError(
            "The frozen test does not contain enough samples to display three deblur and two bypass cases"
        )
    deblur_order = deblur_indices[np.argsort(blur_scores[deblur_indices])]
    bypass_order = bypass_indices[np.argsort(blur_scores[bypass_indices])]
    selected_indices = [
        int(deblur_order[round(q * (len(deblur_order) - 1))]) for q in (0.15, 0.50, 0.85)
    ] + [
        int(bypass_order[round(q * (len(bypass_order) - 1))]) for q in (0.33, 0.67)
    ]
    selected_names = [clear_files[index] for index in selected_indices]
    render_figure(
        args.output_dir / "dynamic_skip_scoring_examples",
        selected_names, args.clear_dir, args.blur_dir,
        clear_score_map, blur_score_map, threshold,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
