"""Render traceable publication assets from a completed TSR nested-CV run.

This is intentionally a reporting script, not an additional model or ablation.
It reads only pooled outer-fold predictions and retains every source image.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SKILL_SCRIPTS = Path(r"C:/Users/user/.codex/skills/nature-figure/scripts")
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


CLASS_ORDER = ("skip", "break", "skew", "loose")
DISPLAY = {"skip": "跳线", "break": "断线", "skew": "线迹歪斜", "loose": "松线", "normal": "正常"}
COLORS = {"skip": "#4C78A8", "break": "#D46A6A", "skew": "#59A14F", "loose": "#B07AA1"}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
    "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.linewidth": 0.8, "axes.spines.right": False, "axes.spines.top": False,
})


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_tables(output: Path, summary: dict) -> None:
    current = summary["conditions"]["clear"]
    image = current["defect_image_classification"]
    with (output / "tsr_main_results_112_oof.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["evaluation_scope", "sample_count", "macro_f1", "accuracy", "alarm_recall", "unit_exact_accuracy"])
        writer.writerow(["112 defect-source outer-fold predictions", image["count"], image["macro_f1"], image["accuracy"], current["defect_alarm_recall"], current["local_unit_exact_accuracy"]])
    with (output / "tsr_per_class_f1_112_oof.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "support", "precision", "recall", "f1"])
        for name in CLASS_ORDER:
            value = image["per_class"][name]
            writer.writerow([name, value["positive"], value["precision"], value["recall"], value["f1"]])
    labels = image["confusion_labels"]
    with (output / "tsr_confusion_matrix_112_oof.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["ground_truth\\predicted", *labels])
        for name, row in zip(labels, image["confusion_matrix"]):
            writer.writerow([name, *row])


def save_single(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(output / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_f1(output: Path, summary: dict) -> None:
    metrics = summary["conditions"]["clear"]["defect_image_classification"]["per_class"]
    values = [metrics[name]["f1"] * 100 for name in CLASS_ORDER]
    fig, ax = plt.subplots(figsize=(88 / 25.4, 54 / 25.4), dpi=600)
    bars = ax.barh(range(4), values, color=[COLORS[x] for x in CLASS_ORDER], height=0.58)
    ax.set_yticks(range(4), [DISPLAY[x] for x in CLASS_ORDER])
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("F1 / %")
    ax.set_xticks(np.arange(0, 101, 20)); ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    for bar, value in zip(bars, values):
        ax.text(min(value + 1.6, 98), bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=7)
    ax.text(0.01, 1.05, "a", transform=ax.transAxes, fontweight="bold", fontsize=8)
    fig.tight_layout(pad=0.6)
    save_single(fig, output, "fig_tsr_per_class_f1_112_oof")


def plot_confusion(output: Path, summary: dict) -> None:
    item = summary["conditions"]["clear"]["defect_image_classification"]
    labels = item["confusion_labels"]
    matrix = np.asarray(item["confusion_matrix"], dtype=int)
    fig, ax = plt.subplots(figsize=(89 / 25.4, 75 / 25.4), dpi=600)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
    ax.set_xticks(range(len(labels)), [DISPLAY[v] for v in labels], rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(labels)), [DISPLAY[v] for v in labels])
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > matrix.max() * 0.52 else "#1F1F1F")
    ax.text(-0.13, 1.05, "b", transform=ax.transAxes, fontweight="bold", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="图像数")
    fig.tight_layout(pad=0.7)
    save_single(fig, output, "fig_tsr_confusion_matrix_112_oof")


def read_mask(path: Path) -> np.ndarray:
    if path.with_suffix(".npy").exists():
        return np.load(path.with_suffix(".npy"))
    return np.asarray(Image.open(path.with_suffix(".png")).convert("L"), dtype=np.float32) / 255.0


def plot_errors(output: Path, predictions: list[dict], image_dir: Path, gt_dir: Path, probability_dir: Path) -> None:
    errors = [row for row in predictions if row["target_class"] != row["predicted_class"]]
    transitions = Counter((row["target_class"], row["predicted_class"]) for row in errors)
    selected = []
    for transition, _ in transitions.most_common(4):
        candidates = [r for r in errors if (r["target_class"], r["predicted_class"]) == transition]
        # Select the most decisive wrong prediction, not a hand-picked favourable example.
        selected.append(max(candidates, key=lambda r: max(r["score_over_threshold"].values())))
    if not selected:
        selected = sorted(predictions, key=lambda r: sorted(r["score_over_threshold"].values(), reverse=True)[0] - sorted(r["score_over_threshold"].values(), reverse=True)[1])[:4]
    n = len(selected)
    fig, axes = plt.subplots(3, n, figsize=(n * 33 / 25.4, 88 / 25.4), dpi=600, squeeze=False)
    for col, row in enumerate(selected):
        sid = row["source_id"]
        rgb = np.asarray(Image.open(image_dir / f"{sid}.png").convert("RGB"))
        gt = read_mask(gt_dir / sid)
        pred = read_mask(probability_dir / sid)
        for ax in axes[:, col]: ax.set_axis_off()
        axes[0, col].imshow(rgb)
        axes[1, col].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[2, col].imshow(pred, cmap="magma", vmin=0, vmax=1)
        axes[0, col].set_title(f"{sid}  真:{DISPLAY[row['target_class']]}\n预测:{DISPLAY[row['predicted_class']]}", fontsize=6.4, pad=3)
    axes[0, 0].set_ylabel("清晰图像", fontsize=7); axes[1, 0].set_ylabel("真实掩码", fontsize=7); axes[2, 0].set_ylabel("ACHNet概率", fontsize=7)
    fig.text(0.01, 0.99, "c", fontweight="bold", fontsize=8, va="top")
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.03, top=0.84, wspace=0.08, hspace=0.07)
    require_matplotlib_panel_alignment(fig, json_out=output / "fig_tsr_typical_misclassifications_112_oof.alignment-layout.json",
        overlay_svg=output / "fig_tsr_typical_misclassifications_112_oof.alignment.svg", tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, strict=True)
    save_single(fig, output, "fig_tsr_typical_misclassifications_112_oof")
    with (output / "typical_misclassification_cases_112_oof.json").open("w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--probability-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = load(args.result_dir / "summary.json")
    predictions = load(args.result_dir / "clear_oof_image_predictions.json")
    write_tables(args.result_dir, summary)
    plot_f1(args.result_dir, summary)
    plot_confusion(args.result_dir, summary)
    plot_errors(args.result_dir, predictions, args.image_dir, args.gt_dir, args.probability_dir)
    print(args.result_dir)


if __name__ == "__main__":
    main()
