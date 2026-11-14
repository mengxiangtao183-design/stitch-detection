"""Paired, unsmoothed restoration curves from frozen OOF outputs.

The ROC panels use threshold-normalized OOF decision scores, not calibrated
probabilities. Dice panels use the same ground-truth binary mask in all paths.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score

from render_btsr_balanced_manuscript import RESULT, DATA, MASK

sys.path.insert(0, str(Path(r"C:\Users\user\.codex\skills\nature-figure\scripts")))
from audit_panel_alignment import require_matplotlib_panel_alignment

OUT = Path(r"H:\C3Netqing\docs\figures\btsr_balanced_112\deblur_curves_tnr8")
CONDITIONS = ("clear", "blur_direct", "deblur_then_aphnet")
CLASSES = ("skip", "break", "skew", "loose")
NAMES = ("Skipped stitch", "Broken stitch", "Skewed stitch", "Loose stitch")
STYLE = {
    "clear": ("Clear", "#7E858B", "--", "o"),
    "blur_direct": ("Blurred", "#C77C3D", "-", "^"),
    "deblur_then_aphnet": ("Restored", "#286C92", "-", "s"),
}
mpl.rcParams.update({
    "font.family": "Times New Roman", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.65, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, stem):
    require_matplotlib_panel_alignment(
        fig, json_out=OUT / f"{stem}.alignment.json",
        overlay_svg=OUT / f"{stem}.alignment.svg", strict=True,
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
    )
    for txt in fig.findobj(match=mpl.text.Text):
        if txt.get_text().strip():
            assert txt.get_fontsize() == 8
    fig.savefig(OUT / f"{stem}.svg")
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=600)
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def axes_grid(ylabel, xlabel):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.65))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.115,
                        top=0.865, wspace=0.31, hspace=0.52)
    for i, ax in enumerate(axes.flat):
        ax.set_title(f"{'abcd'[i]}  {NAMES[i]}", loc="left", pad=7, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", color="#E6E9EC", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(width=0.6, length=3)
    return fig, axes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = {c: json.loads((RESULT / f"{c}_oof_image_predictions.json").read_text())
               for c in CONDITIONS}
    by_id = {c: {r["source_id"]: r for r in rows} for c, rows in records.items()}
    ids = sorted(by_id["clear"])
    assert len(ids) == 112 and all(set(ids) == set(by_id[c]) for c in CONDITIONS)
    assert all(by_id[c][s]["target_class"] == by_id["clear"][s]["target_class"]
               for c in CONDITIONS for s in ids)
    assert all(sum(by_id["clear"][s]["target_class"] == k for s in ids) == 28 for k in CLASSES)

    fig, axes = axes_grid("True-positive rate", "False-positive rate")
    roc_rows, auc_rows = [], []
    for name, ax in zip(CLASSES, axes.flat):
        truth = np.array([by_id["clear"][s]["target_class"] == name for s in ids])
        ax.plot([0, 1], [0, 1], color="#D5DADD", linestyle=":", linewidth=0.7)
        for condition in CONDITIONS:
            label, color, linestyle, _ = STYLE[condition]
            score = np.array([by_id[condition][s]["score_over_threshold"][name] for s in ids])
            assert np.all(np.isfinite(score))
            fpr, tpr, _ = roc_curve(truth, score, drop_intermediate=False)
            ax.step(fpr, tpr, where="post", color=color, linestyle=linestyle,
                    linewidth=1.3 if condition == "deblur_then_aphnet" else 1.0, label=label)
            auc_rows.append({"class": name, "condition": condition,
                             "auc": roc_auc_score(truth, score), "positive": 28, "negative": 84})
            roc_rows.extend({"class": name, "condition": condition, "fpr": x, "tpr": y}
                            for x, y in zip(fpr, tpr))
        ax.set_xlim(-0.015, 1.015); ax.set_ylim(-0.015, 1.025)
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.99))
    save(fig, "fig12_deblur_four_class_oof_roc")
    write_csv("roc_curve_source_data.csv", roc_rows)
    write_csv("roc_auc_summary.csv", auc_rows)

    dice_rows = []
    for sid in ids:
        gt = np.asarray(Image.open(DATA / "masks" / f"{sid}.png").convert("L")) > 127
        assert gt.shape == (320, 320) and gt.any()
        for condition in CONDITIONS:
            predicted = np.asarray(Image.open(MASK[condition] / f"{sid}.png").convert("L")) > 127
            assert predicted.shape == gt.shape
            dice = 2 * np.logical_and(gt, predicted).sum() / (gt.sum() + predicted.sum())
            dice_rows.append({"source_id": sid, "class": by_id["clear"][sid]["target_class"],
                              "fold": by_id[condition][sid]["fold"], "condition": condition,
                              "dice_percent": float(dice * 100)})
    fig, axes = axes_grid("Binary Dice (%)", "Within-class image index")
    low = max(0, np.floor(min(r["dice_percent"] for r in dice_rows) / 10) * 10 - 10)
    for name, ax in zip(CLASSES, axes.flat):
        for condition in CONDITIONS:
            label, color, linestyle, marker = STYLE[condition]
            rows = [r for r in dice_rows if r["class"] == name and r["condition"] == condition]
            ax.plot(np.arange(1, 29), [r["dice_percent"] for r in rows], label=label,
                    color=color, linestyle=linestyle, marker=marker, markersize=2.5,
                    markeredgewidth=0.4, linewidth=0.95)
        ax.set_xlim(0.5, 28.5); ax.set_ylim(low, 100)
        ax.set_xticks([1, 7, 14, 21, 28])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.99))
    save(fig, "fig13_deblur_paired_binary_dice")
    write_csv("paired_dice_source_data.csv", dice_rows)
    note = {
        "source_images": 112, "images_per_defect_class": 28,
        "roc": "One-versus-rest empirical curves from four-fold OOF threshold-normalized decision scores. Not calibrated probabilities; no threshold selection on test data. Curves are diagnostic and do not replace fixed-decision accuracy or alarm recall.",
        "dice": "All 112 images, source-ID order within each class. Same GT mask for clear, blurred and restored paths. Connecting lines aid paired comparison, not temporal continuity. No smoothing, interpolation or selected success examples.",
        "uncertainty": "Individual paired observations / empirical ROC, not a mean across independent runs. No uncertainty interval or significance claim is shown.",
        "font": "Times New Roman 8 pt; requested serif font intentionally overrides the source preflight sans-serif recommendation.",
        "manuscript": "No document modified.",
        "dice_mean": {c: float(np.mean([r["dice_percent"] for r in dice_rows if r["condition"] == c])) for c in CONDITIONS},
        "panel_audit": "All four classes retained; common scales and alignment audited. Dashed gray clear reference, orange blurred control and blue restored path are consistent across figures.",
    }
    (OUT / "figure_notes.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "auc": auc_rows, "dice_mean": note["dice_mean"]}, indent=2))


if __name__ == "__main__":
    main()
