"""Paper figures for the selected binary-topology self-reference unit.
All quantitative labels come from the 112-source nested four-fold OOF run.
Qualitative panels show actual RGB images and ACHNet binary PNG masks; only
the crop/overlay/annotation are presentation operations.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

FIGURE_SCRIPTS = Path(r"C:\Users\user\.codex\skills\nature-figure\scripts")
sys.path.insert(0, str(FIGURE_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

ROOT = Path(r"H:\C3Netqing")
DATA = Path(r"H:\DATASET\quexian\dataset_112_320")
RESULT = ROOT / "results/defect_detection/tsr_defect_nested4fold_joint_clear_deblur_112_binary_topology_v2_balanced"
OUT = ROOT / "docs/figures/btsr_balanced_112/english_tnr8"

RGB = {
    "clear": DATA / "images",
    "blur_direct": DATA / "synblur_k3to15_seed42",
    "deblur_then_aphnet": ROOT / "results/deblurring_ablation/quexian_synblur112_mbdnet_restored",
}
MASK = {
    "clear": ROOT / "results/prediction/runs/m6-afmguided-highres-mobilenetv4-fpn-full-150-seed42_20260915_152506/results/segmentation/binary",
    "blur_direct": ROOT / "results/prediction/runs/m6-afmguided-highres-mobilenetv4-fpn-full-150-seed42_20260915_154128/results/segmentation/binary",
    "deblur_then_aphnet": ROOT / "results/prediction/runs/m6-afmguided-highres-mobilenetv4-fpn-full-150-seed42_20260915_154140/results/segmentation/binary",
}

CLASSES = ("skip", "break", "skew", "loose")
DISPLAY = {"skip": "Skip", "break": "Break", "skew": "Skew", "loose": "Loose", "normal": "Normal"}
INK, BLUE, RED, GRAY = "#263746", "#326D8C", "#B65A51", "#AAB6BF"

# Keep all figure text editable and in Times New Roman at exactly 8 pt.
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "image.composite_image": True,
    "image.interpolation": "nearest",
})



def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rgb(condition: str, sid: str) -> np.ndarray:
    return np.asarray(Image.open(RGB[condition] / f"{sid}.png").convert("RGB"))


def mask(condition: str, sid: str) -> np.ndarray:
    return (np.asarray(Image.open(MASK[condition] / f"{sid}.png").convert("L")) > 127).astype(np.uint8) * 255



def save(fig: plt.Figure, stem: str, *, multipanel: bool = False) -> None:
    # Reject accidental font or size drift before exporting any figure.
    for label in fig.findobj(match=mpl.text.Text):
        if label.get_text().strip() and (label.get_fontsize() != 8 or
                                         label.get_fontfamily() != ["Times New Roman"]):
            raise AssertionError(f"Inconsistent figure text: {label.get_text()!r}")
    if multipanel:
        require_matplotlib_panel_alignment(
            fig, json_out=OUT / f"{stem}.alignment.json",
            overlay_svg=OUT / f"{stem}.alignment.svg", tolerance_pt=1.5,
            gutter_tolerance_pt=1.5, strict=True,
        )
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)




def node(ax, x: float, y: float, w: float, h: float, title: str, detail: str = "") -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#F7FAFB", edgecolor=GRAY, linewidth=1.25))
    ax.text(x + w / 2, y + h * (0.60 if detail else 0.50), title,
            ha="center", va="center", color=INK, fontsize=8, fontweight="bold")
    if detail:
        ax.text(x + w / 2, y + h * 0.29, detail, ha="center", va="center", color="#526370", fontsize=8)


def arrow(ax, a, b, *, dashed=False) -> None:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11,
                                linewidth=1.15, linestyle="--" if dashed else "-", color=BLUE))


def overview() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 1.8))
    ax.set_xlim(0, 10.8); ax.set_ylim(0, 2.7); ax.axis("off")
    node(ax, 0.16, 0.90, 1.30, 0.78, "Input", "RGB image")
    node(ax, 1.82, 0.90, 1.35, 0.78, "Quality gate", "Routing")
    node(ax, 3.53, 1.55, 1.48, 0.78, "MBDNet-P+", "Deblurring")
    node(ax, 5.43, 0.90, 1.37, 0.78, "APHNet", "Segmentation")
    node(ax, 7.18, 0.90, 1.35, 0.78, "Binary mask", "320 × 320")
    node(ax, 8.91, 0.90, 1.70, 0.78, "BTSR", "Defect decision")
    arrow(ax, (1.46, 1.29), (1.82, 1.29))
    arrow(ax, (3.17, 1.29), (3.53, 1.94))
    arrow(ax, (5.01, 1.94), (5.43, 1.29))
    arrow(ax, (3.17, 1.18), (5.43, 1.18), dashed=True)
    arrow(ax, (6.80, 1.29), (7.18, 1.29))
    arrow(ax, (8.53, 1.29), (8.91, 1.29))
    ax.text(3.73, 2.48, "Restore", fontsize=8, color=BLUE)
    ax.text(4.20, 0.90, "Bypass", fontsize=8, color=BLUE, ha="center")
    ax.text(0.18, 0.26, "BTSR uses binary-mask trajectories and topology; RGB images are shown only for visualization.",
            fontsize=8, color="#526370")
    save(fig, "fig1_btsr_three_stage_overview")


def labels() -> dict[str, dict]:
    out = {}
    with (DATA / "defect_review/defect_labels_completed.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["augmentation"] != "original":
                continue
            active = [name for name in CLASSES if int(row[name])]
            if active:
                out[row["sample_id"]] = {"class": active[0], "x0": int(float(row["x0"])), "x1": int(float(row["x1"]))}
    return out


def predictions() -> dict[str, dict[str, dict]]:
    return {
        condition: {row["source_id"]: row for row in load(RESULT / f"{condition}_oof_image_predictions.json")}
        for condition in RGB
    }


def select_examples(pred: dict[str, dict[str, dict]]) -> tuple[dict[str, str], dict[str, str]]:
    stable, recovered = {}, {}
    for name in CLASSES:
        ids = sorted(sid for sid, row in pred["clear"].items() if row["target_class"] == name)
        correct = [sid for sid in ids if all(pred[condition][sid]["predicted_class"] == name for condition in RGB)]
        if not correct:
            raise RuntimeError(f"No stable {name} example")
        stable[name] = correct[len(correct) // 2]
        candidates = [sid for sid in ids if pred["blur_direct"][sid]["predicted_class"] != name
                      and pred["deblur_then_aphnet"][sid]["predicted_class"] == name]
        if not candidates:
            raise RuntimeError(f"No blur-error/restored-correct {name} example")
        gains = {
            sid: pred["deblur_then_aphnet"][sid]["score_over_threshold"][name]
            - pred["blur_direct"][sid]["score_over_threshold"][name]
            for sid in candidates
        }
        median = float(np.median(list(gains.values())))
        recovered[name] = min(candidates, key=lambda sid: (abs(gains[sid] - median), sid))
    (OUT / "figure_example_selection.json").write_text(
        json.dumps({"stable": stable, "blur_error_restored_correct": recovered}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stable, recovered


def crop_range(masks: list[np.ndarray]) -> tuple[int, int]:
    merged = np.logical_or.reduce(masks)
    yy = np.flatnonzero(merged.any(axis=1))
    if not len(yy):
        return 80, 240
    middle = int(np.median(yy))
    return max(0, min(160, middle - 80)), max(160, min(320, middle + 80))


def evidence(stable: dict[str, str], targets: dict[str, dict]) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 3.15))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.92, bottom=0.06, wspace=0.055, hspace=0.13)
    
    for col, name in enumerate(CLASSES):
        sid = stable[name]
        binary = mask("clear", sid)
        y0, y1 = crop_range([binary])
        label = targets[sid]
        
        axes[0, col].imshow(rgb("clear", sid)[y0:y1], aspect="auto")
        axes[1, col].imshow(binary[y0:y1], cmap="gray", vmin=0, vmax=1, aspect="auto")
        
        column_support = binary.any(axis=0).astype(float)
        center = np.full(320, np.nan)
        for x in np.flatnonzero(binary.any(axis=0)):
            center[x] = np.mean(np.flatnonzero(binary[:, x]))
        
        axes[2, col].fill_between(np.arange(320), 0, column_support, color="#D7E6ED", linewidth=0)
        axes[2, col].plot(np.arange(320), center / 320, color=BLUE, linewidth=1.25)
        axes[2, col].set_ylim(0, 1)
        axes[2, col].set_xlim(0, 320)
        
        if name != "skew":
            for ax in axes[:, col]:
                ax.axvspan(label["x0"], label["x1"], edgecolor=RED, facecolor="none", linewidth=1.1)
        
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        axes[0, col].set_title(DISPLAY[name], fontsize=8, pad=4, fontweight="bold")

    for row, title in enumerate(("Clear RGB", "Binary mask", "Trajectory")):
        axes[row, 0].set_ylabel(title, fontsize=8, rotation=90, rotation_mode="anchor", labelpad=9)
    save(fig, "fig8_btsr_four_class_binary_evidence", multipanel=True)


def confusion() -> None:
    item = load(RESULT / "summary.json")["conditions"]["clear"]["defect_image_classification"]
    matrix = np.asarray(item["confusion_matrix"], dtype=int)[:4, :5]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.20)
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=28)
    for i in range(4):
        for j in range(5):
            value = int(matrix[i, j])
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8,
                    color="white" if value >= 15 else INK)
    ax.set_xticks(range(5), [DISPLAY[x] for x in item["confusion_labels"][:5]])
    ax.set_yticks(range(4), [DISPLAY[x] for x in CLASSES])
    ax.set_xlabel("Predicted class", fontsize=8); ax.set_ylabel("True class", fontsize=8)
    ax.tick_params(length=0)
    save(fig, "fig10_btsr_clear_oof_confusion")


def overlay(ax, condition: str, sid: str, yrange: tuple[int, int], xlim: tuple[int, int]) -> None:
    y0, y1 = yrange
    arr = rgb(condition, sid)[y0:y1]
    binary = mask(condition, sid)[y0:y1]
    ax.imshow(arr, extent=(0, 320, y1, y0), aspect="auto")
    tint = np.zeros((*binary.shape, 4), dtype=np.float32)
    tint[..., :3] = (0.98, 0.79, 0.22)
    tint[..., 3] = (binary.astype(np.float32) / 255.0) * 0.64
    ax.imshow(tint, extent=(0, 320, y1, y0), aspect="auto")
    ax.set_xlim(*xlim); ax.set_ylim(y1, y0)
    ax.set_xticks([]); ax.set_yticks([])


def paired(recovered: dict[str, str], pred: dict[str, dict[str, dict]], targets: dict[str, dict]) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(7.2, 5.25))
    fig.subplots_adjust(left=0.06, right=0.995, top=0.88, bottom=0.06, wspace=0.045, hspace=0.15)
    heads = ("Clear", "Blurred", "MBDNet-P+ restored")
    for col, head in enumerate(heads):
        axes[0, col].set_title(head, fontsize=8, pad=5)
    for row, name in enumerate(CLASSES):
        sid = recovered[name]
        conditions = ("clear", "blur_direct", "deblur_then_aphnet")
        yrange = crop_range([mask("clear", sid)])
        xlim = (0, 320)
        for col, condition in enumerate(conditions):
            ax = axes[row, col]
            overlay(ax, condition, sid, yrange, xlim)
            predicted = pred[condition][sid]["predicted_class"]
            ax.text(0.012, 0.96, DISPLAY[predicted], transform=ax.transAxes,
                    ha="left", va="top", color="#FFFFFF", fontsize=8,
                    bbox={"facecolor": BLUE if predicted == name else RED, "edgecolor": "none", "pad": 1.7})
            if name != "skew":
                focus = targets[sid]
                ax.add_patch(Rectangle((focus["x0"], yrange[0] + 1), focus["x1"] - focus["x0"],
                                       yrange[1] - yrange[0] - 2, facecolor="none", edgecolor="#EAEAEA", linewidth=1.0))
        axes[row, 0].set_ylabel(f"{DISPLAY[name]}  {sid}", fontsize=8, rotation=90,
                                rotation_mode="anchor", labelpad=10)
    save(fig, "fig11_btsr_paired_blur_restoration", multipanel=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pred = predictions()
    target = labels()
    stable, recovered = select_examples(pred)
    overview()
    evidence(stable, target)
    confusion()
    paired(recovered, pred, target)
    print(OUT)


if __name__ == "__main__":
    main()
