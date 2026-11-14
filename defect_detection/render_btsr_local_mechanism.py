"""Render a data-grounded BTSR local-decision illustration for the manuscript.

Figure claim: an internal short remnant is interpreted relative to its two
same-image neighbours, then combined with the full 41-D representation for
classification. This single OOF example illustrates mechanism, not accuracy.

Evidence roles: (a) detector-generated six-unit localization, (b) actual
RGB/APHNet-mask appearance, (c) measured foreground-run lengths and decision.
The clear RGB and binary mask are cropped only; no contrast or morphology edits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, r"C:\Users\user\.codex\skills\nature-figure\scripts")

from audit_panel_alignment import require_matplotlib_panel_alignment
from defect_detection.binary_topology_features import support_runs
from defect_detection.train_trajectory_feature_classifier import load_detector, read_mask


SOURCE_ID = "042"
MASK = PROJECT / (
    "results/prediction/runs/"
    "m6-afmguided-highres-mobilenetv4-fpn-full-150-seed42_20260915_152506/"
    "results/segmentation/binary/042.png"
)
RGB = Path(r"H:\DATASET\quexian\dataset_112_320\images\042.png")
CONFIG = PROJECT / "configs/defect_head/tsr_defect_rules.yaml"
OOF = PROJECT / (
    "results/defect_detection/"
    "tsr_defect_nested4fold_joint_clear_deblur_112_binary_topology_v2_balanced/"
    "clear_oof_image_predictions.json"
)
UNIT_OOF = OOF.with_name("clear_oof_unit_predictions.json")
OUT = PROJECT / "docs/figures/btsr_balanced_112"
STEM = "fig9_btsr_local_break_mechanism"

INK = "#253A49"
BLUE = "#41718B"
RED = "#B5544B"
PALE = "#DDE8ED"
GRAY = "#9AAAB5"
YELLOW = (0.96, 0.75, 0.20)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
    "font.size": 9.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.75,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def xmap(pixel: float, start: float, stop: float, left: float, right: float) -> float:
    return left + (pixel - start) / (stop - start) * (right - left)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    binary = np.asarray(Image.open(MASK).convert("L")) > 127
    original = np.asarray(Image.open(RGB).convert("RGB"))
    detector = load_detector(CONFIG)
    result = detector.detect(read_mask(MASK))
    units = result["units"]
    bounds = [(int(u["evidence"]["x0"]), int(u["evidence"]["x1"])) for u in units]
    assert len(bounds) == 6 and bounds[0][0] == 0 and bounds[-1][1] == 320

    image_prediction = next(
        row for row in json.loads(OOF.read_text(encoding="utf-8"))
        if row["source_id"] == SOURCE_ID
    )
    unit_prediction = next(
        row for row in json.loads(UNIT_OOF.read_text(encoding="utf-8"))
        if row["source_id"] == SOURCE_ID and row["unit_id"] == 2
    )
    assert image_prediction["target_class"] == "break"
    assert image_prediction["predicted_class"] == "break"
    assert unit_prediction["prediction"]["break"] is True

    spans: list[int] = []
    runs_global: list[list[tuple[int, int]]] = []
    for x0, x1 in bounds:
        runs = [(int(a + x0), int(b + x0)) for a, b in support_runs(binary[:, x0:x1].any(axis=0))]
        runs_global.append(runs)
        spans.append(max((b - a for a, b in runs), default=0))
    reference = (spans[1] + spans[3]) / 2
    ratio = min(spans[2] / max(reference, 1), 2)
    assert spans[1:4] == [45, 28, 41]

    fig, ax = plt.subplots(figsize=(7.2, 4.55))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    # a: actual binary image and detector-produced logical-unit boundaries.
    ax.text(6, 64.3, "a   六单元定位", color=INK, fontsize=10.1, fontweight="bold", va="top")
    top = (8.0, 88.0, 39.0, 59.0)
    ax.imshow(binary[98:178], cmap="gray", vmin=0, vmax=1,
              extent=top, interpolation="nearest", aspect="auto", zorder=1)
    for index, (x0, x1) in enumerate(bounds):
        left = xmap(x0, 0, 320, top[0], top[1])
        right = xmap(x1, 0, 320, top[0], top[1])
        edge = RED if index == 2 else GRAY
        ax.add_patch(Rectangle((left, top[2]), right - left, top[3] - top[2],
                               fill=False, edgecolor=edge,
                               linewidth=1.4 if index == 2 else 0.65, zorder=3))
        ax.text((left + right) / 2, 56.0, f"U{index}", ha="center", va="center",
                color=RED if index == 2 else "#FFFFFF", fontsize=8.8,
                fontweight="bold" if index == 2 else "normal")
    ax.text(91.2, 48.8, "APHNet\n二值掩码", ha="left", va="center",
            color=INK, fontsize=8.3)

    # b: unchanged source crop with a transparent tint only where the mask is 1.
    ax.text(6, 35.3, "b   缺陷与邻段", color=INK, fontsize=10.1, fontweight="bold", va="top")
    crop_x0, crop_x1, crop_y0, crop_y1 = 46, 216, 98, 178
    left, right, bottom, upper = 7.5, 50.0, 9.2, 29.2
    ax.imshow(original[crop_y0:crop_y1, crop_x0:crop_x1],
              extent=(left, right, bottom, upper), interpolation="nearest",
              aspect="auto", zorder=1)
    crop_mask = binary[crop_y0:crop_y1, crop_x0:crop_x1]
    tint = np.zeros((*crop_mask.shape, 4), dtype=np.float32)
    tint[..., :3] = YELLOW
    tint[..., 3] = crop_mask.astype(np.float32) * 0.67
    ax.imshow(tint, extent=(left, right, bottom, upper), interpolation="nearest",
              aspect="auto", zorder=2)
    for boundary in (bounds[1][1], bounds[2][1]):
        xx = xmap(boundary, crop_x0, crop_x1, left, right)
        ax.plot([xx, xx], [bottom, upper], color="white", linewidth=0.85, zorder=4)
    focus_left = xmap(bounds[2][0], crop_x0, crop_x1, left, right)
    focus_right = xmap(bounds[2][1], crop_x0, crop_x1, left, right)
    ax.add_patch(Rectangle((focus_left, bottom), focus_right - focus_left, upper - bottom,
                           fill=False, edgecolor=RED, linewidth=1.35, zorder=5))
    for index in (1, 2, 3):
        x0, x1 = bounds[index]
        xx = xmap((x0 + x1) / 2, crop_x0, crop_x1, left, right)
        ax.text(xx, 7.5, "短残留 U2" if index == 2 else f"邻段 U{index}",
                ha="center", va="top", color=RED if index == 2 else INK,
                fontsize=8.5, fontweight="bold" if index == 2 else "normal")

    # c: measured run lengths, one transparent feature value, final OOF call.
    ax.text(55, 35.3, "c   同图自参照与判定", color=INK, fontsize=10.1,
            fontweight="bold", va="top")
    ax.text(55, 30.0, "最长连续前景跨度", color=INK, fontsize=8.7, va="top")
    bar_left, bar_width = 65.0, 23.0
    for unit_index, ypos in zip((1, 2, 3), (25.2, 20.0, 14.8)):
        value = spans[unit_index]
        ax.text(55.0, ypos, f"U{unit_index}", ha="left", va="center",
                color=RED if unit_index == 2 else INK, fontsize=9.0)
        ax.add_patch(Rectangle((bar_left, ypos - 1.25), bar_width, 2.5,
                               facecolor=PALE, edgecolor="none"))
        ax.add_patch(Rectangle((bar_left, ypos - 1.25), bar_width * value / 50, 2.5,
                               facecolor=RED if unit_index == 2 else BLUE, edgecolor="none"))
        ax.text(90.0, ypos, f"{value} px", ha="left", va="center", fontsize=8.9, color=INK)
    ax.text(55.0, 9.8, f"相对邻段长度  {ratio:.2f}", color=INK, fontsize=8.7, va="top")
    ax.add_patch(FancyArrowPatch((71.5, 8.0), (71.5, 4.5), arrowstyle="-|>",
                                 mutation_scale=10, linewidth=0.9, color=BLUE))
    ax.text(55.0, 2.0, "轨迹 21 维 + 拓扑 20 维  →  断线（折外判定）",
            color=INK, fontsize=8.9, va="bottom")

    require_matplotlib_panel_alignment(
        fig, json_out=OUT / f"{STEM}.alignment.json",
        overlay_svg=OUT / f"{STEM}.alignment.svg", strict=True,
    )
    fig.savefig(OUT / f"{STEM}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{STEM}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{STEM}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "source_id": SOURCE_ID,
        "condition": "clear",
        "outer_fold": image_prediction["fold"],
        "target_class": image_prediction["target_class"],
        "oof_predicted_class": image_prediction["predicted_class"],
        "oof_unit_2_break_probability": unit_prediction["probability"]["break"],
        "mask": str(MASK), "rgb": str(RGB),
        "display_crop_xyxy": [crop_x0, crop_y0, crop_x1, crop_y1],
        "display_adjustment": "crop and transparent binary-mask overlay only",
        "detector_unit_bounds": bounds,
        "longest_run_px": spans,
        "unit_2_neighbor_mean_px": reference,
        "unit_2_neighbor_ratio": ratio,
        "claim_boundary": "illustrative real OOF example; ratio alone is not a decision rule",
    }
    (OUT / f"{STEM}.provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT / f"{STEM}.png")


if __name__ == "__main__":
    main()
