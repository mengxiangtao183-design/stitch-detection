"""Create publication-ready TSR-Defect visual comparisons.

The figures are deliberately based on the fixed-clear-calibrated v6 OOF
predictions, so the direct-blur versus restored comparison uses identical
operating thresholds.  Each source image is evaluated only once, out of fold.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


DEFECTS = ("skip", "break", "skew", "loose")
LOCAL = ("skip", "break", "loose")
COLORS = {"direct": "#b35c3e", "deblur": "#26734d", "neutral": "#5b6870"}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = rgb.astype(np.float32).copy()
    color = np.array([255, 215, 0], dtype=np.float32)
    alpha = 0.72
    result[mask] = (1 - alpha) * result[mask] + alpha * color
    return result.astype(np.uint8)


def image_predictions(rows: List[Dict]) -> Dict[str, Dict]:
    """Pool unit scores to the final one-type-per-image decision."""
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row["sample_id"]].append(row)
    output = {}
    for sample_id, values in grouped.items():
        local_rows = [row for row in values if row["scope"] == "unit"]
        image_rows = [row for row in values if row["scope"] == "image"]
        thresholds = local_rows[0].get("thresholds") or local_rows[0].get("thresholds_from_clear_only")
        scores = {name: max(float(row["scores"][name]) for row in local_rows) for name in LOCAL}
        scores["skew"] = max(float(row["scores"]["skew"]) for row in image_rows)
        ratios = {name: scores[name] / max(float(thresholds[name]), 1e-6) for name in DEFECTS}
        best = max(DEFECTS, key=lambda name: (ratios[name], scores[name]))
        prediction = best if ratios[best] >= 1.0 else "normal"
        target_types = []
        for name in DEFECTS:
            # Local records only expose skip/break/loose; the single image
            # record exposes skew.  Do not assume a common target schema.
            if any(bool(row["target"].get(name, 0)) for row in values):
                target_types.append(name)
        if len(target_types) != 1:
            raise ValueError(f"{sample_id}: expected one target type, found {target_types}")
        output[sample_id] = {
            "target": target_types[0], "prediction": prediction,
            "scores": scores, "ratios": ratios, "rows": values,
        }
    return output


def choose_examples(direct: Dict[str, Dict], deblur: Dict[str, Dict]) -> Dict[str, str]:
    choices = {}
    for defect in DEFECTS:
        candidates = [sample_id for sample_id, value in direct.items() if value["target"] == defect]
        # Prefer an image for which restoration changes a wrong final type to the correct type.
        recovered = [s for s in candidates if direct[s]["prediction"] != defect and deblur[s]["prediction"] == defect]
        if recovered:
            choices[defect] = max(recovered, key=lambda s: deblur[s]["ratios"][defect] - direct[s]["ratios"][defect])
        else:
            choices[defect] = max(candidates, key=lambda s: deblur[s]["ratios"][defect] - direct[s]["ratios"][defect])
    return choices


def focus_row(rows: Iterable[Dict], defect: str) -> Dict:
    scope = "image" if defect == "skew" else "unit"
    candidates = [row for row in rows if row["scope"] == scope and row["target"][defect]]
    if not candidates:
        raise ValueError(f"No target row for {defect}")
    return candidates[0]


def type_text(value: str) -> str:
    return "No alarm" if value == "normal" else value.capitalize()


def panel(axis, rgb, title: str, *, mask=None, focus=None, value: Dict | None = None, target: str | None = None):
    shown = overlay_mask(rgb, mask) if mask is not None else rgb
    axis.imshow(shown)
    if focus is not None:
        if focus["scope"] == "unit":
            x0, x1 = focus["evidence"]["x0"], focus["evidence"]["x1"]
            axis.add_patch(Rectangle((x0, 2), x1 - x0, shown.shape[0] - 4, fill=False, edgecolor="#e4007f", linewidth=2.0))
        else:
            axis.add_patch(Rectangle((2, 2), shown.shape[1] - 4, shown.shape[0] - 4, fill=False, edgecolor="#e4007f", linewidth=2.0))
    if value is not None and target is not None:
        correct = value["prediction"] == target
        axis.text(0.02, 0.98, f"TSR: {type_text(value['prediction'])}", transform=axis.transAxes,
                  ha="left", va="top", fontsize=9.2, color="white",
                  bbox={"boxstyle": "round,pad=0.25", "facecolor": COLORS["deblur"] if correct else COLORS["direct"], "edgecolor": "none", "alpha": 0.95})
    if title:
        axis.set_title(title, fontsize=10, pad=6)
    axis.set_xticks([])
    axis.set_yticks([])


def render_qualitative(output: Path, choices: Dict[str, str], direct: Dict[str, Dict], deblur: Dict[str, Dict], clear_dir: Path, blur_dir: Path, restored_dir: Path, direct_mask_dir: Path, deblur_mask_dir: Path):
    fig, axes = plt.subplots(4, 5, figsize=(15.3, 11.8), dpi=300)
    columns = ("(a) Clear source", "(b) Direct blur", "(c) Direct blur: M6 + TSR", "(d) Restored image", "(e) Restored: M6 + TSR")
    for j, title in enumerate(columns):
        axes[0, j].set_title(title, fontsize=10.2, fontweight="bold", pad=8)
    for i, defect in enumerate(DEFECTS):
        sample_id = choices[defect]
        focus = focus_row(deblur[sample_id]["rows"], defect)
        titles = columns if i == 0 else ("",) * len(columns)
        panel(axes[i, 0], load_rgb(clear_dir / f"{sample_id}.png"), titles[0], focus=focus)
        panel(axes[i, 1], load_rgb(blur_dir / f"{sample_id}.png"), titles[1], focus=focus)
        panel(axes[i, 2], load_rgb(blur_dir / f"{sample_id}.png"), titles[2], mask=load_mask(direct_mask_dir / f"{sample_id}.png"), focus=focus, value=direct[sample_id], target=defect)
        panel(axes[i, 3], load_rgb(restored_dir / f"{sample_id}.png"), titles[3], focus=focus)
        panel(axes[i, 4], load_rgb(restored_dir / f"{sample_id}.png"), titles[4], mask=load_mask(deblur_mask_dir / f"{sample_id}.png"), focus=focus, value=deblur[sample_id], target=defect)
        axes[i, 0].set_ylabel(f"{defect.capitalize()}\nimage {sample_id}", fontsize=11.0, rotation=0, labelpad=42, va="center", fontweight="bold")
    fig.text(0.5, 0.985, "Qualitative comparison: direct blurred inspection versus deblurring-assisted inspection", ha="center", va="top", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.012, "Yellow: M6 trajectory delivered to TSR-Defect. Magenta: labelled local unit (or whole-image skew). Green: correct final type; red: wrong type / missed alarm.", ha="center", va="bottom", fontsize=9.2)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.95, bottom=0.04, hspace=0.12, wspace=0.035)
    fig.savefig(output / "Fig_qualitative_direct_blur_vs_deblur.png", dpi=300)
    fig.savefig(output / "Fig_qualitative_direct_blur_vs_deblur.pdf")
    plt.close(fig)


def draw_slope(axis, names: List[str], direct_values: List[float], deblur_values: List[float], title: str, xmax: float = 100.0):
    ypos = np.arange(len(names))[::-1]
    for y, before, after in zip(ypos, direct_values, deblur_values):
        axis.plot([before, after], [y, y], color="#a7b1b7", linewidth=2.0, zorder=1)
        axis.scatter(before, y, s=68, color=COLORS["direct"], zorder=2, marker="o")
        axis.scatter(after, y, s=68, color=COLORS["deblur"], zorder=3, marker="o")
        direct_dy, deblur_dy = (.18, -.20) if abs(before - after) < 8 else (.18, .18)
        axis.text(before - 1.2, y + direct_dy, f"{before:.2f}", ha="right", va="center", fontsize=9, color=COLORS["direct"])
        axis.text(after + 1.2, y + deblur_dy, f"{after:.2f}", ha="left", va="center", fontsize=9, color=COLORS["deblur"])
    axis.set_yticks(ypos, names)
    axis.set_xlim(0, xmax)
    axis.set_xlabel("Score (%)")
    axis.set_title(title, fontsize=12, fontweight="bold", pad=9)
    axis.grid(axis="x", color="#d9dfe3", linewidth=.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)


def render_metrics(output: Path, summary: Dict):
    direct = summary["blur_direct"]
    deblur = summary["deblur_then_m6"]
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.0), dpi=300)
    names = ["Type Top-1 accuracy", "Type macro F1", "Defect-alarm recall"]
    before = [100 * direct["type_top1_accuracy"], 100 * direct["type_macro_f1"], 100 * direct["defect_alarm_recall"]]
    after = [100 * deblur["type_top1_accuracy"], 100 * deblur["type_macro_f1"], 100 * deblur["defect_alarm_recall"]]
    draw_slope(axes[0], names, before, after, "(a) Final image-level decision", xmax=104)
    class_names = ["Skip", "Break", "Skew", "Loose"]
    before_f1 = [100 * direct[name.lower()]["f1"] for name in class_names]
    after_f1 = [100 * deblur[name.lower()]["f1"] for name in class_names]
    draw_slope(axes[1], class_names, before_f1, after_f1, "(b) Class-wise F1", xmax=104)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["direct"], markersize=8, label="Direct blur → M6 → TSR-Defect"),
               plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["deblur"], markersize=8, label="Deblur → M6 → TSR-Defect")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.suptitle("Image-level defect recognition under identical fixed thresholds", fontsize=15, fontweight="bold", y=0.99)
    fig.text(0.5, 0.045, "All values are source-grouped 4-fold out-of-fold results on the 48 controlled synthetic-blur counterparts; higher is better.", ha="center", fontsize=8.8)
    fig.subplots_adjust(left=.16, right=.985, top=.84, bottom=.20, wspace=.33)
    fig.savefig(output / "Fig_quantitative_direct_blur_vs_deblur.png", dpi=300)
    fig.savefig(output / "Fig_quantitative_direct_blur_vs_deblur.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render paper figures for direct-blur versus deblur TSR comparison")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--clear-image-dir", type=Path, required=True)
    parser.add_argument("--blur-image-dir", type=Path, required=True)
    parser.add_argument("--restored-image-dir", type=Path, required=True)
    parser.add_argument("--direct-mask-dir", type=Path, required=True)
    parser.add_argument("--deblur-mask-dir", type=Path, required=True)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    direct = image_predictions(load_json(args.result_dir / "blur_direct_oof_predictions.json"))
    deblur = image_predictions(load_json(args.result_dir / "deblur_then_m6_oof_predictions.json"))
    choices = choose_examples(direct, deblur)
    render_qualitative(args.result_dir, choices, direct, deblur, args.clear_image_dir, args.blur_image_dir, args.restored_image_dir, args.direct_mask_dir, args.deblur_mask_dir)
    render_metrics(args.result_dir, load_json(args.result_dir / "single_defect_image_level_summary.json"))
    (args.result_dir / "paper_figure_example_selection.json").write_text(json.dumps(choices, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(choices, ensure_ascii=False))


if __name__ == "__main__":
    main()
