"""Render paper-ready qualitative examples for the end-to-end TSR ablation.

Each row selects one labelled example of skip, break, skew and loose.  It
prefers cases where direct blur causes a miss but deblurring restores the
correct decision.  Every panel overlays the *same condition's M6 binary
segmentation* on its RGB input, highlights the labelled stitch unit, and
prints the frozen-threshold TSR decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

CONDITIONS = ("clear", "blur_direct", "deblur_then_m6")
CONDITION_TITLES = {
    "clear": "Clear image + M6",
    "blur_direct": "Direct blur + M6",
    "deblur_then_m6": "Deblur + M6",
}
DEFECTS = ("skip", "break", "skew", "loose")
LOCAL = ("skip", "break", "loose")


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def index_rows(rows: List[Dict]) -> Dict[Tuple[str, str, int | None], Dict]:
    return {(row["sample_id"], row["scope"], row["unit_id"]): row for row in rows}


def select_examples(indexed: Dict[str, Dict[Tuple[str, str, int | None], Dict]]) -> Dict[str, Tuple[str, str, int | None]]:
    selected = {}
    for defect in DEFECTS:
        scope = "image" if defect == "skew" else "unit"
        candidates = []
        for key, clear in indexed["clear"].items():
            if key[1] != scope or clear["target"][defect] != 1:
                continue
            direct = indexed["blur_direct"][key]
            deblur = indexed["deblur_then_m6"][key]
            clear_ok = int(bool(clear["prediction"][defect]))
            direct_miss = int(not bool(direct["prediction"][defect]))
            deblur_ok = int(bool(deblur["prediction"][defect]))
            score_gain = float(deblur["scores"][defect]) - float(direct["scores"][defect])
            # Main criterion: a direct-blur miss recovered after restoration.
            rank = 100 * direct_miss * deblur_ok + 10 * clear_ok + score_gain
            candidates.append((rank, key))
        if not candidates:
            raise ValueError(f"No positive labelled {defect} example found")
        selected[defect] = max(candidates, key=lambda item: (item[0], item[1]))[1]
    return selected


def prediction_text(row: Dict) -> str:
    names = [name for name, value in row["prediction"].items() if value]
    if not names:
        return "Normal"
    return "+".join(name.capitalize() for name in names)


def image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def binary(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def draw_panel(axis, rgb: np.ndarray, mask: np.ndarray, row: Dict, defect: str, condition: str) -> None:
    axis.imshow(rgb)
    # Yellow overlay marks the trajectory delivered by M6 to TSR-Defect.
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 1] = 0.90
    overlay[..., 3] = mask.astype(np.float32) * 0.72
    axis.imshow(overlay)
    if row["scope"] == "unit":
        x0 = row["evidence"]["x0"]
        x1 = row["evidence"]["x1"]
        axis.add_patch(Rectangle((x0, 1), x1 - x0, mask.shape[0] - 2, fill=False, edgecolor="#ff33cc", linewidth=1.8))
        unit_text = f"U{int(row['unit_id']) + 1}"
    else:
        unit_text = "global"
        axis.add_patch(Rectangle((1, 1), mask.shape[1] - 2, mask.shape[0] - 2, fill=False, edgecolor="#ff33cc", linewidth=1.8))
    actual = f"GT: {defect.capitalize()} ({unit_text})"
    predicted = f"TSR: {prediction_text(row)}"
    correct = bool(row["prediction"].get(defect, False))
    axis.text(
        0.02, 0.98, predicted, transform=axis.transAxes, va="top", ha="left", fontsize=8.7,
        color="white", bbox={"boxstyle": "round,pad=0.22", "facecolor": "#26734d" if correct else "#a61d24", "edgecolor": "none", "alpha": 0.93},
    )
    axis.text(
        0.02, 0.02, actual, transform=axis.transAxes, va="bottom", ha="left", fontsize=8.4,
        color="white", bbox={"boxstyle": "round,pad=0.20", "facecolor": "#292929", "edgecolor": "none", "alpha": 0.91},
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(CONDITION_TITLES[condition], fontsize=10, pad=7)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render qualitative end-to-end TSR-Defect examples")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--clear-image-dir", type=Path, required=True)
    parser.add_argument("--blur-image-dir", type=Path, required=True)
    parser.add_argument("--deblur-image-dir", type=Path, required=True)
    parser.add_argument("--clear-mask-dir", type=Path, required=True)
    parser.add_argument("--blur-mask-dir", type=Path, required=True)
    parser.add_argument("--deblur-mask-dir", type=Path, required=True)
    args = parser.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    indexed = {
        condition: index_rows(load_json(args.result_dir / f"{condition}_oof_predictions.json"))
        for condition in CONDITIONS
    }
    choices = select_examples(indexed)
    image_dirs = {"clear": args.clear_image_dir, "blur_direct": args.blur_image_dir, "deblur_then_m6": args.deblur_image_dir}
    mask_dirs = {"clear": args.clear_mask_dir, "blur_direct": args.blur_mask_dir, "deblur_then_m6": args.deblur_mask_dir}

    fig, axes = plt.subplots(len(DEFECTS), len(CONDITIONS), figsize=(10.4, 13.2), dpi=300)
    for row_index, defect in enumerate(DEFECTS):
        sample_id, scope, unit_id = choices[defect]
        key = (sample_id, scope, unit_id)
        for col_index, condition in enumerate(CONDITIONS):
            draw_panel(
                axes[row_index, col_index], image(image_dirs[condition] / f"{sample_id}.png"),
                binary(mask_dirs[condition] / f"{sample_id}.png"), indexed[condition][key], defect, condition,
            )
        axes[row_index, 0].set_ylabel(f"{defect.capitalize()}\nimage {sample_id}", fontsize=11, rotation=0, labelpad=47, va="center")
    fig.text(0.5, 0.985, "Qualitative end-to-end defect detection under fixed clear-calibrated thresholds", ha="center", va="top", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.008, "Yellow: M6 binary trajectory overlay; magenta: labelled defect unit / global image; green: labelled target detected; red: labelled target missed.", ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.955, bottom=0.035, hspace=0.18, wspace=0.04)
    output = args.result_dir / "qualitative_end_to_end_defect_examples.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    (args.result_dir / "qualitative_example_selection.json").write_text(json.dumps({name: {"sample_id": value[0], "scope": value[1], "unit_id": value[2]} for name, value in choices.items()}, indent=2), encoding="utf-8")
    print(output)
    print(json.dumps({name: value[0] for name, value in choices.items()}))


if __name__ == "__main__":
    main()
