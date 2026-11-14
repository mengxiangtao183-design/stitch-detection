"""Render a traceable input/output image pair for the pipeline schematic.

The label and unit are verified against held-out, source-grouped OOF predictions.
This is an illustrative case, not an additional quantitative evaluation.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(r"H:\C3Netqing")
OUT = Path(__file__).resolve().parent
SOURCE_ID = "032"
IMAGE = Path(r"H:\DATASET\quexian\dataset_112_320\images") / f"{SOURCE_ID}.png"
MASK = (
    ROOT
    / "results/prediction/runs"
    / "m6-afmguided-highres-mobilenetv4-fpn-full-150-seed42_20260915_152506"
    / "results/segmentation/binary"
    / f"{SOURCE_ID}.png"
)
PREDICTIONS = (
    ROOT
    / "results/defect_detection"
    / "tsr_defect_nested4fold_joint_clear_deblur_112_binary_topology_v2_balanced"
)
CONFIG = ROOT / "configs/defect_head/tsr_defect_rules.yaml"

sys.path.insert(0, str(ROOT))
from defect_detection.train_trajectory_feature_classifier import load_detector, read_mask  # noqa: E402


def main() -> None:
    with (PREDICTIONS / "clear_oof_image_predictions.json").open(encoding="utf-8") as stream:
        image_rows = json.load(stream)
    with (PREDICTIONS / "clear_oof_unit_predictions.json").open(encoding="utf-8") as stream:
        unit_rows = json.load(stream)
    image_row = next(row for row in image_rows if row["source_id"] == SOURCE_ID)
    positive_units = [
        row for row in unit_rows
        if row["source_id"] == SOURCE_ID and row["prediction"]["skip"]
    ]
    assert image_row["target_class"] == image_row["predicted_class"] == "skip"
    assert len(positive_units) == 1
    unit_id = int(positive_units[0]["unit_id"])

    img = np.asarray(Image.open(IMAGE).convert("RGB"))
    assert img.shape == (320, 320, 3)
    mask = read_mask(MASK)
    assert mask.shape == (320, 320)
    detector = load_detector(CONFIG)
    result = detector.detect(mask)
    evidence = result["units"][unit_id]["evidence"]
    x0, x1 = int(evidence["x0"]), int(evidence["x1"])

    # The mask is empty inside this skipped unit; vertical limits come from
    # all visible stitch pixels and are padded to show the corresponding gap.
    ys = np.flatnonzero(np.any(mask > 0.5, axis=1))
    y0, y1 = max(0, int(ys.min()) - 8), min(320, int(ys.max()) + 8)

    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(IMAGE, OUT / "input_032_original.png")

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(3.2, 3.2), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img, interpolation="nearest")
    ax.add_patch(Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        fill=False, linewidth=1.7, edgecolor="#EA4D42",
    ))
    ax.text(
        x0 + 2, y0 - 4, "Skip", ha="left", va="bottom",
        color="white", fontsize=8, fontweight="bold", fontfamily="Times New Roman",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#AD3F38", "edgecolor": "none"},
    )
    ax.set_xlim(0, 320)
    ax.set_ylim(320, 0)
    ax.axis("off")
    fig.savefig(OUT / "output_032_oof_skip.png", dpi=fig.dpi, pad_inches=0)
    fig.savefig(OUT / "output_032_oof_skip.svg", pad_inches=0)
    fig.savefig(OUT / "output_032_oof_skip.pdf", pad_inches=0)
    fig.savefig(OUT / "output_032_oof_skip.tiff", dpi=600, pad_inches=0)
    plt.close(fig)

    manifest = {
        "source_id": SOURCE_ID,
        "input_kind": "original clear RGB photograph",
        "source_image": str(IMAGE),
        "segmentation_mask": str(MASK),
        "prediction_source": "source-grouped four-fold held-out clear OOF predictions",
        "predicted_class": image_row["predicted_class"],
        "predicted_unit": unit_id,
        "unit_x0_x1": [x0, x1],
        "display_y0_y1": [y0, y1],
        "note": "Red box is a candidate unit window, not a pixel-accurate defect segmentation.",
    }
    (OUT / "pair_provenance.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
