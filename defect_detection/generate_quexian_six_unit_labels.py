"""Build six-unit defect-label templates from numbered images and LabelMe masks.

The output is deliberately a *review template*: it creates candidate left-to-
right ROIs and inspection overlays, but never assigns skip/break/skew/loose
labels automatically.  Those are manufacturing ground-truth labels and must be
confirmed by a human reviewer.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FIELDNAMES = [
    "sample_id", "image_file", "mask_file", "source_image", "augmentation",
    "unit_id", "x0", "x1", "y0", "y1", "polygon_count", "roi_method",
    "known_structure_anomaly", "valid",
    "skip", "break", "skew", "loose", "review_status", "note",
]
COLOURS = [(231, 76, 60), (230, 126, 34), (241, 196, 15), (46, 204, 113), (52, 152, 219), (155, 89, 182)]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="gbk"))


def _polygon_centres(annotation: dict[str, Any]) -> list[float]:
    centres = []
    for shape in annotation.get("shapes", []):
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if len(points) < 3:
            raise ValueError("Annotation polygon contains fewer than three points")
        centres.append(float(points[:, 0].mean()))
    return sorted(centres)


def _mask_extent(mask: Image.Image) -> tuple[int, int, int, int]:
    foreground = np.asarray(mask) > 0
    ys, xs = np.where(foreground)
    if xs.size == 0:
        raise ValueError("Binary mask has no foreground pixels")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _six_boundaries(
    centres: list[float], x_min: int, x_max: int, num_units: int = 6
) -> list[int]:
    """Return seven monotonic x boundaries.

    Exactly six annotated polygons give reliable centre anchors; otherwise use
    equal spatial slots over the observed trajectory and require manual review.
    """
    if len(centres) == num_units:
        internal = [(centres[index] + centres[index + 1]) / 2 for index in range(num_units - 1)]
        bounds = [float(x_min), *internal, float(x_max + 1)]
    else:
        bounds = np.linspace(float(x_min), float(x_max + 1), num_units + 1).tolist()
    rounded = [int(round(value)) for value in bounds]
    rounded[0] = x_min
    rounded[-1] = x_max + 1
    for index in range(1, len(rounded)):
        rounded[index] = max(rounded[index], rounded[index - 1] + 1)
    return rounded


def _overlay(
    image: Image.Image,
    mask: Image.Image,
    boundaries: list[int],
    y0: int,
    y1: int,
    status: str,
    output: Path,
) -> None:
    """Render a compact human-review image with foreground and six ROI boxes."""
    base = image.convert("RGB")
    foreground = Image.new("RGB", base.size, (255, 255, 255))
    alpha = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 90, mode="L")
    preview = Image.composite(foreground, base, alpha)
    draw = ImageDraw.Draw(preview, "RGBA")
    font = ImageFont.load_default()
    for index, colour in enumerate(COLOURS):
        x0, x1 = boundaries[index], boundaries[index + 1]
        draw.rectangle((x0, y0, x1, y1), outline=(*colour, 255), width=6)
        draw.rectangle((x0 + 4, y0 + 4, x0 + 42, y0 + 28), fill=(*colour, 220))
        draw.text((x0 + 13, y0 + 10), str(index), fill=(0, 0, 0, 255), font=font)
    draw.rectangle((8, 8, 330, 35), fill=(0, 0, 0, 180))
    draw.text((15, 15), status, fill=(255, 255, 255, 255), font=font)
    preview.save(output, quality=90, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="dataset_96_png directory")
    parser.add_argument("--annotations", type=Path, required=True, help="Original LabelMe JSON directory")
    parser.add_argument("--output", type=Path, required=True, help="New review-output directory")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")
    mapping_path = args.dataset / "mapping.csv"
    with mapping_path.open(encoding="utf-8", newline="") as file:
        mapping = list(csv.DictReader(file))
    if len(mapping) != 96:
        raise ValueError(f"Expected 96 mapping rows, found {len(mapping)}")
    annotations = {path.name: _read_json(path) for path in args.annotations.glob("*.json")}

    args.output.mkdir(parents=True, exist_ok=False)
    overlays_dir = args.output / "overlays"
    overlays_dir.mkdir()
    rows: list[dict[str, str]] = []
    summary = {"six_polygons": 0, "five_polygon_discontinuities": 0, "seven_polygon_discontinuities": 0}

    for entry in mapping:
        annotation_name = entry["source_annotation"]
        if annotation_name not in annotations:
            raise FileNotFoundError(f"Missing source annotation: {annotation_name}")
        image_path = args.dataset / entry["image_file"]
        mask_path = args.dataset / entry["mask_file"]
        with Image.open(image_path) as opened_image, Image.open(mask_path) as opened_mask:
            image = opened_image.convert("RGB")
            mask = opened_mask.convert("L")
        x_min, mask_y_min, x_max, mask_y_max = _mask_extent(mask)
        padding = max(24, round((mask_y_max - mask_y_min + 1) * 0.7))
        y0 = max(0, mask_y_min - padding)
        y1 = min(image.height - 1, mask_y_max + padding)
        annotation = annotations[annotation_name]
        source_width = annotation.get("imageWidth")
        if not isinstance(source_width, (int, float)) or source_width <= 0:
            raise ValueError(f"Invalid imageWidth in {annotation_name}")
        # LabelMe points are expressed in the source-image coordinate system.
        # The numbered training copy may be 320x320, so transform x anchors
        # before deriving its six ROI boundaries.
        x_scale = image.width / float(source_width)
        centres = [centre * x_scale for centre in _polygon_centres(annotation)]
        if entry["augmentation"] == "horizontal_flip":
            # Numbering always runs left-to-right in the displayed image.
            centres = sorted(image.width - centre for centre in centres)
        boundaries = _six_boundaries(centres, x_min, x_max)
        polygon_count = len(centres)
        is_exact_six = polygon_count == 6
        if polygon_count == 6:
            roi_method = "polygon_centres"
            known_structure_anomaly = ""
            status = "needs_defect_label"
            summary["six_polygons"] += 1
        elif polygon_count == 5:
            # Per the dataset owner's definition, fewer visible polygons are a
            # true stitch discontinuity, not a failed ROI annotation.
            roi_method = "equal_slots_due_to_discontinuity"
            known_structure_anomaly = "five_polygons_skip_or_break"
            status = "needs_defect_label"
            summary["five_polygon_discontinuities"] += 1
        elif polygon_count == 7:
            roi_method = "equal_slots_due_to_discontinuity"
            known_structure_anomaly = "seven_polygons_skip_or_break"
            status = "needs_defect_label"
            summary["seven_polygon_discontinuities"] += 1
        else:
            raise ValueError(f"Expected 5, 6 or 7 polygons, got {polygon_count}")
        for unit_id in range(6):
            rows.append({
                "sample_id": entry["sample_id"],
                "image_file": entry["image_file"],
                "mask_file": entry["mask_file"],
                "source_image": entry["source_image"],
                "augmentation": entry["augmentation"],
                "unit_id": str(unit_id),
                "x0": str(boundaries[unit_id]),
                "x1": str(boundaries[unit_id + 1]),
                "y0": str(y0),
                "y1": str(y1),
                "polygon_count": str(polygon_count),
                "roi_method": roi_method,
                "known_structure_anomaly": known_structure_anomaly,
                "valid": "1",
                "skip": "",
                "break": "",
                "skew": "",
                "loose": "",
                "review_status": status,
                "note": "",
            })
        _overlay(
            image, mask, boundaries, y0, y1, status,
            overlays_dir / f"{entry['sample_id']}_review.jpg",
        )

    with (args.output / "defect_labels_template.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    summary.update({
        "images": len(mapping), "unit_rows": len(rows),
        "note": "Only fill skip/break/skew/loose after human review. "
                "Five/seven-polygon images are valid discontinuity samples; "
                "confirm the affected unit and whether it is skip or break.",
    })
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
