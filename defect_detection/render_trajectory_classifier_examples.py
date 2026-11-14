"""Render auditable raw-image / M6-mask examples for the local defect classifier."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


LOCAL_CLASSES = ("skip", "break", "loose")


def load_labels(path: Path):
    grouped = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["augmentation"] == "original":
                grouped[row["sample_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["unit_id"]))
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Render representative held-out local-defect decisions")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--m6-masks", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", nargs="+", default=["skip:021", "break:018", "loose:026"])
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    prediction_rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions = {(row["sample_id"], int(row["unit_id"])): row for row in prediction_rows}
    panels = []
    try:
        font = ImageFont.truetype("arial.ttf", 14 * args.scale)
        small_font = ImageFont.truetype("arial.ttf", 11 * args.scale)
    except OSError:
        font = small_font = ImageFont.load_default()
    for token in args.examples:
        requested_class, sample_id = token.split(":", 1)
        raw = Image.open(args.images / f"{sample_id}.png").convert("RGB")
        mask = Image.open(args.m6_masks / f"{sample_id}.png").convert("RGB")
        raw = raw.resize((raw.width * args.scale, raw.height * args.scale), Image.Resampling.BICUBIC)
        mask = mask.resize((mask.width * args.scale, mask.height * args.scale), Image.Resampling.NEAREST)
        width, height = raw.size
        panel = Image.new("RGB", (width, height * 2 + 112 * args.scale), "black")
        panel.paste(raw, (0, 0))
        panel.paste(mask, (0, height))
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, width - 1, 22 * args.scale), fill=(0, 0, 0))
        draw.text((6 * args.scale, 2 * args.scale), f"{requested_class} | image {sample_id}: raw / M6 mask", fill=(255, 220, 0), font=font)
        for row in labels[sample_id]:
            unit_id = int(row["unit_id"])
            x0, x1 = int(row["x0"]) * args.scale, int(row["x1"]) * args.scale
            prediction = predictions[(sample_id, unit_id)]
            target_names = [name for name in LOCAL_CLASSES if int(row[name])]
            pred_names = [name for name in LOCAL_CLASSES if prediction["prediction"][name]]
            correct = target_names == pred_names
            color = (40, 220, 90) if correct else (255, 70, 50)
            draw.line((x0, height, x0, height * 2), fill=color, width=2)
            if unit_id == 5:
                draw.line((x1 - 1, height, x1 - 1, height * 2), fill=color, width=2)
            gt = "+".join(target_names) if target_names else "normal"
            pred = "+".join(pred_names) if pred_names else "normal"
            draw.text((x0 + 3, height * 2 + 5 * args.scale), f"u{unit_id} GT:{gt}", fill=(220, 220, 220), font=small_font)
            draw.text((x0 + 3, height * 2 + 35 * args.scale), f"P:{pred}", fill=color, font=small_font)
            probs = prediction["probability"]
            draw.text((x0 + 3, height * 2 + 65 * args.scale), f"{requested_class[:2]}={probs[requested_class]:.2f}", fill=(140, 190, 255), font=small_font)
        panels.append(panel)
    montage = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "black")
    x = 0
    for panel in panels:
        montage.paste(panel, (x, 0))
        x += panel.width
    args.output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
