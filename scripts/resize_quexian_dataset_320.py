"""Create a non-destructive 320 x 320 copy of the numbered stitch dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source numbered dataset directory")
    parser.add_argument("--output", type=Path, required=True, help="New 320 x 320 dataset directory")
    parser.add_argument("--size", type=int, default=320)
    args = parser.parse_args()
    if args.size <= 0:
        raise ValueError("size must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")

    with (args.input / "mapping.csv").open(encoding="utf-8", newline="") as file:
        mapping = list(csv.DictReader(file))
    if len(mapping) != 96:
        raise ValueError(f"Expected 96 mapping rows, found {len(mapping)}")

    image_output = args.output / "images"
    mask_output = args.output / "masks"
    image_output.mkdir(parents=True, exist_ok=False)
    mask_output.mkdir(parents=True, exist_ok=False)
    target = (args.size, args.size)
    for row in mapping:
        image_source = args.input / row["image_file"]
        mask_source = args.input / row["mask_file"]
        with Image.open(image_source) as opened:
            image = opened.convert("RGB").resize(target, Image.Resampling.LANCZOS)
        with Image.open(mask_source) as opened:
            mask = opened.convert("L").resize(target, Image.Resampling.NEAREST)
        image.save(image_output / Path(row["image_file"]).name)
        mask.save(mask_output / Path(row["mask_file"]).name)

    with (args.output / "mapping.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    summary = {
        "source_dataset": str(args.input),
        "samples": len(mapping),
        "image_size": [args.size, args.size],
        "rgb_resize": "LANCZOS",
        "mask_resize": "NEAREST (binary 0/255 preserved)",
    }
    (args.output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
