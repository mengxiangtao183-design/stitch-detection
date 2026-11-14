"""Standardize the first 50 RXHF JPEG clear images for external evaluation.

Original JPEGs are deliberately retained.  The evaluation protocol references
only 320x320 ``RXHF/test/clear/img001.png`` through ``img050.png`` so the
external domain follows the same folder/name/size convention as DB/XJ/SD/RX.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_key(path: Path) -> list[object]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", path.name)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the first 50 RXHF clear images to standardized PNG files")
    parser.add_argument("--data-root", default="H:/DATASET")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--size", type=int, default=320, help="Square evaluation size; must match the training protocol.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data_root) / "RXHF"
    if args.count != 50:
        raise ValueError("The approved RXHF external protocol uses exactly 50 images.")
    if args.size != 320:
        raise ValueError("The current multi-source protocol fixes every evaluation image at 320x320.")
    source_files = sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES), key=natural_key
    )
    if len(source_files) < args.count:
        raise RuntimeError(f"RXHF has only {len(source_files)} source images; need {args.count}.")
    output_dir = root / "test" / "clear"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str | int]] = []
    for index, source_path in enumerate(source_files[: args.count], start=1):
        output_path = output_dir / f"img{index:03d}.png"
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read RXHF source image: {source_path}")
        interpolation = cv2.INTER_AREA if image.shape[0] >= args.size or image.shape[1] >= args.size else cv2.INTER_CUBIC
        image = cv2.resize(image, (args.size, args.size), interpolation=interpolation)
        if args.overwrite or not output_path.exists():
            if not cv2.imwrite(str(output_path), image):
                raise RuntimeError(f"Cannot write standardized RXHF image: {output_path}")
        records.append(
            {
                "index": index,
                "original": source_path.name,
                "standardized": output_path.relative_to(root).as_posix(),
            }
        )
    (root / "rxhf50_selection.json").write_text(
        json.dumps(
            {"count": args.count, "ordering": "natural filename order", "size": [args.size, args.size], "images": records},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.count} standardized RXHF PNG files to {output_dir}")


if __name__ == "__main__":
    main()
