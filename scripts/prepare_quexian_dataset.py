"""Create a non-destructive, numbered 96-image stitch dataset from LabelMe JSON.

The script keeps the source JPEGs and LabelMe files untouched.  It writes
original samples as 001--048 and their horizontal flips as 049--096, with a
same-stem binary PNG mask for every output image.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageOps


def _numeric_source_key(path: Path) -> tuple[int, str]:
    """Sort timestamp names by their final numeric suffix, then name."""
    try:
        return int(path.stem.rsplit("_", 1)[1]), path.name
    except (IndexError, ValueError):
        return 10**9, path.name


def _read_labelme(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="gbk"))


def labelme_to_binary_mask(annotation: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    """Rasterise every polygon into a uint8 foreground mask (0 or 255)."""
    width, height = size
    if (annotation.get("imageWidth"), annotation.get("imageHeight")) != size:
        raise ValueError(
            "Annotation dimensions do not match its source image: "
            f"json={(annotation.get('imageWidth'), annotation.get('imageHeight'))}, image={size}"
        )
    mask = Image.new("L", size, 0)
    drawer = ImageDraw.Draw(mask)
    for shape in annotation.get("shapes", []):
        if shape.get("shape_type", "polygon") != "polygon":
            raise ValueError(f"Only polygon annotations are supported, got {shape.get('shape_type')!r}")
        points = shape.get("points", [])
        if len(points) < 3:
            raise ValueError("A polygon must contain at least three points")
        drawer.polygon([(float(x), float(y)) for x, y in points], fill=255)
    return mask


def _assert_empty_or_new(output_dir: Path, resume: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if resume:
            return
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}. "
            "Choose another output directory."
        )


def _write_sample(
    image: Image.Image,
    mask: Image.Image,
    image_path: Path,
    mask_path: Path,
) -> None:
    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(image_path, quality=95, subsampling=0)
    else:
        image.save(image_path)
    mask.save(mask_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Source JPEG directory")
    parser.add_argument("--annotations", type=Path, required=True, help="LabelMe JSON directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument(
        "--image-format", choices=("png", "jpeg"), default="png",
        help="Format for generated RGB images (default: png)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue an interrupted run without rewriting completed image/mask pairs",
    )
    args = parser.parse_args()

    source_images = sorted(args.images.glob("*.jpeg"), key=_numeric_source_key)
    if not source_images:
        raise FileNotFoundError(f"No .jpeg files found in {args.images}")
    annotations = {path.stem: path for path in args.annotations.glob("*.json")}
    missing = [path.name for path in source_images if path.stem not in annotations]
    extras = sorted(set(annotations) - {path.stem for path in source_images})
    if missing or extras:
        raise ValueError(f"Image/JSON mismatch; missing={missing}, extra_json={extras}")
    _assert_empty_or_new(args.output, args.resume)

    images_out = args.output / "images"
    masks_out = args.output / "masks"
    images_out.mkdir(parents=True, exist_ok=args.resume)
    masks_out.mkdir(parents=True, exist_ok=args.resume)

    prepared: list[tuple[Path, Image.Image, Image.Image]] = []
    for source in source_images:
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        annotation = _read_labelme(annotations[source.stem])
        mask = labelme_to_binary_mask(annotation, image.size)
        prepared.append((source, image, mask))

    rows: list[dict[str, str]] = []
    for flipped in (False, True):
        for original_index, (source, image, mask) in enumerate(prepared, start=1):
            output_index = original_index + (len(prepared) if flipped else 0)
            sample_id = f"{output_index:03d}"
            out_image = ImageOps.mirror(image) if flipped else image
            out_mask = ImageOps.mirror(mask) if flipped else mask
            image_filename = f"{sample_id}.{args.image_format}"
            image_path = images_out / image_filename
            mask_path = masks_out / f"{sample_id}.png"
            if not (args.resume and image_path.exists() and mask_path.exists()):
                _write_sample(out_image, out_mask, image_path, mask_path)
            rows.append({
                "sample_id": sample_id,
                "image_file": f"images/{image_filename}",
                "mask_file": f"masks/{sample_id}.png",
                "source_image": source.name,
                "source_annotation": annotations[source.stem].name,
                "augmentation": "horizontal_flip" if flipped else "original",
                "source_order": str(original_index),
            })

    with (args.output / "mapping.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source_images": len(prepared),
        "output_images": len(rows),
        "id_scheme": "001-048 originals; 049-096 matching horizontal flips",
        "image_format": f"{args.image_format.upper()} RGB",
        "mask_format": "PNG grayscale, foreground=255, background=0",
    }
    (args.output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
