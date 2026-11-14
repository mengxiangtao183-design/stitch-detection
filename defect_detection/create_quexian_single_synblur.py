"""Create one reproducible, conservative motion-blur image per clear input.

This follows the SynBlur-v1 convention used by NAFNet: a centred line kernel,
small-angle motion only, and ``cv2.BORDER_REFLECT101`` boundary handling.  It
is intentionally an evaluation degradation, not a three-variant pretraining
set: every source image produces exactly one blurred counterpart.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np


def stable_rng(seed: int, key: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def motion_kernel(length: int, angle_degrees: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    centre = length // 2
    angle_radians = np.deg2rad(angle_degrees)
    dx, dy = np.cos(angle_radians), np.sin(angle_radians)
    for index in range(length):
        x = int(round(centre + (index - centre) * dx))
        y = int(round(centre + (index - centre) * dy))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    total = float(kernel.sum())
    if total <= 0:
        kernel[centre, :] = 1.0
        total = float(kernel.sum())
    return kernel / total


def synthesize_motion_blur(image: np.ndarray, length: int, angle_degrees: float) -> np.ndarray:
    return cv2.filter2D(
        image, -1, motion_kernel(length, angle_degrees), borderType=cv2.BORDER_REFLECT101
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one SynBlur-v1-style image for each clear Quxian input")
    parser.add_argument("--input-dir", type=Path, required=True, help="Clear original PNG images")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kernel-min", type=int, default=3)
    parser.add_argument("--kernel-max", type=int, default=15)
    parser.add_argument("--angle-min", type=float, default=-15.0)
    parser.add_argument("--angle-max", type=float, default=15.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kernel_min < 3 or args.kernel_max < args.kernel_min:
        raise ValueError("kernel range must satisfy 3 <= kernel-min <= kernel-max")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {args.output_dir}; use --overwrite only if intentional")
    files = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() == ".png")
    if not files:
        raise ValueError(f"no PNG files found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    odd_lengths = [length for length in range(args.kernel_min, args.kernel_max + 1) if length % 2 == 1]
    if not odd_lengths:
        raise ValueError("the specified kernel range contains no odd length")

    rows = []
    for source_path in files:
        rng = stable_rng(args.seed, source_path.name)
        kernel_length = rng.choice(odd_lengths)
        angle = rng.uniform(args.angle_min, args.angle_max)
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot read: {source_path}")
        blurred = synthesize_motion_blur(image, kernel_length, angle)
        output_path = args.output_dir / source_path.name
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite: {output_path}")
        if not cv2.imwrite(str(output_path), blurred):
            raise RuntimeError(f"cannot write: {output_path}")
        rows.append({
            "source_file": source_path.name, "blurred_file": output_path.name,
            "kernel_length": kernel_length, "angle_degrees": round(angle, 6),
            "boundary_mode": "BORDER_REFLECT101",
        })

    with (args.output_dir / "degradation_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "degradation_protocol.json").write_text(json.dumps({
        "method": "SynBlur-v1-style centred line-kernel motion blur",
        "source_images": len(files), "variants_per_source": 1, "seed": args.seed,
        "kernel_length_choices": odd_lengths, "angle_degrees": [args.angle_min, args.angle_max],
        "boundary_mode": "cv2.BORDER_REFLECT101",
        "label_policy": "The original binary masks and six-unit defect labels are reused; only appearance is degraded.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"generated {len(rows)} blurred images in: {args.output_dir}")


if __name__ == "__main__":
    main()
