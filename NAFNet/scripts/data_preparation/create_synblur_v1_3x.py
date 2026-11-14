"""Create conservative synthetic motion-blur pretraining data for External-RX.

SynBlur-v1 is intentionally narrow and physically plausible for stitch traces:
only small-angle motion blur is synthesized, with three blur-strength variants
per clear image. RX is never used.

The generated pretraining protocol keeps train/val/test roles clean:
  - train.tsv uses synthetic blur generated from the original real train split;
  - val.tsv uses synthetic blur generated from the original held-out val split;
  - test_*.tsv files are copied from the real External-RX protocol unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


TRAIN_DOMAINS = ("DB", "XJ", "SD")
TEST_DOMAINS = ("DB", "XJ", "SD", "RX")
VARIANTS = (
    ("syn1", (5, 7)),
    ("syn2", (9, 11)),
    ("syn3", (13, 15)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create SynBlur-v1 3x data and manifests")
    parser.add_argument("--data-root", default="H:/DATASET")
    parser.add_argument("--base-manifest-dir", default="H:/C3Netqing/MISCFilter/manifests/c3net_external_rx")
    parser.add_argument("--manifest-dir", default="H:/C3Netqing/MISCFilter/manifests/c3net_external_rx_synblur_v1_3x_splitval")
    parser.add_argument("--output-subdir", default="ImgsSynV1_3x")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        target, source = line.split("\t")[:2]
        pairs.append((target, source))
    return pairs


def write_manifest(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{target}\t{source}\n" for target, source in pairs), encoding="utf-8")


def stable_rng(seed: int, key: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def motion_kernel(length: int, angle_degrees: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    rad = np.deg2rad(angle_degrees)
    dx = np.cos(rad)
    dy = np.sin(rad)
    for index in range(length):
        x = int(round(center + (index - center) * dx))
        y = int(round(center + (index - center) * dy))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    total = float(kernel.sum())
    if total <= 0:
        kernel[center, :] = 1.0
        total = float(kernel.sum())
    return kernel / total


def synthesize_motion_blur(image: np.ndarray, length: int, angle_degrees: float) -> np.ndarray:
    kernel = motion_kernel(length, angle_degrees)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT101)


def parse_domain_split(rel_path: str) -> tuple[str, str, str]:
    parts = Path(rel_path.replace("\\", "/")).parts
    if len(parts) < 4:
        raise ValueError(f"Unexpected manifest path: {rel_path}")
    return parts[0], parts[1], parts[-1]


def synthetic_name(filename: str, variant: str) -> str:
    path = Path(filename)
    return f"{path.stem}_{variant}{path.suffix}"


def create_images_for_manifest(args: argparse.Namespace, manifest_name: str) -> list[tuple[str, str]]:
    data_root = Path(args.data_root)
    base_manifest_dir = Path(args.base_manifest_dir)
    input_pairs = read_manifest(base_manifest_dir / manifest_name)
    synthetic_pairs: list[tuple[str, str]] = []
    generated = 0
    skipped = 0

    for target_rel, _source_rel in tqdm(input_pairs, desc=f"SynBlur-v1 3x {manifest_name}"):
        domain, split, filename = parse_domain_split(target_rel)
        if domain not in TRAIN_DOMAINS or split != "train":
            raise RuntimeError(f"SynBlur {manifest_name} must only contain DB/XJ/SD train images, got: {target_rel}")
        clear_path = data_root / target_rel
        image = cv2.imread(str(clear_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read clear image: {clear_path}")

        out_dir = data_root / domain / split / args.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        for variant, kernel_choices in VARIANTS:
            rng = stable_rng(args.seed, f"{target_rel}:{variant}")
            length = rng.choice(kernel_choices)
            angle = rng.uniform(-15.0, 15.0)
            out_name = synthetic_name(filename, variant)
            out_path = out_dir / out_name
            if out_path.exists() and not args.overwrite:
                skipped += 1
            else:
                blurred = synthesize_motion_blur(image, length, angle)
                cv2.imwrite(str(out_path), blurred)
                generated += 1
            synthetic_rel = f"{domain}/{split}/{args.output_subdir}/{out_name}"
            synthetic_pairs.append((target_rel, synthetic_rel))

    print(f"{manifest_name}: generated {generated} images, skipped {skipped} existing images.")
    return synthetic_pairs


def copy_evaluation_manifests(base_manifest_dir: Path, manifest_dir: Path) -> None:
    for name in (f"test_{domain}.tsv" for domain in TEST_DOMAINS):
        source = base_manifest_dir / name
        destination = manifest_dir / name
        if source.is_file():
            shutil.copyfile(source, destination)


def main() -> None:
    args = parse_args()
    base_manifest_dir = Path(args.base_manifest_dir)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    synthetic_train_pairs = create_images_for_manifest(args, "train.tsv")
    synthetic_val_pairs = create_images_for_manifest(args, "val.tsv")
    rng = random.Random(args.seed)
    rng.shuffle(synthetic_train_pairs)
    rng.shuffle(synthetic_val_pairs)
    write_manifest(manifest_dir / "train.tsv", synthetic_train_pairs)
    write_manifest(manifest_dir / "val.tsv", synthetic_val_pairs)
    copy_evaluation_manifests(base_manifest_dir, manifest_dir)

    metadata = {
        "name": "SynBlur-v1-3x-splitval",
        "source_manifest": str(base_manifest_dir / "train.tsv"),
        "source_val_manifest": str(base_manifest_dir / "val.tsv"),
        "train_pairs": len(synthetic_train_pairs),
        "val_pairs": len(synthetic_val_pairs),
        "domains": TRAIN_DOMAINS,
        "variants": [
            {"suffix": variant, "kernel_choices": list(kernel_choices), "angle_degrees": [-15, 15]}
            for variant, kernel_choices in VARIANTS
        ],
        "output_subdir": args.output_subdir,
        "validation_policy": "SynBlur pretraining validates on synthetic blur generated from the held-out DB/XJ/SD val split, not on Imgs1.",
        "test_policy": "Copied test manifests remain real Imgs1 tests for final evaluation only.",
        "rx_policy": "RX is not used for generation, training, or validation; copied test_RX.tsv remains external test only.",
    }
    (manifest_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote synthetic train manifest: {manifest_dir / 'train.tsv'} ({len(synthetic_train_pairs)} pairs)")
    print(f"Wrote synthetic val manifest: {manifest_dir / 'val.tsv'} ({len(synthetic_val_pairs)} pairs)")
    print(f"Copied real test manifests to: {manifest_dir}")


if __name__ == "__main__":
    main()
