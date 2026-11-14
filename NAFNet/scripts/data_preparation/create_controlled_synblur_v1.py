"""Create a controlled synthetic deblurring benchmark for C3 line-trace data.

This protocol is intended for fast model-innovation screening when the real
Imgs1/Imgs2 degradation distributions are inconsistent or not fully reliable.

Protocol:
  - DB/XJ/SD train: 3 synthetic motion-blur variants per clear image.
  - DB/XJ/SD val: 1 synthetic motion-blur variant per held-out clear image.
  - DB/XJ/SD test: 1 synthetic motion-blur variant per clear test image.
  - RX test: 1 synthetic motion-blur variant per clear RX image; RX is never
    used in train/val and remains a pure external synthetic test domain.

All synthetic inputs are generated from the clear image with a small-angle
motion blur kernel. Targets remain the corresponding clear images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


TRAIN_DOMAINS = ("DB", "XJ", "SD")
TEST_DOMAINS = ("DB", "XJ", "SD", "RX")
TRAIN_VARIANTS = (
    ("syn1", (5, 7)),
    ("syn2", (9, 11)),
    ("syn3", (13, 15)),
)
ONE_VARIANT_KERNELS = (5, 7, 9, 11, 13, 15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create controlled SynBlur-v1 benchmark manifests")
    parser.add_argument("--data-root", default="H:/DATASET")
    parser.add_argument("--base-manifest-dir", default="H:/C3Netqing/MISCFilter/manifests/c3net_external_rx")
    parser.add_argument("--manifest-dir", default="H:/C3Netqing/MISCFilter/manifests/c3net_controlled_synblur_v1")
    parser.add_argument("--output-subdir", default="ImgsSynCtrlV1")
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


def synthetic_name(filename: str, suffix: str) -> str:
    path = Path(filename)
    return f"{path.stem}_{suffix}{path.suffix}"


def generate_one(
    data_root: Path,
    target_rel: str,
    output_subdir: str,
    seed: int,
    suffix: str,
    kernel_choices: tuple[int, ...],
    overwrite: bool,
) -> tuple[str, str, bool]:
    domain, split, filename = parse_domain_split(target_rel)
    clear_path = data_root / target_rel
    image = cv2.imread(str(clear_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read clear image: {clear_path}")

    rng = stable_rng(seed, f"{target_rel}:{suffix}")
    length = rng.choice(kernel_choices)
    angle = rng.uniform(-15.0, 15.0)
    out_name = synthetic_name(filename, suffix)
    out_dir = data_root / domain / split / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    wrote = False
    if overwrite or not out_path.exists():
        blurred = synthesize_motion_blur(image, length, angle)
        cv2.imwrite(str(out_path), blurred)
        wrote = True
    synthetic_rel = f"{domain}/{split}/{output_subdir}/{out_name}"
    return target_rel, synthetic_rel, wrote


def generate_manifest(
    data_root: Path,
    source_pairs: list[tuple[str, str]],
    output_subdir: str,
    seed: int,
    mode: str,
    overwrite: bool,
) -> tuple[list[tuple[str, str]], int, int]:
    synthetic_pairs: list[tuple[str, str]] = []
    generated = 0
    skipped = 0
    for target_rel, _source_rel in tqdm(source_pairs, desc=f"Controlled SynBlur {mode}"):
        domain, _split, _filename = parse_domain_split(target_rel)
        if mode in {"train", "val", "test_internal"} and domain not in TRAIN_DOMAINS:
            raise RuntimeError(f"{mode} must only use DB/XJ/SD, got: {target_rel}")
        if mode == "train":
            variants = TRAIN_VARIANTS
        else:
            variants = (("syn", ONE_VARIANT_KERNELS),)
        for suffix, kernel_choices in variants:
            target, source, wrote = generate_one(
                data_root=data_root,
                target_rel=target_rel,
                output_subdir=output_subdir,
                seed=seed,
                suffix=suffix,
                kernel_choices=tuple(kernel_choices),
                overwrite=overwrite,
            )
            synthetic_pairs.append((target, source))
            if wrote:
                generated += 1
            else:
                skipped += 1
    return synthetic_pairs, generated, skipped


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    base_manifest_dir = Path(args.base_manifest_dir)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    train_pairs, gen_train, skip_train = generate_manifest(
        data_root,
        read_manifest(base_manifest_dir / "train.tsv"),
        args.output_subdir,
        args.seed,
        "train",
        args.overwrite,
    )
    val_pairs, gen_val, skip_val = generate_manifest(
        data_root,
        read_manifest(base_manifest_dir / "val.tsv"),
        args.output_subdir,
        args.seed,
        "val",
        args.overwrite,
    )

    rng = random.Random(args.seed)
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    write_manifest(manifest_dir / "train.tsv", train_pairs)
    write_manifest(manifest_dir / "val.tsv", val_pairs)

    test_counts: dict[str, int] = {}
    generated_counts = {"train": gen_train, "val": gen_val}
    skipped_counts = {"train": skip_train, "val": skip_val}
    for domain in TEST_DOMAINS:
        mode = "test_external_rx" if domain == "RX" else "test_internal"
        pairs, generated, skipped = generate_manifest(
            data_root,
            read_manifest(base_manifest_dir / f"test_{domain}.tsv"),
            args.output_subdir,
            args.seed,
            mode,
            args.overwrite,
        )
        write_manifest(manifest_dir / f"test_{domain}.tsv", pairs)
        test_counts[domain] = len(pairs)
        generated_counts[f"test_{domain}"] = generated
        skipped_counts[f"test_{domain}"] = skipped

    metadata = {
        "name": "Controlled-SynBlur-v1",
        "base_manifest_dir": str(base_manifest_dir),
        "output_subdir": args.output_subdir,
        "seed": args.seed,
        "angle_degrees": [-15, 15],
        "train_policy": "DB/XJ/SD train clear images are degraded into 3 variants per image.",
        "val_policy": "Held-out DB/XJ/SD val clear images are degraded into 1 random-kernel variant per image.",
        "internal_test_policy": "DB/XJ/SD test clear images are degraded into 1 random-kernel variant per image.",
        "external_test_policy": "RX test clear images are degraded into 1 random-kernel variant per image and remain external test only.",
        "train_variants": [
            {"suffix": suffix, "kernel_choices": list(kernels)} for suffix, kernels in TRAIN_VARIANTS
        ],
        "one_variant_kernel_choices": list(ONE_VARIANT_KERNELS),
        "counts": {"train": len(train_pairs), "val": len(val_pairs), **test_counts},
        "generated_counts": generated_counts,
        "skipped_counts": skipped_counts,
    }
    (manifest_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote controlled SynBlur-v1 manifests to {manifest_dir}")
    print(json.dumps(metadata["counts"], ensure_ascii=False))
    print("generated:", json.dumps(generated_counts, ensure_ascii=False))
    print("skipped:", json.dumps(skipped_counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
