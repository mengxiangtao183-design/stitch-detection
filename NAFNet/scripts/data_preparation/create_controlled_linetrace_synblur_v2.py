"""Create Controlled LineTrace SynBlur-v2 for C3 line-trace deblurring.

This protocol is intentionally harder than Controlled-SynBlur-v1, but still
recoverable and defensible:

  - The main degradation is structured motion blur, not arbitrary corruption.
  - Blur is stronger around line/edge evidence estimated from the clear target.
  - Top/middle/bottom horizontal stripes receive different blur parameters.
  - Light noise/JPEG/gamma perturbations are rare auxiliary augmentation only.
  - RX remains a pure external synthetic test domain and is never used in train/val.

The script writes normal C3Net/NAFNet manifests plus a JSONL parameter file that
can supervise a future degradation-condition branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


TRAIN_DOMAINS = ("DB", "XJ", "SD")
TEST_DOMAINS = ("DB", "XJ", "SD", "RX")

# Three train variants make the train set larger while keeping the task centered
# on motion deblurring.  The third variant is deliberately harder, giving the
# proposed condition/stripe modules real room to outperform a blind baseline.
TRAIN_VARIANTS = (
    ("v2a", (13, 15, 17)),
    ("v2b", (17, 19, 21)),
    ("v2c", (21, 23, 25)),
)
ONE_VARIANT_KERNELS = (15, 17, 19, 21, 23, 25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Controlled LineTrace SynBlur-v2 manifests")
    parser.add_argument("--data-root", default="H:/DATASET")
    parser.add_argument("--base-manifest-dir", default="H:/C3Netqing/MISCFilter/manifests/c3net_external_rx")
    parser.add_argument(
        "--manifest-dir",
        default="H:/C3Netqing/MISCFilter/manifests/c3net_controlled_linetrace_synblur_v2",
    )
    parser.add_argument("--output-subdir", default="ImgsSynLineV2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--write-preview",
        action="store_true",
        help="Write a small before/after preview grid under the manifest directory.",
    )
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


def make_odd(value: int, minimum: int = 5, maximum: int = 31) -> int:
    value = int(max(minimum, min(maximum, value)))
    if value % 2 == 0:
        value += 1 if value < maximum else -1
    return value


def motion_kernel(length: int, angle_degrees: float) -> np.ndarray:
    length = make_odd(length)
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


def horizontal_stripe_weights(height: int, width: int, feather: int) -> list[np.ndarray]:
    y = np.arange(height, dtype=np.float32)[:, None]
    thirds = [0.0, height / 3.0, 2.0 * height / 3.0, float(height)]
    weights: list[np.ndarray] = []
    for start, end in zip(thirds[:-1], thirds[1:]):
        left = np.clip((y - (start - feather)) / max(1.0, feather), 0.0, 1.0)
        right = np.clip(((end + feather) - y) / max(1.0, feather), 0.0, 1.0)
        weight = np.minimum(left, right)
        weights.append(np.repeat(weight, width, axis=1))
    total = np.maximum(sum(weights), 1e-6)
    return [(w / total).astype(np.float32) for w in weights]


def line_evidence_mask(image: np.ndarray) -> np.ndarray:
    """Estimate stitch/line evidence from the clean target.

    The mask is only used to synthesize realistic non-uniform blur and optional
    future line-prior supervision.  It is not used at test time by baseline
    models unless a proposed method explicitly predicts/uses such a prior.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray_blur, 40, 120)

    sobel_x = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(sobel_x, sobel_y)
    threshold = np.percentile(grad, 82.0)
    grad_mask = (grad >= max(8.0, threshold)).astype(np.uint8) * 255

    mask = cv2.bitwise_or(edges, grad_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 2.0)

    # If the image is very plain, avoid a near-empty mask by falling back to a
    # broad middle prior.  This keeps the degradation recoverable and consistent
    # with the line-trace ROI protocol.
    if float(mask_f.mean()) < 0.01:
        h, w = gray.shape
        mask_f = np.zeros((h, w), dtype=np.float32)
        mask_f[h // 3 : 2 * h // 3, :] = 0.35
        mask_f = cv2.GaussianBlur(mask_f, (0, 0), 5.0)
    return np.clip(mask_f, 0.0, 1.0)


def apply_sparse_auxiliary_degradation(image: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply rare mild degradation so the task remains deblurring-dominant."""

    out = image.astype(np.float32)
    aug: dict[str, Any] = {"type": "none"}
    choice = rng.random()
    if choice < 0.10:
        sigma = rng.uniform(1.0, 4.0)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0.0, sigma, out.shape).astype(np.float32)
        out = out + noise
        aug = {"type": "noise", "sigma": sigma}
    elif choice < 0.20:
        quality = rng.randint(80, 95)
        ok, encoded = cv2.imencode(".jpg", np.clip(out, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                out = decoded.astype(np.float32)
        aug = {"type": "jpeg", "quality": quality}
    elif choice < 0.25:
        gamma = rng.uniform(0.90, 1.12)
        normalized = np.clip(out / 255.0, 0.0, 1.0)
        out = np.power(normalized, gamma) * 255.0
        aug = {"type": "gamma", "gamma": gamma}
    return np.clip(out, 0, 255).astype(np.uint8), aug


def synthesize_linetrace_blur(
    image: np.ndarray,
    rng: random.Random,
    kernel_choices: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = image.shape[:2]
    base_length = rng.choice(kernel_choices)
    base_angle = rng.uniform(-18.0, 18.0)

    # Middle stripe is intentionally a bit harder because the project evaluates
    # line-trace quality in the middle band and future modules can exploit this
    # structure without seeing ground truth at inference.
    stripe_lengths = [
        make_odd(base_length + rng.choice((-4, -2, 0))),
        make_odd(base_length + rng.choice((2, 4, 6))),
        make_odd(base_length + rng.choice((-2, 0, 2))),
    ]
    stripe_angles = [
        base_angle + rng.uniform(-4.0, 4.0),
        base_angle + rng.uniform(-6.0, 6.0),
        base_angle + rng.uniform(-4.0, 4.0),
    ]

    weights = horizontal_stripe_weights(h, w, feather=max(8, h // 32))
    strong = np.zeros_like(image, dtype=np.float32)
    for weight, length, angle in zip(weights, stripe_lengths, stripe_angles):
        blurred = synthesize_motion_blur(image, length, angle).astype(np.float32)
        strong += blurred * weight[:, :, None]

    mild_length = make_odd(base_length - rng.choice((6, 8, 10)), minimum=5, maximum=21)
    mild_angle = base_angle + rng.uniform(-3.0, 3.0)
    mild = synthesize_motion_blur(image, mild_length, mild_angle).astype(np.float32)

    mask = line_evidence_mask(image)[:, :, None]
    line_strength = rng.uniform(0.90, 1.00)
    background_strength = rng.uniform(0.40, 0.62)
    image_f = image.astype(np.float32)
    line_region = image_f * (1.0 - line_strength) + strong * line_strength
    background = image_f * (1.0 - background_strength) + mild * background_strength
    degraded = mask * line_region + (1.0 - mask) * background
    degraded, auxiliary = apply_sparse_auxiliary_degradation(np.clip(degraded, 0, 255).astype(np.uint8), rng)

    params: dict[str, Any] = {
        "base_length": base_length,
        "base_angle": base_angle,
        "stripe_lengths": stripe_lengths,
        "stripe_angles": stripe_angles,
        "mild_length": mild_length,
        "mild_angle": mild_angle,
        "line_strength": line_strength,
        "background_strength": background_strength,
        "line_mask_mean": float(mask.mean()),
        "auxiliary": auxiliary,
    }
    return degraded, params


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
) -> tuple[str, str, bool, dict[str, Any]]:
    domain, split, filename = parse_domain_split(target_rel)
    clear_path = data_root / target_rel
    image = cv2.imread(str(clear_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read clear image: {clear_path}")

    rng = stable_rng(seed, f"{target_rel}:{suffix}:linetrace-v2")
    out_name = synthetic_name(filename, suffix)
    out_dir = data_root / domain / split / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    synthetic_rel = f"{domain}/{split}/{output_subdir}/{out_name}"
    wrote = False
    params: dict[str, Any]
    if overwrite or not out_path.exists():
        degraded, params = synthesize_linetrace_blur(image, rng, kernel_choices)
        cv2.imwrite(str(out_path), degraded)
        wrote = True
    else:
        # Regenerate params deterministically for the JSONL record without
        # writing the image again.
        _, params = synthesize_linetrace_blur(image, rng, kernel_choices)
    record = {
        "target": target_rel,
        "source": synthetic_rel,
        "domain": domain,
        "split": split,
        "suffix": suffix,
        **params,
    }
    return target_rel, synthetic_rel, wrote, record


def generate_manifest(
    data_root: Path,
    source_pairs: list[tuple[str, str]],
    output_subdir: str,
    seed: int,
    mode: str,
    overwrite: bool,
) -> tuple[list[tuple[str, str]], int, int, list[dict[str, Any]]]:
    synthetic_pairs: list[tuple[str, str]] = []
    records: list[dict[str, Any]] = []
    generated = 0
    skipped = 0
    for target_rel, _source_rel in tqdm(source_pairs, desc=f"LineTrace SynBlur-v2 {mode}"):
        domain, _split, _filename = parse_domain_split(target_rel)
        if mode in {"train", "val", "test_internal"} and domain not in TRAIN_DOMAINS:
            raise RuntimeError(f"{mode} must only use DB/XJ/SD, got: {target_rel}")
        if mode == "train":
            variants = TRAIN_VARIANTS
        else:
            variants = (("v2", ONE_VARIANT_KERNELS),)
        for suffix, kernel_choices in variants:
            target, source, wrote, record = generate_one(
                data_root=data_root,
                target_rel=target_rel,
                output_subdir=output_subdir,
                seed=seed,
                suffix=suffix,
                kernel_choices=tuple(kernel_choices),
                overwrite=overwrite,
            )
            synthetic_pairs.append((target, source))
            records.append(record)
            if wrote:
                generated += 1
            else:
                skipped += 1
    return synthetic_pairs, generated, skipped, records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def maybe_write_preview(data_root: Path, manifest_dir: Path, pairs: list[tuple[str, str]]) -> None:
    samples = pairs[:6]
    tiles: list[np.ndarray] = []
    for target_rel, source_rel in samples:
        clear = cv2.imread(str(data_root / target_rel), cv2.IMREAD_COLOR)
        degraded = cv2.imread(str(data_root / source_rel), cv2.IMREAD_COLOR)
        if clear is None or degraded is None:
            continue
        h = min(clear.shape[0], degraded.shape[0], 240)
        w = min(clear.shape[1], degraded.shape[1], 360)
        clear = cv2.resize(clear[:h, :w], (w, h), interpolation=cv2.INTER_AREA)
        degraded = cv2.resize(degraded[:h, :w], (w, h), interpolation=cv2.INTER_AREA)
        tiles.append(np.concatenate([degraded, clear], axis=1))
    if tiles:
        preview = np.concatenate(tiles, axis=0)
        cv2.imwrite(str(manifest_dir / "preview_degraded_left_clear_right.jpg"), preview)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    base_manifest_dir = Path(args.base_manifest_dir)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    train_pairs, gen_train, skip_train, train_records = generate_manifest(
        data_root,
        read_manifest(base_manifest_dir / "train.tsv"),
        args.output_subdir,
        args.seed,
        "train",
        args.overwrite,
    )
    val_pairs, gen_val, skip_val, val_records = generate_manifest(
        data_root,
        read_manifest(base_manifest_dir / "val.tsv"),
        args.output_subdir,
        args.seed,
        "val",
        args.overwrite,
    )
    all_records.extend(train_records)
    all_records.extend(val_records)

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
        pairs, generated, skipped, records = generate_manifest(
            data_root,
            read_manifest(base_manifest_dir / f"test_{domain}.tsv"),
            args.output_subdir,
            args.seed,
            mode,
            args.overwrite,
        )
        write_manifest(manifest_dir / f"test_{domain}.tsv", pairs)
        all_records.extend(records)
        test_counts[domain] = len(pairs)
        generated_counts[f"test_{domain}"] = generated
        skipped_counts[f"test_{domain}"] = skipped

    write_jsonl(manifest_dir / "degradation_params.jsonl", all_records)
    if args.write_preview:
        maybe_write_preview(data_root, manifest_dir, train_pairs)

    metadata = {
        "name": "Controlled-LineTrace-SynBlur-v2",
        "base_manifest_dir": str(base_manifest_dir),
        "output_subdir": args.output_subdir,
        "seed": args.seed,
        "primary_degradation": "line-evidence-weighted, three-horizontal-stripe non-uniform motion blur",
        "angle_degrees": [-18, 18],
        "train_policy": "DB/XJ/SD train clear images are degraded into 3 structured variants per image.",
        "val_policy": "Held-out DB/XJ/SD val clear images are degraded into 1 structured variant per image.",
        "internal_test_policy": "DB/XJ/SD test clear images are degraded into 1 structured variant per image.",
        "external_test_policy": "RX test clear images are degraded into 1 structured variant per image and remain external test only.",
        "train_variants": [
            {"suffix": suffix, "kernel_choices": list(kernels)} for suffix, kernels in TRAIN_VARIANTS
        ],
        "one_variant_kernel_choices": list(ONE_VARIANT_KERNELS),
        "line_strength_range": [0.90, 1.00],
        "background_strength_range": [0.40, 0.62],
        "auxiliary_degradation_policy": {
            "none_probability": 0.75,
            "noise_probability": 0.10,
            "jpeg_probability": 0.10,
            "gamma_probability": 0.05,
            "note": "Auxiliary perturbations are deliberately sparse; the task remains motion-deblurring dominant.",
        },
        "counts": {"train": len(train_pairs), "val": len(val_pairs), **test_counts},
        "generated_counts": generated_counts,
        "skipped_counts": skipped_counts,
        "degradation_params": str(manifest_dir / "degradation_params.jsonl"),
    }
    (manifest_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote Controlled LineTrace SynBlur-v2 manifests to {manifest_dir}")
    print(json.dumps(metadata["counts"], ensure_ascii=False))
    print("generated:", json.dumps(generated_counts, ensure_ascii=False))
    print("skipped:", json.dumps(skipped_counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
