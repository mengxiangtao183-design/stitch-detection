"""Create the fair multi-source LineTrace SynBlur-v2 protocol.

Protocol
--------
* DB/XJ/SD keep their established 360/40/100 clear-image split.
* RX becomes a fourth source domain with a deterministic 360/40/100 split
  made *before* any degradation is generated.
* RXHF and TJML are clear-only, external domains.  They receive exactly one
  fixed synthetic input per clear image and are never used for train, val, or
  model selection.

The clear target is used only to create a paired synthetic input.  No mask,
target, or degradation parameter is supplied to the network at inference.
The old ``c3net_controlled_linetrace_synblur_v2`` protocol is left untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from create_controlled_linetrace_synblur_v2 import (
    ONE_VARIANT_KERNELS,
    TRAIN_VARIANTS,
    read_manifest,
    stable_rng,
    synthesize_linetrace_blur,
    synthetic_name,
    write_manifest,
)


SOURCE_DOMAINS = ("DB", "XJ", "SD", "RX")
EXTERNAL_DOMAINS = ("RXHF", "TJML")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create multi-source Controlled LineTrace SynBlur-v2 manifests")
    parser.add_argument("--data-root", default="H:/DATASET")
    parser.add_argument("--base-manifest-dir", default="H:/C3Netqing/MISCFilter/manifests/c3net_external_rx")
    parser.add_argument(
        "--manifest-dir",
        default="H:/C3Netqing/MISCFilter/manifests/c3net_multisource_linetrace_synblur_v2",
    )
    parser.add_argument("--output-subdir", default="ImgsSynLineV2MS")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--refresh-external",
        nargs="+",
        choices=EXTERNAL_DOMAINS,
        default=(),
        help="Regenerate only named external test domain(s), preserving source-domain manifests and images.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_output_path(data_root: Path, target_rel: str, output_subdir: str, suffix: str) -> tuple[Path, str]:
    """Place the synthetic input beside the target's split when present."""

    target_path = data_root / target_rel
    if target_path.parent.name.lower() == "clear":
        output_dir = target_path.parent.parent / output_subdir
    else:
        output_dir = target_path.parent / output_subdir
    source_path = output_dir / synthetic_name(target_path.name, suffix)
    return source_path, source_path.relative_to(data_root).as_posix()


def generate_one(
    data_root: Path,
    target_rel: str,
    domain: str,
    split: str,
    output_subdir: str,
    seed: int,
    suffix: str,
    kernels: tuple[int, ...],
    overwrite: bool,
) -> tuple[tuple[str, str], bool, dict[str, Any]]:
    target_path = data_root / target_rel
    image = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read clear image: {target_path}")
    source_path, source_rel = source_output_path(data_root, target_rel, output_subdir, suffix)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    rng = stable_rng(seed, f"{domain}:{split}:{target_rel}:{suffix}:linetrace-v2-ms")
    degraded, params = synthesize_linetrace_blur(image, rng, kernels)
    wrote = False
    if overwrite or not source_path.exists():
        if not cv2.imwrite(str(source_path), degraded):
            raise RuntimeError(f"Failed to write synthetic input: {source_path}")
        wrote = True
    record = {
        "target": target_rel,
        "source": source_rel,
        "domain": domain,
        "split": split,
        "suffix": suffix,
        **params,
    }
    return (target_rel, source_rel), wrote, record


def generate_split(
    data_root: Path,
    samples: list[tuple[str, str, str]],
    output_subdir: str,
    seed: int,
    train: bool,
    overwrite: bool,
) -> tuple[list[tuple[str, str]], int, int, list[dict[str, Any]]]:
    pairs: list[tuple[str, str]] = []
    records: list[dict[str, Any]] = []
    generated = skipped = 0
    variants = TRAIN_VARIANTS if train else (("v2", ONE_VARIANT_KERNELS),)
    split_name = samples[0][2] if samples else "empty"
    for target_rel, domain, split in tqdm(samples, desc=f"Multi-source SynBlur-v2 {split_name}"):
        for suffix, kernels in variants:
            pair, wrote, record = generate_one(
                data_root, target_rel, domain, split, output_subdir, seed, suffix, tuple(kernels), overwrite
            )
            pairs.append(pair)
            records.append(record)
            generated += int(wrote)
            skipped += int(not wrote)
    return pairs, generated, skipped, records


def rx_splits(data_root: Path, seed: int) -> dict[str, list[tuple[str, str, str]]]:
    clear_dir = data_root / "RX" / "train" / "clear"
    files = sorted(path for path in clear_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if len(files) != 500:
        raise RuntimeError(f"Expected exactly 500 RX clear images in {clear_dir}, found {len(files)}")
    rng = random.Random(seed)
    rng.shuffle(files)
    named = [(path.relative_to(data_root).as_posix(), "RX") for path in files]
    split_sizes = {"train": 360, "val": 40, "test": 100}
    cursor = 0
    result: dict[str, list[tuple[str, str, str]]] = {}
    for split, size in split_sizes.items():
        chosen = named[cursor : cursor + size]
        cursor += size
        result[split] = [(target, domain, split) for target, domain in chosen]
    assert cursor == len(named)
    # The shipped RX/test/clear folder is a duplicate of RX/train/clear.  Guard
    # against an accidental future source/test overlap if that assumption changes.
    duplicate_dir = data_root / "RX" / "test" / "clear"
    if duplicate_dir.is_dir():
        for target_rel, _domain, _split in result["test"][:3]:
            duplicate = duplicate_dir / Path(target_rel).name
            original = data_root / target_rel
            if duplicate.exists() and sha256(original) != sha256(duplicate):
                raise RuntimeError("RX train/test folders are no longer duplicated; re-audit the RX split before running.")
    return result


def external_samples(data_root: Path, domain: str) -> list[tuple[str, str, str]]:
    directory = data_root / domain
    # RXHF is standardized before this script runs.  Prefer the normalized
    # test/clear structure; TJML remains supported in its existing flat form.
    standardized_dir = directory / "test" / "clear"
    source_dir = standardized_dir if standardized_dir.is_dir() else directory
    files = sorted(path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise RuntimeError(f"No clear images found for external domain: {source_dir}")
    if domain == "RXHF" and len(files) != 50:
        raise RuntimeError(
            f"RXHF must contain exactly 50 standardized external images in {standardized_dir}; found {len(files)}"
        )
    return [(path.relative_to(data_root).as_posix(), domain, "external_test") for path in files]


def base_samples(base_manifest_dir: Path, split: str) -> list[tuple[str, str, str]]:
    filename = "train.tsv" if split == "train" else "val.tsv" if split == "val" else None
    if filename is None:
        raise ValueError(split)
    samples: list[tuple[str, str, str]] = []
    for target_rel, _source_rel in read_manifest(base_manifest_dir / filename):
        domain = Path(target_rel).parts[0]
        if domain not in {"DB", "XJ", "SD"}:
            raise RuntimeError(f"Unexpected base source domain: {target_rel}")
        samples.append((target_rel.replace("\\", "/"), domain, split))
    return samples


def base_test_samples(base_manifest_dir: Path, domain: str) -> list[tuple[str, str, str]]:
    return [(target.replace("\\", "/"), domain, "test") for target, _source in read_manifest(base_manifest_dir / f"test_{domain}.tsv")]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def refresh_external_domains(args: argparse.Namespace, data_root: Path, manifest_dir: Path) -> None:
    """Refresh a changed external test set without touching source-domain data."""

    metadata_path = manifest_dir / "metadata.json"
    records_path = manifest_dir / "degradation_params.jsonl"
    if not metadata_path.is_file() or not records_path.is_file():
        raise RuntimeError("--refresh-external requires an already generated multi-source manifest directory.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = set(args.refresh_external)
    retained_records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("domain") not in selected
    ]
    new_records: list[dict[str, Any]] = []
    for domain in args.refresh_external:
        samples = external_samples(data_root, domain)
        pairs, generated, skipped, records = generate_split(
            data_root, samples, args.output_subdir, args.seed, train=False, overwrite=True
        )
        write_manifest(manifest_dir / f"test_{domain}.tsv", pairs)
        new_records.extend(records)
        metadata["counts"][domain] = len(pairs)
        metadata["generated_counts"][f"test_{domain}"] = generated
        metadata["skipped_counts"][f"test_{domain}"] = skipped
    write_jsonl(records_path, [*retained_records, *new_records])
    metadata["external_image_size"] = [320, 320]
    metadata["last_external_refresh"] = list(args.refresh_external)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Refreshed external domains: {', '.join(args.refresh_external)}")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    base_manifest_dir = Path(args.base_manifest_dir)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_external:
        refresh_external_domains(args, data_root, manifest_dir)
        return

    rx = rx_splits(data_root, args.seed)
    train_samples = [*base_samples(base_manifest_dir, "train"), *rx["train"]]
    val_samples = [*base_samples(base_manifest_dir, "val"), *rx["val"]]
    test_samples: dict[str, list[tuple[str, str, str]]] = {
        domain: base_test_samples(base_manifest_dir, domain) for domain in ("DB", "XJ", "SD")
    }
    test_samples["RX"] = rx["test"]
    test_samples.update({domain: external_samples(data_root, domain) for domain in EXTERNAL_DOMAINS})

    all_records: list[dict[str, Any]] = []
    generated_counts: dict[str, int] = {}
    skipped_counts: dict[str, int] = {}
    train_pairs, generated, skipped, records = generate_split(
        data_root, train_samples, args.output_subdir, args.seed, train=True, overwrite=args.overwrite
    )
    random.Random(args.seed).shuffle(train_pairs)
    write_manifest(manifest_dir / "train.tsv", train_pairs)
    all_records.extend(records)
    generated_counts["train"] = generated
    skipped_counts["train"] = skipped

    val_pairs, generated, skipped, records = generate_split(
        data_root, val_samples, args.output_subdir, args.seed, train=False, overwrite=args.overwrite
    )
    random.Random(args.seed + 1).shuffle(val_pairs)
    write_manifest(manifest_dir / "val.tsv", val_pairs)
    all_records.extend(records)
    generated_counts["val"] = generated
    skipped_counts["val"] = skipped

    test_counts: dict[str, int] = {}
    for domain, samples in test_samples.items():
        pairs, generated, skipped, records = generate_split(
            data_root, samples, args.output_subdir, args.seed, train=False, overwrite=args.overwrite
        )
        write_manifest(manifest_dir / f"test_{domain}.tsv", pairs)
        all_records.extend(records)
        test_counts[domain] = len(pairs)
        generated_counts[f"test_{domain}"] = generated
        skipped_counts[f"test_{domain}"] = skipped

    write_jsonl(manifest_dir / "degradation_params.jsonl", all_records)
    metadata = {
        "name": "MultiSource-Controlled-LineTrace-SynBlur-v2",
        "seed": args.seed,
        "base_manifest_dir": str(base_manifest_dir),
        "output_subdir": args.output_subdir,
        "primary_degradation": "line-evidence-weighted, three-horizontal-stripe non-uniform motion blur",
        "source_domains": list(SOURCE_DOMAINS),
        "external_domains": list(EXTERNAL_DOMAINS),
        "source_split_policy": {
            "DB/XJ/SD": "preserved base 360/40/100 clear-image split",
            "RX": "deterministic 360/40/100 split by clear-image identity before degradation",
        },
        "train_policy": "Every source-domain train clear image receives three V2 variants.",
        "validation_and_internal_test_policy": "Every source-domain val/test clear image receives one deterministic V2 input.",
        "external_test_policy": "RXHF and TJML each receive one deterministic V2 input per clear image and never enter training, validation, selection, or degradation tuning.",
        "no_leakage_rule": "Clear-image identities are disjoint across RX train/val/test; external clear images are test-only.",
        "counts": {"train": len(train_pairs), "val": len(val_pairs), **test_counts},
        "generated_counts": generated_counts,
        "skipped_counts": skipped_counts,
        "degradation_params": str(manifest_dir / "degradation_params.jsonl"),
    }
    (manifest_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote multi-source LineTrace SynBlur-v2 manifests to {manifest_dir}")
    print(json.dumps(metadata["counts"], ensure_ascii=False))
    print("generated:", json.dumps(generated_counts, ensure_ascii=False))
    print("skipped:", json.dumps(skipped_counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
