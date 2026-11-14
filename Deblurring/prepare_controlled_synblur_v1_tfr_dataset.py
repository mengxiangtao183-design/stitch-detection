"""Prepare a TFR/MPRNet-compatible copy of Controlled-SynBlur-v1.

The Deblurring dataloader pairs files by identical filename stem.  The controlled
manifest pairs look like:

    input:  img277_syn3.png
    target: img277.png

This script creates a lightweight mirror dataset where the target clear image is
linked/copied using the synthetic input filename, e.g.:

    input/img277_syn3.png
    clear/img277_syn3.png

No original dataset files are modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Controlled-SynBlur-v1 for Deblurring/TFR")
    parser.add_argument("--data-root", default="H:/DATASET")
    parser.add_argument("--manifest-dir", default="H:/C3Netqing/MISCFilter/manifests/c3net_controlled_synblur_v1")
    parser.add_argument("--output-root", default="H:/DATASET/ControlledSynBlurV1_TFR")
    parser.add_argument("--input-subdir", default="input")
    parser.add_argument("--target-subdir", default="clear")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of hard-linking where possible")
    return parser.parse_args()


def read_manifest(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        target, source = line.split("\t")[:2]
        pairs.append((target, source))
    return pairs


def link_or_copy(source: Path, destination: Path, force_copy: bool = False) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "exists"
    if not force_copy:
        try:
            destination.hardlink_to(source)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copy"


def domain_from_rel(path: str) -> str:
    return Path(path.replace("\\", "/")).parts[0]


def output_split(manifest_name: str) -> str:
    if manifest_name == "train.tsv":
        return "train"
    if manifest_name == "val.tsv":
        return "val"
    if manifest_name.startswith("test_"):
        return "test"
    raise ValueError(f"Unexpected manifest name: {manifest_name}")


def prepare_manifest(
    data_root: Path,
    output_root: Path,
    manifest_dir: Path,
    manifest_name: str,
    input_subdir: str,
    target_subdir: str,
    force_copy: bool,
) -> dict[str, int]:
    counts = {"pairs": 0, "hardlink": 0, "copy": 0, "exists": 0}
    split = output_split(manifest_name)
    for target_rel, source_rel in read_manifest(manifest_dir / manifest_name):
        domain = domain_from_rel(target_rel)
        source_path = data_root / source_rel
        target_path = data_root / target_rel
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing input: {source_path}")
        if not target_path.is_file():
            raise FileNotFoundError(f"Missing target: {target_path}")

        synthetic_name = source_path.name
        destination_input = output_root / domain / split / input_subdir / synthetic_name
        destination_target = output_root / domain / split / target_subdir / synthetic_name
        counts[link_or_copy(source_path, destination_input, force_copy)] += 1
        counts[link_or_copy(target_path, destination_target, force_copy)] += 1
        counts["pairs"] += 1
    return counts


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    manifest_dir = Path(args.manifest_dir)
    output_root = Path(args.output_root)
    manifest_names = [
        "train.tsv",
        "val.tsv",
        "test_DB.tsv",
        "test_XJ.tsv",
        "test_SD.tsv",
        "test_RX.tsv",
    ]
    summary = {
        name: prepare_manifest(
            data_root=data_root,
            output_root=output_root,
            manifest_dir=manifest_dir,
            manifest_name=name,
            input_subdir=args.input_subdir,
            target_subdir=args.target_subdir,
            force_copy=args.copy,
        )
        for name in manifest_names
    }
    metadata = {
        "name": "ControlledSynBlurV1_TFR",
        "source_manifest_dir": str(manifest_dir),
        "output_root": str(output_root),
        "input_subdir": args.input_subdir,
        "target_subdir": args.target_subdir,
        "pairing_policy": "Synthetic input and clear target are mirrored with identical filenames for Deblurring stem matching.",
        "summary": summary,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
