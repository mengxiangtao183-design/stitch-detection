"""Create an External-RX manifest variant whose final test inputs use Imgs2.

Training and validation manifests are copied from the canonical External-RX
protocol, so the only change is the test input folder:

    test_*.tsv: clear -> Imgs2

RX remains an external test-only domain.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}
TEST_DOMAINS = ("DB", "XJ", "SD", "RX")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create External-RX Imgs2 test manifests")
    parser.add_argument("--data-root", default="H:/DATASET")
    parser.add_argument("--base-manifest-dir", default="H:/C3Netqing/MISCFilter/manifests/c3net_external_rx")
    parser.add_argument("--manifest-dir", default="H:/C3Netqing/MISCFilter/manifests/c3net_external_rx_img2test")
    parser.add_argument("--input-folder", default="Imgs2")
    return parser.parse_args()


def image_files(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def pairs(data_root: Path, domain: str, split: str, input_folder: str) -> list[tuple[str, str]]:
    split_root = data_root / domain / split
    inputs = image_files(split_root / input_folder)
    targets = image_files(split_root / "clear")
    names = sorted(inputs.keys() & targets.keys())
    missing_inputs = sorted(set(targets) - set(inputs))
    missing_targets = sorted(set(inputs) - set(targets))
    if missing_inputs or missing_targets:
        raise RuntimeError(
            f"Unpaired {domain}/{split}: "
            f"missing {input_folder} for {len(missing_inputs)} clear images, "
            f"missing clear for {len(missing_targets)} {input_folder} images"
        )
    if not names:
        raise RuntimeError(f"No paired {input_folder}/clear images under {split_root}")
    return [
        (
            f"{domain}/{split}/clear/{targets[name].name}",
            f"{domain}/{split}/{input_folder}/{inputs[name].name}",
        )
        for name in names
    ]


def write_manifest(path: Path, items: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{target}\t{source}\n" for target, source in items), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    base_manifest_dir = Path(args.base_manifest_dir)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for name in ("train.tsv", "val.tsv"):
        shutil.copyfile(base_manifest_dir / name, manifest_dir / name)

    counts: dict[str, int] = {}
    for domain in TEST_DOMAINS:
        domain_pairs = pairs(data_root, domain, "test", args.input_folder)
        write_manifest(manifest_dir / f"test_{domain}.tsv", domain_pairs)
        counts[domain] = len(domain_pairs)

    metadata = {
        "name": "External-RX-Imgs2Test",
        "base_manifest_dir": str(base_manifest_dir),
        "input_folder": args.input_folder,
        "train_val_policy": "train.tsv and val.tsv are copied from canonical External-RX and still use Imgs1.",
        "test_policy": "test_DB/XJ/SD/RX use clear as target and Imgs2 as input.",
        "rx_policy": "RX remains external test only.",
        "test_counts": counts,
    }
    (manifest_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote External-RX Imgs2-test manifests to {manifest_dir}")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
