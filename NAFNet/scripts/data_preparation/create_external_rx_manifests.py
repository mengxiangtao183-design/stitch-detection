"""Create DB/XJ/SD training manifests with RX kept as an external test domain."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}
TRAIN_DOMAINS = ("DB", "XJ", "SD")
TEST_DOMAINS = ("DB", "XJ", "SD", "RX")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create External-RX C3Net manifests")
    parser.add_argument("--data-root", default="H:/DATASET")
    parser.add_argument("--manifest-dir", default="H:/C3Netqing/MISCFilter/manifests/c3net_external_rx")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def image_files(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def pairs(data_root: Path, domain: str, split: str) -> list[tuple[str, str]]:
    split_root = data_root / domain / split
    inputs = image_files(split_root / "Imgs1")
    targets = image_files(split_root / "clear")
    names = sorted(inputs.keys() & targets.keys())
    if not names:
        raise RuntimeError(f"No paired Imgs1/clear images under {split_root}")
    return [
        (
            f"{domain}/{split}/clear/{targets[name].name}",
            f"{domain}/{split}/Imgs1/{inputs[name].name}",
        )
        for name in names
    ]


def write_manifest(path: Path, items: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{target}\t{source}\n" for target, source in items), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    manifest_dir = Path(args.manifest_dir)
    train_pairs: list[tuple[str, str]] = []
    val_pairs: list[tuple[str, str]] = []

    for domain in TRAIN_DOMAINS:
        domain_pairs = pairs(data_root, domain, "train")
        rng = random.Random(f"{args.seed}:{domain}")
        rng.shuffle(domain_pairs)
        val_count = max(1, round(len(domain_pairs) * args.val_ratio))
        val_pairs.extend(domain_pairs[:val_count])
        train_pairs.extend(domain_pairs[val_count:])

    rng = random.Random(args.seed)
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    write_manifest(manifest_dir / "train.tsv", train_pairs)
    write_manifest(manifest_dir / "val.tsv", val_pairs)

    for domain in TEST_DOMAINS:
        write_manifest(manifest_dir / f"test_{domain}.tsv", pairs(data_root, domain, "test"))

    print(f"Wrote External-RX manifests to {manifest_dir}")
    print(f"train.tsv: {len(train_pairs)} images from DB/XJ/SD only")
    print(f"val.tsv: {len(val_pairs)} images from DB/XJ/SD only")
    for domain in TEST_DOMAINS:
        count = len((manifest_dir / f"test_{domain}.tsv").read_text(encoding="utf-8").splitlines())
        print(f"test_{domain}.tsv: {count} images")


if __name__ == "__main__":
    main()
