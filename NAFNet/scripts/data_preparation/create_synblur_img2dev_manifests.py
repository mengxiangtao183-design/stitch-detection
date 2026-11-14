"""Create a quick-development protocol for Imgs2-targeted model search.

Protocol:
  - train.tsv: SynBlur-v1 synthetic training pairs from DB/XJ/SD train split.
  - val.tsv: held-out DB/XJ/SD val identities, but input folder switched to Imgs2.
  - test_*.tsv: DB/XJ/SD/RX test inputs switched to Imgs2.

This is meant for rapid innovation/module screening when the target test
distribution is Imgs2. RX remains external test only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TEST_DOMAINS = ("DB", "XJ", "SD", "RX")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create SynBlur-train + Imgs2-val/test manifests")
    parser.add_argument("--synblur-manifest-dir", default="H:/C3Netqing/MISCFilter/manifests/c3net_external_rx_synblur_v1_3x_splitval")
    parser.add_argument("--img2test-manifest-dir", default="H:/C3Netqing/MISCFilter/manifests/c3net_external_rx_img2test")
    parser.add_argument("--manifest-dir", default="H:/C3Netqing/MISCFilter/manifests/c3net_external_rx_synblur_train_img2dev")
    parser.add_argument("--input-folder", default="Imgs2")
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


def switch_input_folder(pairs: list[tuple[str, str]], input_folder: str) -> list[tuple[str, str]]:
    switched: list[tuple[str, str]] = []
    for target, source in pairs:
        parts = source.replace("\\", "/").split("/")
        if len(parts) < 4:
            raise ValueError(f"Unexpected source path: {source}")
        parts[-2] = input_folder
        switched.append((target, "/".join(parts)))
    return switched


def main() -> None:
    args = parse_args()
    synblur_manifest_dir = Path(args.synblur_manifest_dir)
    img2test_manifest_dir = Path(args.img2test_manifest_dir)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    train_pairs = read_manifest(synblur_manifest_dir / "train.tsv")
    # Use the canonical held-out identities from the Imgs2-test manifest's copied val.tsv,
    # then switch the input side from Imgs1 to Imgs2.
    val_pairs = switch_input_folder(read_manifest(img2test_manifest_dir / "val.tsv"), args.input_folder)

    write_manifest(manifest_dir / "train.tsv", train_pairs)
    write_manifest(manifest_dir / "val.tsv", val_pairs)

    test_counts: dict[str, int] = {}
    for domain in TEST_DOMAINS:
        pairs = read_manifest(img2test_manifest_dir / f"test_{domain}.tsv")
        write_manifest(manifest_dir / f"test_{domain}.tsv", pairs)
        test_counts[domain] = len(pairs)

    metadata = {
        "name": "SynBlurTrain-Imgs2Dev",
        "train_source": str(synblur_manifest_dir / "train.tsv"),
        "val_source": str(img2test_manifest_dir / "val.tsv"),
        "train_policy": "DB/XJ/SD SynBlur-v1 synthetic inputs are used for rapid module training.",
        "val_policy": "Held-out DB/XJ/SD val identities are evaluated with Imgs2 inputs for module selection.",
        "test_policy": "DB/XJ/SD/RX test inputs use Imgs2.",
        "rx_policy": "RX remains external test only.",
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_counts": test_counts,
    }
    (manifest_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote quick-dev manifests to {manifest_dir}")
    print(json.dumps({"train": len(train_pairs), "val": len(val_pairs), **test_counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
