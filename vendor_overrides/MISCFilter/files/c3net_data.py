"""Paired data utilities for the DB/XJ/SD/RX industrial stitch datasets."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}
DEFAULT_DOMAINS = ("DB", "XJ", "SD", "RX")


def _image_files(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def _pairs(data_root: Path, domain: str, split: str) -> list[tuple[str, str]]:
    split_root = data_root / domain / split
    inputs = _image_files(split_root / "Imgs1")
    targets = _image_files(split_root / "clear")
    common = sorted(inputs.keys() & targets.keys())
    if not common:
        raise RuntimeError(f"No paired Imgs1/clear images under {split_root}")

    return [
        (
            f"{domain}/{split}/clear/{targets[name].name}",
            f"{domain}/{split}/Imgs1/{inputs[name].name}",
        )
        for name in common
    ]


def _write_manifest(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{target}\t{source}\n" for target, source in pairs),
        encoding="utf-8",
    )


def prepare_manifests(
    data_root: str | Path,
    manifest_dir: str | Path,
    domains: list[str] | tuple[str, ...],
    val_ratio: float,
    seed: int,
) -> dict[str, Path]:
    """Create deterministic, domain-stratified train/validation/test manifests."""
    data_root = Path(data_root)
    manifest_dir = Path(manifest_dir)
    manifests: dict[str, Path] = {}
    train_pairs: list[tuple[str, str]] = []
    val_pairs: list[tuple[str, str]] = []

    for domain in domains:
        pairs = _pairs(data_root, domain, "train")
        rng = random.Random(f"{seed}:{domain}")
        rng.shuffle(pairs)
        val_count = max(1, round(len(pairs) * val_ratio))
        val_pairs.extend(pairs[:val_count])
        train_pairs.extend(pairs[val_count:])

        test_path = manifest_dir / f"test_{domain}.tsv"
        _write_manifest(test_path, _pairs(data_root, domain, "test"))
        manifests[f"test_{domain}"] = test_path

    rng = random.Random(seed)
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    train_path = manifest_dir / "train.tsv"
    val_path = manifest_dir / "val.tsv"
    _write_manifest(train_path, train_pairs)
    _write_manifest(val_path, val_pairs)
    manifests["train"] = train_path
    manifests["val"] = val_path
    return manifests


def read_manifest(path: str | Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        target, source = line.split("\t")
        pairs.append((target, source))
    if not pairs:
        raise RuntimeError(f"Manifest is empty: {path}")
    return pairs


def _to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _pad_to_size(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, height, width = image.shape
    pad_w = max(0, patch_size - width)
    pad_h = max(0, patch_size - height)
    if not pad_w and not pad_h:
        return image

    # Reflection padding requires pad < input size. Replicate is a safe fallback
    # for unusually small images.
    mode = "reflect" if pad_w < width and pad_h < height else "replicate"
    return F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode=mode).squeeze(0)


class PairedManifestDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        manifest: str | Path,
        patch_size: int | None,
        augment: bool,
    ) -> None:
        self.data_root = Path(data_root)
        self.pairs = read_manifest(manifest)
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        target_rel, source_rel = self.pairs[index]
        target = Image.open(self.data_root / target_rel).convert("RGB")
        source = Image.open(self.data_root / source_rel).convert("RGB")
        if target.size != source.size:
            raise RuntimeError(f"Unaligned pair: {target_rel} / {source_rel}")

        target = _to_tensor(target)
        source = _to_tensor(source)
        if self.patch_size is not None:
            target = _pad_to_size(target, self.patch_size)
            source = _pad_to_size(source, self.patch_size)
            _, height, width = target.shape
            if self.augment:
                top = random.randint(0, height - self.patch_size)
                left = random.randint(0, width - self.patch_size)
            else:
                top = (height - self.patch_size) // 2
                left = (width - self.patch_size) // 2
            target = target[:, top : top + self.patch_size, left : left + self.patch_size]
            source = source[:, top : top + self.patch_size, left : left + self.patch_size]

        if self.augment:
            if random.random() < 0.5:
                target, source = target.flip(2), source.flip(2)
            if random.random() < 0.5:
                target, source = target.flip(1), source.flip(1)
            # Preserve the predominant stitch direction: only use 180-degree rotation.
            if random.random() < 0.5:
                target, source = torch.rot90(target, 2, (1, 2)), torch.rot90(source, 2, (1, 2))
        name = source_rel.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return target, source, name


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3)).clamp_min(1e-10)
    return 10.0 * torch.log10(1.0 / mse)
