from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _read_manifest(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected GT<TAB>LQ, got {line!r}")
        pairs.append((fields[0], fields[1]))
    if not pairs:
        raise ValueError(f"Manifest is empty: {path}")
    return pairs


def _rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _pad_to_patch(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    pad_h = max(0, patch_size - height)
    pad_w = max(0, patch_size - width)
    if pad_h or pad_w:
        # Identical to the full MBDNet-P+ data pipeline: prefer reflection,
        # and use replication only when reflection is not geometrically valid.
        mode = "reflect" if pad_w < width and pad_h < height else "replicate"
        image = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode=mode).squeeze(0)
    return image


class PairedManifestDataset(Dataset):
    """GT/LQ paired dataset shared by every comparison model.

    Manifest columns follow the existing project convention: clear target first,
    degraded input second.  Paired geometry is always transformed identically.
    """

    def __init__(
        self,
        data_root: str | Path,
        manifest: str | Path,
        patch_size: int = 320,
        augment: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.manifest = Path(manifest)
        self.pairs = _read_manifest(self.manifest)
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        gt_rel, lq_rel = self.pairs[index]
        gt_path = self.data_root / Path(gt_rel)
        lq_path = self.data_root / Path(lq_rel)
        if not gt_path.is_file() or not lq_path.is_file():
            raise FileNotFoundError(f"Missing pair: GT={gt_path}, LQ={lq_path}")
        gt = _rgb_tensor(gt_path)
        lq = _rgb_tensor(lq_path)
        if gt.shape != lq.shape:
            raise ValueError(f"Pair shape mismatch: {gt_path} {gt.shape} != {lq_path} {lq.shape}")

        if self.patch_size > 0:
            gt = _pad_to_patch(gt, self.patch_size)
            lq = _pad_to_patch(lq, self.patch_size)
            height, width = gt.shape[-2:]
            if self.augment:
                top = random.randint(0, height - self.patch_size)
                left = random.randint(0, width - self.patch_size)
            else:
                top = max(0, (height - self.patch_size) // 2)
                left = max(0, (width - self.patch_size) // 2)
            gt = gt[:, top : top + self.patch_size, left : left + self.patch_size]
            lq = lq[:, top : top + self.patch_size, left : left + self.patch_size]

        if self.augment:
            if random.random() < 0.5:
                gt, lq = gt.flip(-1), lq.flip(-1)
            if random.random() < 0.5:
                gt, lq = gt.flip(-2), lq.flip(-2)
            if random.random() < 0.5:
                gt, lq = torch.rot90(gt, 2, (-2, -1)), torch.rot90(lq, 2, (-2, -1))

        return gt.contiguous(), lq.contiguous(), Path(lq_rel).stem


def manifest_for_domain(manifest_dir: str | Path, domain: str) -> Path:
    path = Path(manifest_dir) / f"test_{domain}.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing domain manifest: {path}")
    return path
