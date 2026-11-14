"""Create 320x320 edge targets without overwriting the original annotations."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(r"H:\DATASET")
DATASETS = ("DB", "XJ", "SD")
SIZE = (320, 320)


def resize_edge(path: Path) -> np.ndarray:
    """Match the trainer's nearest-neighbor edge alignment exactly."""
    edge = np.asarray(Image.open(path).convert("L"), dtype=np.uint8).copy()
    tensor = torch.from_numpy(edge).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=SIZE, mode="nearest")
    return resized.squeeze(0).squeeze(0).to(torch.uint8).numpy()


def main() -> None:
    total = 0
    for dataset in DATASETS:
        source = ROOT / dataset / "train" / "Edges"
        destination = ROOT / dataset / "train" / "Edges_320"
        if not source.is_dir():
            raise FileNotFoundError(source)
        destination.mkdir(exist_ok=True)
        files = sorted(p for p in source.iterdir() if p.suffix.lower() == ".png")
        for source_path in files:
            target_path = destination / source_path.name
            Image.fromarray(resize_edge(source_path), mode="L").save(target_path)
            total += 1
        print(f"{dataset}: {len(files)} -> {destination}")
    print(f"Generated {total} edge maps at {SIZE[0]}x{SIZE[1]}.")


if __name__ == "__main__":
    main()
