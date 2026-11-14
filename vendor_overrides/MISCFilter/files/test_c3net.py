"""Evaluate a trained MISCFilter checkpoint on DB/XJ/SD/RX test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from c3net_checkpoint import load_checkpoint_any
from c3net_data import DEFAULT_DOMAINS, PairedManifestDataset, prepare_manifests, psnr
from models.MISCFilterNet import MISCKernelNet
from models.layers import window_partitionx, window_reversex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test MISCFilter on industrial stitch images")
    parser.add_argument("--data-root", default="H:/DATASET")
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--manifest-dir", default="manifests/c3net_all")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="results/C3NetAll_MISCFilters")
    parser.add_argument("--window-size", type=int, default=320)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0, help="Nonzero value limits each domain for a smoke test")
    return parser.parse_args()


def load_model(path: str) -> MISCKernelNet:
    checkpoint = load_checkpoint_any(path)
    state = {key.removeprefix("module."): value for key, value in checkpoint["state_dict"].items()}
    model = MISCKernelNet().cuda().eval()
    model.load_state_dict(state)
    return model


@torch.no_grad()
def evaluate_domain(
    model: MISCKernelNet,
    loader: DataLoader,
    output_dir: Path,
    window_size: int,
    max_images: int,
) -> float:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores: list[torch.Tensor] = []
    for image_index, (target, source, names) in enumerate(loader):
        target, source = target.cuda(non_blocking=True), source.cuda(non_blocking=True)
        _, _, height, width = source.shape
        windows, batch_list = window_partitionx(source, window_size)
        restored, _ = model(windows)
        prediction = window_reversex(restored[0], window_size, height, width, batch_list).clamp(0, 1)
        scores.append(psnr(prediction, target))
        image = (prediction[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
        Image.fromarray(image).save(output_dir / f"{names[0]}.png")
        if max_images and image_index + 1 >= max_images:
            break
    return torch.cat(scores).mean().item()


def main() -> None:
    args = parse_args()
    manifests = prepare_manifests(args.data_root, args.manifest_dir, args.domains, 0.1, 42)
    model = load_model(args.checkpoint)
    output_dir = Path(args.output_dir)
    summary: dict[str, float] = {}
    for domain in args.domains:
        dataset = PairedManifestDataset(args.data_root, manifests[f"test_{domain}"], None, False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
        score = evaluate_domain(model, loader, output_dir / domain, args.window_size, args.max_images)
        summary[domain] = score
        print(f"{domain}: PSNR={score:.4f}")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
