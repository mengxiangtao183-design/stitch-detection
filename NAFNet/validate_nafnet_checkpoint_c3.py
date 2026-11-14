"""Validation-only evaluation for C3Net deblurring checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from c3net_data import PairedManifestDataset, psnr, ssim
from test_nafnet_c3 import build_model
from c3net_checkpoint import load_checkpoint_any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on val.tsv only")
    parser.add_argument("--data-root", default="H:/DATASET")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", required=True)
    # Fields consumed by the shared model builder for other architectures.
    parser.add_argument("--enc-blks", nargs="+", type=int, default=[1, 1, 1, 28])
    parser.add_argument("--middle-blks", type=int, default=1)
    parser.add_argument("--dec-blks", nargs="+", type=int, default=[1, 1, 1, 1])
    parser.add_argument("--directional-stages", nargs="+", type=int, default=[3])
    parser.add_argument("--no-directional-middle", action="store_true")
    parser.add_argument("--direction-kernel", type=int, default=7)
    parser.add_argument("--sffr-decoder-index", type=int, default=1)
    parser.add_argument("--sffr-channels", type=int, default=32)
    parser.add_argument("--detail-width", type=int, default=32)
    parser.add_argument("--detail-blocks", type=int, default=3)
    parser.add_argument("--detail-limit", type=float, default=0.14)
    parser.add_argument("--stripe-token-count", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Validation requires CUDA")
    manifest = Path(args.manifest_dir) / "val.tsv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing validation manifest: {manifest}")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = load_checkpoint_any(checkpoint_path)
    state = {key.removeprefix("module."): value for key, value in checkpoint["state_dict"].items()}
    model = build_model(args).cuda().eval()
    model.load_state_dict(state, strict=True)
    dataset = PairedManifestDataset(args.data_root, manifest, patch_size=320, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    psnr_scores: list[torch.Tensor] = []
    ssim_scores: list[torch.Tensor] = []
    for target, source, _ in loader:
        target = target.cuda(non_blocking=True)
        source = source.cuda(non_blocking=True)
        prediction = model(source).clamp(0.0, 1.0)
        psnr_scores.append(psnr(prediction, target))
        ssim_scores.append(ssim(prediction, target))
    result = {
        "split": "val.tsv",
        "images": len(dataset),
        "psnr": torch.cat(psnr_scores).mean().item(),
        "ssim": torch.cat(ssim_scores).mean().item(),
        "checkpoint": str(checkpoint_path),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
