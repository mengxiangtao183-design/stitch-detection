"""Restore one directory of Quxian motion-blurred images with frozen NAFNet.

This is an ablation inference entry point: no training, fine-tuning, or
threshold adjustment occurs here.  It saves images with their original names
and, when clear references are supplied, restoration PSNR/SSIM/MAE summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from skimage.metrics import structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAFNET_ROOT = PROJECT_ROOT / "NAFNet"
sys.path.insert(0, str(NAFNET_ROOT))

from basicsr.models.archs.MBDNetP_arch import MBDNetPPlus
from c3net_checkpoint import load_checkpoint_any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen NAFNet restoration for the Quxian deblurring ablation")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clear-reference-dir", type=Path, default=None)
    parser.add_argument("--arch", choices=("mbdnet-p-plus", "line-trace-mixer"), default="mbdnet-p-plus")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--enc-blks", nargs="+", type=int, default=[1, 1, 1, 28])
    parser.add_argument("--middle-blks", type=int, default=1)
    parser.add_argument("--dec-blks", nargs="+", type=int, default=[1, 1, 1, 1])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    checkpoint = load_checkpoint_any(args.checkpoint)
    state = {name.removeprefix("module."): value for name, value in checkpoint["state_dict"].items()}
    if args.arch == "mbdnet-p-plus":
        model = MBDNetPPlus(img_channel=3, width=args.width)
    else:
        from basicsr.models.archs.LineTraceMixerNAFNet_arch import LineTraceMixerNAFNet
        model = LineTraceMixerNAFNet(
            img_channel=3, width=args.width, enc_blk_nums=args.enc_blks,
            middle_blk_num=args.middle_blks, dec_blk_nums=args.dec_blks,
        )
    model = model.to(device).eval()
    model.load_state_dict(state, strict=True)
    return model


def tensor_from_image(path: Path, device: torch.device) -> tuple[torch.Tensor, tuple[int, int]]:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor, image.shape[:2]


@torch.no_grad()
def restore(model: torch.nn.Module, source: torch.Tensor, original_size: tuple[int, int]) -> torch.Tensor:
    height, width = original_size
    pad_h, pad_w = (-height) % 16, (-width) % 16
    padded = functional.pad(source, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else source
    restored = model(padded).clamp(0.0, 1.0)
    return restored[:, :, :height, :width]


def image_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = prediction.astype(np.float32) - reference.astype(np.float32)
    mse = float(np.mean(difference ** 2))
    return {
        "psnr": float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse)),
        "ssim": float(structural_similarity(prediction, reference, channel_axis=2, data_range=1.0)),
        "mae": float(np.mean(np.abs(difference))),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    files = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() == ".png")
    if not files:
        raise ValueError(f"no PNG inputs found in {args.input_dir}")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args, device)
    rows = []
    for path in files:
        source, original_size = tensor_from_image(path, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        restored = restore(model, source, original_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        restored_array = restored[0].permute(1, 2, 0).cpu().numpy()
        Image.fromarray((restored_array * 255.0).round().astype(np.uint8)).save(args.output_dir / path.name)
        row = {"filename": path.name, "inference_ms": round(1000.0 * (time.perf_counter() - started), 4)}
        if args.clear_reference_dir is not None:
            reference_path = args.clear_reference_dir / path.name
            if not reference_path.is_file():
                raise FileNotFoundError(reference_path)
            reference = np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.float32) / 255.0
            row.update(image_metrics(restored_array, reference))
        rows.append(row)
    with (args.output_dir / "restoration_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric_keys = [key for key in rows[0] if key != "filename"]
    summary = {
        "model": args.arch, "checkpoint": str(args.checkpoint), "input_images": len(files),
        "device": str(device), "metrics_mean": {key: float(np.mean([row[key] for row in rows])) for key in numeric_keys},
        "protocol": "Frozen inference on one synthetic-blur counterpart per clear image; no fine-tuning or parameter selection.",
    }
    (args.output_dir / "restoration_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
