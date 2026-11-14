from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data_protocol import PairedManifestDataset, manifest_for_domain
from metric_protocol import middle_third, psnr, ssim
from model_factory import available_models, build_model


ROOT = Path(__file__).resolve().parent
REGISTRY = json.loads((ROOT / "model_registry.json").read_text(encoding="utf-8"))
DEFAULT_DOMAINS = ["DB", "XJ", "SD", "RX", "RXHF", "TJML"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair MultiSource-v2 deblurring comparison evaluation")
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=REGISTRY["protocol"]["data_root"])
    parser.add_argument("--manifest-dir", default=REGISTRY["protocol"]["manifest_dir"])
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--window-size", type=int, default=320)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("model_name") not in (None, args.model):
        raise ValueError(f"Checkpoint model mismatch: {checkpoint.get('model_name')} != {args.model}")
    model = build_model(args.model).cuda().eval()
    state = checkpoint.get("state_dict", checkpoint.get("params", checkpoint))
    state = {str(key).removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def window_starts(length: int, window_size: int) -> list[int]:
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, window_size))
    if starts[-1] != length - window_size:
        starts.append(length - window_size)
    return starts


@torch.no_grad()
def infer_tiled(model, source: torch.Tensor, window_size: int, amp: bool) -> torch.Tensor:
    _, _, height, width = source.shape
    if height <= window_size and width <= window_size:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            return model(source)
    output = torch.zeros_like(source)
    weight = torch.zeros_like(source)
    for top in window_starts(height, window_size):
        for left in window_starts(width, window_size):
            patch = source[:, :, top : top + window_size, left : left + window_size]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                restored = model(patch).float()
            output[:, :, top : top + window_size, left : left + window_size] += restored
            weight[:, :, top : top + window_size, left : left + window_size] += 1
    return output / weight.clamp_min(1)


@torch.no_grad()
def evaluate_domain(model, args: argparse.Namespace, domain: str, destination: Path) -> dict:
    dataset = PairedManifestDataset(args.data_root, manifest_for_domain(args.manifest_dir, domain), patch_size=0)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    full_psnr, full_ssim, mid_psnr, mid_ssim, timings = [], [], [], [], []
    image_dir = destination / domain
    if args.save_images:
        image_dir.mkdir(parents=True, exist_ok=True)
    for index, (target, source, names) in enumerate(loader, 1):
        target = target.cuda(non_blocking=True)
        source = source.cuda(non_blocking=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = infer_tiled(model, source, args.window_size, args.amp).float().clamp(0, 1)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        full_psnr.append(psnr(prediction, target))
        full_ssim.append(ssim(prediction, target))
        prediction_mid, target_mid = middle_third(prediction), middle_third(target)
        mid_psnr.append(psnr(prediction_mid, target_mid))
        mid_ssim.append(ssim(prediction_mid, target_mid))
        if args.save_images:
            image = (prediction[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(image).save(image_dir / f"{names[0]}.png")
        if args.max_images and index >= args.max_images:
            break
    return {
        "images": len(timings),
        "psnr_full": torch.cat(full_psnr).mean().item(),
        "ssim_full": torch.cat(full_ssim).mean().item(),
        "psnr_mid": torch.cat(mid_psnr).mean().item(),
        "ssim_mid": torch.cat(mid_ssim).mean().item(),
        "inference_ms": 1000.0 * sum(timings) / len(timings),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "results" / f"{args.model}_multisource_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args)
    summary = {
        "model": args.model,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        # Keep the same metric and timing contract used by the MBDNet-P+
        # evaluation script: RGB [0,1], full image plus fixed middle third,
        # per-image mean, and CUDA-synchronized FP32 timing.
        "protocol": {
            "range": "RGB [0,1]",
            "full": "entire aligned prediction/GT image",
            "mid": "fixed middle horizontal third: rows [H//3, 2H//3)",
            "ssim": "11x11 Gaussian window, sigma=1.5, RGB channel mean",
            "aggregation": "per-image metric, then mean",
            "inference_precision": "FP16 autocast" if args.amp else "FP32",
            "memory_format": "contiguous",
            "cuda_synchronized_timing": True,
            "window_size": args.window_size,
            "domains": args.domains,
        },
        "domains": {},
    }
    for domain in args.domains:
        result = evaluate_domain(model, args, domain, output_dir / "images")
        summary["domains"][domain] = result
        print(json.dumps({"domain": domain, **result}), flush=True)
    domains = list(summary["domains"].values())
    summary["macro"] = {
        key: float(np.mean([domain[key] for domain in domains]))
        for key in ("psnr_full", "ssim_full", "psnr_mid", "ssim_mid", "inference_ms")
    }
    total_images = sum(int(domain["images"]) for domain in domains)
    summary["weighted"] = {
        key: float(sum(domain[key] * int(domain["images"]) for domain in domains) / total_images)
        for key in ("psnr_full", "ssim_full", "psnr_mid", "ssim_mid", "inference_ms")
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"macro": summary["macro"], "weighted": summary["weighted"]}), flush=True)


if __name__ == "__main__":
    main()
    # On this Windows + CUDA stack, a few third-party architectures complete
    # evaluation successfully but may fault while Python tears down native
    # CUDA objects.  At this point metrics, images and summary.json are fully
    # written and CUDA work is synchronized, so exit directly to return a
    # truthful success status to the batch PowerShell runner.
    torch.cuda.synchronize()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
