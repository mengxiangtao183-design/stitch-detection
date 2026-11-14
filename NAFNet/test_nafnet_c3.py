"""Evaluate a trained NAFNet checkpoint on DB/XJ/SD/RX test manifests."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from basicsr.models.archs.NAFNet_arch import NAFNet
from basicsr.models.archs.MBDNetP_arch import MBDNetP, MBDNetPPlus, MBDNetPPlusR
from c3net_checkpoint import load_checkpoint_any
from c3net_data import DEFAULT_DOMAINS, PairedManifestDataset, middle_third, psnr, ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test NAFNet on industrial stitch images")
    parser.add_argument("--data-root", default="H:/DATASET")
    parser.add_argument("--manifest-dir", default="H:/C3Netqing/MISCFilter/manifests/c3net_all")
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="results/C3NetAll_NAFNet_width32")
    parser.add_argument("--window-size", type=int, default=320)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--amp-inference",
        action="store_true",
        help="Run CUDA inference under FP16 autocast; metrics are still calculated in FP32.",
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Use NHWC memory layout for convolution acceleration on Tensor Core GPUs.",
    )
    parser.add_argument(
        "--deploy-reparam",
        action="store_true",
        help="Fuse MBDNet-P+R training branches into single depthwise 3x3 convolutions before inference.",
    )
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--enc-blks", nargs="+", type=int, default=[1, 1, 1, 28])
    parser.add_argument("--middle-blks", type=int, default=1)
    parser.add_argument("--dec-blks", nargs="+", type=int, default=[1, 1, 1, 1])
    parser.add_argument(
        "--arch",
        choices=("nafnet", "mbdnet-p", "mbdnet-p-plus", "mbdnet-p-plus-r"),
        default="nafnet",
    )
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


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.arch == "mbdnet-p":
        return MBDNetP(img_channel=3, width=args.width)
    if args.arch == "mbdnet-p-plus":
        return MBDNetPPlus(img_channel=3, width=args.width)
    if args.arch == "mbdnet-p-plus-r":
        return MBDNetPPlusR(img_channel=3, width=args.width)
    return NAFNet(
        img_channel=3,
        width=args.width,
        enc_blk_nums=args.enc_blks,
        middle_blk_num=args.middle_blks,
        dec_blk_nums=args.dec_blks,
    )


def load_model(args: argparse.Namespace) -> NAFNet:
    checkpoint = load_checkpoint_any(args.checkpoint)
    state = {key.removeprefix("module."): value for key, value in checkpoint["state_dict"].items()}
    model = build_model(args).cuda().eval()
    model.load_state_dict(state)
    if args.deploy_reparam:
        if not hasattr(model, "switch_to_deploy"):
            raise ValueError("--deploy-reparam is supported only by re-parameterizable architectures")
        model.switch_to_deploy()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def window_starts(length: int, window_size: int) -> list[int]:
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, window_size))
    if starts[-1] != length - window_size:
        starts.append(length - window_size)
    return starts


@torch.no_grad()
def infer_tiled(
    model: NAFNet,
    source: torch.Tensor,
    window_size: int,
    amp_inference: bool,
) -> torch.Tensor:
    _, _, height, width = source.shape
    if height <= window_size and width <= window_size:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_inference):
            return model(source)

    output = torch.zeros_like(source)
    weight = torch.zeros_like(source)
    for top in window_starts(height, window_size):
        for left in window_starts(width, window_size):
            patch = source[:, :, top : top + window_size, left : left + window_size]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_inference):
                restored = model(patch)
            restored = restored.float()
            output[:, :, top : top + window_size, left : left + window_size] += restored
            weight[:, :, top : top + window_size, left : left + window_size] += 1
    return output / weight.clamp_min(1)


@torch.no_grad()
def evaluate_domain(
    model: NAFNet,
    loader: DataLoader,
    output_dir: Path,
    window_size: int,
    max_images: int,
    amp_inference: bool,
    channels_last: bool,
) -> dict[str, float | int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_psnr_scores: list[torch.Tensor] = []
    full_ssim_scores: list[torch.Tensor] = []
    mid_psnr_scores: list[torch.Tensor] = []
    mid_ssim_scores: list[torch.Tensor] = []
    timings: list[float] = []
    for image_index, (target, source, names) in enumerate(loader):
        target = target.cuda(non_blocking=True)
        source = source.cuda(non_blocking=True)
        if channels_last:
            source = source.contiguous(memory_format=torch.channels_last)
        torch.cuda.synchronize()
        started = time.perf_counter()
        # Convert outputs to FP32 before PSNR/SSIM and serialization, so the
        # precision option changes execution speed rather than metric math.
        prediction = infer_tiled(model, source, window_size, amp_inference).float().clamp(0, 1)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)

        prediction_mid = middle_third(prediction)
        target_mid = middle_third(target)
        full_psnr_scores.append(psnr(prediction, target))
        full_ssim_scores.append(ssim(prediction, target))
        mid_psnr_scores.append(psnr(prediction_mid, target_mid))
        mid_ssim_scores.append(ssim(prediction_mid, target_mid))
        image = (prediction[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(image).save(output_dir / f"{names[0]}.png")
        if max_images and image_index + 1 >= max_images:
            break
    return {
        "images": len(timings),
        "psnr_full": torch.cat(full_psnr_scores).mean().item(),
        "ssim_full": torch.cat(full_ssim_scores).mean().item(),
        "psnr_mid": torch.cat(mid_psnr_scores).mean().item(),
        "ssim_mid": torch.cat(mid_ssim_scores).mean().item(),
        "inference_ms": 1000.0 * sum(timings) / len(timings),
    }


def main() -> None:
    args = parse_args()
    model = load_model(args)
    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)
    domains: dict[str, dict[str, float | int]] = {}
    for domain in args.domains:
        manifest = manifest_dir / f"test_{domain}.tsv"
        dataset = PairedManifestDataset(args.data_root, manifest, None, False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
        metrics = evaluate_domain(
            model,
            loader,
            output_dir / domain,
            args.window_size,
            args.max_images,
            args.amp_inference,
            args.channels_last,
        )
        domains[domain] = metrics
        print(
            f"{domain}: Full PSNR={metrics['psnr_full']:.4f}, SSIM={metrics['ssim_full']:.4f}; "
            f"Mid PSNR={metrics['psnr_mid']:.4f}, SSIM={metrics['ssim_mid']:.4f}; "
            f"images={metrics['images']}, inference_ms={metrics['inference_ms']:.2f}",
            flush=True,
        )

    total_images = sum(int(item["images"]) for item in domains.values())
    macro = {
        key: sum(float(item[key]) for item in domains.values()) / len(domains)
        for key in ("psnr_full", "ssim_full", "psnr_mid", "ssim_mid", "inference_ms")
    }
    weighted = {
        key: sum(float(item[key]) * int(item["images"]) for item in domains.values()) / total_images
        for key in ("psnr_full", "ssim_full", "psnr_mid", "ssim_mid", "inference_ms")
    }
    summary = {
        "protocol": {
            "range": "RGB [0,1]",
            "full": "entire aligned prediction/GT image",
            "mid": "fixed middle horizontal third: rows [H//3, 2H//3)",
            "ssim": "11x11 Gaussian window, sigma=1.5, RGB channel mean",
            "aggregation": "per-image metric, then mean",
            "inference_precision": "FP16 autocast" if args.amp_inference else "FP32",
            "memory_format": "channels_last" if args.channels_last else "contiguous",
            "reparameterized_deploy": bool(args.deploy_reparam),
        },
        "domains": domains,
        "macro": macro,
        "weighted": weighted,
    }
    print(
        f"Weighted: Full PSNR={weighted['psnr_full']:.4f}, SSIM={weighted['ssim_full']:.4f}; "
        f"Mid PSNR={weighted['psnr_mid']:.4f}, SSIM={weighted['ssim_mid']:.4f}",
        flush=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
