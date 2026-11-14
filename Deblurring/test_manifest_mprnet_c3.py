"""Evaluate canonical MPRNet under the shared MultiSource SynBlur-v2 protocol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import utils
from MPRNet import MPRNet


DEFAULT_DOMAINS = ("DB", "XJ", "SD", "RX", "RXHF", "TJML")


def read_pairs(manifest: Path, data_root: Path) -> list[tuple[Path, Path, str]]:
    pairs = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"Expected clear<TAB>blur at {manifest}:{line_number}")
        target_rel, source_rel = fields
        target = data_root / target_rel
        source = data_root / source_rel
        if not target.is_file() or not source.is_file():
            raise FileNotFoundError(f"Missing manifest pair: {target} / {source}")
        pairs.append((target, source, Path(source_rel).stem))
    if not pairs:
        raise ValueError(f"Empty manifest: {manifest}")
    return pairs


class ManifestTestDataset(Dataset):
    def __init__(self, manifest: Path, data_root: Path) -> None:
        self.pairs = read_pairs(manifest, data_root)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        target_path, source_path, name = self.pairs[index]
        target = np.asarray(Image.open(target_path).convert("RGB"), dtype=np.float32) / 255.0
        source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float32) / 255.0
        if target.shape != source.shape:
            raise RuntimeError(f"Unaligned test pair: {target_path} / {source_path}")
        target_tensor = torch.from_numpy(target).permute(2, 0, 1).contiguous()
        source_tensor = torch.from_numpy(source).permute(2, 0, 1).contiguous()
        return target_tensor, source_tensor, name


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction - target).square().mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return 10.0 * torch.log10(1.0 / mse)


def middle_third(image: torch.Tensor) -> torch.Tensor:
    height = image.shape[-2]
    return image[..., height // 3 : (2 * height) // 3, :]


def ssim(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"SSIM shape mismatch: {prediction.shape} != {target.shape}")
    coordinates = torch.arange(window_size, device=prediction.device, dtype=prediction.dtype)
    coordinates = coordinates - (window_size - 1) / 2
    kernel_1d = torch.exp(-coordinates.square() / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d).expand(prediction.shape[1], 1, window_size, window_size)

    def blur(image: torch.Tensor) -> torch.Tensor:
        return F.conv2d(image, kernel, groups=image.shape[1])

    mu_x, mu_y = blur(prediction), blur(target)
    sigma_x = blur(prediction.square()) - mu_x.square()
    sigma_y = blur(target.square()) - mu_y.square()
    sigma_xy = blur(prediction * target) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean(dim=(1, 2, 3))


def make_model() -> MPRNet:
    # Exactly the original MPRNet path: all optional project modules remain off.
    return MPRNet(
        n_feat=96,
        scale_unetfeats=48,
        scale_orsnetfeats=32,
        num_cab=8,
        edge_refine=False,
        frequency_refine=False,
        light_adapt=False,
        trajectory_lightweight=False,
        stage1_tristripe=False,
    )


@torch.no_grad()
def infer(model: MPRNet, source: torch.Tensor) -> torch.Tensor:
    _, _, height, width = source.shape
    padded_height = (height + 7) // 8 * 8
    padded_width = (width + 7) // 8 * 8
    if padded_height != height or padded_width != width:
        source = F.pad(source, (0, padded_width - width, 0, padded_height - height), mode="reflect")
    prediction = model(source)[0]
    return prediction[..., :height, :width]


@torch.no_grad()
def evaluate_domain(model: MPRNet, loader: DataLoader, output_dir: Path, max_images: int) -> dict[str, float | int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_psnr, full_ssim, mid_psnr, mid_ssim, timings = [], [], [], [], []
    for index, (target, source, names) in enumerate(loader):
        target = target.cuda(non_blocking=True)
        source = source.cuda(non_blocking=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = infer(model, source).float().clamp(0.0, 1.0)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        full_psnr.append(psnr(prediction, target))
        full_ssim.append(ssim(prediction, target))
        mid_psnr.append(psnr(middle_third(prediction), middle_third(target)))
        mid_ssim.append(ssim(middle_third(prediction), middle_third(target)))
        image = (prediction[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(image).save(output_dir / f"{names[0]}.png")
        if max_images and index + 1 >= max_images:
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
    parser = argparse.ArgumentParser(description="Evaluate original MPRNet using shared paired manifests")
    parser.add_argument("--data-root", default="H:/DATASET")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = make_model().cuda().eval()
    utils.load_checkpoint(model, str(checkpoint))
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Original MPRNet parameters: {model_parameters / 1e6:.3f}M", flush=True)

    manifest_dir, data_root, output_dir = Path(args.manifest_dir), Path(args.data_root), Path(args.output_dir)
    domains = {}
    for domain in args.domains:
        manifest = manifest_dir / f"test_{domain}.tsv"
        dataset = ManifestTestDataset(manifest, data_root)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
        metrics = evaluate_domain(model, loader, output_dir / domain, args.max_images)
        domains[domain] = metrics
        print(
            f"{domain}: Full PSNR={metrics['psnr_full']:.4f}, SSIM={metrics['ssim_full']:.4f}; "
            f"Mid PSNR={metrics['psnr_mid']:.4f}, SSIM={metrics['ssim_mid']:.4f}; "
            f"images={metrics['images']}, inference_ms={metrics['inference_ms']:.2f}",
            flush=True,
        )

    total_images = sum(int(metrics["images"]) for metrics in domains.values())
    metric_keys = ("psnr_full", "ssim_full", "psnr_mid", "ssim_mid", "inference_ms")
    macro = {key: sum(float(metrics[key]) for metrics in domains.values()) / len(domains) for key in metric_keys}
    weighted = {key: sum(float(metrics[key]) * int(metrics["images"]) for metrics in domains.values()) / total_images for key in metric_keys}
    summary = {
        "protocol": {
            "model": "original MPRNet (n_feat=96, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8)",
            "optional_modules": "all disabled",
            "range": "RGB [0,1]",
            "full": "entire aligned prediction/GT image",
            "mid": "fixed middle horizontal third: rows [H//3, 2H//3)",
            "ssim": "11x11 Gaussian window, sigma=1.5, RGB channel mean",
            "aggregation": "per-image metric, then mean",
            "inference_precision": "FP32",
        },
        "parameters": model_parameters,
        "domains": domains,
        "macro": macro,
        "weighted": weighted,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Weighted: Full PSNR={weighted['psnr_full']:.4f}, SSIM={weighted['ssim_full']:.4f}; "
        f"Mid PSNR={weighted['psnr_mid']:.4f}, SSIM={weighted['ssim_mid']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
