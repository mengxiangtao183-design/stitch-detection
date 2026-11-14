"""Evaluate stitch-specific clarity metrics for restored image folders.

The metrics are auxiliary, task-oriented measurements for fine stitch traces.
They do not replace PSNR/SSIM; they complement them by focusing on the middle
stitch band, directional gradients, local contrast, and Laplacian sharpness.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from c3net_data import DEFAULT_DOMAINS, middle_third, psnr, ssim


DEFAULT_RUNS = {
    "A0_NAFNet": "ExternalRX_A0_NAFNet_Scratch30_S42_midroi",
    "A1_SPB": "ExternalRX_A1_STDR_SPB_Scratch30_S42_midroi",
    "A2_SPB_DFRB": "ExternalRX_A2_STDR_SPB_DFRB_Scratch30_S42_midroi",
    "A2Light_DFRB1": "ExternalRX_A2Light_STDR_SPB_DFRB1_Scratch30_S42_midroi",
    "A4_TSFD": "ExternalRX_A4_STDR_SPB_DFRB_TSFD_Scratch30_S42_midroi",
    "A5_DAME": "ExternalRX_A5_STDR_SPB_DFRB_DAME_Scratch30_S42_midroi",
    "A6_AFBG": "ExternalRX_A6_STDR_SPB_DFRB_AFBG_Scratch30_S42_midroi",
    "A7_CSDM": "ExternalRX_A7_STDR_SPB_DFRB_CSDM_Scratch30_S42_midroi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare stitch clarity metrics for External-RX result folders")
    parser.add_argument("--data-root", default="H:/DATASET")
    parser.add_argument("--manifest-dir", default="H:/C3Netqing/MISCFilter/manifests/c3net_external_rx")
    parser.add_argument("--results-root", default="H:/C3Netqing/NAFNet/results")
    parser.add_argument("--output-dir", default="H:/C3Netqing/NAFNet/results/stci_external_rx_compare")
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument(
        "--runs",
        nargs="*",
        default=[],
        help="Optional items as label=folder or folder. Defaults to known External-RX A0-A7 folders.",
    )
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.clamp(0, 1)


def luminance(image: torch.Tensor) -> torch.Tensor:
    return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]


def sobel_xy(gray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sobel_x = gray.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4.0
    sobel_y = sobel_x.t()
    weight_x = sobel_x.view(1, 1, 3, 3)
    weight_y = sobel_y.view(1, 1, 3, 3)
    return F.conv2d(gray, weight_x, padding=1), F.conv2d(gray, weight_y, padding=1)


def laplacian(gray: torch.Tensor) -> torch.Tensor:
    kernel = gray.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    return F.conv2d(gray, kernel.view(1, 1, 3, 3), padding=1)


def closeness_ratio(value: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    ratio = (value + eps) / (reference + eps)
    return torch.minimum(ratio, 1.0 / ratio).clamp(0.0, 1.0)


def directional_gradient_similarity(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_gray = luminance(prediction)
    target_gray = luminance(target)
    pred_x, pred_y = sobel_xy(pred_gray)
    target_x, target_y = sobel_xy(target_gray)

    jxx = target_x.square().mean()
    jyy = target_y.square().mean()
    jxy = (target_x * target_y).mean()
    theta = 0.5 * torch.atan2(2.0 * jxy, jxx - jyy + 1e-8)
    direction_x = torch.cos(theta)
    direction_y = torch.sin(theta)

    pred_projected = (direction_x * pred_x + direction_y * pred_y).abs()
    target_projected = (direction_x * target_x + direction_y * target_y).abs()
    error = (pred_projected - target_projected).abs().mean()
    reference = target_projected.abs().mean().clamp_min(1e-8)
    return (1.0 - error / reference).clamp(0.0, 1.0)


def local_contrast_recovery(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_gray = luminance(prediction)
    target_gray = luminance(target)
    pred_mean = F.avg_pool2d(pred_gray, kernel_size=15, stride=1, padding=7)
    target_mean = F.avg_pool2d(target_gray, kernel_size=15, stride=1, padding=7)
    pred_contrast = (pred_gray - pred_mean).abs().mean()
    target_contrast = (target_gray - target_mean).abs().mean()
    return closeness_ratio(pred_contrast, target_contrast)


def laplacian_variance_recovery(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_lap = laplacian(luminance(prediction))
    target_lap = laplacian(luminance(target))
    pred_var = pred_lap.var(unbiased=False)
    target_var = target_lap.var(unbiased=False)
    return closeness_ratio(pred_var, target_var)


def stitch_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_mid = middle_third(prediction)
    target_mid = middle_third(target)
    mid_ssim = ssim(pred_mid, target_mid).mean()
    dgs = directional_gradient_similarity(pred_mid, target_mid)
    lcr = local_contrast_recovery(pred_mid, target_mid)
    lvr = laplacian_variance_recovery(pred_mid, target_mid)
    stci = 0.40 * dgs + 0.25 * lcr + 0.15 * lvr + 0.20 * mid_ssim
    return {
        "psnr_full": psnr(prediction, target).mean().item(),
        "ssim_full": ssim(prediction, target).mean().item(),
        "psnr_mid": psnr(pred_mid, target_mid).mean().item(),
        "ssim_mid": mid_ssim.item(),
        "dgs": dgs.item(),
        "lcr": lcr.item(),
        "lvr": lvr.item(),
        "stci": stci.item(),
    }


def read_manifest(manifest: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        target_rel, source_rel = line.split("\t")[:2]
        pairs.append((target_rel, source_rel))
    return pairs


def parse_runs(items: list[str], results_root: Path) -> dict[str, Path]:
    if not items:
        return {
            label: results_root / folder
            for label, folder in DEFAULT_RUNS.items()
            if (results_root / folder).is_dir()
        }
    runs: dict[str, Path] = {}
    for item in items:
        if "=" in item:
            label, folder = item.split("=", 1)
        else:
            folder = item
            label = Path(item).name
        path = Path(folder)
        if not path.is_absolute():
            path = results_root / path
        runs[label] = path
    return runs


def mean_records(records: list[dict[str, float]]) -> dict[str, float]:
    keys = records[0].keys()
    return {key: float(sum(record[key] for record in records) / len(records)) for key in keys}


def evaluate_run(
    label: str,
    run_dir: Path,
    data_root: Path,
    manifest_dir: Path,
    domains: list[str],
    max_images: int,
) -> dict:
    domain_summary: dict[str, dict[str, float | int]] = {}
    for domain in domains:
        manifest = manifest_dir / f"test_{domain}.tsv"
        if not manifest.is_file():
            continue
        records: list[dict[str, float]] = []
        for index, (target_rel, _source_rel) in enumerate(read_manifest(manifest)):
            target_path = data_root / target_rel
            restored_path = run_dir / domain / Path(target_rel).name
            if not restored_path.is_file():
                raise FileNotFoundError(f"Missing restored image: {restored_path}")
            target = load_image(target_path)
            prediction = load_image(restored_path)
            if prediction.shape[-2:] != target.shape[-2:]:
                prediction = F.interpolate(prediction, size=target.shape[-2:], mode="bilinear", align_corners=False)
            records.append(stitch_metrics(prediction, target))
            if max_images and index + 1 >= max_images:
                break
        summary = mean_records(records)
        summary["images"] = len(records)
        domain_summary[domain] = summary

    metric_keys = [key for key in next(iter(domain_summary.values())).keys() if key != "images"]
    macro = {
        key: float(sum(float(domain_summary[domain][key]) for domain in domain_summary) / len(domain_summary))
        for key in metric_keys
    }
    total_images = sum(int(item["images"]) for item in domain_summary.values())
    weighted = {
        key: float(
            sum(float(item[key]) * int(item["images"]) for item in domain_summary.values()) / total_images
        )
        for key in metric_keys
    }
    return {"label": label, "path": str(run_dir), "domains": domain_summary, "macro": macro, "weighted": weighted}


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    manifest_dir = Path(args.manifest_dir)
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = parse_runs(args.runs, results_root)
    results = {
        label: evaluate_run(label, run_dir, data_root, manifest_dir, args.domains, args.max_images)
        for label, run_dir in runs.items()
    }
    payload = {
        "protocol": {
            "roi": "middle horizontal third",
            "dgs": "directional Sobel-gradient similarity in the stitch band, direction estimated from GT structure tensor",
            "lcr": "local contrast recovery closeness to GT in the stitch band",
            "lvr": "Laplacian variance recovery closeness to GT in the stitch band",
            "stci": "0.40*DGS + 0.25*LCR + 0.15*LVR + 0.20*MidSSIM",
            "note": "STCI is an auxiliary stitch-clarity metric and does not replace PSNR/SSIM.",
        },
        "runs": results,
    }
    (output_dir / "stci_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = []
    for label, result in results.items():
        macro = result["macro"]
        rows.append(
            {
                "run": label,
                "psnr_full": macro["psnr_full"],
                "ssim_full": macro["ssim_full"],
                "psnr_mid": macro["psnr_mid"],
                "ssim_mid": macro["ssim_mid"],
                "dgs": macro["dgs"],
                "lcr": macro["lcr"],
                "lvr": macro["lvr"],
                "stci": macro["stci"],
            }
        )
    rows.sort(key=lambda item: item["stci"], reverse=True)
    with (output_dir / "stci_macro.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print("Macro ranking by STCI:", flush=True)
    for row in rows:
        print(
            f"{row['run']}: STCI={row['stci']:.5f}, DGS={row['dgs']:.5f}, "
            f"LCR={row['lcr']:.5f}, LVR={row['lvr']:.5f}, "
            f"MidPSNR={row['psnr_mid']:.4f}, MidSSIM={row['ssim_mid']:.5f}",
            flush=True,
        )
    print(f"Saved: {output_dir / 'stci_summary.json'}", flush=True)
    print(f"Saved: {output_dir / 'stci_macro.tsv'}", flush=True)


if __name__ == "__main__":
    main()
