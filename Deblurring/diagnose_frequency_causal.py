import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data_RGB import get_test_data
from MPRNet import build_mprnet_from_config
from quality_fusion import load_yaml, model_config_from_training_yaml, resolve_path
from test_quality_fusion import (
    collect_reference_files,
    compute_center_roi_metrics,
    compute_image_metrics,
    image_to_float,
    tensor_to_float_image,
)


CONFIG_PATH = Path(__file__).resolve().parent.parent / "_archive" / "deblurring_nonpaper_yaml_2026-11-14" / "diagnosis_frequency_causal_m5v3.yml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="No-training causal scale diagnosis for an MPRNet frequency residual."
    )
    parser.add_argument("--config", default=str(CONFIG_PATH), type=str)
    parser.add_argument("--max_images", default=0, type=int)
    return parser.parse_args()


def scale_name(value):
    return f"{float(value):g}"


def mean_finite(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(values)) if values else None


def summarize(rows, scale, dataset=None):
    selected = [
        row
        for row in rows
        if row["scale"] == scale and (dataset is None or row["dataset"] == dataset)
    ]
    return {
        "dataset": dataset if dataset is not None else "overall",
        "scale": scale,
        "samples": len(selected),
        "psnr": mean_finite([row["psnr"] for row in selected]),
        "ssim": mean_finite([row["ssim"] for row in selected]),
        "mae": mean_finite([row["mae"] for row in selected]),
        "mse": mean_finite([row["mse"] for row in selected]),
        "roi_psnr": mean_finite([row["roi_psnr"] for row in selected]),
        "roi_ssim": mean_finite([row["roi_ssim"] for row in selected]),
        "roi_mae": mean_finite([row["roi_mae"] for row in selected]),
        "roi_mse": mean_finite([row["roi_mse"] for row in selected]),
        "psnr_vs_scale0": mean_finite([row["psnr_vs_scale0"] for row in selected]),
        "roi_psnr_vs_scale0": mean_finite(
            [row["roi_psnr_vs_scale0"] for row in selected]
        ),
        "improved_vs_scale0_rate": mean_finite(
            [row["improved_vs_scale0"] for row in selected]
        ),
        "roi_improved_vs_scale0_rate": mean_finite(
            [row["roi_improved_vs_scale0"] for row in selected]
        ),
        "residual_abs_mean": mean_finite(
            [row["residual_abs_mean"] for row in selected]
        ),
    }


def write_outputs(result_dir, rows, summaries, macro_summaries, metadata):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "per_image_scale_metrics.csv"
    fields = [
        "dataset",
        "filename",
        "scale",
        "psnr",
        "ssim",
        "mae",
        "mse",
        "roi_psnr",
        "roi_ssim",
        "roi_mae",
        "roi_mse",
        "psnr_vs_scale0",
        "roi_psnr_vs_scale0",
        "improved_vs_scale0",
        "roi_improved_vs_scale0",
        "residual_abs_mean",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = result_dir / "frequency_causal_summary.json"
    payload = dict(metadata)
    payload.update(
        {
            "summaries": summaries,
            "macro_domain_summaries": macro_summaries,
            "per_image_csv": str(csv_path),
        }
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return csv_path, summary_path


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    base_dir = cfg_path.parent

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(gpu) for gpu in cfg.get("GPU", [0])
    )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model_config_path = resolve_path(cfg["MODEL_CONFIG"], base_dir)
    weights_path = resolve_path(cfg["WEIGHTS"], base_dir)
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"M5-v3 weights not found: {weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("USE_AMP", True)) and device.type == "cuda"
    model = build_mprnet_from_config(
        model_config_from_training_yaml(model_config_path)
    )
    utils.load_checkpoint(model, weights_path, strict=True)
    model.to(device).eval()
    if not getattr(model, "frequency_refine_enabled", False):
        raise RuntimeError("Configured model does not enable frequency refinement.")

    captured = {}

    def capture_frequency_io(_module, inputs, output):
        captured["base"] = inputs[0].detach()
        captured["full"] = output.detach()

    hook = model.frequency_refine.register_forward_hook(capture_frequency_io)
    scales = [float(value) for value in cfg.get("SCALES", [0, 0.25, 0.5, 0.75, 1])]
    if 0.0 not in scales or 1.0 not in scales:
        raise ValueError("SCALES must include 0 and 1 for causal validation.")

    data_root = Path(resolve_path(cfg.get("DATA_ROOT", "H:/DATASET"), base_dir))
    datasets = cfg.get("DATASETS", ["DB", "XJ", "SD", "RX"])
    split = str(cfg.get("SPLIT", "test"))
    input_subdir = str(cfg.get("INPUT_SUBDIR", "Imgs1"))
    target_subdir = str(cfg.get("TARGET_SUBDIR", "clear"))
    roi_top = float(cfg.get("ROI_TOP_RATIO", 1.0 / 3.0))
    roi_bottom = float(cfg.get("ROI_BOTTOM_RATIO", 2.0 / 3.0))
    max_images = int(args.max_images or cfg.get("MAX_IMAGES", 0))
    num_workers = int(cfg.get("NUM_WORKERS", 0))
    rows = []

    try:
        with torch.no_grad():
            for dataset in datasets:
                input_dir = data_root / dataset / split / input_subdir
                target_dir = data_root / dataset / split / target_subdir
                if not input_dir.is_dir():
                    print(f"===> Skipping {dataset}: missing input directory {input_dir}")
                    continue
                references = collect_reference_files(target_dir)
                if not references:
                    print(f"===> Skipping {dataset}: missing references {target_dir}")
                    continue

                loader = DataLoader(
                    get_test_data(str(input_dir), img_options={}),
                    batch_size=1,
                    shuffle=False,
                    num_workers=num_workers,
                    drop_last=False,
                    pin_memory=device.type == "cuda",
                )
                dataset_rows = []
                for index, (input_tensor, filenames) in enumerate(
                    tqdm(loader, desc=dataset, dynamic_ncols=True)
                ):
                    if max_images > 0 and index >= max_images:
                        break
                    input_tensor = input_tensor.to(device, non_blocking=True)
                    height, width = input_tensor.shape[-2:]
                    padded_height = ((height + 7) // 8) * 8
                    padded_width = ((width + 7) // 8) * 8
                    pad_h = padded_height - height
                    pad_w = padded_width - width
                    padded = input_tensor
                    if pad_h > 0 or pad_w > 0:
                        padded = F.pad(padded, (0, pad_w, 0, pad_h), mode="reflect")

                    captured.clear()
                    with autocast(enabled=use_amp):
                        model(padded, return_intermediate=False)
                    if "base" not in captured or "full" not in captured:
                        raise RuntimeError("Frequency hook did not capture module input/output.")

                    base = captured["base"][:, :, :height, :width].float()
                    full = captured["full"][:, :, :height, :width].float()
                    residual = full - base
                    residual_abs_mean = float(residual.abs().mean().cpu())
                    filename = filenames[0]
                    reference_path = references.get(filename)
                    if reference_path is None:
                        continue
                    target = image_to_float(reference_path)

                    per_scale = {}
                    for scale in scales:
                        restored = torch.clamp(base + scale * residual, 0.0, 1.0)
                        restored_image = tensor_to_float_image(restored)
                        if target.shape != restored_image.shape:
                            target = image_to_float(
                                reference_path, expected_shape=restored_image.shape
                            )
                        metrics = compute_image_metrics(restored_image, target)
                        roi_metrics = compute_center_roi_metrics(
                            restored_image, target, roi_top, roi_bottom
                        )
                        per_scale[scale] = (metrics, roi_metrics)

                    base_metrics, base_roi_metrics = per_scale[0.0]
                    for scale in scales:
                        metrics, roi_metrics = per_scale[scale]
                        row = {
                            "dataset": dataset,
                            "filename": filename,
                            "scale": scale,
                            "psnr": metrics["psnr"],
                            "ssim": metrics["ssim"],
                            "mae": metrics["mae"],
                            "mse": metrics["mse"],
                            "roi_psnr": roi_metrics["psnr"],
                            "roi_ssim": roi_metrics["ssim"],
                            "roi_mae": roi_metrics["mae"],
                            "roi_mse": roi_metrics["mse"],
                            "psnr_vs_scale0": metrics["psnr"] - base_metrics["psnr"],
                            "roi_psnr_vs_scale0": (
                                roi_metrics["psnr"] - base_roi_metrics["psnr"]
                            ),
                            "improved_vs_scale0": float(metrics["mse"] < base_metrics["mse"]),
                            "roi_improved_vs_scale0": float(
                                roi_metrics["mse"] < base_roi_metrics["mse"]
                            ),
                            "residual_abs_mean": residual_abs_mean,
                        }
                        dataset_rows.append(row)
                        rows.append(row)

                for scale in scales:
                    summary = summarize(dataset_rows, scale, dataset)
                    print(
                        f"===> {dataset} lambda={scale_name(scale)}: "
                        f"PSNR={summary['psnr']:.4f}, "
                        f"ROI={summary['roi_psnr']:.4f}, "
                        f"dPSNR={summary['psnr_vs_scale0']:+.4f}, "
                        f"improved={summary['improved_vs_scale0_rate']:.1%}"
                    )
    finally:
        hook.remove()

    summaries = []
    for dataset in datasets:
        for scale in scales:
            summary = summarize(rows, scale, dataset)
            if summary["samples"]:
                summaries.append(summary)
    for scale in scales:
        summaries.append(summarize(rows, scale))

    macro_summaries = []
    for scale in scales:
        domain_summaries = [
            item
            for item in summaries
            if item["scale"] == scale and item["dataset"] not in {"overall"}
        ]
        macro_summaries.append(
            {
                "scale": scale,
                "domains": len(domain_summaries),
                "macro_psnr": mean_finite([item["psnr"] for item in domain_summaries]),
                "macro_ssim": mean_finite([item["ssim"] for item in domain_summaries]),
                "macro_roi_psnr": mean_finite(
                    [item["roi_psnr"] for item in domain_summaries]
                ),
                "macro_roi_ssim": mean_finite(
                    [item["roi_ssim"] for item in domain_summaries]
                ),
            }
        )

    result_dir = resolve_path(cfg["RESULT_DIR"], base_dir)
    csv_path, summary_path = write_outputs(
        result_dir,
        rows,
        summaries,
        macro_summaries,
        {
            "diagnostic_only": True,
            "training_performed": False,
            "weights": weights_path,
            "model_config": model_config_path,
            "formula": "clamp(pre_frequency_output + scale * frequency_residual, 0, 1)",
            "scales": scales,
            "data": {
                "root": str(data_root),
                "datasets": datasets,
                "split": split,
                "roi_top_ratio": roi_top,
                "roi_bottom_ratio": roi_bottom,
            },
            "test_set_warning": (
                "This run is a retrospective causal diagnosis. Do not select a final "
                "scale from test results; tune any deployable scale on source validation only."
            ),
        },
    )
    print(f"===> Saved causal per-image metrics: {csv_path}")
    print(f"===> Saved causal summary: {summary_path}")


if __name__ == "__main__":
    main()
