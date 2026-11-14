import argparse
import csv
import json
import math
import os
from collections import OrderedDict
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


CONFIG_PATH = Path(__file__).with_name("diagnosis_frequency_hybrid_m2_m5v3.yml")
VARIANTS = (
    "m2_shared_frequency_off",
    "m2_shared_m5_frequency_on",
    "m5_shared_frequency_off",
    "m5_full",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="No-training M2/M5-v3 shared-backbone versus frequency-head diagnosis."
    )
    parser.add_argument("--config", default=str(CONFIG_PATH), type=str)
    parser.add_argument("--max_images", default=0, type=int)
    return parser.parse_args()


def checkpoint_state(path):
    checkpoint = utils.load_checkpoint_file(path)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    return OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in state.items()
    )


def build_hybrid(m5_config, m5_weights, m2_weights):
    model = build_mprnet_from_config(model_config_from_training_yaml(m5_config))
    m5_state = checkpoint_state(m5_weights)
    m2_state = checkpoint_state(m2_weights)
    model.load_state_dict(m5_state, strict=True)

    hybrid_state = model.state_dict()
    copied = []
    missing_shared = []
    for key in hybrid_state:
        if key.startswith("frequency_refine."):
            continue
        if key not in m2_state or m2_state[key].shape != hybrid_state[key].shape:
            missing_shared.append(key)
            continue
        hybrid_state[key] = m2_state[key].clone()
        copied.append(key)
    if missing_shared:
        raise RuntimeError(
            "M2 checkpoint cannot define the complete shared M5 backbone: "
            + ", ".join(missing_shared[:10])
        )
    model.load_state_dict(hybrid_state, strict=True)

    current = model.state_dict()
    mismatched = [
        key for key in copied if not torch.equal(current[key].cpu(), m2_state[key].cpu())
    ]
    if mismatched:
        raise RuntimeError(f"Hybrid shared-weight verification failed: {mismatched[:10]}")
    return model, len(copied)


def attach_frequency_hook(model, storage):
    def hook(_module, inputs, output):
        storage["base"] = inputs[0].detach()
        storage["full"] = output.detach()

    return model.frequency_refine.register_forward_hook(hook)


def mean_finite(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(values)) if values else None


def summarize(rows, variant, dataset=None):
    selected = [
        row
        for row in rows
        if row["variant"] == variant
        and (dataset is None or row["dataset"] == dataset)
    ]
    return {
        "dataset": dataset if dataset is not None else "overall",
        "variant": variant,
        "samples": len(selected),
        "psnr": mean_finite([row["psnr"] for row in selected]),
        "ssim": mean_finite([row["ssim"] for row in selected]),
        "roi_psnr": mean_finite([row["roi_psnr"] for row in selected]),
        "roi_ssim": mean_finite([row["roi_ssim"] for row in selected]),
        "psnr_vs_m2": mean_finite([row["psnr_vs_m2"] for row in selected]),
        "roi_psnr_vs_m2": mean_finite(
            [row["roi_psnr_vs_m2"] for row in selected]
        ),
        "improved_vs_m2_rate": mean_finite(
            [row["improved_vs_m2"] for row in selected]
        ),
    }


def write_results(result_dir, rows, summaries, metadata):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "per_image_hybrid_metrics.csv"
    fields = [
        "dataset",
        "filename",
        "variant",
        "psnr",
        "ssim",
        "roi_psnr",
        "roi_ssim",
        "psnr_vs_m2",
        "roi_psnr_vs_m2",
        "improved_vs_m2",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = result_dir / "frequency_hybrid_summary.json"
    payload = dict(metadata)
    payload["summaries"] = summaries
    payload["per_image_csv"] = str(csv_path)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("USE_AMP", True)) and device.type == "cuda"

    m5_config = resolve_path(cfg["M5_CONFIG"], base_dir)
    m5_weights = resolve_path(cfg["M5_WEIGHTS"], base_dir)
    m2_weights = resolve_path(cfg["M2_WEIGHTS"], base_dir)
    for path in (m5_config, m5_weights, m2_weights):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    native = build_mprnet_from_config(model_config_from_training_yaml(m5_config))
    utils.load_checkpoint(native, m5_weights, strict=True)
    hybrid, copied_shared_tensors = build_hybrid(m5_config, m5_weights, m2_weights)
    native.to(device).eval()
    hybrid.to(device).eval()

    native_capture = {}
    hybrid_capture = {}
    native_hook = attach_frequency_hook(native, native_capture)
    hybrid_hook = attach_frequency_hook(hybrid, hybrid_capture)

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
                references = collect_reference_files(target_dir)
                if not input_dir.is_dir() or not references:
                    print(f"===> Skipping {dataset}: missing input/reference data")
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
                    pad_h = ((height + 7) // 8) * 8 - height
                    pad_w = ((width + 7) // 8) * 8 - width
                    padded = input_tensor
                    if pad_h > 0 or pad_w > 0:
                        padded = F.pad(padded, (0, pad_w, 0, pad_h), mode="reflect")

                    native_capture.clear()
                    hybrid_capture.clear()
                    with autocast(enabled=use_amp):
                        native(padded, return_intermediate=False)
                        hybrid(padded, return_intermediate=False)
                    for capture in (native_capture, hybrid_capture):
                        if "base" not in capture or "full" not in capture:
                            raise RuntimeError("Frequency hook failed to capture output.")

                    tensors = {
                        "m2_shared_frequency_off": hybrid_capture["base"],
                        "m2_shared_m5_frequency_on": hybrid_capture["full"],
                        "m5_shared_frequency_off": native_capture["base"],
                        "m5_full": native_capture["full"],
                    }
                    images = {
                        name: tensor_to_float_image(
                            torch.clamp(tensor[:, :, :height, :width].float(), 0.0, 1.0)
                        )
                        for name, tensor in tensors.items()
                    }
                    filename = filenames[0]
                    reference_path = references.get(filename)
                    if reference_path is None:
                        continue
                    target = image_to_float(
                        reference_path,
                        expected_shape=images["m2_shared_frequency_off"].shape,
                    )
                    per_variant = {}
                    for name, image in images.items():
                        per_variant[name] = (
                            compute_image_metrics(image, target),
                            compute_center_roi_metrics(
                                image, target, roi_top, roi_bottom
                            ),
                        )
                    m2_metrics, m2_roi = per_variant["m2_shared_frequency_off"]
                    for name in VARIANTS:
                        metrics, roi_metrics = per_variant[name]
                        row = {
                            "dataset": dataset,
                            "filename": filename,
                            "variant": name,
                            "psnr": metrics["psnr"],
                            "ssim": metrics["ssim"],
                            "roi_psnr": roi_metrics["psnr"],
                            "roi_ssim": roi_metrics["ssim"],
                            "psnr_vs_m2": metrics["psnr"] - m2_metrics["psnr"],
                            "roi_psnr_vs_m2": roi_metrics["psnr"] - m2_roi["psnr"],
                            "improved_vs_m2": float(metrics["mse"] < m2_metrics["mse"]),
                        }
                        dataset_rows.append(row)
                        rows.append(row)

                for name in VARIANTS:
                    summary = summarize(dataset_rows, name, dataset)
                    print(
                        f"===> {dataset} {name}: PSNR={summary['psnr']:.4f}, "
                        f"ROI={summary['roi_psnr']:.4f}, "
                        f"dM2={summary['psnr_vs_m2']:+.4f}, "
                        f"improved={summary['improved_vs_m2_rate']:.1%}"
                    )
    finally:
        native_hook.remove()
        hybrid_hook.remove()

    summaries = []
    for dataset in datasets:
        for name in VARIANTS:
            summary = summarize(rows, name, dataset)
            if summary["samples"]:
                summaries.append(summary)
    for name in VARIANTS:
        summaries.append(summarize(rows, name))

    csv_path, summary_path = write_results(
        resolve_path(cfg["RESULT_DIR"], base_dir),
        rows,
        summaries,
        {
            "diagnostic_only": True,
            "training_performed": False,
            "m2_weights": m2_weights,
            "m5_weights": m5_weights,
            "m5_config": m5_config,
            "copied_and_verified_shared_tensors": copied_shared_tensors,
            "variant_definitions": {
                "m2_shared_frequency_off": "M2 shared weights; M5 frequency head bypassed.",
                "m2_shared_m5_frequency_on": "M2 shared weights plus trained M5-v3 frequency head.",
                "m5_shared_frequency_off": "M5-v3 shared weights; frequency head bypassed.",
                "m5_full": "Unmodified M5-v3 checkpoint.",
            },
            "test_set_warning": (
                "Retrospective causal diagnosis only; do not tune a final architecture "
                "or threshold on these test results."
            ),
        },
    )
    print(f"===> Saved hybrid per-image metrics: {csv_path}")
    print(f"===> Saved hybrid summary: {summary_path}")


if __name__ == "__main__":
    main()
