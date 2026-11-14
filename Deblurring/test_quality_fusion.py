import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_RGB import get_test_data
from quality_fusion import (
    QualityFusionGate,
    build_frozen_expert,
    forward_padded,
    load_yaml,
    resolve_path,
)


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CONFIG_PATH = Path(__file__).resolve().parent.parent / "_archive" / "deblurring_nonpaper_yaml_2026-11-14" / "training_quality_fusion_fast100.yml"


def parse_args():
    parser = argparse.ArgumentParser(description="Test quality-aware expert fusion.")
    parser.add_argument("--config", default=str(CONFIG_PATH), type=str)
    parser.add_argument("--weights", default="", type=str, help="Optional gate checkpoint override.")
    parser.add_argument("--max_images", default=0, type=int)
    return parser.parse_args()


def collect_reference_files(reference_dir):
    reference_dir = Path(reference_dir)
    if not reference_dir.exists():
        return {}
    return {
        path.stem: path
        for path in sorted(reference_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    }


def image_to_float(path, expected_shape=None):
    image = Image.open(path).convert("RGB")
    if expected_shape is not None:
        h, w = expected_shape[:2]
        if image.size != (w, h):
            image = image.resize((w, h), Image.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def tensor_to_float_image(tensor):
    array = tensor.detach().clamp(0, 1).cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    return np.transpose(array, (1, 2, 0)).astype(np.float32)


def save_float_image(array, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(array, 0.0, 1.0)
    Image.fromarray((array * 255.0 + 0.5).astype(np.uint8)).save(path)


def compute_image_metrics(pred, target):
    pred = np.clip(pred.astype(np.float32), 0.0, 1.0)
    target = np.clip(target.astype(np.float32), 0.0, 1.0)
    diff = pred - target
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
    ssim = float(structural_similarity(pred, target, channel_axis=2, data_range=1.0))
    return {"psnr": psnr, "ssim": ssim, "mae": mae, "mse": mse}


def compute_center_roi_metrics(pred, target, top_ratio, bottom_ratio):
    height = pred.shape[0]
    top = int(round(height * float(top_ratio)))
    bottom = int(round(height * float(bottom_ratio)))
    top = max(0, min(height - 1, top))
    bottom = max(top + 1, min(height, bottom))
    return compute_image_metrics(pred[top:bottom], target[top:bottom])


def mean_or_none(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(values)) if values else None


def std_or_none(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.std(values)) if values else None


def summarize_rows(rows, dataset=None):
    selected = [row for row in rows if dataset is None or row["dataset"] == dataset]
    keys = [
        "restored_psnr",
        "restored_ssim",
        "restored_mae",
        "restored_mse",
        "restored_roi_psnr",
        "restored_roi_ssim",
        "restored_roi_mae",
        "restored_roi_mse",
        "input_psnr",
        "input_ssim",
        "input_mae",
        "input_mse",
        "input_roi_psnr",
        "input_roi_ssim",
        "input_roi_mae",
        "input_roi_mse",
        "expert_a_psnr",
        "expert_a_ssim",
        "expert_a_roi_psnr",
        "expert_a_roi_ssim",
        "expert_b_psnr",
        "expert_b_ssim",
        "expert_b_roi_psnr",
        "expert_b_roi_ssim",
        "oracle_psnr",
        "oracle_ssim",
        "oracle_roi_psnr",
        "oracle_roi_ssim",
        "oracle_selected_b",
    ]
    summary = {
        "dataset": dataset if dataset is not None else "overall",
        "samples": len(selected),
        "alpha_mean": mean_or_none([row["alpha"] for row in selected]),
        "alpha_std": std_or_none([row["alpha"] for row in selected]),
    }
    for key in keys:
        summary[key] = mean_or_none([row[key] for row in selected])
    return summary


def write_metrics(metrics_dir, rows, summaries, cfg, gate_weights):
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "per_image_metrics.csv"
    fieldnames = [
        "dataset",
        "filename",
        "alpha",
        "restored_psnr",
        "restored_ssim",
        "restored_mae",
        "restored_mse",
        "restored_roi_psnr",
        "restored_roi_ssim",
        "restored_roi_mae",
        "restored_roi_mse",
        "input_psnr",
        "input_ssim",
        "input_mae",
        "input_mse",
        "input_roi_psnr",
        "input_roi_ssim",
        "input_roi_mae",
        "input_roi_mse",
        "expert_a_psnr",
        "expert_a_ssim",
        "expert_a_roi_psnr",
        "expert_a_roi_ssim",
        "expert_b_psnr",
        "expert_b_ssim",
        "expert_b_roi_psnr",
        "expert_b_roi_ssim",
        "oracle_psnr",
        "oracle_ssim",
        "oracle_roi_psnr",
        "oracle_roi_ssim",
        "oracle_selected_b",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = metrics_dir / "eval_summary.json"
    payload = {
        "gate_weights": str(gate_weights),
        "expert_a": cfg["EXPERTS"]["A"],
        "expert_b": cfg["EXPERTS"]["B"],
        "roi": {
            "top_ratio": cfg["FUSION"].get("ROI_TOP_RATIO", 1.0 / 3.0),
            "bottom_ratio": cfg["FUSION"].get("ROI_BOTTOM_RATIO", 2.0 / 3.0),
            "description": "Fixed centre-band metrics; full-image metrics remain primary.",
        },
        "summaries": summaries,
        "per_image_csv": str(csv_path),
        "oracle_note": "Oracle is an analysis upper bound selecting the better expert per image by PSNR.",
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return csv_path, summary_path


def load_gate(weights, fusion_cfg, device):
    gate = QualityFusionGate(
        in_channels=int(fusion_cfg.get("IN_CHANNELS", 12)),
        hidden=tuple(fusion_cfg.get("HIDDEN", [32, 48, 64])),
        mlp_hidden=int(fusion_cfg.get("MLP_HIDDEN", 16)),
        use_descriptors=bool(fusion_cfg.get("USE_DESCRIPTORS", False)),
        descriptor_hidden=int(fusion_cfg.get("DESCRIPTOR_HIDDEN", 16)),
        roi_top_ratio=float(fusion_cfg.get("ROI_TOP_RATIO", 1.0 / 3.0)),
        roi_bottom_ratio=float(fusion_cfg.get("ROI_BOTTOM_RATIO", 2.0 / 3.0)),
    )
    checkpoint = torch.load(weights, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    cleaned = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    gate.load_state_dict(cleaned)
    gate.to(device)
    gate.eval()
    return gate


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    base_dir = cfg_path.parent

    gpus = ",".join(str(gpu) for gpu in cfg.get("GPU", [0]))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    data_cfg = cfg["DATA"]
    fusion_cfg = cfg["FUSION"]
    test_cfg = cfg.get("TEST", {})
    expert_a_cfg = cfg["EXPERTS"]["A"]
    expert_b_cfg = cfg["EXPERTS"]["B"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(test_cfg.get("USE_AMP", True)) and device.type == "cuda"

    gate_weights = args.weights or fusion_cfg.get("WEIGHTS", "")
    if not gate_weights:
        gate_weights = str(Path(resolve_path(fusion_cfg["MODEL_DIR"], base_dir)) / "model_best.pth")
    gate_weights = resolve_path(gate_weights, base_dir)
    if not Path(gate_weights).exists():
        raise FileNotFoundError(f"Fusion gate weights not found: {gate_weights}")

    expert_a = build_frozen_expert(
        resolve_path(expert_a_cfg["CONFIG"], base_dir),
        resolve_path(expert_a_cfg["WEIGHTS"], base_dir),
        device,
    )
    expert_b = build_frozen_expert(
        resolve_path(expert_b_cfg["CONFIG"], base_dir),
        resolve_path(expert_b_cfg["WEIGHTS"], base_dir),
        device,
    )
    gate = load_gate(gate_weights, fusion_cfg, device)

    result_dir = Path(resolve_path(fusion_cfg["RESULT_DIR"], base_dir))
    metrics_rows = []
    summary_rows = []
    datasets = test_cfg.get("DATASETS", ["DB", "XJ", "SD", "RX"])
    input_root = Path(data_cfg.get("ROOT", "H:/DATASET"))
    split = test_cfg.get("SPLIT", "test")
    input_subdir = data_cfg.get("INPUT_SUBDIR", "Imgs1")
    target_subdir = data_cfg.get("TARGET_SUBDIR", "clear")
    roi_top_ratio = float(fusion_cfg.get("ROI_TOP_RATIO", 1.0 / 3.0))
    roi_bottom_ratio = float(fusion_cfg.get("ROI_BOTTOM_RATIO", 2.0 / 3.0))
    inference_temperature = float(fusion_cfg.get("INFERENCE_ALPHA_TEMPERATURE", 1.0))

    with torch.no_grad():
        for dataset in datasets:
            rgb_dir_test = input_root / dataset / split / input_subdir
            clear_dir_test = input_root / dataset / split / target_subdir
            if not rgb_dir_test.exists():
                print(f"===> Skipping {dataset}: input directory not found: {rgb_dir_test}")
                continue
            reference_files = collect_reference_files(clear_dir_test)
            if not reference_files:
                print(f"===> Skipping metrics for {dataset}: reference directory missing or empty: {clear_dir_test}")

            test_dataset = get_test_data(str(rgb_dir_test), img_options={})
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=int(test_cfg.get("NUM_WORKERS", 0)),
                drop_last=False,
                pin_memory=device.type == "cuda",
            )
            dataset_result_dir = result_dir / dataset
            dataset_result_dir.mkdir(parents=True, exist_ok=True)
            dataset_rows = []

            print(f"===> Testing fusion {dataset}: {rgb_dir_test} -> {dataset_result_dir}")
            for index, data_test in enumerate(tqdm(test_loader, desc=dataset, dynamic_ncols=True)):
                max_images = int(args.max_images or test_cfg.get("MAX_IMAGES", 0))
                if max_images > 0 and index >= max_images:
                    break

                input_tensor = data_test[0].to(device, non_blocking=True)
                filename = data_test[1][0]
                out_a = forward_padded(expert_a, input_tensor, use_amp=use_amp)
                out_b = forward_padded(expert_b, input_tensor, use_amp=use_amp)
                fused, alpha = gate(
                    input_tensor,
                    out_a,
                    out_b,
                    temperature=inference_temperature,
                )
                fused = torch.clamp(fused, 0.0, 1.0)

                input_img = tensor_to_float_image(input_tensor)
                out_a_img = tensor_to_float_image(out_a)
                out_b_img = tensor_to_float_image(out_b)
                fused_img = tensor_to_float_image(fused)
                save_float_image(fused_img, dataset_result_dir / f"{filename}.png")

                reference_path = reference_files.get(filename)
                if reference_path is None:
                    continue

                target = image_to_float(reference_path, expected_shape=fused_img.shape)
                input_metrics = compute_image_metrics(input_img, target)
                input_roi_metrics = compute_center_roi_metrics(
                    input_img, target, roi_top_ratio, roi_bottom_ratio
                )
                fused_metrics = compute_image_metrics(fused_img, target)
                fused_roi_metrics = compute_center_roi_metrics(
                    fused_img, target, roi_top_ratio, roi_bottom_ratio
                )
                expert_a_metrics = compute_image_metrics(out_a_img, target)
                expert_a_roi_metrics = compute_center_roi_metrics(
                    out_a_img, target, roi_top_ratio, roi_bottom_ratio
                )
                expert_b_metrics = compute_image_metrics(out_b_img, target)
                expert_b_roi_metrics = compute_center_roi_metrics(
                    out_b_img, target, roi_top_ratio, roi_bottom_ratio
                )
                oracle_uses_b = float(expert_b_metrics["psnr"] > expert_a_metrics["psnr"])
                oracle_metrics = expert_b_metrics if oracle_uses_b else expert_a_metrics
                oracle_roi_metrics = expert_b_roi_metrics if oracle_uses_b else expert_a_roi_metrics

                row = {
                    "dataset": dataset,
                    "filename": filename,
                    "alpha": float(alpha.view(-1)[0].detach().cpu()),
                    "restored_psnr": fused_metrics["psnr"],
                    "restored_ssim": fused_metrics["ssim"],
                    "restored_mae": fused_metrics["mae"],
                    "restored_mse": fused_metrics["mse"],
                    "restored_roi_psnr": fused_roi_metrics["psnr"],
                    "restored_roi_ssim": fused_roi_metrics["ssim"],
                    "restored_roi_mae": fused_roi_metrics["mae"],
                    "restored_roi_mse": fused_roi_metrics["mse"],
                    "input_psnr": input_metrics["psnr"],
                    "input_ssim": input_metrics["ssim"],
                    "input_mae": input_metrics["mae"],
                    "input_mse": input_metrics["mse"],
                    "input_roi_psnr": input_roi_metrics["psnr"],
                    "input_roi_ssim": input_roi_metrics["ssim"],
                    "input_roi_mae": input_roi_metrics["mae"],
                    "input_roi_mse": input_roi_metrics["mse"],
                    "expert_a_psnr": expert_a_metrics["psnr"],
                    "expert_a_ssim": expert_a_metrics["ssim"],
                    "expert_a_roi_psnr": expert_a_roi_metrics["psnr"],
                    "expert_a_roi_ssim": expert_a_roi_metrics["ssim"],
                    "expert_b_psnr": expert_b_metrics["psnr"],
                    "expert_b_ssim": expert_b_metrics["ssim"],
                    "expert_b_roi_psnr": expert_b_roi_metrics["psnr"],
                    "expert_b_roi_ssim": expert_b_roi_metrics["ssim"],
                    "oracle_psnr": oracle_metrics["psnr"],
                    "oracle_ssim": oracle_metrics["ssim"],
                    "oracle_roi_psnr": oracle_roi_metrics["psnr"],
                    "oracle_roi_ssim": oracle_roi_metrics["ssim"],
                    "oracle_selected_b": oracle_uses_b,
                }
                dataset_rows.append(row)
                metrics_rows.append(row)

            if dataset_rows:
                summary = summarize_rows(dataset_rows, dataset=dataset)
                summary_rows.append(summary)
                print(
                    f"===> {dataset}: "
                    f"Fusion={summary['restored_psnr']:.4f}/{summary['restored_ssim']:.5f}, "
                    f"ROI={summary['restored_roi_psnr']:.4f}, "
                    f"A={summary['expert_a_psnr']:.4f}, "
                    f"B={summary['expert_b_psnr']:.4f}, "
                    f"Oracle={summary['oracle_psnr']:.4f}, "
                    f"alpha={summary['alpha_mean']:.3f}+/-{summary['alpha_std']:.3f}"
                )

    if metrics_rows:
        summary_rows.append(summarize_rows(metrics_rows))
        metrics_dir = result_dir / "metrics"
        csv_path, summary_path = write_metrics(metrics_dir, metrics_rows, summary_rows, cfg, gate_weights)
        print(f"===> Saved per-image metrics: {csv_path}")
        print(f"===> Saved metric summary: {summary_path}")


if __name__ == "__main__":
    main()
