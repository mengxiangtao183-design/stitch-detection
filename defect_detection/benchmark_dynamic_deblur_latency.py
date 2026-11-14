"""Measure real per-image latency of the frozen sharpness router plus MBDNet-P+.

Timing starts before image decoding and ends when the routed RGB output is
available as a uint8 NumPy array. Model loading, calibration and warm-up are
excluded. Optionally, PNG writing is timed as a separate stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAFNET_ROOT = PROJECT_ROOT / "NAFNet"
sys.path.insert(0, str(NAFNET_ROOT))

from basicsr.models.archs.MBDNetP_arch import MBDNetPPlusR
from c3net_checkpoint import load_checkpoint_any
from test_nafnet_c3 import infer_tiled
from evaluate_dynamic_deblur_skip import FEATURE_NAMES, quality_features_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--router-summary", type=Path,
        default=Path("H:/C3Netqing/results/deblurring_ablation/quexian_synblur48_dynamic_skip/dynamic_skip_summary.json"),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=NAFNET_ROOT / "checkpoints/MultiSourceV2_MBDNetPPlusR12_S30_S42/model_best.pth",
    )
    parser.add_argument(
        "--clear-dir", type=Path,
        default=Path("H:/DATASET/quexian/dataset_96_320/inference_original48"),
    )
    parser.add_argument(
        "--blur-dir", type=Path,
        default=Path("H:/DATASET/quexian/dataset_96_320/synblur_original48_k3to15"),
    )
    parser.add_argument("--workload", choices=("mixed", "blurred", "clear"), default="mixed")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("H:/C3Netqing/results/deblurring_ablation/quexian_synblur48_dynamic_skip_latency"),
    )
    parser.add_argument("--window-size", type=int, default=320)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=0, help="0 measures the complete workload")
    parser.add_argument("--save-outputs", action="store_true", help="Also time writing each PNG to disk")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def input_rows(args: argparse.Namespace) -> list[tuple[str, Path]]:
    clear_files = {path.name: path for path in args.clear_dir.glob("*.png")}
    blur_files = {path.name: path for path in args.blur_dir.glob("*.png")}
    if args.workload == "mixed":
        if not clear_files or clear_files.keys() != blur_files.keys():
            raise ValueError("Mixed workload requires matching nonempty clear/blur PNG filenames")
        rows = [item for name in sorted(clear_files) for item in
                (("clear", clear_files[name]), ("blurred", blur_files[name]))]
    elif args.workload == "clear":
        rows = [("clear", clear_files[name]) for name in sorted(clear_files)]
    else:
        rows = [("blurred", blur_files[name]) for name in sorted(blur_files)]
    if not rows:
        raise ValueError(f"No PNG images found for workload {args.workload}")
    return rows[:args.max_images] if args.max_images > 0 else rows


def frozen_router(summary_path: Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    calibration = summary["calibration"]
    if tuple(calibration["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Router feature names/order differ from the frozen calibration")
    mean = np.asarray(calibration["feature_mean"], dtype=np.float64)
    scale = np.asarray(calibration["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(calibration["logistic_coefficients"], dtype=np.float64)
    if mean.shape != (len(FEATURE_NAMES),) or scale.shape != mean.shape or coefficients.shape != mean.shape:
        raise ValueError("Invalid frozen router coefficient dimensions")
    if np.any(scale <= 0):
        raise ValueError("Invalid frozen router feature scales")
    return (float(calibration["threshold"]), mean, scale, coefficients,
            float(calibration["logistic_intercept"]))


def clear_probability(
    features: np.ndarray, mean: np.ndarray, scale: np.ndarray,
    coefficients: np.ndarray, intercept: float,
) -> float:
    logit = float(np.dot((features - mean) / scale, coefficients) + intercept)
    return 1.0 / (1.0 + math.exp(-logit))


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    checkpoint = load_checkpoint_any(checkpoint_path)
    state = {name.removeprefix("module."): tensor for name, tensor in checkpoint["state_dict"].items()}
    model = MBDNetPPlusR(img_channel=3, width=64).cuda().eval()
    model.load_state_dict(state, strict=True)
    model.switch_to_deploy()
    return model


def synchronize() -> None:
    torch.cuda.synchronize()


@torch.inference_mode()
def run_image(
    condition: str, path: Path, model: torch.nn.Module, threshold: float,
    mean: np.ndarray, scale: np.ndarray, coefficients: np.ndarray,
    intercept: float, window_size: int, output_dir: Path | None,
) -> dict[str, str | float]:
    synchronize()
    started = time.perf_counter()
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    decoded = time.perf_counter()

    features = quality_features_rgb(rgb)
    score = clear_probability(features, mean, scale, coefficients, intercept)
    route = "bypass" if score >= threshold else "deblur"
    routed = time.perf_counter()

    if route == "deblur":
        source = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
        source = source.to(device="cuda", dtype=torch.float32).div_(255.0)
        prediction = infer_tiled(model, source, window_size, amp_inference=False).float().clamp_(0, 1)
        synchronize()
        inferred = time.perf_counter()
        output = (prediction[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    else:
        inferred = routed
        output = rgb
    ready = time.perf_counter()

    if output_dir is not None:
        destination = output_dir / condition
        destination.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination / path.name), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    finished = time.perf_counter()

    return {
        "condition": condition,
        "filename": path.name,
        "route": route,
        "clear_probability": score,
        "decode_ms": 1000.0 * (decoded - started),
        "router_ms": 1000.0 * (routed - decoded),
        "model_path_ms": 1000.0 * (inferred - routed),
        "finalize_ms": 1000.0 * (ready - inferred),
        "output_ready_ms": 1000.0 * (ready - started),
        "png_write_ms": 1000.0 * (finished - ready),
        "total_with_save_ms": 1000.0 * (finished - started),
    }


def time_stats(rows: list[dict], key: str) -> dict[str, float] | None:
    if not rows:
        return None
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA to match the existing FP32 GPU protocol")
    if args.window_size <= 0 or args.warmup < 0 or args.max_images < 0:
        raise ValueError("window-size must be positive; warmup and max-images must be nonnegative")
    for required in (args.router_summary, args.checkpoint):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; use --overwrite")
    rows = input_rows(args)
    threshold, mean, scale, coefficients, intercept = frozen_router(args.router_summary)
    model = load_model(args.checkpoint)

    warmup_rgb = cv2.cvtColor(cv2.imread(str(rows[0][1]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    for _ in range(args.warmup):
        quality_features_rgb(warmup_rgb)
        source = torch.from_numpy(warmup_rgb.copy()).permute(2, 0, 1).unsqueeze(0)
        source = source.to(device="cuda", dtype=torch.float32).div_(255.0)
        infer_tiled(model, source, args.window_size, amp_inference=False)
    synchronize()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_images = args.output_dir / "images" if args.save_outputs else None
    timings = []
    for condition, path in rows:
        row = run_image(
            condition, path, model, threshold, mean, scale, coefficients,
            intercept, args.window_size, output_images,
        )
        timings.append(row)
        print(
            f"{condition:7s} {path.name:12s} {row['route']:6s} "
            f"q={row['clear_probability']:.3f} "
            f"output_ready={row['output_ready_ms']:.2f} ms",
            flush=True,
        )

    with (args.output_dir / "per_image_latency.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timings[0]))
        writer.writeheader()
        writer.writerows(timings)

    groups = {name: [row for row in timings if row["route"] == name] for name in ("deblur", "bypass")}
    conditions = {name: [row for row in timings if row["condition"] == name] for name in ("clear", "blurred")}
    summary = {
        "protocol": {
            "workload": args.workload,
            "timed_interval": "Before PNG decode to routed RGB uint8 output ready in CPU memory",
            "model_precision": "FP32",
            "model_deploy_reparameterized": True,
            "batch_size": 1,
            "window_size": args.window_size,
            "warmup_runs_excluded": args.warmup,
            "png_saving_included_in_total_with_save_ms": bool(args.save_outputs),
            "router_summary": str(args.router_summary),
            "checkpoint": str(args.checkpoint),
            "gpu": torch.cuda.get_device_name(0),
        },
        "threshold": threshold,
        "images": len(timings),
        "route_counts": {name: len(group) for name, group in groups.items()},
        "condition_counts": {name: len(group) for name, group in conditions.items()},
        "blurred_bypass_count": sum(row["route"] == "bypass" for row in conditions["blurred"]),
        "bypass_rate": len(groups["bypass"]) / len(timings),
        "all_output_ready_ms": time_stats(timings, "output_ready_ms"),
        "clear_output_ready_ms": time_stats(conditions["clear"], "output_ready_ms"),
        "blurred_output_ready_ms": time_stats(conditions["blurred"], "output_ready_ms"),
        "deblur_output_ready_ms": time_stats(groups["deblur"], "output_ready_ms"),
        "bypass_output_ready_ms": time_stats(groups["bypass"], "output_ready_ms"),
        "all_decode_ms": time_stats(timings, "decode_ms"),
        "all_router_ms": time_stats(timings, "router_ms"),
        "deblur_model_path_ms": time_stats(groups["deblur"], "model_path_ms"),
        "all_total_with_save_ms": time_stats(timings, "total_with_save_ms") if args.save_outputs else None,
    }
    (args.output_dir / "latency_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
