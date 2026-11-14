import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import ConfigurationManager
from models.c3net import C3Net
from utils.checkpoint_utils import load_model_checkpoint, normalize_state_dict_keys
from utils.data_loader import get_test_loaders


def parse_thresholds(spec: str) -> List[float]:
    if ":" in spec:
        start, stop, step = [float(part) for part in spec.split(":")]
        values = []
        value = start
        while value <= stop + 1e-9:
            values.append(round(value, 6))
            value += step
        return values
    return [float(part) for part in spec.split(",") if part.strip()]


def f1_score(precision: float, recall: float) -> float:
    denom = precision + recall
    return 0.0 if denom <= 0 else 2.0 * precision * recall / denom


def binary_prf_iou(pred_bin: np.ndarray, gt_bin: np.ndarray) -> Tuple[float, float, float, float]:
    pred_sum = int(pred_bin.sum())
    gt_sum = int(gt_bin.sum())
    if pred_sum == 0 and gt_sum == 0:
        return 1.0, 1.0, 1.0, 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0, 0.0, 0.0, 0.0
    tp = int(np.logical_and(pred_bin, gt_bin).sum())
    union = int(np.logical_or(pred_bin, gt_bin).sum())
    precision = float(tp / pred_sum)
    recall = float(tp / gt_sum)
    return precision, recall, f1_score(precision, recall), float(tp / union)


def tolerance_f1(pred_bin: np.ndarray, gt_bin: np.ndarray, radius: int) -> float:
    pred_sum = int(pred_bin.sum())
    gt_sum = int(gt_bin.sum())
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0
    if radius <= 0:
        return binary_prf_iou(pred_bin, gt_bin)[2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    pred_u8 = pred_bin.astype(np.uint8)
    gt_u8 = gt_bin.astype(np.uint8)
    gt_near = cv2.dilate(gt_u8, kernel) > 0
    pred_near = cv2.dilate(pred_u8, kernel) > 0
    precision = float(np.logical_and(pred_bin, gt_near).sum() / pred_sum)
    recall = float(np.logical_and(gt_bin, pred_near).sum() / gt_sum)
    return f1_score(precision, recall)


def mask_boundary(mask_bin: np.ndarray) -> np.ndarray:
    mask_u8 = mask_bin.astype(np.uint8)
    if int(mask_u8.sum()) == 0:
        return mask_u8.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return (mask_u8 - eroded) > 0


def empty_meter(thresholds: Iterable[float]) -> Dict[float, Dict[str, float]]:
    return {
        threshold: {
            "binary_iou": 0.0,
            "binary_f1": 0.0,
            "trajectory_f1_1px": 0.0,
            "boundary_f1_1px": 0.0,
            "count": 0.0,
        }
        for threshold in thresholds
    }


def update_meter(meter: Dict[float, Dict[str, float]], prob: np.ndarray, gt: np.ndarray) -> None:
    gt_bin = gt >= 0.5
    for threshold, stats in meter.items():
        pred_bin = prob >= threshold
        _, _, binary_f1, binary_iou = binary_prf_iou(pred_bin, gt_bin)
        boundary_f1 = tolerance_f1(mask_boundary(pred_bin), mask_boundary(gt_bin), radius=1)
        trajectory_f1 = tolerance_f1(pred_bin, gt_bin, radius=1)
        stats["binary_iou"] += binary_iou
        stats["binary_f1"] += binary_f1
        stats["trajectory_f1_1px"] += trajectory_f1
        stats["boundary_f1_1px"] += boundary_f1
        stats["count"] += 1.0


def average_meter(meter: Dict[float, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    averaged = {}
    for threshold, stats in meter.items():
        count = max(stats["count"], 1.0)
        averaged[f"{threshold:.3f}"] = {
            key: value / count
            for key, value in stats.items()
            if key != "count"
        }
    return averaged


def best_by_iou(averaged: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    return max(averaged.items(), key=lambda item: item[1]["binary_iou"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan binary thresholds for C3Net outputs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--thresholds", default="0.25:0.75:0.025")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    model_path = Path(args.model)
    thresholds = parse_thresholds(args.thresholds)

    config = ConfigurationManager.load_config(config_path)
    checkpoint_config = ConfigurationManager.load_model_config(model_path)
    config["model"].update(checkpoint_config["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C3Net(config["model"]).to(device)
    checkpoint = load_model_checkpoint(model_path, map_location=device)
    state_dict = normalize_state_dict_keys(checkpoint["model_state_dict"])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected}")
    model.eval()

    loaders = get_test_loaders(
        dataset_dirs=config["evaluation"]["datasets"],
        model_config=config["model"],
        batch_size=config["evaluation"].get("batch_size", 1),
        num_workers=config["evaluation"].get("num_workers", 0),
    )

    results = {}
    macro_meter = empty_meter(thresholds)

    with torch.inference_mode():
        for dataset_name, loader in loaders.items():
            meter = empty_meter(thresholds)
            for batch in tqdm(loader, desc=f"Scanning {dataset_name}"):
                images = batch["images"].to(device, non_blocking=True)
                masks = [mask.to(device, non_blocking=True) for mask in batch["masks"]]
                outputs = model(images)
                logits = outputs["final_mask"]
                for idx, gt in enumerate(masks):
                    resized = F.interpolate(
                        logits[idx : idx + 1],
                        size=gt.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    prob = resized.sigmoid().squeeze().detach().cpu().numpy()
                    gt_np = gt.squeeze().detach().cpu().numpy()
                    update_meter(meter, prob, gt_np)
                    update_meter(macro_meter, prob, gt_np)
            averaged = average_meter(meter)
            threshold, metrics = best_by_iou(averaged)
            results[dataset_name] = {
                "best_threshold": float(threshold),
                "best_metrics": metrics,
                "thresholds": averaged,
            }

    macro_averaged = average_meter(macro_meter)
    macro_threshold, macro_metrics = best_by_iou(macro_averaged)
    results["macro"] = {
        "best_threshold": float(macro_threshold),
        "best_metrics": macro_metrics,
        "thresholds": macro_averaged,
    }

    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "results" / "threshold_scan" /
        f"{model_path.parent.parent.parent.name}_threshold_scan.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nBest threshold by Binary IoU")
    for name, item in results.items():
        metrics = item["best_metrics"]
        print(
            f"{name:>8}: thr={item['best_threshold']:.3f}, "
            f"IoU={metrics['binary_iou'] * 100:.3f}, "
            f"BinaryF1={metrics['binary_f1'] * 100:.3f}, "
            f"TrajF1@1={metrics['trajectory_f1_1px'] * 100:.3f}, "
            f"BoundaryF1@1={metrics['boundary_f1_1px'] * 100:.3f}"
        )
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
