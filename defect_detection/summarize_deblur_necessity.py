"""Create paper-ready three-layer evidence for deblurring necessity.

The output separates restoration fidelity, downstream segmentation quality and
end-to-end defect recognition.  Segmentation masks are evaluated against the
same 48 binary GT masks under clear, direct-blur and deblurred conditions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred_sum, target_sum = int(prediction.sum()), int(target.sum())
    if not pred_sum or not target_sum:
        return {"precision": 0.0, "recall": 0.0, "binary_f1": 0.0, "binary_iou": 0.0, "trajectory_f1_1px": 0.0, "boundary_f1_1px": 0.0}
    tp = int(np.logical_and(prediction, target).sum())
    precision, recall = tp / pred_sum, tp / target_sum
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pred_u8, target_u8 = prediction.astype(np.uint8), target.astype(np.uint8)
    target_near = cv2.dilate(target_u8, kernel) > 0
    pred_near = cv2.dilate(pred_u8, kernel) > 0
    trajectory = f1(float(np.logical_and(prediction, target_near).sum() / pred_sum), float(np.logical_and(target, pred_near).sum() / target_sum))
    rectangle = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    pred_edge = (pred_u8 - cv2.erode(pred_u8, rectangle)) > 0
    target_edge = (target_u8 - cv2.erode(target_u8, rectangle)) > 0
    pred_edge_sum, target_edge_sum = int(pred_edge.sum()), int(target_edge.sum())
    if pred_edge_sum and target_edge_sum:
        target_edge_near = cv2.dilate(target_edge.astype(np.uint8), kernel) > 0
        pred_edge_near = cv2.dilate(pred_edge.astype(np.uint8), kernel) > 0
        boundary = f1(float(np.logical_and(pred_edge, target_edge_near).sum() / pred_edge_sum), float(np.logical_and(target_edge, pred_edge_near).sum() / target_edge_sum))
    else:
        boundary = 0.0
    return {"precision": precision, "recall": recall, "binary_f1": f1(precision, recall), "binary_iou": tp / int(np.logical_or(prediction, target).sum()), "trajectory_f1_1px": trajectory, "boundary_f1_1px": boundary}


def evaluate_masks(mask_dir: Path, gt_dir: Path) -> Dict[str, float]:
    values = []
    for gt_path in sorted(gt_dir.glob("*.png"))[:48]:
        prediction_path = mask_dir / gt_path.name
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        target = np.asarray(Image.open(gt_path).convert("L")) > 127
        prediction = np.asarray(Image.open(prediction_path).convert("L")) > 127
        values.append(binary_metrics(prediction, target))
    return {name: float(np.mean([item[name] for item in values])) for name in values[0]} | {"image_count": len(values)}


def save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise restoration, segmentation and defect evidence")
    parser.add_argument("--gt-mask-dir", type=Path, required=True)
    parser.add_argument("--clear-mask-dir", type=Path, required=True)
    parser.add_argument("--blur-mask-dir", type=Path, required=True)
    parser.add_argument("--deblur-mask-dir", type=Path, required=True)
    parser.add_argument("--restoration-summary", type=Path, required=True)
    parser.add_argument("--defect-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    segmentation = {
        "clear": evaluate_masks(args.clear_mask_dir, args.gt_mask_dir),
        "blur_direct": evaluate_masks(args.blur_mask_dir, args.gt_mask_dir),
        "deblur_then_m6": evaluate_masks(args.deblur_mask_dir, args.gt_mask_dir),
    }
    restoration = json.loads(args.restoration_summary.read_text(encoding="utf-8"))
    defect = json.loads(args.defect_summary.read_text(encoding="utf-8"))
    output = {"restoration": restoration, "segmentation": segmentation, "defect_detection": defect,
              "interpretation": "All segmentation and defect comparisons use the same 48 clear-source images and their one-to-one synthetic motion-blur counterparts."}
    save_json(args.output_dir / "deblur_necessity_evidence.json", output)
    seg_columns = ["condition", "image_count", "binary_iou", "binary_f1", "precision", "recall", "trajectory_f1_1px", "boundary_f1_1px"]
    with (args.output_dir / "deblur_necessity_segmentation.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=seg_columns)
        writer.writeheader()
        for condition, values in segmentation.items():
            writer.writerow({"condition": condition, **values})
    defect_columns = ["condition", "macro_f1_mixed_scope", "mean_class_accuracy_mixed_scope", "local_unit_exact_decision_accuracy", "image_complete_decision_accuracy"]
    with (args.output_dir / "deblur_necessity_defect.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=defect_columns)
        writer.writeheader()
        for condition, values in defect.items():
            writer.writerow({"condition": condition, **{name: values[name] for name in defect_columns[1:]}})
    print(json.dumps(segmentation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
