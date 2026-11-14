"""Apply validation-only threshold calibration to completed defect-head folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_stitch_defect_head import DEFECT_CLASSES, calibrate_thresholds, metrics_from_rows


def save_json(path: Path, content: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate completed folds using validation predictions only")
    parser.add_argument("--root", type=Path, required=True, help="Cross-validation result root containing fold_* directories")
    args = parser.parse_args()
    calibrated_macro_f1 = []
    for fold_dir in sorted(args.root.glob("fold_*")):
        val_path, test_path = fold_dir / "validation_predictions.json", fold_dir / "test_predictions.json"
        if not val_path.exists() or not test_path.exists():
            print(f"Skip {fold_dir.name}: prediction files not found")
            continue
        validation_rows = json.loads(val_path.read_text(encoding="utf-8"))
        test_rows = json.loads(test_path.read_text(encoding="utf-8"))
        thresholds = calibrate_thresholds(validation_rows)
        save_json(fold_dir / "calibrated_thresholds.json", dict(zip(DEFECT_CLASSES, thresholds)))
        save_json(fold_dir / "validation_metrics_calibrated.json", metrics_from_rows(validation_rows, thresholds))
        save_json(fold_dir / "test_metrics_fixed_0_5.json", metrics_from_rows(test_rows, 0.5))
        calibrated_metrics = metrics_from_rows(test_rows, thresholds)
        save_json(fold_dir / "test_metrics_calibrated.json", calibrated_metrics)
        calibrated_macro_f1.append(calibrated_metrics["macro_f1_active_classes"])
        print(f"{fold_dir.name}: " + ", ".join(f"{name}={value:.3f}" for name, value in zip(DEFECT_CLASSES, thresholds)))
    if calibrated_macro_f1:
        save_json(args.root / "calibrated_cross_validation_summary.json", {
            "mean_fold_macro_f1": sum(calibrated_macro_f1) / len(calibrated_macro_f1),
            "fold_macro_f1": calibrated_macro_f1,
            "note": "Each fold uses thresholds selected only from that fold's validation predictions."
        })
        print("Saved validation-calibrated metrics for each fold and their cross-validation summary.")


if __name__ == "__main__":
    main()
