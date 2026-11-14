"""Run the new TSR-Defect geometry unit on C3Net binary/probability masks."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.trajectory_self_reference_detector import TSRConfig, TrajectorySelfReferenceDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TSR-Defect trajectory self-reference rules")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True, help="C3Net final-mask PNG directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    allowed = {field.name for field in fields(TSRConfig)}
    config = TSRConfig(**{key: value for key, value in source.items() if key in allowed})
    detector = TrajectorySelfReferenceDetector(config)
    results = {}
    for path in sorted(args.input_dir.glob("*.png")):
        mask = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        results[path.stem] = detector.detect(mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Processed {len(results)} masks -> {args.output}")


if __name__ == "__main__":
    main()
