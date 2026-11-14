"""Build a source-grouped, mask-level controlled augmentation set.

The set is deliberately restricted to the 36 development source images.  It
does not alter raw images and never uses the 12 internal hold-out sources.
Each synthetic change is written to a binary mask and its six unit labels are
carried forward, with only the intended target unit's skip/break/loose label
changed.  This makes the provenance of every training example auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


PRIMARY_CLASSES: Tuple[str, ...] = ("skip", "break", "loose")
ALL_CLASSES: Tuple[str, ...] = ("skip", "break", "skew", "loose")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_labels(path: Path) -> tuple[List[str], Dict[str, List[Dict[str, str]]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in reader:
            if row["augmentation"] == "original":
                grouped[row["sample_id"]].append(dict(row))
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["unit_id"]))
        if len(rows) != 6:
            raise ValueError("each original source must have exactly six unit labels")
    return fieldnames, dict(grouped)


def stratified_split(labels: Dict[str, List[Dict[str, str]]], seed: int, holdout_per_type: int) -> tuple[List[str], List[str]]:
    """Reserve the same number of source images from each image-level type."""
    buckets: Dict[str, List[str]] = {name: [] for name in ALL_CLASSES}
    for source_id, rows in labels.items():
        active = [name for name in ALL_CLASSES if any(int(row[name]) for row in rows)]
        if len(active) != 1:
            raise ValueError(f"{source_id}: expected exactly one image-level defect family, got {active}")
        buckets[active[0]].append(source_id)
    rng = random.Random(seed)
    holdout: List[str] = []
    for name, source_ids in buckets.items():
        source_ids = sorted(source_ids)
        if len(source_ids) <= holdout_per_type:
            raise ValueError(f"{name}: not enough sources for the requested hold-out")
        rng.shuffle(source_ids)
        holdout.extend(source_ids[:holdout_per_type])
    return sorted(set(labels) - set(holdout)), sorted(holdout)


def erase_range(mask: np.ndarray, x0: int, x1: int) -> np.ndarray:
    result = mask.copy()
    result[:, max(0, x0):min(mask.shape[1], x1)] = 0
    return result


def synthesize_skip(mask: np.ndarray, x0: int, x1: int, rng: random.Random) -> tuple[np.ndarray, Dict[str, object]]:
    width = max(x1 - x0, 8)
    gap_width = max(4, int(round(width * rng.uniform(0.18, 0.30))))
    center = int(round(x0 + width * rng.uniform(0.35, 0.65)))
    left = int(np.clip(center - gap_width // 2, x0 + 2, x1 - 3))
    right = min(left + gap_width, x1 - 1)
    return erase_range(mask, left, right), {"gap_x0": left, "gap_x1": right}


def synthesize_break(mask: np.ndarray, x0: int, x1: int, rng: random.Random) -> tuple[np.ndarray, Dict[str, object]]:
    """Create a partial, terminal trajectory rather than an empty slot.

    The short-residual branch represents difficult broken threads that retain
    only a small contiguous fragment.  It is distinct from a skip because the
    remaining support touches one terminal edge and contains no internal gap.
    """
    width = max(x1 - x0, 8)
    short_residual = rng.random() < 0.70
    retained_fraction = rng.uniform(0.18, 0.42) if short_residual else rng.uniform(0.48, 0.68)
    retained = max(3, int(round(width * retained_fraction)))
    if rng.random() < 0.5:
        # Retain the terminal fragment at the right; remove the left portion.
        left, right, side = x0 + 1, max(x0 + 2, x1 - retained), "left"
    else:
        # Retain the terminal fragment at the left; remove the right portion.
        left, right, side = min(x1 - 2, x0 + retained), x1 - 1, "right"
    return erase_range(mask, left, right), {
        "removed_x0": left, "removed_x1": right, "side": side,
        "retained_fraction": round(retained / width, 4), "mode": "short_residual" if short_residual else "partial_residual",
    }


def synthesize_loose(mask: np.ndarray, x0: int, x1: int, rng: random.Random) -> tuple[np.ndarray, Dict[str, object]]:
    """Move only the target ribbon along a smooth continuous U or S profile."""
    result = mask.copy()
    height, width = result.shape
    left, right = max(0, x0 + 1), min(width, x1 - 1)
    if right - left < 6:
        return result, {"amplitude_px": 0, "profile": "none"}
    patch = mask[:, left:right].copy()
    result[:, left:right] = 0
    amplitude = rng.uniform(5.5, 11.0) * (-1.0 if rng.random() < 0.5 else 1.0)
    profile = "u" if rng.random() < 0.65 else "s"
    t = np.linspace(0.0, 1.0, right - left, dtype=np.float32)
    offset = amplitude * (np.sin(np.pi * t) if profile == "u" else np.sin(2.0 * np.pi * t))
    for local_x, shift in enumerate(np.rint(offset).astype(int)):
        ys = np.flatnonzero(patch[:, local_x] > 0)
        target_y = ys + shift
        target_y = target_y[(target_y >= 0) & (target_y < height)]
        result[target_y, left + local_x] = 255
    return result, {"amplitude_px": round(float(amplitude), 3), "profile": profile}


def choose_normal_unit(rows: Iterable[Dict[str, str]], rng: random.Random) -> Dict[str, str]:
    eligible = [
        row for row in rows
        if int(row["valid"]) and not any(int(row[name]) for name in PRIMARY_CLASSES)
    ]
    if not eligible:
        raise ValueError("no eligible normal unit available for controlled synthesis")
    return rng.choice(eligible)


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.where(mask > 0, 255, 0).astype(np.uint8), mode="L").save(path)


def add_label_rows(
    output: List[Dict[str, str]], source_rows: List[Dict[str, str]], sample_id: str,
    mask_relative: str, mode: str, synthetic_class: str, target_unit: int | None, params: Dict[str, object],
) -> None:
    for source_row in source_rows:
        row = deepcopy(source_row)
        row["sample_id"] = sample_id
        row["mask_file"] = mask_relative
        row["augmentation"] = mode
        row["source_sample_id"] = source_rows[0]["sample_id"]
        row["synthetic_class"] = synthetic_class
        row["synthetic_parameters"] = json.dumps(params, ensure_ascii=False, sort_keys=True)
        if target_unit is not None and int(row["unit_id"]) == target_unit:
            row[synthetic_class] = "1"
        output.append(row)


def create_preview(output: Path, records: List[Tuple[str, np.ndarray]]) -> None:
    if not records:
        return
    font = None
    panels = []
    for title, mask in records:
        image = Image.fromarray(mask).convert("RGB")
        image = Image.new("RGB", (image.width, image.height + 28), "black") if False else image
        canvas = Image.new("RGB", (mask.shape[1], mask.shape[0] + 28), "black")
        canvas.paste(Image.fromarray(mask).convert("RGB"), (0, 28))
        ImageDraw.Draw(canvas).text((6, 6), title, fill=(255, 210, 0), font=font)
        panels.append(canvas)
    montage = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "black")
    x = 0
    for panel in panels:
        montage.paste(panel, (x, 0))
        x += panel.width
    montage.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create controlled skip/break/loose mask augmentation from 36 source images")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-per-type", type=int, default=3)
    parser.add_argument("--variants-per-class", type=int, default=2)
    args = parser.parse_args()

    fieldnames, labels = read_labels(args.labels)
    development_ids, holdout_ids = stratified_split(labels, args.seed, args.holdout_per_type)
    args.output.mkdir(parents=True, exist_ok=True)
    mask_output = args.output / "masks"
    if mask_output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {mask_output}")
    mask_output.mkdir()

    rng = random.Random(args.seed)
    all_rows: List[Dict[str, str]] = []
    manifest: List[Dict[str, object]] = []
    preview_records: List[Tuple[str, np.ndarray]] = []
    preview_classes: set[str] = set()
    transformations = {"skip": synthesize_skip, "break": synthesize_break, "loose": synthesize_loose}

    for source_id in development_ids:
        source_rows = labels[source_id]
        source_mask_path = args.mask_dir / f"{source_id}.png"
        if not source_mask_path.exists():
            raise FileNotFoundError(source_mask_path)
        base_mask = np.asarray(Image.open(source_mask_path).convert("L"), dtype=np.uint8)
        variants: List[Tuple[str, str, int | None, np.ndarray, Dict[str, object]]] = [
            (f"aug_{source_id}_base", "controlled_base", None, base_mask, {}),
        ]
        for defect in PRIMARY_CLASSES:
            for variant_index in range(args.variants_per_class):
                unit = choose_normal_unit(source_rows, rng)
                x0, x1 = int(unit["x0"]), int(unit["x1"])
                modified, params = transformations[defect](base_mask, x0, x1, rng)
                variants.append((
                    f"aug_{source_id}_{defect}_{variant_index + 1:02d}",
                    f"controlled_{defect}", int(unit["unit_id"]), modified, params,
                ))
        for sample_id, mode, target_unit, mask, params in variants:
            relative = f"masks/{sample_id}.png"
            save_mask(args.output / relative, mask)
            synthetic_class = mode.removeprefix("controlled_") if mode != "controlled_base" else "normal"
            add_label_rows(all_rows, source_rows, sample_id, relative, mode, synthetic_class, target_unit, params)
            manifest.append({
                "sample_id": sample_id, "source_sample_id": source_id, "mode": mode,
                "synthetic_class": synthetic_class, "target_unit": target_unit, "parameters": params,
            })
            if source_id == development_ids[0] and synthetic_class not in preview_classes:
                preview_records.append((synthetic_class, mask))
                preview_classes.add(synthetic_class)

    output_fields = fieldnames + ["source_sample_id", "synthetic_class", "synthetic_parameters"]
    with (args.output / "labels.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(args.output / "split.json", {
        "seed": args.seed, "development_source_ids": development_ids, "internal_holdout_source_ids": holdout_ids,
        "note": "The internal hold-out is excluded from controlled augmentation. It is not a final independent test because prior development inspected all 48 sources.",
    })
    write_json(args.output / "manifest.json", manifest)
    create_preview(args.output / "controlled_mask_examples.png", preview_records)
    print(f"development sources: {len(development_ids)}, internal hold-out sources: {len(holdout_ids)}")
    print(f"controlled masks: {len(manifest)}, label rows: {len(all_rows)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
