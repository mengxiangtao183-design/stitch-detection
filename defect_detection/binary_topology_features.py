"""Binary-mask topology and same-image neighbour features for TSR.

This extractor is designed for ACHNet's predicted 0/255 PNG masks.  It
extends the established trajectory representation with foreground-run and
8-connected-component evidence.  No label, RGB image, or test-fold statistic
enters feature construction.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from defect_detection.train_trajectory_feature_classifier import (
    FEATURE_NAMES as TRAJECTORY_FEATURE_NAMES,
    feature_vector as trajectory_feature_vector,
)


FEATURE_NAMES = (*TRAJECTORY_FEATURE_NAMES,
    "binary_longest_run_ratio", "binary_second_run_ratio",
    "binary_internal_gap_ratio", "binary_run_count",
    "binary_reappearance_ratio", "binary_component_span_ratio",
    "binary_largest_component_fraction", "binary_significant_component_count",
    "binary_neighbor_shorter_run_ratio", "binary_neighbor_median_run_ratio",
    "binary_terminal_gap_asymmetry", "binary_bridged_center_gap_ratio",
)

FEATURE_NAMES_V2 = (*FEATURE_NAMES,
    "binary_left_third_occupancy", "binary_middle_third_occupancy",
    "binary_right_third_occupancy", "binary_largest_gap_position",
    "binary_flanked_gap_balance", "binary_foreground_center_offset",
    "binary_left_terminal_missing", "binary_right_terminal_missing",
)


def support_runs(support: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.r_[False, support.astype(bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))


def _one_unit(mask: np.ndarray, x0: int, x1: int) -> dict[str, float]:
    region = np.ascontiguousarray(mask[:, x0:x1] >= 0.5, dtype=np.uint8)
    width = max(int(region.shape[1]), 1)
    support = np.any(region > 0, axis=0)
    runs = support_runs(support)
    lengths = sorted((end - start for start, end in runs), reverse=True)
    foreground = int(region.sum())
    longest = lengths[0] if lengths else 0
    second = lengths[1] if len(lengths) > 1 else 0
    if len(runs) >= 2:
        internal_gap = max(runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1))
    else:
        internal_gap = 0
    terminal_left = runs[0][0] if runs else width
    terminal_right = width - runs[-1][1] if runs else width
    thirds = np.array_split(support, 3)
    third_occupancy = [float(np.mean(part)) if len(part) else 0.0 for part in thirds]
    if len(runs) >= 2:
        gaps = [(runs[i][1], runs[i + 1][0]) for i in range(len(runs) - 1)]
        largest_gap_start, largest_gap_end = max(gaps, key=lambda pair: pair[1] - pair[0])
        left_foreground = int(np.count_nonzero(support[:largest_gap_start]))
        right_foreground = int(np.count_nonzero(support[largest_gap_end:]))
        flanked_balance = min(left_foreground, right_foreground) / max(left_foreground + right_foreground, 1)
        largest_gap_position = 0.5 * (largest_gap_start + largest_gap_end) / width
    else:
        flanked_balance = 0.0
        largest_gap_position = 0.5
    foreground_x = np.flatnonzero(support)
    foreground_center_offset = (
        abs(float(np.mean(foreground_x)) / width - 0.5) if foreground_x.size else 0.5
    )

    if foreground:
        count, _, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
        areas = stats[1:count, cv2.CC_STAT_AREA]
        spans = stats[1:count, cv2.CC_STAT_WIDTH]
        biggest = int(np.argmax(areas))
        largest_area = int(areas[biggest])
        component_span = float(spans[biggest]) / width
        component_fraction = largest_area / foreground
        significant = int(np.count_nonzero(areas >= max(2, 0.15 * largest_area)))
    else:
        component_span = component_fraction = 0.0
        significant = 0

    return {
        "longest": float(longest),
        "longest_ratio": longest / width,
        "second_ratio": second / width,
        "internal_gap_ratio": internal_gap / width,
        "run_count": float(min(len(runs), 6)) / 6.0,
        "reappearance_ratio": min(longest, second) / width,
        "component_span_ratio": component_span,
        "component_fraction": component_fraction,
        "significant_components": float(min(significant, 6)) / 6.0,
        "terminal_asymmetry": abs(terminal_left - terminal_right) / width,
        "left_third_occupancy": third_occupancy[0],
        "middle_third_occupancy": third_occupancy[1],
        "right_third_occupancy": third_occupancy[2],
        "largest_gap_position": largest_gap_position,
        "flanked_gap_balance": flanked_balance,
        "foreground_center_offset": foreground_center_offset,
        "left_terminal_missing": terminal_left / width,
        "right_terminal_missing": terminal_right / width,
    }


def binary_topology_feature_vector(result: dict[str, Any], unit: dict[str, Any], mask: np.ndarray) -> list[float]:
    binary = np.asarray(mask)
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2-D binary mask, got {binary.shape}")
    if binary.max(initial=0) > 1:
        binary = binary.astype(np.float32) / 255.0
    height, width = binary.shape
    summaries = result.get("_binary_topology_summaries")
    if summaries is None:
        summaries = [
            _one_unit(binary, int(item["evidence"]["x0"]), int(item["evidence"]["x1"]))
            for item in result["units"]
        ]
        # All six feature vectors for an image share this image-local summary.
        result["_binary_topology_summaries"] = summaries
    index = int(unit["unit_id"])
    local = summaries[index]
    # Use both immediate neighbours only for internal units.  End units may be
    # shortened by camera framing and must not establish a break reference.
    if 0 < index < len(summaries) - 1:
        neighbours = [summaries[index - 1]["longest"], summaries[index + 1]["longest"]]
        shorter = min(neighbours)
        median = float(np.median(neighbours))
        shorter_ratio = min(local["longest"] / max(shorter, 1.0), 2.0)
        median_ratio = min(local["longest"] / max(median, 1.0), 2.0)
    else:
        shorter_ratio = median_ratio = 1.0
    bridge = float(unit["evidence"].get("bridged_center_gap_ratio", 0.0))
    extension = [
        local["longest_ratio"], local["second_ratio"],
        local["internal_gap_ratio"], local["run_count"],
        local["reappearance_ratio"], local["component_span_ratio"],
        local["component_fraction"], local["significant_components"],
        shorter_ratio, median_ratio, local["terminal_asymmetry"],
        min(bridge, 3.0),
    ]
    return trajectory_feature_vector(result, unit, height, width) + extension


def binary_topology_v2_feature_vector(result: dict[str, Any], unit: dict[str, Any], mask: np.ndarray) -> list[float]:
    original = binary_topology_feature_vector(result, unit, mask)
    local = result["_binary_topology_summaries"][int(unit["unit_id"])]
    return original + [
        local["left_third_occupancy"], local["middle_third_occupancy"],
        local["right_third_occupancy"], local["largest_gap_position"],
        local["flanked_gap_balance"], local["foreground_center_offset"],
        local["left_terminal_missing"], local["right_terminal_missing"],
    ]
