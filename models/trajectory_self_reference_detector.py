"""Trajectory Self-Reference Defect Reasoner (TSR-Defect).

This is a rule-based post-segmentation defect unit designed for the six-unit
Quxian images.  It is intentionally different from pitch/centroid measurement
modules: it never converts pixels to millimetres, does not count connected
components as stitches, and has no area or vertical-neighbour constraint.

Instead it turns a C3Net probability mask into a soft one-dimensional
trajectory, removes the image-wide trend, and compares each unit against the
other units in the *same image*.  This makes the decision interpretable and
useful before a large defect-classification dataset is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


DEFECT_CLASSES: Tuple[str, ...] = ("skip", "break", "skew", "loose")
LOCAL_DEFECT_CLASSES: Tuple[str, ...] = ("skip", "break", "loose")


@dataclass(frozen=True)
class TSRConfig:
    num_units: int = 6
    mask_threshold: float = 0.50
    # A 2--5 px stitch in a 320 px image occupies roughly 0.006--0.016 of a
    # column after averaging over height, hence this must remain small.
    min_column_mass: float = 0.005
    # Scores are continuous; these are operating thresholds, to be calibrated
    # on validation source images only when enough data are available.
    skip_threshold: float = 0.50
    break_threshold: float = 0.50
    skew_threshold: float = 0.50
    loose_threshold: float = 0.50
    # Enable only when capture pose is controlled or has been ruled out during
    # annotation.  The current Quxian labels meet that condition.
    enable_global_skew: bool = True
    # The present labels define skew at image level (all six units are marked).
    # Keep local-slope detection separate until local-skew labels are collected.
    enable_local_skew: bool = False
    # Keep global slope as a continuous score for calibration.  The earlier
    # binary-style ramp saturated most diagonal trajectories at one and made
    # threshold calibration impossible.
    global_slope_score_scale: float = 0.40
    # A global slope above this fraction of image height per image width is a
    # whole-image skew candidate.  It can be disabled by setting a large value.
    global_slope_start: float = 0.035
    global_slope_full: float = 0.110
    # Infer six units from valleys in the one-dimensional trajectory support.
    # This preserves genuinely shorter first/last stitches caused by framing,
    # instead of forcing all six units to have equal pixel widths.
    adaptive_slots: bool = True
    projection_min_run_fraction: float = 0.15
    # In the present annotated dataset every local unit is either normal or
    # belongs to exactly one of skip/break/loose.  Enforcing this semantic
    # constraint prevents a single local trajectory from being reported as
    # several mutually incompatible defect types.
    local_defects_mutually_exclusive: bool = True


def _ramp(value: float, start: float, full: float) -> float:
    if full <= start:
        return float(value >= full)
    return float(np.clip((value - start) / (full - start), 0.0, 1.0))


def decide_defect_classes(
    scores: Dict[str, float], thresholds: Dict[str, float],
    local_defects_mutually_exclusive: bool = True,
) -> Dict[str, bool]:
    """Threshold TSR scores, enforcing the annotated local-label semantics.

    When multiple local classes pass their operating thresholds, retain only
    the one with the greatest score-to-threshold ratio.  This makes scores
    with different calibrated threshold scales comparable.  Global skew stays
    independent because it is an image-level label, not a local alternative.
    """
    decisions = {name: bool(value >= thresholds.get(name, 0.5)) for name, value in scores.items()}
    present_local = [name for name in LOCAL_DEFECT_CLASSES if name in scores]
    if not local_defects_mutually_exclusive or not present_local:
        return decisions
    candidates = [name for name in present_local if decisions[name]]
    if len(candidates) <= 1:
        return decisions
    class_order = {name: index for index, name in enumerate(LOCAL_DEFECT_CLASSES)}
    selected = max(
        candidates,
        key=lambda name: (
            scores[name] / max(thresholds.get(name, 0.5), 1e-6),
            scores[name],
            -class_order[name],
        ),
    )
    for name in present_local:
        decisions[name] = name == selected
    return decisions


def _median_reference(values: np.ndarray, index: int, fallback: float = 0.0) -> float:
    """Robust same-image reference excluding the unit under examination."""
    others = np.delete(values, index)
    finite = others[np.isfinite(others)]
    return float(np.median(finite)) if finite.size else fallback


def _largest_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _interpolate_small_gaps(y: np.ndarray, valid: np.ndarray, max_gap: int = 4) -> np.ndarray:
    """Fill only very short gaps for curvature measurement, retaining breaks."""
    result = y.copy()
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        return result
    for left, right in zip(indices[:-1], indices[1:]):
        if 1 < right - left <= max_gap + 1:
            result[left:right + 1] = np.linspace(y[left], y[right], right - left + 1)
    return result


def _support_runs(valid: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous one-dimensional support runs, represented as [start, end)."""
    runs: List[Tuple[int, int]] = []
    start = None
    for index, present in enumerate(valid):
        if present and start is None:
            start = index
        if start is not None and (not present or index == len(valid) - 1):
            end = index if not present else index + 1
            runs.append((start, end))
            start = None
    return runs


def _adaptive_slot_edges(valid: np.ndarray, num_units: int) -> Tuple[np.ndarray, str]:
    """Locate unit boundaries from support valleys, with a safe equal-slot fallback.

    Tiny isolated support runs are segmentation speckles.  A missing stitch is
    represented by a virtual centre inserted into the largest observed gap so
    the output always contains six ordered slots.
    """
    width = len(valid)
    runs = _support_runs(valid)
    lengths = np.asarray([end - start for start, end in runs], dtype=np.float32)
    if not len(runs):
        return np.linspace(0, width, num_units + 1).round().astype(int), "equal_slots_no_support"
    min_length = max(3, int(np.floor(np.median(lengths) * 0.15)))
    runs = [(start, end) for start, end in runs if end - start >= min_length]
    if not runs:
        return np.linspace(0, width, num_units + 1).round().astype(int), "equal_slots_filtered_support"
    if len(runs) > num_units:
        # Retain the dominant trajectory runs; this only happens after noise
        # fragments survive filtering.
        runs = sorted(sorted(runs, key=lambda item: item[1] - item[0], reverse=True)[:num_units])
    centres = [0.5 * (start + end - 1) for start, end in runs]
    while len(centres) < num_units and len(centres) >= 2:
        gaps = np.diff(centres)
        positive_gaps = gaps[gaps > 0]
        if not len(positive_gaps):
            break
        expected = float(np.median(positive_gaps))
        insert_after = int(np.argmax(gaps))
        # The largest gap is the most plausible missing unit location.  Even
        # if perspective makes it imperfect, the fallback below remains safe.
        centre = 0.5 * (centres[insert_after] + centres[insert_after + 1])
        centres.insert(insert_after + 1, centre)
        if expected <= 0:
            break
    if len(centres) != num_units:
        return np.linspace(0, width, num_units + 1).round().astype(int), "equal_slots_ambiguous_runs"
    boundaries = [0]
    boundaries.extend(int(round((left + right) * 0.5)) for left, right in zip(centres[:-1], centres[1:]))
    boundaries.append(width)
    edges = np.asarray(boundaries, dtype=int)
    edges = np.maximum.accumulate(np.clip(edges, 0, width))
    if np.any(np.diff(edges) < 1):
        return np.linspace(0, width, num_units + 1).round().astype(int), "equal_slots_degenerate_runs"
    return edges, "projection_runs"


class TrajectorySelfReferenceDetector:
    """Rule detector based on local trajectory deformation, not pitch rules.

    The input may be C3Net probability output in ``[0, 1]`` or an 8-bit binary
    mask.  Each output unit contains continuous scores, binary decisions and
    auditable geometric evidence.
    """

    def __init__(self, config: TSRConfig | None = None) -> None:
        self.config = config or TSRConfig()

    @staticmethod
    def _as_probability(mask: np.ndarray) -> np.ndarray:
        array = np.asarray(mask)
        if array.ndim == 3:
            array = array[..., 0]
        if array.ndim != 2:
            raise ValueError(f"expected a 2-D mask, got {array.shape}")
        array = array.astype(np.float32)
        if array.max(initial=0.0) > 1.0:
            array /= 255.0
        return np.clip(array, 0.0, 1.0)

    def _trajectory(self, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate one soft y coordinate per x without connected components."""
        height, width = probability.shape
        y_axis = np.arange(height, dtype=np.float32)[:, None]
        mass = probability.sum(axis=0) / max(height, 1)
        valid = mass >= self.config.min_column_mass
        weighted = (probability * y_axis).sum(axis=0)
        y = np.full(width, np.nan, dtype=np.float32)
        y[valid] = weighted[valid] / probability[:, valid].sum(axis=0).clip(1e-6)
        return y, valid, mass

    def detect(self, mask: np.ndarray) -> Dict[str, Any]:
        probability = self._as_probability(mask)
        height, width = probability.shape
        y, valid, mass = self._trajectory(probability)
        x = np.arange(width, dtype=np.float32)
        valid_x = x[valid]
        valid_y = y[valid]
        if valid_x.size >= 2:
            global_slope, global_intercept = np.polyfit(valid_x, valid_y, 1)
        else:
            global_slope, global_intercept = 0.0, height / 2.0
        trend = global_slope * x + global_intercept
        residual = y - trend

        # Slots are normally recovered from valleys in the 1-D support rather
        # than equal partitioning.  This matters for this dataset because the
        # first and last stitch can be physically truncated by the camera.
        if valid_x.size:
            left_edge, right_edge = int(valid_x.min()), int(valid_x.max()) + 1
        else:
            left_edge, right_edge = 0, width
        if self.config.adaptive_slots:
            edges, slot_method = _adaptive_slot_edges(valid, self.config.num_units)
        else:
            edges = np.linspace(left_edge, right_edge, self.config.num_units + 1).round().astype(int)
            edges[0], edges[-1] = left_edge, right_edge
            slot_method = "equal_observed_span"

        raw: List[Dict[str, float]] = []
        for unit in range(self.config.num_units):
            x0, x1 = int(edges[unit]), int(edges[unit + 1])
            x1 = max(x1, x0 + 1)
            local_valid = valid[x0:x1]
            local_y = y[x0:x1]
            local_residual = residual[x0:x1]
            unit_width = max(x1 - x0, 1)
            coverage = float(local_valid.mean())
            gap_ratio = _largest_run(~local_valid) / unit_width
            observed_x = np.flatnonzero(local_valid)

            if observed_x.size >= 2:
                absolute_x = observed_x + x0
                observed_y = local_y[local_valid]
                dx = np.diff(absolute_x.astype(np.float32))
                dy = np.diff(observed_y)
                # Only consecutive / near-consecutive samples describe a real
                # trajectory; a large gap must not be bridged into arc length.
                near = dx <= 2.0
                arc_length = float(np.sqrt(dx[near] ** 2 + dy[near] ** 2).sum())
                chord_length = float(np.hypot(absolute_x[-1] - absolute_x[0], observed_y[-1] - observed_y[0]))
                arc_ratio = arc_length / max(chord_length, 1e-6)
                profile = _interpolate_small_gaps(local_y, local_valid)
                residual_profile = _interpolate_small_gaps(local_residual, local_valid)
                finite_residual = residual_profile[np.isfinite(residual_profile)]
                deviation = float(np.percentile(np.abs(finite_residual), 90)) if finite_residual.size else 0.0
                thirds = np.array_split(np.flatnonzero(np.isfinite(residual_profile)), 3)
                third_values = [float(np.median(residual_profile[idx])) if idx.size else np.nan for idx in thirds]
                sag = third_values[1] - (third_values[0] + third_values[2]) / 2.0 if all(np.isfinite(third_values)) else 0.0
                # A signed curvature proxy: it identifies a U-shape but does
                # not require all loose stitches to have that exact shape.
                bend_energy = float(np.mean(np.abs(np.diff(profile[np.isfinite(profile)], n=2)))) if np.isfinite(profile).sum() >= 3 else 0.0
                local_slope = float((observed_y[-1] - observed_y[0]) / max(absolute_x[-1] - absolute_x[0], 1))
                span_ratio = float((absolute_x[-1] - absolute_x[0] + 1) / unit_width)
                # Separate a simple sloped segment from a genuinely bent one.
                # Normalising x makes the residual invariant to each adaptive
                # slot's width, which is critical after restoration changes
                # individual segment boundaries by a few pixels.
                local_x_norm = (absolute_x.astype(np.float32) - absolute_x.mean()) / max(float(np.ptp(absolute_x)), 1.0)
                line_coeff = np.polyfit(local_x_norm, observed_y, 1)
                line_fit = np.polyval(line_coeff, local_x_norm)
                line_rmse = float(np.sqrt(np.mean((observed_y - line_fit) ** 2)))
                if observed_x.size >= 4:
                    quad_coeff = np.polyfit(local_x_norm, observed_y, 2)
                    quad_fit = np.polyval(quad_coeff, local_x_norm)
                    quad_rmse = float(np.sqrt(np.mean((observed_y - quad_fit) ** 2)))
                    quadratic_gain = float(np.clip((line_rmse - quad_rmse) / max(line_rmse, 1e-6), 0.0, 1.0))
                else:
                    quadratic_gain = 0.0
                # The internal gap topology distinguishes a terminal partial
                # remnant (break) from an interrupted trajectory with recovery
                # (skip).  Gap values before the first or after the last valid
                # column are not counted as internal interruption.
                first_valid, last_valid = int(observed_x[0]), int(observed_x[-1])
                interior = local_valid[first_valid:last_valid + 1]
                internal_gap_ratio = _largest_run(~interior) / unit_width
                terminal_missing_ratio = max(first_valid, unit_width - 1 - last_valid) / unit_width
                center_x = float(np.median(absolute_x))
                center_y = float(np.median(observed_y))
            else:
                arc_length = chord_length = arc_ratio = deviation = sag = bend_energy = local_slope = span_ratio = 0.0
                line_rmse = quadratic_gain = 0.0
                internal_gap_ratio = 1.0
                terminal_missing_ratio = 1.0
                center_x = center_y = float("nan")
            raw.append({
                "x0": x0, "x1": x1, "coverage": coverage, "gap_ratio": gap_ratio,
                "span_ratio": span_ratio, "arc_ratio": arc_ratio, "deviation_px": deviation,
                "sag_px": float(sag), "bend_energy": bend_energy, "local_slope": local_slope,
                "line_rmse_px": line_rmse, "quadratic_gain": quadratic_gain,
                "internal_gap_ratio": internal_gap_ratio, "terminal_missing_ratio": terminal_missing_ratio,
                "observed_columns": int(observed_x.size),
                "center_x": center_x, "center_y": center_y,
                "edge_truncated": bool((unit == 0 and local_valid.size and local_valid[0]) or (
                    unit == self.config.num_units - 1 and local_valid.size and local_valid[-1]
                )),
            })

        spans = np.asarray([item["span_ratio"] for item in raw], dtype=np.float32)
        coverages = np.asarray([item["coverage"] for item in raw], dtype=np.float32)
        gaps = np.asarray([item["gap_ratio"] for item in raw], dtype=np.float32)
        deviations = np.asarray([item["deviation_px"] for item in raw], dtype=np.float32)
        arcs = np.asarray([item["arc_ratio"] for item in raw], dtype=np.float32)
        bends = np.asarray([item["bend_energy"] for item in raw], dtype=np.float32)
        sags = np.asarray([abs(item["sag_px"]) for item in raw], dtype=np.float32)
        slopes = np.asarray([item["local_slope"] for item in raw], dtype=np.float32)
        line_rmses = np.asarray([item["line_rmse_px"] for item in raw], dtype=np.float32)
        quadratic_gains = np.asarray([item["quadratic_gain"] for item in raw], dtype=np.float32)
        internal_gaps = np.asarray([item["internal_gap_ratio"] for item in raw], dtype=np.float32)
        terminal_missing = np.asarray([item["terminal_missing_ratio"] for item in raw], dtype=np.float32)
        center_xs = np.asarray([item["center_x"] for item in raw], dtype=np.float32)
        center_ys = np.asarray([item["center_y"] for item in raw], dtype=np.float32)
        # A segment has a reliable centre only when it is sufficiently
        # continuous.  Short break remnants deliberately do not become
        # reference points for their neighbours.
        center_valid = np.asarray([
            np.isfinite(item["center_x"]) and item["coverage"] >= 0.35 and item["span_ratio"] >= 0.35
            for item in raw
        ], dtype=bool)
        adjacent_center_gaps = [
            center_xs[index + 1] - center_xs[index]
            for index in range(self.config.num_units - 1)
            if center_valid[index] and center_valid[index + 1]
        ]
        expected_center_gap = float(np.median(adjacent_center_gaps)) if adjacent_center_gaps else width / max(self.config.num_units, 1)
        if int(center_valid.sum()) >= 2:
            center_slope = float(np.polyfit(center_xs[center_valid], center_ys[center_valid], 1)[0])
        else:
            center_slope = float(global_slope)
        center_global_slope_norm = abs(center_slope) * width / max(height, 1)
        global_slope_norm = abs(float(global_slope)) * width / max(height, 1)

        units = []
        for index, evidence in enumerate(raw):
            span_ref = max(_median_reference(spans, index, 1.0), 1e-4)
            coverage_ref = max(_median_reference(coverages, index, 1.0), 1e-4)
            gap_ref = _median_reference(gaps, index, 0.0)
            deviation_ref = max(_median_reference(deviations, index, 1.0), 1.0)
            arc_ref = max(_median_reference(arcs, index, 1.0), 1.0)
            bend_ref = max(_median_reference(bends, index, 1.0), 1e-4)
            sag_ref = max(_median_reference(sags, index, 1.0), 1.0)
            slope_ref = _median_reference(slopes, index, float(global_slope))
            line_rmse_ref = max(_median_reference(line_rmses, index, 1.0), 1.0)
            quadratic_gain_ref = _median_reference(quadratic_gains, index, 0.0)
            internal_gap_ref = _median_reference(internal_gaps, index, 0.0)
            terminal_missing_ref = _median_reference(terminal_missing, index, 0.0)
            # A difficult broken thread can leave only a short residual (even
            # a few supported columns).  Unlike a jump stitch, support does
            # not recover inside the slot.  Compare it to *both* immediate
            # neighbours; edge slots are excluded because the camera may crop
            # them.  This is a geometry rule, independent of absolute pitch.
            immediate = [raw[j] for j in (index - 1, index + 1) if 0 <= j < len(raw)]
            neighbour_shortest_span = min((float(item["span_ratio"]) for item in immediate), default=0.0)
            neighbours_well_supported = bool(immediate) and all(float(item["coverage"]) >= 0.45 for item in immediate)
            short_residual_break = bool(
                0 < int(evidence["observed_columns"]) <= max(2, int(0.35 * max(evidence["x1"] - evidence["x0"], 1)))
                and float(evidence["span_ratio"]) < 0.35 * neighbour_shortest_span
                and neighbours_well_supported
                and 0 < index < self.config.num_units - 1
            )
            evidence["short_residual_break"] = short_residual_break
            # Centre-point reference: a missing middle segment causes its
            # valid neighbours to be farther apart in x than the same-image
            # centre spacing.  This distinguishes a jump from a short break.
            left_valid = next((j for j in range(index - 1, -1, -1) if center_valid[j]), None)
            right_valid = next((j for j in range(index + 1, self.config.num_units) if center_valid[j]), None)
            bridged_center_gap = (
                float(center_xs[right_valid] - center_xs[left_valid])
                if left_valid is not None and right_valid is not None else 0.0
            )
            center_gap_jump = bool(
                not center_valid[index] and left_valid is not None and right_valid is not None
                and bridged_center_gap > 1.45 * max(expected_center_gap, 1.0)
                and 0 < index < self.config.num_units - 1
            )
            center_y_deviation = 0.0
            if center_valid[index] and left_valid is not None and right_valid is not None:
                ratio = (center_xs[index] - center_xs[left_valid]) / max(center_xs[right_valid] - center_xs[left_valid], 1e-6)
                expected_y = center_ys[left_valid] + ratio * (center_ys[right_valid] - center_ys[left_valid])
                center_y_deviation = abs(float(center_ys[index] - expected_y)) / max(height, 1)
            evidence["center_valid"] = bool(center_valid[index])
            evidence["center_gap_jump"] = center_gap_jump
            evidence["bridged_center_gap_ratio"] = bridged_center_gap / max(expected_center_gap, 1.0)
            evidence["center_y_deviation_norm"] = center_y_deviation

            shortfall = max(0.0, 1.0 - evidence["span_ratio"] / span_ref)
            coverage_drop = max(0.0, 1.0 - evidence["coverage"] / coverage_ref)
            excess_gap = max(0.0, evidence["gap_ratio"] - gap_ref)
            internal_gap_excess = max(0.0, evidence["internal_gap_ratio"] - internal_gap_ref)
            terminal_missing_excess = max(0.0, evidence["terminal_missing_ratio"] - terminal_missing_ref)
            # Skip is an isolated support drought with recovery, whereas break
            # prioritises a shortened trajectory.  The two may both be raised
            # only when the image genuinely supplies evidence for both events.
            skip_score = (
                0.55 * _ramp(internal_gap_excess, 0.04, 0.25)
                + 0.25 * _ramp(excess_gap, 0.06, 0.30)
                + 0.20 * _ramp(coverage_drop, 0.10, 0.50)
            )
            # A completely missing unit is a skip, not a break.  Breaks retain
            # a partial trajectory that is shorter / less direct than neighbours.
            partial_trajectory = _ramp(evidence["coverage"], 0.06, 0.30) * _ramp(evidence["span_ratio"], 0.05, 0.30)
            arc_deficit = max(0.0, 1.0 - evidence["arc_ratio"] / arc_ref)
            arc_excess = max(0.0, evidence["arc_ratio"] / arc_ref - 1.0)
            break_score = partial_trajectory * (
                0.45 * _ramp(shortfall, 0.08, 0.45)
                + 0.30 * _ramp(excess_gap, 0.06, 0.30)
                + 0.15 * _ramp(terminal_missing_excess, 0.06, 0.35)
                + 0.10 * _ramp(arc_deficit, 0.08, 0.35)
            )
            # A highly meandering residue is inconsistent with a simple
            # thread break.  Suppress this common skip/loose false positive
            # before threshold calibration, while leaving short straight
            # residuals available as break candidates.
            break_score *= 1.0 - 0.85 * _ramp(arc_excess, 0.35, 1.20)
            if short_residual_break:
                break_score = 1.0
            # Framing can truncate only the outermost stitch.  It supplies no
            # evidence for a defect and must not be confused with a local gap.
            if evidence["edge_truncated"]:
                skip_score = 0.0
                break_score = 0.0
            local_slope_delta = abs(evidence["local_slope"] - slope_ref) * width / max(height, 1)
            global_skew_score = (
                float(np.clip(max(global_slope_norm, center_global_slope_norm) / max(self.config.global_slope_score_scale, 1e-6), 0.0, 1.0))
                if self.config.enable_global_skew else 0.0
            )
            local_skew_score = _ramp(local_slope_delta, 0.025, 0.110) if self.config.enable_local_skew else 0.0
            skew_score = max(global_skew_score, local_skew_score)
            deformation = max(0.0, evidence["deviation_px"] / deviation_ref - 1.0)
            line_deformation = max(0.0, evidence["line_rmse_px"] / line_rmse_ref - 1.0)
            quadratic_excess = max(0.0, evidence["quadratic_gain"] - quadratic_gain_ref)
            bend_excess = max(0.0, evidence["bend_energy"] / bend_ref - 1.0)
            sag_excess = max(0.0, abs(evidence["sag_px"]) / sag_ref - 1.0)
            # Loose thread needs a continuous ribbon.  This gate prevents a
            # skip/break's fragmented mask from being reclassified as loose.
            continuous_trajectory = _ramp(evidence["coverage"], 0.45, 0.70) * _ramp(evidence["span_ratio"], 0.45, 0.75)
            # Deformation covers U, S and off-centre bending.  Sag is helpful
            # but not mandatory; it only supplies an additional cue for U forms.
            loose_score = continuous_trajectory * (
                0.40 * _ramp(deformation, 0.18, 0.95)
                + 0.25 * _ramp(line_deformation, 0.12, 0.90)
                + 0.15 * _ramp(quadratic_excess, 0.03, 0.30)
                + 0.12 * _ramp(sag_excess, 0.20, 1.20)
                + 0.08 * _ramp(arc_excess, 0.04, 0.25)
            )
            if center_valid[index] and 0 < index < self.config.num_units - 1:
                loose_score = max(loose_score, _ramp(center_y_deviation, 0.012, 0.055))
            scores = {
                "skip": float(np.clip(skip_score, 0.0, 1.0)),
                "break": float(np.clip(break_score, 0.0, 1.0)),
                "skew": float(np.clip(skew_score, 0.0, 1.0)),
                "loose": float(np.clip(loose_score, 0.0, 1.0)),
            }
            thresholds = {
                "skip": self.config.skip_threshold,
                "break": self.config.break_threshold,
                "skew": self.config.skew_threshold,
                "loose": self.config.loose_threshold,
            }
            decisions = decide_defect_classes(
                scores, thresholds,
                local_defects_mutually_exclusive=self.config.local_defects_mutually_exclusive,
            )
            units.append({"unit_id": index, "scores": scores, "decisions": decisions, "evidence": evidence})

        return {
            "detector": "TSR-Defect",
            "config": asdict(self.config),
            "global_trajectory": {
                "x_start": left_edge, "x_end": right_edge,
                "global_slope_px_per_x": float(global_slope),
                "global_slope_normalized": global_slope_norm,
                "center_slope_normalized": center_global_slope_norm,
                "expected_center_gap_px": expected_center_gap,
                "slot_method": slot_method,
            },
            "units": units,
        }
