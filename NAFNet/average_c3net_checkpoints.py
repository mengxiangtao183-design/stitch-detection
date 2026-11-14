"""Create a validation-selected average from compatible C3Net checkpoints.

This utility averages floating-point model weights only.  Non-floating values
are copied from the final checkpoint, which is safe for the current MBDNet-P+
state dictionaries and avoids corrupting counters should they be introduced in
future architectures.  It never evaluates a test manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from c3net_checkpoint import load_checkpoint_any, save_checkpoint_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average compatible C3Net model checkpoints")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(item) for item in args.checkpoints]
    if len(input_paths) < 2:
        raise ValueError("At least two checkpoints are required for averaging")
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    weights = args.weights or [1.0] * len(input_paths)
    if len(weights) != len(input_paths) or any(value <= 0.0 for value in weights):
        raise ValueError("weights must be positive and match checkpoints exactly")
    total_weight = float(sum(weights))
    normalized = [value / total_weight for value in weights]

    checkpoints = [load_checkpoint_any(path) for path in input_paths]
    reference = checkpoints[-1]["state_dict"]
    reference_keys = set(reference)
    for path, checkpoint in zip(input_paths, checkpoints):
        if set(checkpoint["state_dict"]) != reference_keys:
            raise RuntimeError(f"State dictionary keys do not match: {path}")

    averaged: dict[str, torch.Tensor] = {}
    for key in sorted(reference):
        values = [checkpoint["state_dict"][key] for checkpoint in checkpoints]
        if any(value.shape != values[0].shape for value in values[1:]):
            raise RuntimeError(f"Shape mismatch for {key}")
        if values[0].is_floating_point():
            accumulator = torch.zeros_like(values[0], dtype=torch.float64)
            for weight, value in zip(normalized, values):
                accumulator.add_(value.to(dtype=torch.float64), alpha=weight)
            averaged[key] = accumulator.to(dtype=values[0].dtype)
        else:
            averaged[key] = values[-1].clone()

    final = checkpoints[-1]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint_npz(
        output,
        epoch=int(final["epoch"]),
        state_dict=averaged,
        scheduler_state=final.get("scheduler") or {},
        global_step=int(final.get("global_step", 0)),
        epoch_complete=True,
    )
    print(
        {
            "output": str(output),
            "checkpoints": [str(path) for path in input_paths],
            "normalized_weights": normalized,
            "epoch": int(final["epoch"]),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
