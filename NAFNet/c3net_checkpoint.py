"""Lightweight checkpoint helpers for NAFNet C3 training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


FORMAT = "nafnet_c3_npz_checkpoint_v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_checkpoint_npz(
    path: str | Path,
    *,
    epoch: int,
    state_dict: dict[str, torch.Tensor],
    scheduler_state: dict[str, Any],
    global_step: int,
    epoch_complete: bool,
) -> None:
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)

    metadata = {
        "format": FORMAT,
        "epoch": int(epoch),
        "scheduler": _jsonable(scheduler_state),
        "global_step": int(global_step),
        "epoch_complete": bool(epoch_complete),
    }
    arrays: dict[str, np.ndarray] = {
        "__metadata__": np.array(json.dumps(metadata), dtype=np.unicode_),
    }
    for name, tensor in state_dict.items():
        arrays[f"state::{name}"] = tensor.detach().cpu().contiguous().numpy()

    with temporary_path.open("wb") as handle:
        np.savez(handle, **arrays)

    load_checkpoint_any(temporary_path)
    temporary_path.replace(path)


def load_checkpoint_any(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "__metadata__" in archive.files:
                metadata = json.loads(str(archive["__metadata__"].item()))
                state = {
                    name.removeprefix("state::"): torch.from_numpy(archive[name].copy())
                    for name in archive.files
                    if name.startswith("state::")
                }
                return {
                    "state_dict": state,
                    "epoch": int(metadata["epoch"]),
                    "scheduler": metadata.get("scheduler"),
                    "global_step": int(metadata.get("global_step", 0)),
                    "epoch_complete": bool(metadata.get("epoch_complete", True)),
                }
    except Exception:
        pass

    return torch.load(path, map_location="cpu", weights_only=False)
