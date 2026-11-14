"""Checkpoint helpers shared by training, evaluation, and prediction.

Model checkpoints are written with ``safetensors`` even when the established
``.pth`` filename is retained.  This avoids the intermittent Windows CPython /
PyTorch pickle-writer crash while keeping existing command lines unchanged.
Metadata needed to reconstruct a model is stored atomically in a JSON sidecar.
Legacy PyTorch ``.pth`` checkpoints remain readable.
"""

from collections import OrderedDict
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors.torch import load_file, save_file


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip a leading DataParallel ``module.`` prefix when present."""
    normalized = OrderedDict()
    has_module_prefix = False

    for key, value in state_dict.items():
        if key.startswith('module.'):
            normalized[key[7:]] = value
            has_module_prefix = True
        else:
            normalized[key] = value

    return normalized if has_module_prefix else state_dict


def _metadata_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.name}.json")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize checkpoint metadata of type {type(value).__name__}")


def save_model_checkpoint(
    checkpoint_path: Path,
    state_dict: Dict[str, torch.Tensor],
    metadata: Dict[str, Any],
) -> None:
    """Atomically save CPU model weights plus JSON metadata without pickle.

    The ``.pth`` suffix is intentionally preserved for experiment continuity;
    :func:`load_model_checkpoint` detects this safe tensor payload before trying
    legacy ``torch.load``.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state_dict = {
        key: value.detach().to(device='cpu', copy=True).contiguous()
        for key, value in state_dict.items()
    }
    temp_path = checkpoint_path.with_name(
        f".{checkpoint_path.name}.{os.getpid()}.tmp"
    )
    save_file(cpu_state_dict, str(temp_path), metadata={'format': 'c3net-safetensors-v1'})
    os.replace(temp_path, checkpoint_path)

    metadata_path = _metadata_path(checkpoint_path)
    metadata_temp_path = metadata_path.with_name(
        f".{metadata_path.name}.{os.getpid()}.tmp"
    )
    with metadata_temp_path.open('w', encoding='utf-8') as file:
        json.dump(metadata, file, indent=2, default=_json_default)
    os.replace(metadata_temp_path, metadata_path)


def load_model_checkpoint(
    checkpoint_path: Path | str,
    map_location: torch.device | str = 'cpu',
) -> Dict[str, Any]:
    """Load a safe-tensor checkpoint or fall back to a legacy PyTorch one."""
    checkpoint_path = Path(checkpoint_path)
    try:
        state_dict = load_file(str(checkpoint_path), device='cpu')
    except Exception:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    metadata_path = _metadata_path(checkpoint_path)
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        with metadata_path.open('r', encoding='utf-8') as file:
            metadata = json.load(file)
    metadata['model_state_dict'] = state_dict
    return metadata
