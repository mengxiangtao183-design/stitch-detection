"""PIDNet-S wrapper for the C3Net training/evaluation interface.

The PIDNet core is vendored unchanged from the authors' official repository.
This adapter only maps its final, proportional and derivative predictions to
the project's three supervised logits.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .third_party.PIDNet.models.pidnet import PIDNet


class PIDNetSWrapper(nn.Module):
    """Official PIDNet-S architecture, trained end-to-end for binary masks."""

    def __init__(self, config: Dict):
        super().__init__()
        self.model = PIDNet(
            m=2,
            n=3,
            num_classes=1,
            planes=32,
            ppm_planes=96,
            head_planes=128,
            augment=True,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        size = x.shape[-2:]
        # [proportional auxiliary, final PI fusion, derivative boundary]
        proportional, final, derivative = self.model(x)
        return {
            "final_mask": F.interpolate(final, size=size, mode="bilinear", align_corners=False),
            "object_mask": F.interpolate(proportional, size=size, mode="bilinear", align_corners=False),
            "edge_mask": F.interpolate(derivative, size=size, mode="bilinear", align_corners=False),
        }
