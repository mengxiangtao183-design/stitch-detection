"""Mask-guided, six-unit sewing-stitch defect detector.

This head deliberately lives *after* C3Net.  It does not attempt to repeat
pixel segmentation: it turns a C3Net trajectory mask into six ordered stitch
unit tokens and classifies the geometry and masked RGB evidence of each token.

The target use case is a macro image containing six adjacent stitch units.  It
is not a calibrated microscope measurement module; all geometric evidence is
normalised by the current image size.  That distinction is important because a
pixel-to-mm rule learnt on microscope images is not valid for this dataset.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


DEFECT_CLASSES: Tuple[str, ...] = ("skip", "break", "skew", "loose")
"""Order used by all logits: 跳线/漏针, 断线, 线迹歪斜, 浮线或松线."""


class _DepthwiseBlock(nn.Module):
    """Small residual block suitable for the low-data post-segmentation head."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.norm = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pointwise(self.act(self.norm(self.depthwise(x))))


class SixUnitTokenizer(nn.Module):
    """Convert a mask-guided feature map into ordered stitch-unit tokens.

    ``adaptive_avg_pool2d`` makes the six ROIs resolution independent.  The
    dataset is acquired with the sewing direction approximately horizontal, so
    uniform left-to-right cells are stable and are much less brittle than
    connected-component counting when a defect itself breaks a component.
    """

    def __init__(
        self,
        channels: int,
        num_units: int = 6,
        token_height: int = 16,
        token_width: int = 16,
    ) -> None:
        super().__init__()
        if num_units < 2:
            raise ValueError("num_units must be at least 2")
        self.num_units = num_units
        self.token_height = token_height
        self.token_width = token_width
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, features: Tensor) -> Tensor:
        features = self.proj(features)
        pooled = F.adaptive_avg_pool2d(
            features,
            (self.token_height, self.num_units * self.token_width),
        )
        b, c, h, _ = pooled.shape
        # B,C,H,(six*W) -> B,six,C,H,W, then preserve both mean and peak cue.
        cells = pooled.reshape(b, c, h, self.num_units, self.token_width)
        cells = cells.permute(0, 3, 1, 2, 4).contiguous()
        mean_token = cells.mean(dim=(-2, -1))
        max_token = cells.amax(dim=(-2, -1))
        return torch.cat((mean_token, max_token), dim=-1)


def _soft_unit_geometry(
    mask_probability: Tensor,
    centerline_probability: Tensor,
    num_units: int,
    eps: float = 1e-6,
) -> Tensor:
    """Extract differentiable, image-normalised geometry for each stitch cell.

    Returned cues are foreground mass, centreline mass, y-centroid, y-spread
    (a width/float proxy), left-right centreline displacement, horizontal total
    variation and longitudinal occupancy.  They are evidence for the learned
    classifier, not hand-written defect decisions.
    """
    if mask_probability.shape != centerline_probability.shape:
        raise ValueError("mask and centerline probabilities must have the same shape")
    b, _, height, width = mask_probability.shape
    pooled_width = num_units * 12
    mask = F.adaptive_avg_pool2d(mask_probability, (24, pooled_width))
    line = F.adaptive_avg_pool2d(centerline_probability, (24, pooled_width))
    cell_width = pooled_width // num_units
    y = torch.linspace(-1.0, 1.0, 24, device=mask.device, dtype=mask.dtype).view(1, 1, 24, 1)
    result = []
    for index in range(num_units):
        left, right = index * cell_width, (index + 1) * cell_width
        m = mask[..., left:right]
        c = line[..., left:right]
        mass = m.mean(dim=(-2, -1))
        line_mass = c.mean(dim=(-2, -1))
        weights = m / (m.sum(dim=(-2, -1), keepdim=True) + eps)
        y_mean = (weights * y).sum(dim=(-2, -1))
        y_std = (weights * (y - y_mean[..., None, None]).square()).sum(dim=(-2, -1)).sqrt()

        # A broken/shifted trajectory produces different mean centreline height
        # in the two halves of a cell.  The value is signed for skew direction.
        first, second = torch.chunk(c, 2, dim=-1)
        first_mass = first.sum(dim=(-2, -1), keepdim=True) + eps
        second_mass = second.sum(dim=(-2, -1), keepdim=True) + eps
        first_y = (first * y).sum(dim=(-2, -1), keepdim=True) / first_mass
        second_y = (second * y).sum(dim=(-2, -1), keepdim=True) / second_mass
        endpoint_shift = (second_y - first_y).squeeze(-1).squeeze(-1)

        variation = (m[..., 1:] - m[..., :-1]).abs().mean(dim=(-2, -1))
        occupancy = m.amax(dim=-2).mean(dim=-1)
        result.append(torch.cat(
            (mass, line_mass, y_mean, y_std, endpoint_shift, variation, occupancy), dim=1
        ))
    return torch.stack(result, dim=1).reshape(b, num_units, 7)


class StitchDefectHead(nn.Module):
    """Classify six ordered stitch units from C3Net segmentation outputs.

    Args:
        num_units: Number of stitch units in one image.  Keep this at six for
            the supplied ``quexian`` dataset; changing it requires relabelling.
        hidden_channels: Width of the mask-guided image encoder.
        token_dim: Width of each sequence token.
        transformer_layers: Number of cross-unit relation layers.

    Forward inputs are raw C3Net logits so this module can remain in a PyTorch
    graph.  ``rgb`` is optional for geometry-only ablations; in production pass
    the same restored image tensor that entered C3Net, resized if necessary.

    Output ``unit_logits`` has shape ``[B, 6, 4]`` in ``DEFECT_CLASSES`` order.
    Defects are multi-label: a loose stitch may also be skewed.  ``image_logits``
    is max pooled from the six unit predictions and is useful for a one-row-per-
    image production report.
    """

    def __init__(
        self,
        num_units: int = 6,
        hidden_channels: int = 48,
        token_dim: int = 96,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if token_dim % transformer_heads:
            raise ValueError("token_dim must be divisible by transformer_heads")
        self.num_units = num_units
        self.hidden_channels = hidden_channels
        # mask probability, centreline probability, mask-gated RGB (3) and RGB
        # detail weighted by the predicted line (3): eight channels total.
        self.encoder = nn.Sequential(
            nn.Conv2d(8, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            _DepthwiseBlock(hidden_channels),
            _DepthwiseBlock(hidden_channels),
        )
        self.tokenizer = SixUnitTokenizer(hidden_channels, num_units=num_units)
        self.token_projection = nn.Sequential(
            nn.Linear(2 * hidden_channels + 7, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position = nn.Parameter(torch.zeros(1, num_units, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=transformer_heads,
            dim_feedforward=token_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers, enable_nested_tensor=False
        )
        self.unit_classifier = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, len(DEFECT_CLASSES))
        )
        nn.init.trunc_normal_(self.position, std=0.02)

    @staticmethod
    def _as_probability(logits: Tensor, name: str) -> Tensor:
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"{name} must have shape [B, 1, H, W], got {tuple(logits.shape)}")
        return torch.sigmoid(logits)

    def forward(
        self,
        final_mask_logits: Tensor,
        centerline_logits: Optional[Tensor] = None,
        rgb: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Run mask-guided defect recognition after C3Net."""
        mask_probability = self._as_probability(final_mask_logits, "final_mask_logits")
        if centerline_logits is None:
            # No pseudo-skeleton is manufactured here.  Using the mask as an
            # explicit fallback makes an ablation valid while preserving shape.
            centerline_probability = mask_probability
        else:
            if centerline_logits.shape[-2:] != final_mask_logits.shape[-2:]:
                centerline_logits = F.interpolate(
                    centerline_logits, size=final_mask_logits.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            centerline_probability = self._as_probability(centerline_logits, "centerline_logits")

        b, _, height, width = final_mask_logits.shape
        if rgb is None:
            rgb = final_mask_logits.new_zeros((b, 3, height, width))
        if rgb.ndim != 4 or rgb.shape[0] != b or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [B, 3, H, W]")
        if rgb.shape[-2:] != (height, width):
            rgb = F.interpolate(rgb, size=(height, width), mode="bilinear", align_corners=False)

        # The foreground gate stops the patterned fabric from dominating the
        # appearance branch, while the second gated image keeps fine line cues.
        masked_rgb = rgb * mask_probability
        line_rgb = rgb * centerline_probability
        features = self.encoder(torch.cat(
            (mask_probability, centerline_probability, masked_rgb, line_rgb), dim=1
        ))
        visual_tokens = self.tokenizer(features)
        geometry = _soft_unit_geometry(mask_probability, centerline_probability, self.num_units)
        tokens = self.token_projection(torch.cat((visual_tokens, geometry), dim=-1))
        encoded = self.sequence_encoder(tokens + self.position)
        unit_logits = self.unit_classifier(encoded)
        return {
            "unit_logits": unit_logits,
            "image_logits": unit_logits.amax(dim=1),
            "unit_probabilities": torch.sigmoid(unit_logits),
            "geometry": geometry,
        }


def multilabel_defect_loss(
    unit_logits: Tensor,
    targets: Tensor,
    valid_units: Optional[Tensor] = None,
    positive_weight: Optional[Sequence[float]] = None,
) -> Tensor:
    """Class-balanced BCE loss for ``[B, 6, 4]`` unit annotations.

    ``valid_units`` has shape ``[B, 6]`` and allows an image edge/crop to be
    ignored without inventing a negative label.  Class weights should be
    estimated from the *training split only*, never from validation/test data.
    """
    if unit_logits.shape != targets.shape:
        raise ValueError("unit_logits and targets must have the same shape")
    if unit_logits.ndim != 3 or unit_logits.shape[-1] != len(DEFECT_CLASSES):
        raise ValueError("expected unit logits with shape [B, units, 4]")
    pos_weight = None
    if positive_weight is not None:
        pos_weight = unit_logits.new_tensor(positive_weight)
        if pos_weight.numel() != len(DEFECT_CLASSES):
            raise ValueError("positive_weight must have four elements")
    loss = F.binary_cross_entropy_with_logits(unit_logits, targets.float(), pos_weight=pos_weight, reduction="none")
    if valid_units is None:
        return loss.mean()
    if valid_units.shape != unit_logits.shape[:2]:
        raise ValueError("valid_units must have shape [B, units]")
    weights = valid_units.to(loss.dtype).unsqueeze(-1)
    return (loss * weights).sum() / (weights.sum() * loss.shape[-1]).clamp_min(1.0)

