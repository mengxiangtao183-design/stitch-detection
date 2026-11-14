"""
Implements the main C3Net (Context-Contrast COD Network) model architecture.

This model serves as the central orchestrator, integrating several key components:
1.  A MobileNetV4-Hybrid-Large backbone (via timm) for hierarchical feature extraction,
    replacing the original DINOv2 encoder for improved efficiency.
2.  Coordinate Attention (CoordAtt) modules applied to each backbone output stage
    for enhanced spatial-channel feature refinement.
3.  A shared upsampling module used across the decoder for detail-preserving, content-
    aware upsampling based on learned sampling locations (DySample; Liu et al., ICCV 2023).
4.  A specialized dual-branch decoder:
    - `EdgeBranch` (Edge Refinement Pathway - ERP): Focuses on fine boundary
      details using early encoder features and Sobel/Laplacian initialized
      enhancement blocks (EEMs).
    - `ObjectLocalizationBranch` (Contextual Localization Pathway - CLP): Focuses
      on semantic context and localization using late encoder features, incorporating
      the novel Image-based Context Guidance (ICG) module for integrated salient
      object suppression.
5.  A `FinalFusionHead` (Attentive Fusion Module - AFM) that synergistically
    combines the outputs of the two decoder branches using an attention-gating
    mechanism.
6.  Intermediate prediction heads for enabling deep supervision during training.

Author  : Baber Jan  (original C3Net architecture)
Modified: MobileNetV4 backbone + CoordAtt integration
Date    : 01-05-2025 (original), 05-06-2026 (modified)
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Set
import logging

import timm

from models.utils import DySample, ConvGNReLU
from models.edge_branch import (
    DirectionalConfidenceEdgeRefine,
    EdgeBranch,
    EdgeEnhanceBlock,
)
from models.localization_branch import (
    ObjectLocalizationBranch,
    ImageBasedContextGuidance,
    SemanticEnhanceBlock,
)
from models.fusion_head import FinalFusionHead, PlainFusionHead

# Initialize logger for this module
logger = logging.getLogger(__name__)

# =============================================================================
# HuggingFace download configuration (applied before timm model creation)
# Timeout: 60s, Retries: 10 鈥?handles slow/unstable network conditions
# =============================================================================
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '60')
os.environ.setdefault('HF_HUB_REQUEST_TIMEOUT', '60')

# Patch http_backoff default max_retries from 5 鈫?10
try:
    from huggingface_hub.utils import _http as _hf_http
    if hasattr(_hf_http.http_backoff, '__kwdefaults__') and \
       _hf_http.http_backoff.__kwdefaults__ is not None:
        _hf_http.http_backoff.__kwdefaults__['max_retries'] = 10
        logger.info("HuggingFace download: timeout=60s, retries=10")
except Exception:
    pass  # Non-critical; fall back to default retry count


# =============================================================================
# Coordinate Attention Module (CoordAtt)
# Paper: "Coordinate Attention for Efficient Mobile Network Design" (CVPR 2021)
# Encodes channel relationships along horizontal and vertical spatial directions,
# enabling precise spatial localization with channel-aware attention.
# reduction=16 as specified.
# =============================================================================
class CoordAtt(nn.Module):
    """
    Coordinate Attention Block.

    Decomposes global pooling into a pair of 1D feature encodings along the
    horizontal (W) and vertical (H) axes. This captures long-range dependencies
    along one spatial direction while preserving precise position information
    along the other, producing direction-sensitive and position-sensitive
    attention maps that complement channel attention.

    Args:
        inp (int): Number of input channels.
        oup (int): Number of output channels.
        reduction (int): Reduction ratio for the intermediate bottleneck
                         dimension. Default: 16.
    """
    def __init__(self, inp: int, oup: int, reduction: int = 16):
        super().__init__()
        # Adaptive pooling to (H, 1) and (1, W) for per-direction aggregation
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # Intermediate channel dimension with reduction
        mip = max(8, inp // reduction)

        # Shared 1脳1 convolution for cross-channel interaction after concat
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)  # Using ReLU as in the official implementation

        # Separate 1脳1 convolutions for height and width attention scores
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize convolution weights with Kaiming initialization."""
        for m in [self.conv1, self.conv_h, self.conv_w]:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(self.bn1, 'weight') and self.bn1.weight is not None:
            nn.init.constant_(self.bn1.weight, 1)
        if hasattr(self.bn1, 'bias') and self.bn1.bias is not None:
            nn.init.constant_(self.bn1.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Coordinate Attention.

        Args:
            x (torch.Tensor): Input feature map (B, C, H, W).

        Returns:
            torch.Tensor: Attention-refined feature map (B, C, H, W).
        """
        identity = x
        n, c, h, w = x.size()

        # --- Pool along each spatial dimension ---
        # (B, C, H, 1)
        x_h = self.pool_h(x)
        # (B, C, 1, W) 鈫?permute to (B, C, W, 1) for concatenation along H+W axis
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # --- Concatenate along spatial axis and transform ---
        # Shape: (B, mip, H + W, 1)  鈫?actually (B, C, H + W, 1) before conv
        y = torch.cat([x_h, x_w], dim=2)          # (B, C, H+W, 1)
        y = self.conv1(y)                          # (B, mip, H+W, 1)
        y = self.bn1(y)
        y = self.act(y)

        # --- Split back into height and width branches ---
        x_h, x_w = torch.split(y, [h, w], dim=2)  # (B, mip, H, 1), (B, mip, W, 1)
        x_w = x_w.permute(0, 1, 3, 2)              # (B, mip, 1, W)

        # --- Generate attention maps ---
        a_h = self.conv_h(x_h).sigmoid()           # (B, oup, H, 1)
        a_w = self.conv_w(x_w).sigmoid()           # (B, oup, 1, W)

        # --- Apply attention ---
        out = identity * a_h * a_w                 # (B, C, H, W)
        return out


class DepthwiseSeparableRefine(nn.Module):
    """Light spatial refinement block used by the MobileNet FPN decoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            nn.GroupNorm(max(1, channels // 32), channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(max(1, channels // 32), channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StitchIntervalAwareDifferenceConv(nn.Module):
    """Multi-scale contrast convolution for discontinuous, fine stitch patterns.

    Unlike directional/axial convolutions, SIAD-Conv compares local appearance
    against two concentric contextual receptive fields.  This makes the block
    target the stitch--gap--background contrast that is common in short,
    repeated stitch segments without reproducing a snowflake-shaped sampler.
    """

    def __init__(
        self,
        channels: int,
        dilations: Tuple[int, int] = (2, 4),
        gate_channels: int = 48,
    ):
        super().__init__()
        if len(dilations) != 2 or any(d < 1 for d in dilations):
            raise ValueError("SIAD-Conv needs two positive dilation values.")
        if gate_channels < 1:
            raise ValueError("SIAD-Conv gate_channels must be positive.")

        def depthwise_branch(dilation: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    channels, channels, kernel_size=3, padding=dilation,
                    dilation=dilation, groups=channels, bias=False,
                ),
                nn.GroupNorm(max(1, channels // 32), channels),
                nn.ReLU(inplace=True),
            )

        self.local_branch = depthwise_branch(1)
        self.mid_branch = depthwise_branch(int(dilations[0]))
        self.context_branch = depthwise_branch(int(dilations[1]))

        # The gate receives the feature itself and local-to-context contrasts,
        # then selects the most useful receptive field at each spatial point.
        self.scale_gate = nn.Sequential(
            ConvGNReLU(channels * 3, gate_channels, kernel_size=1, padding=0),
            nn.Conv2d(gate_channels, 3, kernel_size=1),
        )
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(max(1, channels // 32), channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local_branch(x)
        mid = self.mid_branch(x)
        context = self.context_branch(x)

        gate_input = torch.cat([x, local - mid, local - context], dim=1)
        weights = torch.softmax(self.scale_gate(gate_input), dim=1)
        fused = (
            weights[:, 0:1] * local
            + weights[:, 1:2] * mid
            + weights[:, 2:3] * context
        )
        return self.activation(x + self.output_proj(fused))


class C2PContextAlignment(nn.Module):
    """Content-adaptive alignment between lateral FPN features and context."""

    def __init__(self, channels: int, gate_channels: int = 48):
        super().__init__()
        if gate_channels < 1:
            raise ValueError("C2P alignment gate_channels must be positive.")
        self.context_gate = nn.Sequential(
            ConvGNReLU(channels * 3, gate_channels, kernel_size=1, padding=0),
            nn.Conv2d(gate_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        # Begin close to ordinary top-down addition: sigmoid(2) ≈ 0.88.
        nn.init.constant_(self.context_gate[1].bias, 2.0)
        self.discrepancy_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            nn.GroupNorm(max(1, channels // 32), channels),
            nn.ReLU(inplace=True),
        )
        self.correction_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, lateral: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if lateral.shape[-2:] != context.shape[-2:]:
            raise ValueError(
                "C2PContextAlignment requires matched spatial sizes, got "
                f"{lateral.shape[-2:]} and {context.shape[-2:]}"
            )
        discrepancy = lateral - context
        gate = self.context_gate(torch.cat([lateral, context, discrepancy], dim=1))
        correction = self.discrepancy_refine(discrepancy)
        return lateral + gate * context + self.correction_scale * correction


class CrossScaleDeformableAlignment(nn.Module):
    """Align an upsampled semantic map to its lateral FPN feature.

    This is a single-offset feature warp, not a replacement for FPN
    interpolation.  Its zero-initialized predictor starts as ordinary
    top-down addition and learns sub-feature-map corrections only where the
    lateral/detail feature disagrees with the semantic feature.
    """

    def __init__(
        self,
        channels: int,
        gate_channels: int = 48,
        max_offset: float = 2.0,
    ):
        super().__init__()
        if channels < 1 or gate_channels < 1 or max_offset <= 0:
            raise ValueError("invalid cross-scale alignment configuration")
        self.max_offset = float(max_offset)
        self.offset_gate = nn.Sequential(
            ConvGNReLU(channels * 3, gate_channels, kernel_size=1, padding=0),
            nn.Conv2d(gate_channels, 3, kernel_size=1),
        )
        # Offset=(0, 0) and gate=1 reproduce lateral + context at start.
        nn.init.zeros_(self.offset_gate[-1].weight)
        nn.init.zeros_(self.offset_gate[-1].bias)

    def forward(self, lateral: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if lateral.shape != context.shape:
            raise ValueError(
                "CrossScaleDeformableAlignment requires matching feature shapes, got "
                f"{lateral.shape} and {context.shape}"
            )
        batch, _, height, width = lateral.shape
        controls = self.offset_gate(
            torch.cat([lateral, context, lateral - context], dim=1)
        )
        offsets = torch.tanh(controls[:, :2]) * self.max_offset
        gate = 1.0 + 0.25 * torch.tanh(controls[:, 2:3])

        y, x = torch.meshgrid(
            torch.arange(height, device=lateral.device, dtype=lateral.dtype),
            torch.arange(width, device=lateral.device, dtype=lateral.dtype),
            indexing='ij',
        )
        x = x.unsqueeze(0) + offsets[:, 0]
        y = y.unsqueeze(0) + offsets[:, 1]
        x = 2.0 * x / max(width - 1, 1) - 1.0
        y = 2.0 * y / max(height - 1, 1) - 1.0
        grid = torch.stack([x, y], dim=-1)
        aligned = F.grid_sample(
            context,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        return lateral + gate * aligned


class DirectionAwareCrossScaleDeformableAlignment(nn.Module):
    """Direction-aware P4-to-P3 alignment compatible with the D1 module.

    The original D1 alignment predicts one 2-D sampling offset and one gate per
    P3 location.  This module keeps that interface intact, then constrains the
    learned offset by a predicted stitch tangent direction:

    - tangent component is preserved;
    - normal component is suppressed;
    - low direction-confidence areas fall back to the original unconstrained
      offset.

    It therefore replaces only the P4->P3 alignment logic and keeps feature
    shapes, channels and downstream branches identical to D1.
    """

    def __init__(
        self,
        channels: int,
        gate_channels: int = 48,
        max_offset: float = 2.0,
        normal_scale: float = 0.3,
        direction_channels: Optional[List[int]] = None,
    ):
        super().__init__()
        if channels < 1 or gate_channels < 1 or max_offset <= 0:
            raise ValueError("invalid direction-aware alignment configuration")
        if not (0.0 <= normal_scale <= 1.0):
            raise ValueError("normal_scale must be in [0, 1]")
        self.max_offset = float(max_offset)
        self.normal_scale = float(normal_scale)

        self.offset_gate = nn.Sequential(
            ConvGNReLU(channels * 3, gate_channels, kernel_size=1, padding=0),
            nn.Conv2d(gate_channels, 3, kernel_size=1),
        )
        # Keep D1's safe initialization: offset=(0,0), gate=1.
        nn.init.zeros_(self.offset_gate[-1].weight)
        nn.init.zeros_(self.offset_gate[-1].bias)

        direction_channels = direction_channels or [192, 96, 48]
        blocks: List[nn.Module] = []
        in_channels = channels * 2
        for out_channels in direction_channels:
            out_channels = int(out_channels)
            if out_channels < 1:
                raise ValueError("direction predictor channels must be positive")
            blocks.extend([
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                ),
                # Batch size is 2 in the current training setup; GroupNorm is
                # more stable than BatchNorm while preserving the intended
                # depthwise-conv + normalization + GELU design.
                nn.GroupNorm(max(1, in_channels // 32), in_channels),
                nn.GELU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(max(1, out_channels // 32), out_channels),
                nn.GELU(),
            ])
            in_channels = out_channels
        self.direction_encoder = nn.Sequential(*blocks)
        self.angle_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.confidence_head = nn.Conv2d(in_channels, 1, kernel_size=1)

        for module in self.direction_encoder.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.angle_head.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.angle_head.bias)
        nn.init.kaiming_normal_(self.confidence_head.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.confidence_head.bias)

    def forward(self, lateral: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if lateral.shape != context.shape:
            raise ValueError(
                "DirectionAwareCrossScaleDeformableAlignment requires matching "
                f"feature shapes, got {lateral.shape} and {context.shape}"
            )
        batch, _, height, width = lateral.shape
        controls = self.offset_gate(
            torch.cat([lateral, context, lateral - context], dim=1)
        )
        offset_base = torch.tanh(controls[:, :2]) * self.max_offset
        gate = 1.0 + 0.25 * torch.tanh(controls[:, 2:3])

        direction_features = self.direction_encoder(torch.cat([lateral, context], dim=1))
        theta = torch.tanh(self.angle_head(direction_features)) * (math.pi / 2.0)
        direction_confidence = torch.sigmoid(self.confidence_head(direction_features))

        tangent = torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
        normal = torch.cat([-torch.sin(theta), torch.cos(theta)], dim=1)
        tangent_component = (offset_base * tangent).sum(dim=1, keepdim=True)
        normal_component = (offset_base * normal).sum(dim=1, keepdim=True)
        offset_aligned = (
            tangent_component * tangent
            + self.normal_scale * normal_component * normal
        )
        offsets = (
            direction_confidence * offset_aligned
            + (1.0 - direction_confidence) * offset_base
        )

        y, x = torch.meshgrid(
            torch.arange(height, device=lateral.device, dtype=lateral.dtype),
            torch.arange(width, device=lateral.device, dtype=lateral.dtype),
            indexing='ij',
        )
        x = x.unsqueeze(0) + offsets[:, 0]
        y = y.unsqueeze(0) + offsets[:, 1]
        x = 2.0 * x / max(width - 1, 1) - 1.0
        y = 2.0 * y / max(height - 1, 1) - 1.0
        grid = torch.stack([x, y], dim=-1)
        aligned = F.grid_sample(
            context,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        return lateral + gate * aligned


class LocalCorrelationAlignmentCalibrator(nn.Module):
    """Calibrate D1-aligned P3 features by local cross-scale correspondence.

    D1 learns one continuous sampling displacement.  On repeated woven texture,
    a displacement can be locally plausible but semantically wrong.  This
    module therefore compares every P3 location against a small 3x3
    neighbourhood in the already-upsampled P4 map and aggregates the most
    compatible context.  Its final projection is zero-initialized, so D1 is
    exactly preserved at the start of training.
    """

    def __init__(
        self,
        channels: int,
        correlation_channels: int = 32,
        kernel_size: int = 3,
    ):
        super().__init__()
        if channels < 1 or correlation_channels < 1:
            raise ValueError("correlation calibration channels must be positive")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("correlation calibration kernel_size must be odd and >= 3")
        self.kernel_size = int(kernel_size)
        self.query_proj = ConvGNReLU(
            channels, correlation_channels, kernel_size=1, padding=0
        )
        self.key_proj = ConvGNReLU(
            channels, correlation_channels, kernel_size=1, padding=0
        )
        self.value_proj = ConvGNReLU(channels, channels, kernel_size=1, padding=0)
        self.refine = DepthwiseSeparableRefine(channels)
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, aligned_p3: torch.Tensor, semantic_p4_up: torch.Tensor) -> torch.Tensor:
        if aligned_p3.shape != semantic_p4_up.shape:
            raise ValueError(
                "LocalCorrelationAlignmentCalibrator requires matching feature "
                f"shapes, got {aligned_p3.shape} and {semantic_p4_up.shape}"
            )
        batch, _, height, width = aligned_p3.shape
        locations = height * width
        neighbourhood = self.kernel_size * self.kernel_size
        # B,C,L,1 so each P3 location compares against its KxK P4 candidates.
        query = self.query_proj(aligned_p3).flatten(2).unsqueeze(-1)
        keys = F.unfold(
            self.key_proj(semantic_p4_up), kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        ).view(batch, -1, neighbourhood, locations).permute(0, 1, 3, 2)
        scores = (query * keys).sum(dim=1) / math.sqrt(query.shape[1])
        weights = torch.softmax(scores, dim=-1)
        values = F.unfold(
            self.value_proj(semantic_p4_up), kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        ).view(batch, -1, neighbourhood, locations).permute(0, 1, 3, 2)
        matched = (values * weights.unsqueeze(1)).sum(dim=-1).view(
            batch, -1, height, width
        )
        return aligned_p3 + self.output_proj(self.refine(matched))


class SemanticDetailDecomposedAlignment(nn.Module):
    """P3-only semantic-detail decoupled deformable alignment.

    The P3 lateral feature is decomposed into a locally smoothed semantic
    component and its complementary high-frequency detail residual.  Offsets
    are predicted only from the semantic component, while the detail residual
    bypasses warping intact.  This prevents P4 semantics from moving fragile
    stitch boundaries, a failure mode observed when alignment is extended to
    P2.  Zero-initialization preserves the original FPN addition at startup.
    """

    def __init__(
        self,
        channels: int,
        gate_channels: int = 48,
        max_offset: float = 2.0,
        lowpass_kernel_size: int = 5,
    ):
        super().__init__()
        if channels < 1 or gate_channels < 1 or max_offset <= 0:
            raise ValueError("invalid semantic-detail alignment configuration")
        if lowpass_kernel_size < 3 or lowpass_kernel_size % 2 == 0:
            raise ValueError("lowpass_kernel_size must be odd and >= 3")

        self.max_offset = float(max_offset)
        self.lowpass = nn.AvgPool2d(
            kernel_size=lowpass_kernel_size,
            stride=1,
            padding=lowpass_kernel_size // 2,
            count_include_pad=False,
        )
        self.offset_gate = nn.Sequential(
            ConvGNReLU(channels * 3, gate_channels, kernel_size=1, padding=0),
            nn.Conv2d(gate_channels, 3, kernel_size=1),
        )
        # offset=(0, 0), gate=1 -> detail + semantic + context = L3 + P4-up.
        nn.init.zeros_(self.offset_gate[-1].weight)
        nn.init.zeros_(self.offset_gate[-1].bias)

    def forward(self, lateral: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if lateral.shape != context.shape:
            raise ValueError(
                "SemanticDetailDecomposedAlignment requires matching feature "
                f"shapes, got {lateral.shape} and {context.shape}"
            )
        batch, _, height, width = lateral.shape
        semantic_low = self.lowpass(lateral)
        detail_high = lateral - semantic_low
        controls = self.offset_gate(
            torch.cat([semantic_low, context, semantic_low - context], dim=1)
        )
        offsets = torch.tanh(controls[:, :2]) * self.max_offset
        gate = 1.0 + 0.25 * torch.tanh(controls[:, 2:3])

        y, x = torch.meshgrid(
            torch.arange(height, device=lateral.device, dtype=lateral.dtype),
            torch.arange(width, device=lateral.device, dtype=lateral.dtype),
            indexing='ij',
        )
        x = x.unsqueeze(0) + offsets[:, 0]
        y = y.unsqueeze(0) + offsets[:, 1]
        grid = torch.stack([
            2.0 * x / max(width - 1, 1) - 1.0,
            2.0 * y / max(height - 1, 1) - 1.0,
        ], dim=-1)
        aligned_context = F.grid_sample(
            context,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        return detail_high + semantic_low + gate * aligned_context


class P3SemanticDetailReinforcement(nn.Module):
    """Residual semantic-detail reinforcement after D1 P4-to-P3 alignment.

    D1 solves cross-scale misalignment at the P3 fusion point.  This module does
    not replace that alignment.  It takes the already D1-aligned P3 feature and
    strengthens it with two complementary streams:

    - semantic stream: low-frequency P3 plus P4-up context for mask continuity;
    - detail stream: high-frequency P3 residual for thin trajectory boundaries.

    A learnable residual scale starts at zero, so enabling the module preserves
    D1 at initialization while still allowing the new branch to receive useful
    gradients through the scale parameter.
    """

    def __init__(
        self,
        channels: int,
        semantic_channels: int = 48,
        detail_channels: int = 48,
        gate_channels: int = 48,
        lowpass_kernel_size: int = 5,
        num_blocks: int = 1,
        residual_scale_init: float = 0.0,
    ):
        super().__init__()
        if min(channels, semantic_channels, detail_channels, gate_channels) < 1:
            raise ValueError("P3 semantic-detail reinforcement channels must be positive")
        if lowpass_kernel_size < 3 or lowpass_kernel_size % 2 == 0:
            raise ValueError("lowpass_kernel_size must be odd and >= 3")

        self.lowpass = nn.AvgPool2d(
            kernel_size=lowpass_kernel_size,
            stride=1,
            padding=lowpass_kernel_size // 2,
            count_include_pad=False,
        )
        self.semantic_stream = nn.Sequential(
            ConvGNReLU(channels * 3, semantic_channels, kernel_size=1, padding=0),
            *[
                DepthwiseSeparableRefine(semantic_channels)
                for _ in range(max(1, int(num_blocks)))
            ],
            ConvGNReLU(semantic_channels, channels, kernel_size=1, padding=0),
        )
        self.detail_stream = nn.Sequential(
            ConvGNReLU(channels, detail_channels, kernel_size=1, padding=0),
            nn.Conv2d(
                detail_channels,
                detail_channels,
                kernel_size=(1, 7),
                padding=(0, 3),
                groups=detail_channels,
                bias=False,
            ),
            nn.GroupNorm(max(1, detail_channels // 32), detail_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                detail_channels,
                detail_channels,
                kernel_size=(7, 1),
                padding=(3, 0),
                groups=detail_channels,
                bias=False,
            ),
            nn.GroupNorm(max(1, detail_channels // 32), detail_channels),
            nn.ReLU(inplace=True),
            ConvGNReLU(detail_channels, channels, kernel_size=1, padding=0),
        )
        self.stream_gate = nn.Sequential(
            ConvGNReLU(channels * 3, gate_channels, kernel_size=1, padding=0),
            nn.Conv2d(gate_channels, 2, kernel_size=1),
        )
        self.output_refine = DepthwiseSeparableRefine(channels)
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )

    def forward(self, p3: torch.Tensor, p4_context: torch.Tensor) -> torch.Tensor:
        if p3.shape != p4_context.shape:
            raise ValueError(
                "P3SemanticDetailReinforcement requires matching feature shapes, "
                f"got {p3.shape} and {p4_context.shape}"
            )
        p3_semantic = self.lowpass(p3)
        p3_detail = p3 - p3_semantic
        semantic_delta = self.semantic_stream(
            torch.cat([p3_semantic, p4_context, p3_semantic - p4_context], dim=1)
        )
        detail_delta = self.detail_stream(p3_detail)
        weights = torch.softmax(
            self.stream_gate(torch.cat([p3, semantic_delta, detail_delta], dim=1)),
            dim=1,
        )
        delta = weights[:, 0:1] * semantic_delta + weights[:, 1:2] * detail_delta
        delta = self.output_refine(delta)
        return p3 + torch.tanh(self.residual_scale) * delta


class MultiDirectionalDetailBlock(nn.Module):
    """Non-gated directional detail extractor for the high-resolution stream."""

    def __init__(self, channels: int):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(max(1, channels // 32), channels),
            nn.ReLU(inplace=True),
        )
        self.horizontal = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 7), padding=(0, 3), groups=channels, bias=False),
            nn.GroupNorm(max(1, channels // 32), channels),
            nn.ReLU(inplace=True),
        )
        self.vertical = nn.Sequential(
            nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels, bias=False),
            nn.GroupNorm(max(1, channels // 32), channels),
            nn.ReLU(inplace=True),
        )
        self.project = ConvGNReLU(channels * 3, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.project(torch.cat([
            self.local(x), self.horizontal(x), self.vertical(x)
        ], dim=1))


class DualResolutionDetailSemanticEncoder(nn.Module):
    """Persistent 1/2-resolution detail stream with C2/C3 semantic exchange.

    This encoder-side module does not alter the pretrained MobileNet stages.
    A detail stream is built directly from RGB at 1/2 resolution, propagated to
    1/4 and 1/8 resolution, then exchanged with C2/C3 through concatenative
    residual projections.  It is intentionally separate from the final
    HighRes output head and contains no frequency-domain residual correction.
    """

    def __init__(
        self,
        c2_channels: int,
        c3_channels: int,
        detail_channels: int = 32,
    ):
        super().__init__()
        detail_channels = max(16, int(detail_channels))
        self.detail_stem = nn.Sequential(
            ConvGNReLU(3, detail_channels, kernel_size=3, stride=2),
            MultiDirectionalDetailBlock(detail_channels),
        )
        self.to_c2 = ConvGNReLU(detail_channels, c2_channels, kernel_size=3, stride=2)
        self.c2_fuse = nn.Sequential(
            ConvGNReLU(c2_channels * 2, c2_channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(c2_channels),
        )
        self.c2_update = ConvGNReLU(c2_channels, c2_channels, kernel_size=1, padding=0)
        self.to_c3 = ConvGNReLU(c2_channels, c3_channels, kernel_size=3, stride=2)
        self.c3_fuse = nn.Sequential(
            ConvGNReLU(c3_channels * 2, c3_channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(c3_channels),
        )
        self.c3_update = ConvGNReLU(c3_channels, c3_channels, kernel_size=1, padding=0)
        # Both exchanges start as exact identity, preserving the MobileNetV4
        # encoder distribution and allowing the new stream to earn its use.
        self.c2_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.c3_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(
        self, image: torch.Tensor, features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        if len(features) != 4:
            raise ValueError("DualResolutionDetailSemanticEncoder expects C2-C5 features")
        c2, c3, c4, c5 = features
        detail_half = self.detail_stem(image)
        detail_c2 = self.to_c2(detail_half)
        if detail_c2.shape[-2:] != c2.shape[-2:]:
            detail_c2 = F.interpolate(
                detail_c2, size=c2.shape[-2:], mode='bilinear', align_corners=False
            )
        fused_c2 = self.c2_fuse(torch.cat([c2, detail_c2], dim=1))
        c2_enhanced = c2 + torch.tanh(self.c2_scale) * self.c2_update(fused_c2)

        detail_c3 = self.to_c3(fused_c2)
        if detail_c3.shape[-2:] != c3.shape[-2:]:
            detail_c3 = F.interpolate(
                detail_c3, size=c3.shape[-2:], mode='bilinear', align_corners=False
            )
        fused_c3 = self.c3_fuse(torch.cat([c3, detail_c3], dim=1))
        c3_enhanced = c3 + torch.tanh(self.c3_scale) * self.c3_update(fused_c3)
        return [c2_enhanced, c3_enhanced, c4, c5]


class BidirectionalSemanticDetailExchange(nn.Module):
    """P3/P4 bidirectional semantic-detail exchange for thin stitch masks.

    The initial semantic-to-detail alignment is the D1 mechanism.  Subsequent
    exchange blocks feed P3 details back to P4 through learned strided
    projection, then reinject the updated P4 semantics into P3 through aligned
    residual updates.  P2 is deliberately excluded to preserve its fragile
    high-frequency boundary information.
    """

    def __init__(
        self,
        channels: int,
        gate_channels: int = 48,
        max_offset: float = 2.0,
        num_blocks: int = 2,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()
        if channels < 1 or gate_channels < 1 or num_blocks < 1:
            raise ValueError("invalid BSDE configuration")

        self.initial_semantic_to_detail = CrossScaleDeformableAlignment(
            channels, gate_channels, max_offset
        )
        self.detail_to_semantic = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    channels, channels, kernel_size=3, stride=2, padding=1,
                    groups=channels, bias=False,
                ),
                nn.GroupNorm(max(1, channels // 32), channels),
                nn.ReLU(inplace=True),
                ConvGNReLU(channels, channels, kernel_size=1, padding=0),
            )
            for _ in range(num_blocks)
        ])
        self.semantic_updates = nn.ModuleList([
            nn.Sequential(
                ConvGNReLU(channels * 2, channels, kernel_size=3),
                DepthwiseSeparableRefine(channels),
            )
            for _ in range(num_blocks)
        ])
        self.semantic_to_detail = nn.ModuleList([
            CrossScaleDeformableAlignment(channels, gate_channels, max_offset)
            for _ in range(num_blocks)
        ])
        self.detail_updates = nn.ModuleList([
            nn.Sequential(
                ConvGNReLU(channels * 2, channels, kernel_size=3),
                DepthwiseSeparableRefine(channels),
            )
            for _ in range(num_blocks)
        ])
        self.semantic_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(float(residual_scale_init)))
            for _ in range(num_blocks)
        ])
        self.detail_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(float(residual_scale_init)))
            for _ in range(num_blocks)
        ])

    def forward(
        self, lateral_p3: torch.Tensor, semantic_p4: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        semantic_up = F.interpolate(
            semantic_p4, size=lateral_p3.shape[-2:],
            mode='bilinear', align_corners=False,
        )
        detail = self.initial_semantic_to_detail(lateral_p3, semantic_up)
        semantic = semantic_p4

        for down, semantic_update, align, detail_update, s_scale, d_scale in zip(
            self.detail_to_semantic,
            self.semantic_updates,
            self.semantic_to_detail,
            self.detail_updates,
            self.semantic_scales,
            self.detail_scales,
        ):
            detail_down = down(detail)
            semantic_delta = semantic_update(torch.cat([semantic, detail_down], dim=1))
            semantic = semantic + torch.tanh(s_scale) * semantic_delta

            semantic_up = F.interpolate(
                semantic, size=detail.shape[-2:],
                mode='bilinear', align_corners=False,
            )
            # Alignment returns detail + aligned semantic; remove the identity
            # term so the update contains semantic information only.
            aligned_semantic = align(detail, semantic_up) - detail
            detail_delta = detail_update(
                torch.cat([detail, aligned_semantic], dim=1)
            )
            detail = detail + torch.tanh(d_scale) * detail_delta

        return semantic, detail


class MRFADTEBlock(nn.Module):
    """
    Multi-Receptive-Field Directional Trajectory Enhancement block.

    This block is designed for stitch trajectories rather than generic object
    edges. It combines local texture, horizontal/vertical strip kernels and
    dilated spatial context, then learns a spatially varying gate over these
    branches. The output keeps a residual path for stable fine-tuning from the
    current C3Net-MobileNetV4 baseline.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        strip_kernel_size: int = 7,
        dilations: Tuple[int, ...] = (2, 3),
        gate_channels: int = 32,
        residual_scale_init: float = 0.0,
    ):
        super().__init__()
        if strip_kernel_size < 3 or strip_kernel_size % 2 == 0:
            raise ValueError(
                f"strip_kernel_size must be odd and >= 3, got {strip_kernel_size}"
            )

        self.input_proj = ConvGNReLU(
            in_channels, out_channels, kernel_size=1, padding=0
        )

        branches: List[nn.Module] = [
            self._dw_branch(out_channels, kernel_size=3, padding=1),
            self._dw_branch(
                out_channels,
                kernel_size=(1, strip_kernel_size),
                padding=(0, strip_kernel_size // 2),
            ),
            self._dw_branch(
                out_channels,
                kernel_size=(strip_kernel_size, 1),
                padding=(strip_kernel_size // 2, 0),
            ),
        ]
        for dilation in dilations:
            dilation = int(dilation)
            if dilation < 1:
                raise ValueError(f"dilation must be >= 1, got {dilation}")
            branches.append(
                self._dw_branch(
                    out_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                )
            )
        self.branches = nn.ModuleList(branches)
        num_branches = len(branches)

        gate_channels = max(8, int(gate_channels))
        self.branch_gate = nn.Sequential(
            ConvGNReLU(
                out_channels * num_branches,
                gate_channels,
                kernel_size=1,
                padding=0,
            ),
            nn.Conv2d(gate_channels, num_branches, kernel_size=1, bias=True),
        )
        self.output_mix = ConvGNReLU(
            out_channels, out_channels, kernel_size=1, padding=0
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )

    @staticmethod
    def _norm(channels: int) -> nn.GroupNorm:
        groups = 32 if channels >= 32 and channels % 32 == 0 else 1
        return nn.GroupNorm(groups, channels)

    @classmethod
    def _dw_branch(
        cls,
        channels: int,
        kernel_size,
        padding,
        dilation=1,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            cls._norm(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.input_proj(x)
        branch_outputs = [branch(base) for branch in self.branches]
        branch_stack = torch.stack(branch_outputs, dim=1)  # B, N, C, H, W
        gate_logits = self.branch_gate(torch.cat(branch_outputs, dim=1))
        gate = torch.softmax(gate_logits, dim=1).unsqueeze(2)
        fused = (branch_stack * gate).sum(dim=1)
        fused = self.output_mix(fused)
        return base + torch.tanh(self.residual_scale) * fused


class ContrastContextTrajectoryEnhancer(nn.Module):
    """Local-to-context contrast routing for discontinuous stitch trajectories.

    The block models a stitch as a local response that must remain distinct
    from its near and far context, rather than relying on directional strip
    kernels.  It is intentionally used once in the edge pathway, where this
    contrast is most informative and computationally affordable.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: Tuple[int, int] = (2, 4),
        gate_channels: int = 32,
        residual_scale_init: float = 0.2,
    ):
        super().__init__()
        if len(dilations) != 2 or any(int(d) < 1 for d in dilations):
            raise ValueError("CCTE requires two positive context dilations.")
        self.input_proj = ConvGNReLU(in_channels, out_channels, kernel_size=1, padding=0)

        def depthwise(dilation: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3,
                          padding=dilation, dilation=dilation,
                          groups=out_channels, bias=False),
                nn.GroupNorm(max(1, out_channels // 32), out_channels),
                nn.ReLU(inplace=True),
            )

        self.local_branch = depthwise(1)
        self.near_context_branch = depthwise(int(dilations[0]))
        self.far_context_branch = depthwise(int(dilations[1]))
        self.contrast_gate = nn.Sequential(
            ConvGNReLU(out_channels * 2, max(8, int(gate_channels)), kernel_size=1, padding=0),
            nn.Conv2d(max(8, int(gate_channels)), 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.output_mix = ConvGNReLU(out_channels, out_channels, kernel_size=1, padding=0)
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.input_proj(x)
        local = self.local_branch(base)
        near_context = self.near_context_branch(base)
        far_context = self.far_context_branch(base)
        context = 0.5 * (near_context + far_context)
        contrast = local - context
        gate = self.contrast_gate(torch.cat([base, contrast], dim=1))
        enhanced = self.output_mix(local + gate * contrast)
        return base + torch.tanh(self.residual_scale) * enhanced


class ThicknessAwareBranchRouter(nn.Module):
    """Sample-adaptive thickness router for edge/localization features.

    Dataset inspection shows that the remaining IoU loss concentrates on
    narrow stitch samples: M6 improves wide/large-area masks more reliably
    than <4.5px-thick masks.  This router therefore predicts one supervised
    sample-level normalized half-width prior and uses it to route the shared
    edge/localization representation through two experts:

    - thin/detail expert: anisotropic strip filters preserve fragile stitches;
    - wide/context expert: dilated context stabilizes thicker masks.

    The update projections are zero-initialized, so enabling the module starts
    exactly from the input edge/loc features while the auxiliary thickness
    prior learns an interpretable routing signal.
    """

    @staticmethod
    def _norm(channels: int) -> nn.GroupNorm:
        groups = 32 if channels >= 32 and channels % 32 == 0 else 1
        return nn.GroupNorm(groups, channels)

    @classmethod
    def _dw_block(cls, channels: int, kernel_size, padding, dilation=1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            cls._norm(channels),
            nn.ReLU(inplace=True),
        )

    def __init__(
        self,
        edge_channels: int,
        loc_channels: int,
        hidden_channels: int = 64,
        image_channels: int = 12,
        strip_kernel_size: int = 7,
        dilation_rates: Tuple[int, int] = (2, 4),
        width_center: float = 0.45,
        width_temperature: float = 0.08,
    ):
        super().__init__()
        if min(edge_channels, loc_channels, hidden_channels, image_channels) < 1:
            raise ValueError("Thickness router channels must be positive")
        if strip_kernel_size < 3 or strip_kernel_size % 2 == 0:
            raise ValueError("Thickness router strip_kernel_size must be odd and >= 3")
        if len(dilation_rates) != 2 or any(int(d) < 1 for d in dilation_rates):
            raise ValueError("Thickness router requires two positive dilation rates")
        if width_temperature <= 0:
            raise ValueError("Thickness router width_temperature must be > 0")

        self.width_center = float(width_center)
        self.width_temperature = float(width_temperature)
        self.image_proj = nn.Sequential(
            ConvGNReLU(3, image_channels, kernel_size=3),
            DepthwiseSeparableRefine(image_channels),
        )
        input_channels = edge_channels + loc_channels + image_channels
        self.shared = nn.Sequential(
            ConvGNReLU(input_channels, hidden_channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(hidden_channels),
        )

        pad = strip_kernel_size // 2
        self.thin_detail_expert = nn.Sequential(
            self._dw_block(hidden_channels, kernel_size=(1, strip_kernel_size), padding=(0, pad)),
            self._dw_block(hidden_channels, kernel_size=(strip_kernel_size, 1), padding=(pad, 0)),
            ConvGNReLU(hidden_channels, hidden_channels, kernel_size=1, padding=0),
        )
        rate1, rate2 = (int(dilation_rates[0]), int(dilation_rates[1]))
        self.wide_context_expert = nn.Sequential(
            self._dw_block(hidden_channels, kernel_size=3, padding=rate1, dilation=rate1),
            self._dw_block(hidden_channels, kernel_size=3, padding=rate2, dilation=rate2),
            ConvGNReLU(hidden_channels, hidden_channels, kernel_size=1, padding=0),
        )

        self.width_prior_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Sequential(
                nn.Conv2d(
                    hidden_channels,
                    max(16, hidden_channels // 2),
                    kernel_size=1,
                    bias=False,
                ),
                nn.GroupNorm(1, max(16, hidden_channels // 2)),
                nn.ReLU(inplace=True),
            ),
            nn.Conv2d(max(16, hidden_channels // 2), 1, kernel_size=1),
        )
        self.edge_update = nn.Conv2d(hidden_channels, edge_channels, kernel_size=1)
        self.loc_update = nn.Conv2d(hidden_channels, loc_channels, kernel_size=1)
        nn.init.zeros_(self.edge_update.weight)
        nn.init.zeros_(self.edge_update.bias)
        nn.init.zeros_(self.loc_update.weight)
        nn.init.zeros_(self.loc_update.bias)

    def forward(
        self,
        edge_features: torch.Tensor,
        loc_features: torch.Tensor,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_features.shape[-2:] != loc_features.shape[-2:]:
            raise ValueError(
                "Thickness router requires matching edge/loc spatial sizes, got "
                f"{edge_features.shape[-2:]} and {loc_features.shape[-2:]}"
            )
        target_size = edge_features.shape[-2:]
        image_input = image if image.shape[-2:] == target_size else F.interpolate(
            image, size=target_size, mode='bilinear', align_corners=False
        )
        image_detail = self.image_proj(image_input)
        shared = self.shared(torch.cat([edge_features, loc_features, image_detail], dim=1))

        width_logits = self.width_prior_head(shared)
        width_prior = torch.sigmoid(width_logits)
        thin_weight = torch.sigmoid(
            (self.width_center - width_prior) / self.width_temperature
        )

        detail_delta = self.thin_detail_expert(shared)
        context_delta = self.wide_context_expert(shared)
        routed = thin_weight * detail_delta + (1.0 - thin_weight) * context_delta

        edge_refined = edge_features + self.edge_update(routed)
        loc_refined = loc_features + self.loc_update(routed)
        return edge_refined, loc_refined, width_logits


class StyleConditionedDecoderAdapter(nn.Module):
    """Image-style conditioned adapter for edge/localization decoder features.

    The four stitch datasets differ more in fabric texture/style than in the
    centerline topology itself.  A fixed extra branch often helps one domain and
    hurts another.  This adapter therefore estimates a per-image style code from
    the current edge/localization features plus RGB detail, routes the feature
    through several lightweight experts, and applies only zero-initialized
    residual updates to the two decoder branches.

    It is intentionally placed before AFM/HighRes so the final prediction head
    remains unchanged while the branch features become sample-adaptive.
    """

    @staticmethod
    def _norm(channels: int) -> nn.GroupNorm:
        groups = 32 if channels >= 32 and channels % 32 == 0 else 1
        return nn.GroupNorm(groups, channels)

    @classmethod
    def _dw_block(cls, channels: int, kernel_size, padding, dilation=1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            cls._norm(channels),
            nn.ReLU(inplace=True),
        )

    def __init__(
        self,
        edge_channels: int,
        loc_channels: int,
        hidden_channels: int = 64,
        image_channels: int = 12,
        style_channels: int = 32,
        strip_kernel_size: int = 7,
        dilation_rate: int = 3,
    ):
        super().__init__()
        if min(edge_channels, loc_channels, hidden_channels, image_channels, style_channels) < 1:
            raise ValueError("Style adapter channels must be positive")
        if strip_kernel_size < 3 or strip_kernel_size % 2 == 0:
            raise ValueError("style adapter strip_kernel_size must be odd and >= 3")
        if dilation_rate < 1:
            raise ValueError("style adapter dilation_rate must be positive")

        self.image_proj = nn.Sequential(
            ConvGNReLU(3, image_channels, kernel_size=3),
            DepthwiseSeparableRefine(image_channels),
        )
        self.shared = nn.Sequential(
            ConvGNReLU(
                edge_channels + loc_channels + image_channels,
                hidden_channels,
                kernel_size=1,
                padding=0,
            ),
            DepthwiseSeparableRefine(hidden_channels),
        )

        pad = strip_kernel_size // 2
        self.direction_detail_expert = nn.Sequential(
            self._dw_block(hidden_channels, kernel_size=(1, strip_kernel_size), padding=(0, pad)),
            self._dw_block(hidden_channels, kernel_size=(strip_kernel_size, 1), padding=(pad, 0)),
            ConvGNReLU(hidden_channels, hidden_channels, kernel_size=1, padding=0),
        )
        self.texture_suppress_expert = nn.Sequential(
            self._dw_block(hidden_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate),
            ConvGNReLU(hidden_channels, hidden_channels, kernel_size=1, padding=0),
        )
        self.context_balance_expert = nn.Sequential(
            ConvGNReLU(hidden_channels, hidden_channels, kernel_size=3),
            DepthwiseSeparableRefine(hidden_channels),
        )

        self.style_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Sequential(
                nn.Conv2d(hidden_channels, style_channels, kernel_size=1, bias=False),
                nn.GroupNorm(1, style_channels),
                nn.ReLU(inplace=True),
            ),
            nn.Conv2d(style_channels, 3, kernel_size=1),
        )
        self.mix = ConvGNReLU(hidden_channels, hidden_channels, kernel_size=1, padding=0)
        self.edge_update = nn.Conv2d(hidden_channels, edge_channels, kernel_size=1)
        self.loc_update = nn.Conv2d(hidden_channels, loc_channels, kernel_size=1)
        nn.init.zeros_(self.style_head[-1].weight)
        nn.init.zeros_(self.style_head[-1].bias)
        nn.init.zeros_(self.edge_update.weight)
        nn.init.zeros_(self.edge_update.bias)
        nn.init.zeros_(self.loc_update.weight)
        nn.init.zeros_(self.loc_update.bias)

    def forward(
        self,
        edge_features: torch.Tensor,
        loc_features: torch.Tensor,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_features.shape[-2:] != loc_features.shape[-2:]:
            raise ValueError(
                "Style adapter requires matching edge/loc spatial sizes, got "
                f"{edge_features.shape[-2:]} and {loc_features.shape[-2:]}"
            )
        target_size = edge_features.shape[-2:]
        image_input = image if image.shape[-2:] == target_size else F.interpolate(
            image, size=target_size, mode='bilinear', align_corners=False
        )
        image_detail = self.image_proj(image_input)
        shared = self.shared(torch.cat([edge_features, loc_features, image_detail], dim=1))

        style_logits = self.style_head(shared)
        style_weights = torch.softmax(style_logits, dim=1)
        experts = torch.stack(
            [
                self.direction_detail_expert(shared),
                self.texture_suppress_expert(shared),
                self.context_balance_expert(shared),
            ],
            dim=1,
        )
        routed = (experts * style_weights.unsqueeze(2)).sum(dim=1)
        update_features = self.mix(shared + routed)

        edge_refined = edge_features + self.edge_update(update_features)
        loc_refined = loc_features + self.loc_update(update_features)
        return edge_refined, loc_refined, style_weights


class HighResStitchRefineHead(nn.Module):
    """
    Lightweight high-resolution refinement head for stitch masks.

    The engineering defect rules downstream need clean geometry at image
    resolution: centerline continuity, local width, direction and spacing.
    This head keeps the heavy context work at P2, then performs only cheap
    depthwise refinement after upsampling to the requested output stride.
    """

    def __init__(
        self,
        edge_channels: int,
        loc_channels: int,
        refine_channels: int = 48,
        image_channels: int = 16,
        num_blocks: int = 1,
        output_stride: int = 1,
        pattern_channels: int = 0,
        use_afm_prior: bool = False,
    ):
        super().__init__()
        if output_stride < 1:
            raise ValueError(f"output_stride must be >= 1, got {output_stride}")

        self.output_stride = output_stride
        self.use_afm_prior = bool(use_afm_prior)
        self.edge_proj = ConvGNReLU(edge_channels, refine_channels, kernel_size=1, padding=0)
        self.loc_proj = ConvGNReLU(loc_channels, refine_channels, kernel_size=1, padding=0)
        self.image_proj = nn.Sequential(
            ConvGNReLU(3, image_channels, kernel_size=3),
            DepthwiseSeparableRefine(image_channels),
        )
        self.pattern_proj = (
            ConvGNReLU(pattern_channels, refine_channels, kernel_size=1, padding=0)
            if pattern_channels > 0 else None
        )
        # Pattern information is added as a residual to the original HighRes
        # pathway.  A zero start preserves the M4 output path while the
        # component-structure heads learn their targets.
        self.pattern_scale = (
            nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            if pattern_channels > 0 else None
        )

        self.fuse = ConvGNReLU(
            refine_channels * 2 + image_channels + (2 if self.use_afm_prior else 0),
            refine_channels,
            kernel_size=1,
            padding=0,
        )
        self.refine_blocks = nn.Sequential(*[
            DepthwiseSeparableRefine(refine_channels)
            for _ in range(max(0, num_blocks))
        ])
        self.final_pred = nn.Conv2d(refine_channels, 1, kernel_size=1)
        self.edge_pred = nn.Conv2d(refine_channels, 1, kernel_size=1)
        self.centerline_pred = nn.Conv2d(refine_channels, 1, kernel_size=1)

    def forward(
        self,
        edge_features: torch.Tensor,
        loc_features: torch.Tensor,
        image: torch.Tensor,
        pattern_features: Optional[torch.Tensor] = None,
        afm_logits: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        if self.output_stride == 1:
            target_size = image.shape[-2:]
        else:
            target_size = (
                max(1, image.shape[-2] // self.output_stride),
                max(1, image.shape[-1] // self.output_stride),
            )

        edge_up = F.interpolate(
            self.edge_proj(edge_features),
            size=target_size,
            mode='bilinear',
            align_corners=False,
        )
        loc_up = F.interpolate(
            self.loc_proj(loc_features),
            size=target_size,
            mode='bilinear',
            align_corners=False,
        )
        image_input = image
        if image_input.shape[-2:] != target_size:
            image_input = F.interpolate(
                image_input, size=target_size, mode='bilinear', align_corners=False
            )
        image_feat = self.image_proj(image_input)

        fuse_inputs = [edge_up, loc_up, image_feat]
        if self.use_afm_prior:
            if afm_logits is None:
                raise ValueError("HighRes AFM prior is enabled but afm_logits is None")
            afm_prior = afm_logits
            if afm_prior.shape[-2:] != target_size:
                afm_prior = F.interpolate(
                    afm_prior, size=target_size, mode='bilinear', align_corners=False
                )
            afm_prob = torch.sigmoid(afm_prior)
            afm_uncertainty = (4.0 * afm_prob * (1.0 - afm_prob)).clamp_(0.0, 1.0)
            fuse_inputs.extend([afm_prob, afm_uncertainty])

        refined = self.fuse(torch.cat(fuse_inputs, dim=1))
        if self.pattern_proj is not None:
            if pattern_features is None:
                raise ValueError("HighRes pattern projection requires pattern_features")
            pattern_up = F.interpolate(
                self.pattern_proj(pattern_features),
                size=target_size,
                mode='bilinear',
                align_corners=False,
            )
            refined = refined + torch.tanh(self.pattern_scale) * pattern_up
        refined = self.refine_blocks(refined)
        outputs = (
            self.final_pred(refined),
            self.edge_pred(refined),
            self.centerline_pred(refined),
        )
        return (*outputs, refined) if return_features else outputs


class MaskGuidedNarrowBandResamplingHead(nn.Module):
    """Mask-guided local-band resampling calibration after HighRes.

    HighRes already produces a strong full-resolution stitch mask.  This module
    does not replace it.  Instead it builds a soft narrow band from the current
    mask uncertainty, performs a lightweight 2x resampling refinement inside
    that band, and writes back a bounded residual only near uncertain stitch
    boundaries.  The final projection is zero-initialized, so the module starts
    exactly as the base HighRes prediction.
    """

    def __init__(
        self,
        highres_channels: int,
        hidden_channels: int = 32,
        image_channels: int = 12,
        upsample_scale: int = 2,
        uncertainty_temperature: float = 8.0,
        band_dilation: int = 3,
        max_residual: float = 1.5,
    ):
        super().__init__()
        if min(highres_channels, hidden_channels, image_channels) < 1:
            raise ValueError("narrow-band resampling channels must be positive")
        if upsample_scale < 1:
            raise ValueError("upsample_scale must be >= 1")
        if uncertainty_temperature <= 0 or max_residual <= 0:
            raise ValueError("uncertainty_temperature and max_residual must be > 0")
        if band_dilation < 1 or band_dilation % 2 == 0:
            raise ValueError("band_dilation must be an odd positive integer")

        self.upsample_scale = int(upsample_scale)
        self.uncertainty_temperature = float(uncertainty_temperature)
        self.band_dilation = int(band_dilation)
        self.max_residual = float(max_residual)
        self.image_proj = nn.Sequential(
            ConvGNReLU(3, image_channels, kernel_size=3),
            DepthwiseSeparableRefine(image_channels),
        )
        self.fuse = ConvGNReLU(
            highres_channels + image_channels + 3,
            hidden_channels,
            kernel_size=1,
            padding=0,
        )
        self.refine = nn.Sequential(
            DepthwiseSeparableRefine(hidden_channels),
            DepthwiseSeparableRefine(hidden_channels),
        )
        self.residual_pred = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        nn.init.zeros_(self.residual_pred.weight)
        nn.init.zeros_(self.residual_pred.bias)

    def _narrow_band(self, logits: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        # Peak at p=0.5, decays toward confident foreground/background.
        uncertainty = torch.exp(
            -self.uncertainty_temperature * torch.abs(prob - 0.5)
        )
        if self.band_dilation > 1:
            pad = self.band_dilation // 2
            uncertainty = F.max_pool2d(
                uncertainty,
                kernel_size=self.band_dilation,
                stride=1,
                padding=pad,
            )
        return uncertainty.clamp(0.0, 1.0)

    def forward(
        self,
        base_logits: torch.Tensor,
        highres_features: torch.Tensor,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if base_logits.shape[-2:] != highres_features.shape[-2:]:
            raise ValueError(
                "narrow-band resampling expects HighRes features at final-mask "
                f"resolution, got {highres_features.shape[-2:]} and "
                f"{base_logits.shape[-2:]}"
            )
        target_size = base_logits.shape[-2:]
        band = self._narrow_band(base_logits)

        if self.upsample_scale > 1:
            local_size = (
                target_size[0] * self.upsample_scale,
                target_size[1] * self.upsample_scale,
            )
            local_logits = F.interpolate(
                base_logits, size=local_size, mode='bilinear', align_corners=False
            )
            local_features = F.interpolate(
                highres_features, size=local_size, mode='bilinear', align_corners=False
            )
            local_band = F.interpolate(
                band, size=local_size, mode='bilinear', align_corners=False
            )
            local_image = F.interpolate(
                image, size=local_size, mode='bilinear', align_corners=False
            )
        else:
            local_logits = base_logits
            local_features = highres_features
            local_band = band
            local_image = image if image.shape[-2:] == target_size else F.interpolate(
                image, size=target_size, mode='bilinear', align_corners=False
            )

        image_feat = self.image_proj(local_image)
        local_prob = torch.sigmoid(local_logits)
        local_uncertainty = (4.0 * local_prob * (1.0 - local_prob)).clamp(0.0, 1.0)
        features = torch.cat(
            [local_features, image_feat, local_prob, local_uncertainty, local_band],
            dim=1,
        )
        residual = torch.tanh(self.residual_pred(self.refine(self.fuse(features))))
        residual = residual * self.max_residual * local_band
        if residual.shape[-2:] != target_size:
            residual = F.interpolate(
                residual, size=target_size, mode='bilinear', align_corners=False
            )
            residual = residual * band
        return base_logits + residual, residual


class ForegroundCompletionResidualHead(nn.Module):
    """Positive foreground-completion residual after the D1 HighRes head.

    This head targets the main Binary IoU failure mode: foreground underfill.
    It never subtracts from the base mask logits.  Instead, it predicts a
    bounded positive residual from HighRes features and applies it only where
    the localization prior and base-mask uncertainty indicate plausible missing
    foreground.
    """

    def __init__(
        self,
        highres_channels: int,
        hidden_channels: int = 32,
        max_residual: float = 2.0,
        residual_scale_init: float = -4.0,
        uncertainty_floor: float = 0.35,
    ):
        super().__init__()
        if highres_channels < 1 or hidden_channels < 1:
            raise ValueError("FCRR channels must be positive")
        if max_residual <= 0:
            raise ValueError("FCRR max_residual must be > 0")
        if not (0.0 <= uncertainty_floor <= 1.0):
            raise ValueError("FCRR uncertainty_floor must be in [0, 1]")
        self.max_residual = float(max_residual)
        self.uncertainty_floor = float(uncertainty_floor)
        self.refine = nn.Sequential(
            ConvGNReLU(highres_channels + 3, hidden_channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(hidden_channels),
            DepthwiseSeparableRefine(hidden_channels),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )

    def forward(
        self,
        base_logits: torch.Tensor,
        highres_features: torch.Tensor,
        loc_logits: torch.Tensor,
        edge_logits: torch.Tensor,
    ) -> torch.Tensor:
        if base_logits.shape[-2:] != highres_features.shape[-2:]:
            raise ValueError(
                "FCRR expects highres features at the final-mask resolution, got "
                f"{highres_features.shape[-2:]} and {base_logits.shape[-2:]}"
            )
        target_size = base_logits.shape[-2:]
        loc_up = F.interpolate(
            loc_logits, size=target_size, mode='bilinear', align_corners=False
        )
        edge_up = edge_logits
        if edge_up.shape[-2:] != target_size:
            edge_up = F.interpolate(
                edge_up, size=target_size, mode='bilinear', align_corners=False
            )

        # These priors guide where completion is allowed, but are detached so
        # the residual head cannot destabilize D1's existing loc/edge branches.
        base_prob = torch.sigmoid(base_logits.detach())
        loc_prior = torch.sigmoid(loc_up.detach())
        edge_prior = torch.sigmoid(edge_up.detach())
        uncertainty = 1.0 - 2.0 * torch.abs(base_prob - 0.5)
        uncertainty = uncertainty.clamp_(0.0, 1.0)
        uncertainty_gate = self.uncertainty_floor + (
            1.0 - self.uncertainty_floor
        ) * uncertainty
        gate = loc_prior * uncertainty_gate * (0.5 + 0.5 * edge_prior)

        residual_input = torch.cat([
            highres_features,
            base_prob,
            loc_prior,
            edge_prior,
        ], dim=1)
        positive_residual = self.max_residual * torch.sigmoid(
            self.refine(residual_input)
        )
        residual_scale = torch.sigmoid(self.residual_scale)
        return base_logits + residual_scale * gate * positive_residual


class WidthAwareReconstructionDecoder(nn.Module):
    """B7 width-aware reconstruction decoder after the M4 HighRes head.

    The validated M4 HighRes path already localizes stitch trajectories well.
    WRD keeps that prediction as the base logit map and learns a bounded
    full-resolution residual for the remaining Binary-IoU errors: local
    under/over-width, tiny holes, broken foreground and fabric-texture leaks.

    Unlike the earlier positive-only completion head, WRD can both add and
    subtract foreground evidence.  It uses anisotropic strip convolutions for
    stitch-width/continuity cues, dilated context for small gaps, and a compact
    local U-Net branch for pixel-level reconstruction.  The final residual
    predictor is zero-initialized, so enabling the module starts exactly from
    the M4 HighRes output and then learns only useful corrections.
    """

    @staticmethod
    def _gn_relu(channels: int) -> nn.Sequential:
        groups = 32 if channels >= 32 and channels % 32 == 0 else 1
        return nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
        )

    @classmethod
    def _conv_gn_relu(
        cls,
        in_channels: int,
        out_channels: int,
        kernel_size,
        padding,
        dilation=1,
        stride=1,
        bias: bool = False,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            ),
            cls._gn_relu(out_channels),
        )

    def __init__(
        self,
        highres_channels: int,
        edge_channels: int,
        loc_channels: int,
        hidden_channels: int = 64,
        prior_channels: int = 24,
        image_channels: int = 16,
        strip_kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        dilation_rates: Tuple[int, ...] = (2, 4),
        num_refine_blocks: int = 2,
        max_residual: float = 2.0,
        uncertainty_floor: float = 0.15,
    ):
        super().__init__()
        if min(highres_channels, edge_channels, loc_channels, hidden_channels) < 1:
            raise ValueError("WRD channels must be positive")
        if max_residual <= 0:
            raise ValueError("WRD max_residual must be > 0")
        if not (0.0 <= uncertainty_floor <= 1.0):
            raise ValueError("WRD uncertainty_floor must be in [0, 1]")

        self.max_residual = float(max_residual)
        self.uncertainty_floor = float(uncertainty_floor)
        strip_kernel_sizes = tuple(int(k) for k in strip_kernel_sizes)
        if not strip_kernel_sizes or any(k < 3 or k % 2 == 0 for k in strip_kernel_sizes):
            raise ValueError("WRD strip_kernel_sizes must be odd integers >= 3")
        dilation_rates = tuple(int(d) for d in dilation_rates)
        if not dilation_rates or any(d < 1 for d in dilation_rates):
            raise ValueError("WRD dilation_rates must be positive integers")

        self.edge_proj = ConvGNReLU(edge_channels, prior_channels, kernel_size=1, padding=0)
        self.loc_proj = ConvGNReLU(loc_channels, prior_channels, kernel_size=1, padding=0)
        self.image_proj = nn.Sequential(
            ConvGNReLU(3, image_channels, kernel_size=3),
            DepthwiseSeparableRefine(image_channels),
        )

        input_channels = highres_channels + prior_channels * 2 + image_channels + 2
        self.stem = nn.Sequential(
            ConvGNReLU(input_channels, hidden_channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(hidden_channels),
        )

        strip_blocks = []
        for kernel_size in strip_kernel_sizes:
            pad = kernel_size // 2
            strip_blocks.append(nn.Sequential(
                self._conv_gn_relu(
                    hidden_channels, hidden_channels,
                    kernel_size=(1, kernel_size), padding=(0, pad),
                ),
                self._conv_gn_relu(
                    hidden_channels, hidden_channels,
                    kernel_size=(kernel_size, 1), padding=(pad, 0),
                ),
            ))
        self.strip_blocks = nn.ModuleList(strip_blocks)

        self.context_blocks = nn.ModuleList([
            ConvGNReLU(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=rate,
                dilation=rate,
            )
            for rate in dilation_rates
        ])

        self.local_down = nn.Sequential(
            ConvGNReLU(hidden_channels, hidden_channels, kernel_size=3, stride=2),
            DepthwiseSeparableRefine(hidden_channels),
        )
        self.local_mid = nn.Sequential(
            ConvGNReLU(hidden_channels, hidden_channels, kernel_size=3),
            DepthwiseSeparableRefine(hidden_channels),
        )
        self.local_up_fuse = ConvGNReLU(
            hidden_channels * 2, hidden_channels, kernel_size=1, padding=0
        )

        fuse_channels = hidden_channels * (1 + len(strip_blocks) + len(dilation_rates) + 1)
        refine_layers: List[nn.Module] = [
            ConvGNReLU(fuse_channels, hidden_channels, kernel_size=1, padding=0)
        ]
        refine_layers.extend([
            DepthwiseSeparableRefine(hidden_channels)
            for _ in range(max(0, int(num_refine_blocks)))
        ])
        self.refine = nn.Sequential(*refine_layers)
        self.residual_pred = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        # Exact M4 behaviour at iteration 0; gradients still flow into this
        # predictor because final_logits = base_logits + gate * tanh(pred).
        nn.init.zeros_(self.residual_pred.weight)
        nn.init.zeros_(self.residual_pred.bias)

    def forward(
        self,
        base_logits: torch.Tensor,
        highres_features: torch.Tensor,
        edge_features: torch.Tensor,
        loc_features: torch.Tensor,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_size = base_logits.shape[-2:]
        if highres_features.shape[-2:] != target_size:
            raise ValueError(
                "WRD expects HighRes features at final-mask resolution, got "
                f"{highres_features.shape[-2:]} and {target_size}"
            )

        edge_up = F.interpolate(
            self.edge_proj(edge_features), size=target_size,
            mode='bilinear', align_corners=False,
        )
        loc_up = F.interpolate(
            self.loc_proj(loc_features), size=target_size,
            mode='bilinear', align_corners=False,
        )
        image_input = image if image.shape[-2:] == target_size else F.interpolate(
            image, size=target_size, mode='bilinear', align_corners=False
        )
        image_detail = self.image_proj(image_input)

        base_prob = torch.sigmoid(base_logits.detach())
        uncertainty = 4.0 * base_prob * (1.0 - base_prob)
        uncertainty = uncertainty.clamp_(0.0, 1.0)
        residual_gate = self.uncertainty_floor + (
            1.0 - self.uncertainty_floor
        ) * uncertainty

        features = self.stem(torch.cat([
            highres_features,
            edge_up,
            loc_up,
            image_detail,
            base_prob,
            uncertainty,
        ], dim=1))

        branch_outputs: List[torch.Tensor] = [features]
        branch_outputs.extend(block(features) for block in self.strip_blocks)
        branch_outputs.extend(block(features) for block in self.context_blocks)

        local = self.local_down(features)
        local = self.local_mid(local)
        local = F.interpolate(
            local, size=target_size, mode='bilinear', align_corners=False
        )
        branch_outputs.append(self.local_up_fuse(torch.cat([features, local], dim=1)))

        residual_raw = self.residual_pred(self.refine(torch.cat(branch_outputs, dim=1)))
        residual_logits = residual_gate * self.max_residual * torch.tanh(residual_raw)
        return base_logits + residual_logits, residual_logits


class BasePreservingBoundaryResidualHead(nn.Module):
    """D7 low-cost boundary correction without replacing the HighRes result.

    The original HighRes logits remain the segmentation decision.  D7 predicts
    only a zero-initialized residual at 160x160 from D1-calibrated semantics,
    P2 edge/localization features, RGB detail, and the base prediction.  A
    fixed uncertainty band limits that residual to ambiguous mask boundaries,
    avoiding D6's full-image re-segmentation of repetitive fabric texture.
    """

    def __init__(
        self,
        edge_channels: int,
        loc_channels: int,
        p3_channels: int,
        hidden_channels: int = 32,
        semantic_channels: int = 24,
        image_channels: int = 12,
        num_blocks: int = 1,
    ):
        super().__init__()
        if hidden_channels < 1 or semantic_channels < 1 or image_channels < 1:
            raise ValueError("D7 residual head channels must be positive")
        self.edge_proj = ConvGNReLU(
            edge_channels, semantic_channels, kernel_size=1, padding=0
        )
        self.loc_proj = ConvGNReLU(
            loc_channels, semantic_channels, kernel_size=1, padding=0
        )
        self.p3_proj = ConvGNReLU(
            p3_channels, semantic_channels, kernel_size=1, padding=0
        )
        self.image_proj = nn.Sequential(
            ConvGNReLU(3, image_channels, kernel_size=3),
            DepthwiseSeparableRefine(image_channels),
        )
        self.fuse = ConvGNReLU(
            semantic_channels * 3 + image_channels + 2,
            hidden_channels,
            kernel_size=1,
            padding=0,
        )
        self.refine = nn.Sequential(*[
            DepthwiseSeparableRefine(hidden_channels)
            for _ in range(max(1, num_blocks))
        ])
        self.residual_pred = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        # D7 starts as the exact M4+D1 HighRes output.
        nn.init.zeros_(self.residual_pred.weight)
        nn.init.zeros_(self.residual_pred.bias)

    def forward(
        self,
        base_logits: torch.Tensor,
        edge_features: torch.Tensor,
        loc_features: torch.Tensor,
        aligned_p3: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        target_size = (
            max(1, base_logits.shape[-2] // 2),
            max(1, base_logits.shape[-1] // 2),
        )
        base_mid = F.interpolate(
            base_logits, size=target_size, mode='bilinear', align_corners=False,
        )
        # Highest near logit=0 (uncertain boundary), lowest on confident
        # foreground/background.  This is a deterministic spatial constraint,
        # not an additional learned attention gate.
        uncertainty = 1.0 - torch.abs(torch.sigmoid(base_mid) * 2.0 - 1.0)
        edge_mid = F.interpolate(
            self.edge_proj(edge_features), size=target_size,
            mode='bilinear', align_corners=False,
        )
        loc_mid = F.interpolate(
            self.loc_proj(loc_features), size=target_size,
            mode='bilinear', align_corners=False,
        )
        p3_mid = F.interpolate(
            self.p3_proj(aligned_p3), size=target_size,
            mode='bilinear', align_corners=False,
        )
        image_mid = F.interpolate(
            image, size=target_size, mode='bilinear', align_corners=False,
        )
        residual_features = self.fuse(torch.cat([
            edge_mid, loc_mid, p3_mid, self.image_proj(image_mid),
            base_mid, uncertainty,
        ], dim=1))
        residual_mid = uncertainty * self.residual_pred(self.refine(residual_features))
        residual = F.interpolate(
            residual_mid, size=base_logits.shape[-2:], mode='bilinear',
            align_corners=False,
        )
        return base_logits + residual


class AgreementGuidedAreaBoundaryCalibrator(nn.Module):
    """Calibrate HighRes logits only where area and boundary evidence disagree.

    M6 already has an area-oriented localization path, an edge-oriented detail
    path and an AFM coarse mask.  This head does not introduce RGB again or
    rebuild the full-resolution mask.  It uses their detached probability
    disagreement to select a narrow correction region, then exchanges the
    feature evidence between the area and boundary paths to predict a small,
    zero-initialized residual to the HighRes result.
    """
    def __init__(
        self,
        highres_channels: int,
        edge_channels: int,
        loc_channels: int,
        semantic_channels: int = 16,
        hidden_channels: int = 32,
        max_residual: float = 1.0,
    ):
        super().__init__()
        if min(highres_channels, edge_channels, loc_channels, semantic_channels, hidden_channels) < 1:
            raise ValueError("agreement calibration channels must be positive")
        if max_residual <= 0:
            raise ValueError("agreement calibration max_residual must be positive")
        self.max_residual = float(max_residual)
        self.highres_proj = ConvGNReLU(highres_channels, semantic_channels, kernel_size=1, padding=0)
        self.edge_proj = ConvGNReLU(edge_channels, semantic_channels, kernel_size=1, padding=0)
        self.loc_proj = ConvGNReLU(loc_channels, semantic_channels, kernel_size=1, padding=0)
        # Detail-to-area and area-to-detail preserve the distinct roles of the
        # two M6 branches before their correction is fused.
        self.detail_to_area = nn.Sequential(
            ConvGNReLU(semantic_channels * 3, hidden_channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(hidden_channels),
        )
        self.area_to_detail = nn.Sequential(
            ConvGNReLU(semantic_channels * 3, hidden_channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(hidden_channels),
        )
        self.fuse = nn.Sequential(
            ConvGNReLU(hidden_channels * 2 + semantic_channels, hidden_channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(hidden_channels),
        )
        self.residual_pred = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        # Exact M6 behaviour at initialization; the correction head starts
        # learning through its final predictor without perturbing M6 output.
        nn.init.zeros_(self.residual_pred.weight)
        nn.init.zeros_(self.residual_pred.bias)

    @staticmethod
    def _resize(tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        return tensor if tensor.shape[-2:] == size else F.interpolate(
            tensor, size=size, mode='bilinear', align_corners=False
        )

    def forward(
        self,
        base_logits: torch.Tensor,
        highres_features: torch.Tensor,
        edge_features: torch.Tensor,
        loc_features: torch.Tensor,
        afm_logits: torch.Tensor,
        loc_logits: torch.Tensor,
        edge_logits: torch.Tensor,
    ) -> torch.Tensor:
        target_size = base_logits.shape[-2:]
        if highres_features.shape[-2:] != target_size:
            raise ValueError("agreement calibration expects HighRes features at final resolution")

        highres = self.highres_proj(highres_features)
        edge = self._resize(self.edge_proj(edge_features), target_size)
        loc = self._resize(self.loc_proj(loc_features), target_size)

        # Probability maps are detached gates: this residual cannot destabilize
        # the existing AFM, localization, and edge supervision paths.
        base_prob = torch.sigmoid(base_logits.detach())
        afm_prob = torch.sigmoid(self._resize(afm_logits.detach(), target_size))
        loc_prob = torch.sigmoid(self._resize(loc_logits.detach(), target_size))
        edge_prob = torch.sigmoid(self._resize(edge_logits.detach(), target_size))
        coarse_disagreement = 0.5 * (
            torch.abs(base_prob - afm_prob) + torch.abs(base_prob - loc_prob)
        )
        uncertainty = (4.0 * base_prob * (1.0 - base_prob)).clamp_(0.0, 1.0)
        # Edge logits are thin.  A deterministic 5x5 dilation creates a local
        # support band without another learned spatial attention mechanism.
        boundary_support = F.max_pool2d(edge_prob, kernel_size=5, stride=1, padding=2)
        correction_gate = coarse_disagreement * (0.25 + 0.75 * uncertainty) * (
            0.25 + 0.75 * boundary_support
        )

        detail_to_area = self.detail_to_area(torch.cat([
            edge, loc, torch.abs(edge - loc)
        ], dim=1))
        area_to_detail = self.area_to_detail(torch.cat([
            loc, edge, highres
        ], dim=1))
        residual_features = self.fuse(torch.cat([
            detail_to_area, area_to_detail, highres
        ], dim=1))
        residual = torch.tanh(self.residual_pred(residual_features))
        return base_logits + self.max_residual * correction_gate * residual


class InstanceConditionedHighResDecoder(nn.Module):
    """RGB-driven 160x160 instance-conditioned decoder guided by D1 P3.

    Unlike D6/D7, this branch builds the 160x160 representation directly from
    the input image, conditions it with aligned P3 semantics, predicts multiple
    trajectory logits, and aggregates them back to one binary mask residual.
    """

    def __init__(
        self,
        p3_channels: int,
        detail_channels: int = 32,
        semantic_channels: int = 48,
        mask_channels: int = 48,
        num_instances: int = 6,
        num_detail_blocks: int = 2,
        residual_scale_init: float = 0.0,
        residual_max_scale: float = 1.0,
        uncertainty_gate: bool = False,
        uncertainty_gate_power: float = 1.0,
    ):
        super().__init__()
        if min(p3_channels, detail_channels, semantic_channels, mask_channels) < 1:
            raise ValueError("ICHRD channels must be positive")
        if num_instances < 1:
            raise ValueError("ICHRD num_instances must be >= 1")
        if residual_max_scale < 0:
            raise ValueError("ICHRD residual_max_scale must be >= 0")
        if uncertainty_gate_power <= 0:
            raise ValueError("ICHRD uncertainty_gate_power must be > 0")

        self.num_instances = int(num_instances)
        self.mask_channels = int(mask_channels)
        self.residual_max_scale = float(residual_max_scale)
        self.uncertainty_gate = bool(uncertainty_gate)
        self.uncertainty_gate_power = float(uncertainty_gate_power)
        self.detail_stem = nn.Sequential(
            ConvGNReLU(3, detail_channels, kernel_size=3),
            *[
                MultiDirectionalDetailBlock(detail_channels)
                for _ in range(max(0, int(num_detail_blocks)))
            ],
        )
        self.semantic_proj = nn.Sequential(
            ConvGNReLU(p3_channels, semantic_channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(semantic_channels),
        )
        self.mask_feature_proj = nn.Sequential(
            ConvGNReLU(
                detail_channels + semantic_channels,
                mask_channels,
                kernel_size=1,
                padding=0,
            ),
            DepthwiseSeparableRefine(mask_channels),
        )
        query_hidden = max(mask_channels, semantic_channels)
        self.query_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvGNReLU(p3_channels, query_hidden, kernel_size=1, padding=0),
            nn.Conv2d(
                query_hidden,
                self.num_instances * self.mask_channels,
                kernel_size=1,
            ),
        )
        self.query_bias = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(p3_channels, self.num_instances, kernel_size=1),
        )
        self.aggregate_refine = DepthwiseSeparableRefine(1)
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )

    def forward(
        self,
        base_logits: torch.Tensor,
        p3: torch.Tensor,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_size = (
            max(1, image.shape[-2] // 2),
            max(1, image.shape[-1] // 2),
        )
        detail_input = F.interpolate(
            image, size=target_size, mode='bilinear', align_corners=False
        )
        detail_features = self.detail_stem(detail_input)
        semantic_features = F.interpolate(
            self.semantic_proj(p3),
            size=target_size,
            mode='bilinear',
            align_corners=False,
        )
        mask_features = self.mask_feature_proj(
            torch.cat([detail_features, semantic_features], dim=1)
        )

        batch = image.shape[0]
        queries = self.query_proj(p3).view(
            batch, self.num_instances, self.mask_channels
        )
        bias = self.query_bias(p3).view(batch, self.num_instances, 1, 1)
        instance_logits = torch.einsum(
            'bkc,bchw->bkhw', queries, mask_features
        ) / math.sqrt(float(self.mask_channels))
        instance_logits = instance_logits + bias

        aggregate_logits = torch.logsumexp(
            instance_logits, dim=1, keepdim=True
        ) - math.log(float(self.num_instances))
        aggregate_logits = self.aggregate_refine(aggregate_logits)
        aggregate_up = F.interpolate(
            aggregate_logits,
            size=base_logits.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        if self.uncertainty_gate:
            # Restrict the instance-conditioned residual to pixels where the
            # base HighRes prediction is uncertain.  This preserves confident
            # foreground/background regions and makes the branch behave as a
            # boundary/detail corrector instead of a second full mask decoder.
            base_prob = torch.sigmoid(base_logits.detach())
            uncertainty = 1.0 - 2.0 * torch.abs(base_prob - 0.5)
            uncertainty = uncertainty.clamp_(0.0, 1.0)
            if self.uncertainty_gate_power != 1.0:
                uncertainty = uncertainty.pow(self.uncertainty_gate_power)
            aggregate_up = aggregate_up * uncertainty

        residual_scale = torch.tanh(self.residual_scale) * self.residual_max_scale
        final_logits = base_logits + residual_scale * aggregate_up
        return final_logits, aggregate_up, instance_logits


class CenterlineWidthRibbonHead(nn.Module):
    """G1 geometry renderer: reconstruct a stitch ribbon from centerline + width.

    The HighRes segmentation logits remain the base prediction.  This head
    predicts a centerline and a spatial half-width map, then softly renders a
    ribbon by accumulating nearby centerline candidates.  The rendered mask is
    supervised directly, while a zero-initialized scalar adds it to the base
    logits only after it becomes useful.
    """

    def __init__(
        self,
        highres_channels: int,
        p3_channels: int,
        hidden_channels: int = 32,
        semantic_channels: int = 24,
        max_half_width: float = 5.5,
        min_half_width: float = 0.5,
        renderer_radius: int = 5,
        renderer_temperature: float = 0.75,
        positive_only_residual: bool = False,
        detach_inputs: bool = False,
    ):
        super().__init__()
        if hidden_channels < 1 or semantic_channels < 1:
            raise ValueError("G1 ribbon head channels must be positive")
        if not (0.0 < min_half_width < max_half_width):
            raise ValueError("G1 requires 0 < min_half_width < max_half_width")
        if renderer_radius < 1 or renderer_temperature <= 0:
            raise ValueError("invalid G1 renderer radius or temperature")
        self.min_half_width = float(min_half_width)
        self.max_half_width = float(max_half_width)
        self.renderer_radius = int(renderer_radius)
        self.renderer_temperature = float(renderer_temperature)
        self.positive_only_residual = bool(positive_only_residual)
        self.detach_inputs = bool(detach_inputs)
        self.p3_proj = ConvGNReLU(
            p3_channels, semantic_channels, kernel_size=1, padding=0
        )
        self.fuse = ConvGNReLU(
            highres_channels + semantic_channels, hidden_channels,
            kernel_size=1, padding=0,
        )
        self.refine = nn.Sequential(
            DepthwiseSeparableRefine(hidden_channels),
            DepthwiseSeparableRefine(hidden_channels),
        )
        self.centerline_pred = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.width_pred = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        # Keep the base HighRes output exact at initialization.  Geometry heads
        # still receive gradients through their explicit centerline/ribbon loss.
        # In G2's positive-only calibration mode the tiny positive start keeps
        # D1 logits effectively unchanged while allowing a non-negative scale
        # to receive gradients immediately.
        scale_init = -6.0 if self.positive_only_residual else 0.0
        self.geometry_scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

        offsets = []
        for dy in range(-self.renderer_radius, self.renderer_radius + 1):
            for dx in range(-self.renderer_radius, self.renderer_radius + 1):
                distance = math.sqrt(float(dx * dx + dy * dy))
                if distance <= self.renderer_radius:
                    offsets.append((dy, dx, distance))
        self.register_buffer(
            'offsets', torch.tensor(offsets, dtype=torch.float32), persistent=False
        )

    def _render_ribbon(
        self, centerline_logits: torch.Tensor, half_width: torch.Tensor
    ) -> torch.Tensor:
        """Softly dilate variable-width centerline candidates into a ribbon."""
        _, _, height, width = centerline_logits.shape
        radius = self.renderer_radius
        center_pad = F.pad(
            centerline_logits, (radius, radius, radius, radius), mode='replicate'
        )
        width_pad = F.pad(
            half_width, (radius, radius, radius, radius), mode='replicate'
        )
        aggregate = None
        for dy, dx, distance in self.offsets.tolist():
            y0 = radius + int(dy)
            x0 = radius + int(dx)
            center_shift = center_pad[:, :, y0:y0 + height, x0:x0 + width]
            width_shift = width_pad[:, :, y0:y0 + height, x0:x0 + width]
            score = center_shift + (width_shift - float(distance)) / self.renderer_temperature
            aggregate = score if aggregate is None else torch.logaddexp(aggregate, score)
        return aggregate - math.log(float(self.offsets.shape[0]))

    def forward(
        self,
        base_logits: torch.Tensor,
        highres_features: torch.Tensor,
        aligned_p3: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.detach_inputs:
            highres_features = highres_features.detach()
            aligned_p3 = aligned_p3.detach()
        p3_up = F.interpolate(
            self.p3_proj(aligned_p3), size=highres_features.shape[-2:],
            mode='bilinear', align_corners=False,
        )
        features = self.refine(self.fuse(torch.cat([highres_features, p3_up], dim=1)))
        centerline_logits = self.centerline_pred(features)
        half_width = self.min_half_width + (
            self.max_half_width - self.min_half_width
        ) * torch.sigmoid(self.width_pred(features))
        ribbon_logits = self._render_ribbon(centerline_logits, half_width)
        if self.positive_only_residual:
            # A geometric proposal can only restore marginal background pixels;
            # it can never lower a D1 foreground logit below its base value.
            positive_residual = F.relu(ribbon_logits)
            final_logits = base_logits + F.softplus(self.geometry_scale) * positive_residual
        else:
            final_logits = base_logits + torch.tanh(self.geometry_scale) * ribbon_logits
        return final_logits, ribbon_logits, centerline_logits, half_width


class ProgressiveD1HighResRefineHead(nn.Module):
    """D6 progressive reconstruction head conditioned on the aligned D1 P3.

    M4 projects its P2 edge/localization streams directly from 80x80 to the
    image plane.  D6 keeps those streams and the HighRes output role, but
    inserts a 160x160 semantic-boundary reconstruction stage.  The aligned P3
    (the output of D1's P4-to-P3 fusion) is injected at that stage, where its
    semantic support can resolve a stitch from repetitive fabric texture before
    the final 320x320 detail reconstruction.
    """

    def __init__(
        self,
        edge_channels: int,
        loc_channels: int,
        p3_channels: int,
        mid_channels: int = 32,
        p3_semantic_channels: int = 24,
        image_channels: int = 16,
        num_blocks: int = 1,
        pattern_channels: int = 0,
    ):
        super().__init__()
        self.edge_proj = ConvGNReLU(edge_channels, mid_channels, kernel_size=1, padding=0)
        self.loc_proj = ConvGNReLU(loc_channels, mid_channels, kernel_size=1, padding=0)
        self.p3_proj = ConvGNReLU(
            p3_channels, p3_semantic_channels, kernel_size=1, padding=0
        )
        self.image_mid_proj = nn.Sequential(
            ConvGNReLU(3, image_channels, kernel_size=3),
            DepthwiseSeparableRefine(image_channels),
        )
        self.mid_fuse = ConvGNReLU(
            mid_channels * 2 + p3_semantic_channels + image_channels,
            mid_channels,
            kernel_size=1,
            padding=0,
        )
        self.mid_refine = nn.Sequential(*[
            DepthwiseSeparableRefine(mid_channels)
            for _ in range(max(1, num_blocks))
        ])

        self.image_full_proj = nn.Sequential(
            ConvGNReLU(3, image_channels, kernel_size=3),
            DepthwiseSeparableRefine(image_channels),
        )
        self.full_fuse = ConvGNReLU(
            mid_channels + image_channels, mid_channels, kernel_size=1, padding=0
        )
        self.full_refine = nn.Sequential(*[
            DepthwiseSeparableRefine(mid_channels)
            for _ in range(max(1, num_blocks))
        ])
        self.pattern_proj = (
            ConvGNReLU(pattern_channels, mid_channels, kernel_size=1, padding=0)
            if pattern_channels > 0 else None
        )
        self.pattern_scale = (
            nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            if pattern_channels > 0 else None
        )
        self.final_pred = nn.Conv2d(mid_channels, 1, kernel_size=1)
        self.edge_pred = nn.Conv2d(mid_channels, 1, kernel_size=1)
        self.centerline_pred = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(
        self,
        edge_features: torch.Tensor,
        loc_features: torch.Tensor,
        image: torch.Tensor,
        p3_features: torch.Tensor,
        pattern_features: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        mid_size = (
            max(1, image.shape[-2] // 2),
            max(1, image.shape[-1] // 2),
        )
        edge_mid = F.interpolate(
            self.edge_proj(edge_features), size=mid_size,
            mode='bilinear', align_corners=False,
        )
        loc_mid = F.interpolate(
            self.loc_proj(loc_features), size=mid_size,
            mode='bilinear', align_corners=False,
        )
        p3_mid = F.interpolate(
            self.p3_proj(p3_features), size=mid_size,
            mode='bilinear', align_corners=False,
        )
        image_mid = F.interpolate(
            image, size=mid_size, mode='bilinear', align_corners=False,
        )
        mid_features = self.mid_fuse(torch.cat([
            edge_mid, loc_mid, p3_mid, self.image_mid_proj(image_mid),
        ], dim=1))
        mid_features = self.mid_refine(mid_features)

        full_features = F.interpolate(
            mid_features, size=image.shape[-2:], mode='bilinear', align_corners=False,
        )
        full_features = self.full_fuse(torch.cat([
            full_features, self.image_full_proj(image),
        ], dim=1))
        if self.pattern_proj is not None:
            if pattern_features is None:
                raise ValueError("D6 pattern projection requires pattern_features")
            pattern_full = F.interpolate(
                self.pattern_proj(pattern_features), size=image.shape[-2:],
                mode='bilinear', align_corners=False,
            )
            full_features = full_features + torch.tanh(self.pattern_scale) * pattern_full
        full_features = self.full_refine(full_features)
        outputs = (
            self.final_pred(full_features),
            self.edge_pred(full_features),
            self.centerline_pred(full_features),
        )
        return (*outputs, full_features) if return_features else outputs


class SDFBoundaryBandHead(nn.Module):
    """Training-time auxiliary head for truncated signed-distance prediction.

    The head reads the P2 localization feature before HighRes refinement.  It is
    deliberately independent of the final mask path: SDF supervision shapes the
    shared decoder representation but never replaces or post-processes the
    segmentation logits during inference.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        hidden_channels = max(16, int(hidden_channels))
        self.refine = nn.Sequential(
            ConvGNReLU(in_channels, hidden_channels, kernel_size=3),
            DepthwiseSeparableRefine(hidden_channels),
        )
        self.predict = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Raw, normalized SDF logits.  C3NetLoss applies tanh before the
        # regression loss so the prediction remains in the target range [-1, 1].
        return self.predict(self.refine(features))


class D1ConditionedContourRefiner(nn.Module):
    """Full-resolution background/boundary/core reconstruction after HighRes.

    The current HighRes logits provide a coarse mask.  P3 features already
    aligned by D1 provide semantic evidence, while the RGB branch preserves the
    pixel-level thread contour.  A three-class contour decoder predicts only a
    residual correction weighted by coarse-mask uncertainty, leaving confident
    interior/background pixels on the validated HighRes path.
    """

    def __init__(
        self,
        highres_channels: int,
        p3_channels: int,
        image_channels: int = 16,
        hidden_channels: int = 48,
    ):
        super().__init__()
        semantic_channels = max(16, hidden_channels // 2)
        self.semantic_proj = ConvGNReLU(p3_channels, semantic_channels, kernel_size=1, padding=0)
        self.image_proj = nn.Sequential(
            ConvGNReLU(3, image_channels, kernel_size=3),
            DepthwiseSeparableRefine(image_channels),
        )
        self.fuse = nn.Sequential(
            ConvGNReLU(
                highres_channels + semantic_channels + image_channels + 1,
                hidden_channels,
                kernel_size=3,
            ),
            DepthwiseSeparableRefine(hidden_channels),
        )
        self.classifier = nn.Conv2d(hidden_channels, 3, kernel_size=1)
        self.residual_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(
        self,
        highres_features: torch.Tensor,
        aligned_p3: torch.Tensor,
        image: torch.Tensor,
        coarse_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_size = coarse_logits.shape[-2:]
        semantic = F.interpolate(
            self.semantic_proj(aligned_p3), size=target_size,
            mode='bilinear', align_corners=False,
        )
        image_input = image if image.shape[-2:] == target_size else F.interpolate(
            image, size=target_size, mode='bilinear', align_corners=False
        )
        image_detail = self.image_proj(image_input)
        contour_logits = self.classifier(self.fuse(torch.cat([
            highres_features, semantic, image_detail, coarse_logits
        ], dim=1)))
        foreground_logit = torch.logsumexp(contour_logits[:, 1:], dim=1, keepdim=True)
        contour_residual = foreground_logit - contour_logits[:, :1]
        coarse_prob = torch.sigmoid(coarse_logits)
        uncertainty = 4.0 * coarse_prob * (1.0 - coarse_prob)
        final_logits = coarse_logits + (
            torch.tanh(self.residual_scale) * uncertainty * contour_residual
        )
        return final_logits, contour_logits


class StitchPatternRelationDecoder(nn.Module):
    """High-resolution periodic relation decoder for separated stitch instances.

    Instead of assuming a fixed component count or pitch, the module predicts a
    local component direction and normalized neighbour spacing.  It samples
    feature evidence from the two preceding and following predicted stitch
    positions, then fuses this relation feature with the local P2 detail.
    """

    def __init__(
        self,
        edge_channels: int,
        loc_channels: int,
        relation_channels: int = 64,
        min_spacing_norm: float = 0.06,
        max_spacing_norm: float = 0.30,
    ):
        super().__init__()
        if relation_channels < 16:
            raise ValueError("relation_channels must be >= 16")
        if not (0 < min_spacing_norm < max_spacing_norm < 1):
            raise ValueError("Require 0 < min_spacing_norm < max_spacing_norm < 1")
        self.min_spacing_norm = float(min_spacing_norm)
        self.max_spacing_norm = float(max_spacing_norm)
        self.edge_proj = ConvGNReLU(edge_channels, relation_channels, kernel_size=1, padding=0)
        self.loc_proj = ConvGNReLU(loc_channels, relation_channels, kernel_size=1, padding=0)
        self.local_fuse = nn.Sequential(
            ConvGNReLU(relation_channels * 2, relation_channels, kernel_size=3),
            DepthwiseSeparableRefine(relation_channels),
        )
        self.center_head = nn.Conv2d(relation_channels, 1, kernel_size=1)
        self.direction_head = nn.Conv2d(relation_channels, 2, kernel_size=1)
        self.spacing_head = nn.Conv2d(relation_channels, 1, kernel_size=1)
        self.relation_fuse = nn.Sequential(
            ConvGNReLU(relation_channels * 5, relation_channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(relation_channels),
        )
        self.relation_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        # Initial neighbour locations collapse to the local feature.  Auxiliary
        # direction supervision then establishes a meaningful sampling axis.
        nn.init.zeros_(self.direction_head.weight)
        nn.init.zeros_(self.direction_head.bias)

    def forward(
        self, edge_features: torch.Tensor, loc_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_size = (edge_features.shape[-2] * 2, edge_features.shape[-1] * 2)
        edge = F.interpolate(
            self.edge_proj(edge_features), size=target_size,
            mode='bilinear', align_corners=False,
        )
        loc = F.interpolate(
            self.loc_proj(loc_features), size=target_size,
            mode='bilinear', align_corners=False,
        )
        local = self.local_fuse(torch.cat([edge, loc], dim=1))
        center_logits = self.center_head(local)
        direction_raw = self.direction_head(local)
        spacing_logits = self.spacing_head(local)

        raw_direction = torch.tanh(direction_raw)
        direction_norm = torch.linalg.vector_norm(raw_direction, dim=1, keepdim=True)
        unit_direction = raw_direction / direction_norm.clamp_min(1e-4)
        spacing_norm = self.min_spacing_norm + (
            self.max_spacing_norm - self.min_spacing_norm
        ) * torch.sigmoid(spacing_logits)

        batch, _, height, width = local.shape
        y, x = torch.meshgrid(
            torch.arange(height, device=local.device, dtype=local.dtype),
            torch.arange(width, device=local.device, dtype=local.dtype),
            indexing='ij',
        )
        base_x = x.unsqueeze(0)
        base_y = y.unsqueeze(0)
        spacing_pixels = spacing_norm * float(width)
        neighbour_features = [local]
        for step in (-2.0, -1.0, 1.0, 2.0):
            sample_x = base_x + step * spacing_pixels[:, 0] * unit_direction[:, 0]
            sample_y = base_y + step * spacing_pixels[:, 0] * unit_direction[:, 1]
            grid = torch.stack([
                2.0 * sample_x / max(width - 1, 1) - 1.0,
                2.0 * sample_y / max(height - 1, 1) - 1.0,
            ], dim=-1)
            neighbour_features.append(F.grid_sample(
                local, grid, mode='bilinear', padding_mode='border', align_corners=True
            ))
        relation = self.relation_fuse(torch.cat(neighbour_features, dim=1))
        refined = local + torch.tanh(self.relation_scale) * relation
        return refined, center_logits, direction_raw, spacing_logits


class DeepContextGate(nn.Module):
    """
    Gated residual injection of P5 semantics into P4 localization context.

    The residual scale is initialized to zero by default, so enabling this
    module starts from the original P4-only behavior and learns to use P5 only
    when it helps.
    """

    def __init__(
        self,
        channels: int,
        gate_channels: int = 64,
        residual_scale_init: float = 0.0,
    ):
        super().__init__()
        gate_channels = max(16, int(gate_channels))
        self.p5_proj = ConvGNReLU(channels, channels, kernel_size=1, padding=0)
        self.gate = nn.Sequential(
            ConvGNReLU(channels * 2, gate_channels, kernel_size=1, padding=0),
            nn.Conv2d(gate_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )

    def forward(self, p4: torch.Tensor, p5: torch.Tensor) -> torch.Tensor:
        p5_context = F.interpolate(
            p5, size=p4.shape[-2:], mode='bilinear', align_corners=False
        )
        p5_context = self.p5_proj(p5_context)
        gate = self.gate(torch.cat([p4, p5_context], dim=1))
        return p4 + torch.tanh(self.residual_scale) * gate * p5_context


class TextureInvariantLateralProjection(nn.Module):
    """Style-robust lateral projection for shallow FPN stages.

    Early C2/C3 features carry fabric color, illumination and periodic texture.
    The projection keeps a content-preserving GN branch and an appearance-
    suppressing InstanceNorm branch, then recombines them before the FPN top-down
    path.  It is used only when explicitly enabled for controlled experiments.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("TextureInvariantLateralProjection channels must be positive")
        self.content_proj = ConvGNReLU(
            in_channels, out_channels, kernel_size=1, padding=0
        )
        self.style_invariant_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
        )
        self.recombine = ConvGNReLU(
            out_channels * 2, out_channels, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.recombine(torch.cat([
            self.content_proj(x),
            self.style_invariant_proj(x),
        ], dim=1))


class TextureStructurePurification(nn.Module):
    """Semantic-conditioned soft purification for textile-background features.

    The module is designed for the C3/C4 lateral features that enter the
    Mobile-FPN.  It does *not* suppress every locally repetitive response:
    stitch intervals can be repetitive too.  Instead, it factorizes a lateral
    feature into (1) a semantic-supported, directionally coherent structure
    update and (2) a local high-frequency texture residual.  Only texture that
    has weak structural and semantic support is removed through a bounded,
    learned residual.

    ``semantic_context`` is supplied by L5.  Consequently, a repeated response
    on a likely stitch trajectory remains available to the structure branch;
    densely repeated background fabric is preferentially sent to the texture
    suppression branch.  The output shape is identical to the input shape.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 64,
        texture_kernel_size: int = 5,
        residual_scale_init: float = 0.05,
    ) -> None:
        super().__init__()
        if channels < 1 or hidden_channels < 1:
            raise ValueError("TextureStructurePurification channels must be positive")
        if texture_kernel_size < 3 or texture_kernel_size % 2 == 0:
            raise ValueError("texture_kernel_size must be an odd integer >= 3")

        self.texture_kernel_size = int(texture_kernel_size)
        self.semantic_projection = ConvGNReLU(
            channels, channels, kernel_size=1, padding=0
        )

        # Structure uses the lateral response together with L5 semantic support.
        self.structure_encoder = nn.Sequential(
            ConvGNReLU(channels * 2, hidden_channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(hidden_channels),
            ConvGNReLU(hidden_channels, hidden_channels, kernel_size=3),
        )
        self.structure_update = nn.Conv2d(
            hidden_channels, channels, kernel_size=1, bias=False
        )
        self.structure_confidence = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Local high-pass residual captures the dense, repeated fabric response.
        # It is never removed by itself: suppression is conditioned on both the
        # structure confidence and the L5 semantic support below.
        self.texture_encoder = nn.Sequential(
            ConvGNReLU(channels * 2, hidden_channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(hidden_channels),
            ConvGNReLU(hidden_channels, hidden_channels, kernel_size=3),
        )
        self.texture_update = nn.Conv2d(
            hidden_channels, channels, kernel_size=1, bias=False
        )
        self.texture_confidence = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.semantic_confidence = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Bounded residual scales keep the module a soft purifier rather than a
        # hard texture filter and stabilize training on very thin stitches.
        self.structure_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )
        self.texture_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )

    def forward(
        self,
        lateral: torch.Tensor,
        semantic_context: torch.Tensor,
    ) -> torch.Tensor:
        if lateral.ndim != 4 or semantic_context.ndim != 4:
            raise ValueError("TextureStructurePurification expects BCHW tensors")
        if lateral.shape[0] != semantic_context.shape[0]:
            raise ValueError("lateral and semantic_context batch sizes must match")

        semantic = F.interpolate(
            semantic_context, size=lateral.shape[-2:], mode='bilinear',
            align_corners=False,
        )
        semantic = self.semantic_projection(semantic)

        structure_features = self.structure_encoder(
            torch.cat([lateral, semantic], dim=1)
        )
        structure_confidence = self.structure_confidence(structure_features)

        padding = self.texture_kernel_size // 2
        local_mean = F.avg_pool2d(
            lateral, kernel_size=self.texture_kernel_size, stride=1, padding=padding
        )
        local_high_frequency = lateral - local_mean
        texture_features = self.texture_encoder(
            torch.cat([lateral, local_high_frequency], dim=1)
        )
        texture_confidence = self.texture_confidence(texture_features)
        semantic_confidence = self.semantic_confidence(semantic)

        # Repeated responses are treated as removable texture only when neither
        # the structure branch nor deep semantics supports them as a stitch.
        background_texture_confidence = texture_confidence * (
            1.0 - structure_confidence
        ) * (1.0 - semantic_confidence)

        structure_residual = self.structure_update(structure_features)
        texture_residual = self.texture_update(texture_features)
        return (
            lateral
            + torch.tanh(self.structure_scale) * structure_confidence * structure_residual
            - torch.tanh(self.texture_scale) * background_texture_confidence * texture_residual
        )


class CrossScaleStitchTexturePrototypeContext(nn.Module):
    """P4-supported, P3-resolved prototype context for stitch localization.

    P4 provides global semantic support, while prototype matching and residual
    correction are performed at P3 so thin stitches are not classified solely
    on the coarse 20x20 grid.  The correction is added to the original
    ICG-enhanced P4-up context and starts as an identity mapping.
    """

    def __init__(
        self,
        channels: int,
        embedding_channels: int = 64,
        foreground_prototypes: int = 1,
        background_prototypes: int = 4,
        temperature: float = 0.10,
        residual_scale_init: float = 0.05,
    ) -> None:
        super().__init__()
        if min(channels, embedding_channels, foreground_prototypes, background_prototypes) < 1:
            raise ValueError("STPC channels and prototype counts must be positive")
        if temperature <= 0:
            raise ValueError("STPC temperature must be positive")
        self.foreground_prototypes = int(foreground_prototypes)
        self.background_prototypes = int(background_prototypes)
        self.temperature = float(temperature)
        total_prototypes = self.foreground_prototypes + self.background_prototypes

        self.p4_support = ConvGNReLU(channels, embedding_channels, kernel_size=1, padding=0)
        self.p3_query = nn.Sequential(
            ConvGNReLU(channels + embedding_channels, embedding_channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(embedding_channels),
            nn.Conv2d(embedding_channels, embedding_channels, kernel_size=1, bias=False),
        )
        self.prototypes = nn.Parameter(torch.empty(total_prototypes, embedding_channels))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)
        self.prototype_condition = nn.Linear(
            embedding_channels, total_prototypes * embedding_channels, bias=True
        )
        nn.init.zeros_(self.prototype_condition.weight)
        nn.init.zeros_(self.prototype_condition.bias)
        self.update_refine = nn.Sequential(
            ConvGNReLU(embedding_channels * 2 + 2, channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(channels),
        )
        self.update_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.update_proj.weight)
        nn.init.zeros_(self.update_proj.bias)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(
        self, p4_context: torch.Tensor, p3_detail: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if p4_context.ndim != 4 or p3_detail.ndim != 4:
            raise ValueError("STPC expects BCHW P4 and P3 features")
        if p4_context.shape[0] != p3_detail.shape[0]:
            raise ValueError("STPC P4/P3 batch sizes must match")
        p4_embed = self.p4_support(p4_context)
        p4_up = F.interpolate(
            p4_embed, size=p3_detail.shape[-2:], mode='bilinear', align_corners=False
        )
        query = F.normalize(self.p3_query(torch.cat([p3_detail, p4_up], dim=1)), dim=1)
        p4_summary = F.normalize(p4_embed.mean(dim=(-2, -1)), dim=1)
        prototype_shift = self.prototype_condition(p4_summary).view(
            p3_detail.shape[0], -1, query.shape[1]
        )
        prototypes = F.normalize(self.prototypes.unsqueeze(0) + prototype_shift, dim=-1)
        similarity = torch.einsum('bchw,bkc->bkhw', query, prototypes)
        foreground_similarity = similarity[:, :self.foreground_prototypes].amax(dim=1, keepdim=True)
        background_similarity = similarity[:, self.foreground_prototypes:].amax(dim=1, keepdim=True)
        prototype_logits = (foreground_similarity - background_similarity) / self.temperature
        assignment = torch.softmax(similarity / self.temperature, dim=1)
        retrieved = torch.einsum('bkhw,bkc->bchw', assignment, prototypes)
        update_features = self.update_refine(torch.cat([
            query, retrieved, foreground_similarity, background_similarity
        ], dim=1))
        base_context_p3 = F.interpolate(
            p4_context, size=p3_detail.shape[-2:], mode='bilinear', align_corners=False
        )
        corrected_p3 = base_context_p3 + torch.tanh(self.residual_scale) * self.update_proj(update_features)
        return corrected_p3, prototype_logits


class HorizontalContinuityEnhancement(nn.Module):
    """P4 horizontal long-range context with identity-safe residual output."""

    def __init__(self, channels: int, kernel_size: int = 15) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("horizontal continuity kernel must be odd and >= 3")
        self.horizontal_dw = nn.Conv2d(
            channels, channels, kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2), groups=channels, bias=False,
        )
        self.horizontal_pw = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(max(1, channels // 32), channels),
            nn.ReLU(inplace=True),
        )
        self.update = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.update.weight)
        nn.init.zeros_(self.update.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.mean(dim=2, keepdim=True)
        horizontal = self.horizontal_pw(self.horizontal_dw(sequence))
        horizontal = horizontal.expand(-1, -1, x.shape[-2], -1)
        return x + self.update(horizontal)


class HardNegativePrototypeResidualSuppression(nn.Module):
    """Protective P3 residual suppression driven by hard-negative prototypes.

    Three learned background prototypes are matched through normalized 1x1
    convolution.  They do not reconstruct or replace P3.  Instead, they gate
    a zero-initialized negative residual and bypass pixels with high coarse
    localization confidence.
    """

    def __init__(
        self,
        channels: int,
        embedding_channels: int = 64,
        num_prototypes: int = 3,
        threshold: float = 0.50,
        temperature: float = 0.10,
        safety_threshold: float = 0.60,
        safety_temperature: float = 0.10,
        max_suppression: float = 0.20,
    ) -> None:
        super().__init__()
        if min(channels, embedding_channels, num_prototypes) < 1:
            raise ValueError("hard-negative suppression channels must be positive")
        if temperature <= 0 or safety_temperature <= 0:
            raise ValueError("hard-negative suppression temperatures must be positive")
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.safety_threshold = float(safety_threshold)
        self.safety_temperature = float(safety_temperature)
        self.max_suppression = float(max_suppression)
        self.query = nn.Sequential(
            ConvGNReLU(channels, embedding_channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(embedding_channels),
            nn.Conv2d(embedding_channels, embedding_channels, kernel_size=1, bias=False),
        )
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, embedding_channels))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)
        self.negative_residual = nn.Sequential(
            ConvGNReLU(channels, channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(channels),
        )
        self.residual_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)
        # Coarse P4/ICG context is the pre-correction localization safety cue.
        self.safety_head = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.safety_head.weight)
        nn.init.zeros_(self.safety_head.bias)

    def forward(
        self, p4_icg_context: torch.Tensor, p3_detail: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_context = F.interpolate(
            p4_icg_context, size=p3_detail.shape[-2:], mode='bilinear', align_corners=False
        )
        query = F.normalize(self.query(p3_detail), dim=1)
        kernels = F.normalize(self.prototypes, dim=1).view(
            self.prototypes.shape[0], query.shape[1], 1, 1
        )
        similarities = F.conv2d(query, kernels)
        hard_similarity = similarities.amax(dim=1, keepdim=True)
        hard_gate_logits = (hard_similarity - self.threshold) / self.temperature
        hard_gate = torch.sigmoid(hard_gate_logits)
        safety_logits = self.safety_head(base_context)
        safety_probability = torch.sigmoid(safety_logits).detach()
        # Soft complement is differentiable with respect to neither safety nor
        # the gate.  Strong coarse stitch evidence therefore bypasses removal.
        safety_bypass = 1.0 - torch.sigmoid(
            (safety_probability - self.safety_threshold) / self.safety_temperature
        )
        residual = self.residual_proj(self.negative_residual(p3_detail))
        suppressed_context = base_context - (
            self.max_suppression * hard_gate * safety_bypass * residual
        )
        return suppressed_context, hard_gate_logits, safety_logits


class ThreadBidirectionalReconstructionPyramid(nn.Module):
    """One bottom-up structural reconstruction pass for the Mobile-FPN.

    Mobile-FPN already passes semantic context from P5 down to P2.  This
    module adds the complementary direction once: a high-frequency structural
    residual extracted at P2 is downsampled to P3 and then P4, where it
    reconstructs their semantic features through zero-initialized residuals.
    P2 itself is not modified, preserving the established edge pathway.
    """

    def __init__(self, channels: int, hidden_channels: int = 96) -> None:
        super().__init__()
        if min(channels, hidden_channels) < 1:
            raise ValueError("TBRP channels must be positive")
        self.p2_structure = nn.Sequential(
            ConvGNReLU(channels * 2, hidden_channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(hidden_channels),
        )
        self.p2_to_p3 = nn.Sequential(
            ConvGNReLU(hidden_channels, hidden_channels, kernel_size=3, stride=2),
            DepthwiseSeparableRefine(hidden_channels),
        )
        self.p3_reconstruct = nn.Sequential(
            ConvGNReLU(channels + hidden_channels, channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(channels),
        )
        self.p3_update = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.p3_to_p4 = nn.Sequential(
            ConvGNReLU(channels, hidden_channels, kernel_size=3, stride=2),
            DepthwiseSeparableRefine(hidden_channels),
        )
        self.p4_reconstruct = nn.Sequential(
            ConvGNReLU(channels + hidden_channels, channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(channels),
        )
        self.p4_update = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        for projection in (self.p3_update, self.p4_update):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(
        self, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        expected_sizes = (
            p2.shape[-2] == p3.shape[-2] * 2
            and p2.shape[-1] == p3.shape[-1] * 2
            and p3.shape[-2] == p4.shape[-2] * 2
            and p3.shape[-1] == p4.shape[-1] * 2
        )
        if not expected_sizes:
            raise ValueError("TBRP expects a consecutive P2(2x)->P3(2x)->P4 pyramid")
        local_mean = F.avg_pool2d(p2, kernel_size=3, stride=1, padding=1)
        structure_p2 = self.p2_structure(torch.cat([p2, p2 - local_mean], dim=1))
        structure_p3 = self.p2_to_p3(structure_p2)
        p3_state = self.p3_reconstruct(torch.cat([p3, structure_p3], dim=1))
        refined_p3 = p3 + self.p3_update(p3_state)
        structure_p4 = self.p3_to_p4(p3_state)
        p4_state = self.p4_reconstruct(torch.cat([p4, structure_p4], dim=1))
        refined_p4 = p4 + self.p4_update(p4_state)
        return refined_p3, refined_p4


class BoundaryControlledTriplePathAdapter(nn.Module):
    """Detail/context/boundary-control adapter for the mobile decoder.

    The existing decoder already has edge(detail) and localization(context)
    branches.  This adapter adds a third boundary-control path from P2/P3 and
    uses it to exchange only a residual correction between the two branches.
    Zero-initialized update projections make the initial network identical to
    the base M6 decoder.
    """

    def __init__(
        self,
        fpn_channels: int,
        edge_channels: int,
        loc_channels: int,
        control_channels: int = 64,
    ):
        super().__init__()
        if min(fpn_channels, edge_channels, loc_channels, control_channels) < 1:
            raise ValueError("Triple-path decoder channels must be positive")
        self.boundary_path = nn.Sequential(
            ConvGNReLU(fpn_channels * 3, control_channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(control_channels),
        )
        self.detail_proj = ConvGNReLU(
            edge_channels, control_channels, kernel_size=1, padding=0
        )
        self.context_proj = ConvGNReLU(
            loc_channels, control_channels, kernel_size=1, padding=0
        )
        self.control_refine = nn.Sequential(
            ConvGNReLU(control_channels * 3, control_channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(control_channels),
        )
        self.boundary_gate = nn.Conv2d(control_channels, 1, kernel_size=1)
        self.edge_update = nn.Conv2d(control_channels, edge_channels, kernel_size=1)
        self.loc_update = nn.Conv2d(control_channels, loc_channels, kernel_size=1)
        nn.init.zeros_(self.edge_update.weight)
        nn.init.zeros_(self.edge_update.bias)
        nn.init.zeros_(self.loc_update.weight)
        nn.init.zeros_(self.loc_update.bias)

    def forward(
        self,
        edge_features: torch.Tensor,
        loc_features: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_size = p2.shape[-2:]
        p3_up = F.interpolate(
            p3, size=target_size, mode='bilinear', align_corners=False
        )
        boundary_features = self.boundary_path(torch.cat([
            p2,
            p3_up,
            torch.abs(p2 - p3_up),
        ], dim=1))
        boundary_logits = self.boundary_gate(boundary_features)
        boundary_gate = torch.sigmoid(boundary_logits)

        detail = self.detail_proj(edge_features)
        context = self.context_proj(loc_features)
        controlled = self.control_refine(torch.cat([
            boundary_features,
            boundary_gate * detail,
            (1.0 - boundary_gate) * context,
        ], dim=1))

        edge_refined = edge_features + self.edge_update(controlled)
        loc_refined = loc_features + self.loc_update(controlled)
        return edge_refined, loc_refined, boundary_logits


class SemanticAreaLocalizationPyramid(nn.Module):
    """Multi-scale semantic area enhancement for the localization branch.

    Binary IoU is mainly affected by foreground area/width recovery, while the
    current edge and boundary metrics are already high.  This module therefore
    leaves the edge path untouched and injects only a zero-initialized semantic
    residual into loc_features.  P3/P4/P5 are upsampled to the localization
    resolution and aggregated with multiple dilation rates to provide area-level
    context that is less tied to shallow fabric texture.
    """

    @staticmethod
    def _norm(channels: int) -> nn.GroupNorm:
        groups = 32 if channels >= 32 and channels % 32 == 0 else 1
        return nn.GroupNorm(groups, channels)

    @classmethod
    def _dw_context(cls, channels: int, dilation: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            cls._norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            cls._norm(channels),
            nn.ReLU(inplace=True),
        )

    def __init__(
        self,
        fpn_channels: int,
        loc_channels: int,
        hidden_channels: int = 64,
        dilations: Tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()
        if min(fpn_channels, loc_channels, hidden_channels) < 1:
            raise ValueError("Semantic area localization channels must be positive")
        if not dilations or any(int(d) < 1 for d in dilations):
            raise ValueError("Semantic area localization dilations must be positive")

        self.semantic_reduce = ConvGNReLU(
            fpn_channels * 3, hidden_channels, kernel_size=1, padding=0
        )
        self.context_branches = nn.ModuleList([
            self._dw_context(hidden_channels, int(dilation))
            for dilation in dilations
        ])
        self.context_fuse = ConvGNReLU(
            hidden_channels * len(dilations),
            hidden_channels,
            kernel_size=1,
            padding=0,
        )
        self.loc_proj = ConvGNReLU(
            loc_channels, hidden_channels, kernel_size=1, padding=0
        )
        self.area_fuse = nn.Sequential(
            ConvGNReLU(hidden_channels * 2, hidden_channels, kernel_size=1, padding=0),
            DepthwiseSeparableRefine(hidden_channels),
        )
        self.area_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.loc_update = nn.Conv2d(hidden_channels, loc_channels, kernel_size=1)
        nn.init.zeros_(self.loc_update.weight)
        nn.init.zeros_(self.loc_update.bias)

    def forward(
        self,
        loc_features: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_size = loc_features.shape[-2:]
        p3_up = F.interpolate(p3, size=target_size, mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)
        p5_up = F.interpolate(p5, size=target_size, mode='bilinear', align_corners=False)

        semantic = self.semantic_reduce(torch.cat([p3_up, p4_up, p5_up], dim=1))
        multi_context = self.context_fuse(torch.cat([
            branch(semantic) for branch in self.context_branches
        ], dim=1))
        loc_context = self.loc_proj(loc_features)
        area_features = self.area_fuse(torch.cat([multi_context, loc_context], dim=1))
        area_logits = self.area_head(area_features)
        loc_refined = loc_features + self.loc_update(area_features)
        return loc_refined, area_logits


class TextureSemanticContradictionBranch(nn.Module):
    """Parameter-free texture-repeat cue with semantic safety for the edge path.

    Fabric distractors tend to recur in more than one non-collinear direction,
    whereas a stitch normally has high local consistency only along its tangent.
    The repeat cue itself therefore contains no learned filter: it is computed
    from multi-offset cosine correlations of P2.  A detached localization
    probability protects pixels already supported by the semantic path.

    The branch never overwrites localization features.  It applies a bounded,
    zero-start residual attenuation to edge features and supplies an additional
    texture evidence feature to a zero-initialized three-evidence adapter.
    Thus an enabled-but-untrained branch is exactly equivalent to M6.
    """

    def __init__(
        self,
        p2_channels: int,
        edge_channels: int,
        loc_channels: int,
        texture_channels: int = 128,
        offsets: Tuple[int, ...] = (1, 2, 3),
        repeat_threshold: float = 0.55,
        repeat_temperature: float = 0.10,
        semantic_gamma: float = 1.5,
        max_suppression: float = 0.25,
        suppression_init_logit: float = -8.0,
        tri_evidence_enable: bool = True,
    ) -> None:
        super().__init__()
        if not offsets or any(int(offset) < 1 for offset in offsets):
            raise ValueError("TSCB offsets must be a non-empty sequence of positive integers")
        if not 0.0 < repeat_temperature:
            raise ValueError("TSCB repeat_temperature must be positive")
        if semantic_gamma <= 0.0:
            raise ValueError("TSCB semantic_gamma must be positive")
        if not 0.0 < max_suppression <= 1.0:
            raise ValueError("TSCB max_suppression must be in (0, 1]")

        self.offsets = tuple(int(offset) for offset in offsets)
        self.repeat_threshold = float(repeat_threshold)
        self.repeat_temperature = float(repeat_temperature)
        self.semantic_gamma = float(semantic_gamma)
        self.max_suppression = float(max_suppression)
        self.tri_evidence_enable = bool(tri_evidence_enable)

        # Optional M22 adapter.  M23 disables it to isolate the intended
        # parameter-free suppression mechanism from a learned recall-biased
        # feature residual.
        if self.tri_evidence_enable:
            self.texture_project = ConvGNReLU(
                p2_channels + 2, texture_channels, kernel_size=1, padding=0
            )
            self.tri_evidence_update = nn.Conv2d(
                edge_channels + loc_channels + texture_channels,
                edge_channels,
                kernel_size=1,
                bias=True,
            )
            nn.init.zeros_(self.tri_evidence_update.weight)
            nn.init.zeros_(self.tri_evidence_update.bias)
        else:
            self.texture_project = None
            self.tri_evidence_update = None
        self.suppression_logit = nn.Parameter(
            torch.tensor(float(suppression_init_logit))
        )

    @staticmethod
    def _shift_with_validity(
        features: torch.Tensor, dy: int, dx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shift without circular boundary leakage and return a valid-pixel map."""
        shifted = torch.roll(features, shifts=(dy, dx), dims=(-2, -1))
        validity = features.new_ones((1, 1, features.shape[-2], features.shape[-1]))
        if dy > 0:
            shifted[..., :dy, :] = 0.0
            validity[..., :dy, :] = 0.0
        elif dy < 0:
            shifted[..., dy:, :] = 0.0
            validity[..., dy:, :] = 0.0
        if dx > 0:
            shifted[..., :, :dx] = 0.0
            validity[..., :, :dx] = 0.0
        elif dx < 0:
            shifted[..., :, dx:] = 0.0
            validity[..., :, dx:] = 0.0
        return shifted, validity

    def _direction_score(
        self, normalized: torch.Tensor, dy: int, dx: int
    ) -> torch.Tensor:
        responses = []
        for offset in self.offsets:
            shifted, validity = self._shift_with_validity(
                normalized, dy * offset, dx * offset
            )
            # Normalization is performed before shifting, so this is a true
            # channel-wise cosine similarity at every valid spatial location.
            response = (normalized * shifted).sum(dim=1, keepdim=True)
            responses.append(response * validity + (-1.0) * (1.0 - validity))
        return torch.stack(responses, dim=0).mean(dim=0)

    def forward(
        self,
        p2_features: torch.Tensor,
        edge_features: torch.Tensor,
        loc_features: torch.Tensor,
        loc_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_features.shape[-2:] != p2_features.shape[-2:]:
            raise ValueError("TSCB requires edge and P2 features at the same resolution")
        if loc_features.shape[-2:] != edge_features.shape[-2:]:
            raise ValueError("TSCB requires localization and edge features at the same resolution")

        normalized = F.normalize(p2_features.float(), p=2, dim=1, eps=1e-6)
        # Four non-collinear directions.  Taking the second largest response
        # rejects structures that are coherent in only one tangent direction.
        direction_scores = torch.cat([
            self._direction_score(normalized, 0, 1),
            self._direction_score(normalized, 1, 0),
            self._direction_score(normalized, 1, 1),
            self._direction_score(normalized, 1, -1),
        ], dim=1)
        repeat_score = torch.topk(direction_scores, k=2, dim=1).values[:, 1:2]
        repeat_probability = torch.sigmoid(
            (repeat_score - self.repeat_threshold) / self.repeat_temperature
        ).to(dtype=p2_features.dtype)

        semantic_probability = torch.sigmoid(loc_logits.detach())
        texture_confidence = repeat_probability * (
            1.0 - semantic_probability
        ).clamp_min(0.0).pow(self.semantic_gamma)

        # This scale is constrained to attenuation-only and starts at ~0.
        suppression = self.max_suppression * torch.sigmoid(self.suppression_logit)
        edge_suppressed = edge_features * (1.0 - suppression * texture_confidence)

        if not self.tri_evidence_enable:
            return edge_suppressed, texture_confidence
        texture_features = self.texture_project(torch.cat([
            p2_features,
            repeat_probability,
            texture_confidence,
        ], dim=1))
        edge_update = self.tri_evidence_update(torch.cat([
            edge_suppressed, loc_features, texture_features
        ], dim=1))
        return edge_suppressed + edge_update, texture_confidence


class SemanticGuidedWeakStitchRecovery(nn.Module):
    """Recover weak stitch evidence before the localization prediction.

    P4 contains the stable semantic cue but is too coarse for very thin,
    low-contrast stitches.  This branch uses its P4 context as a query over a
    *local* P2/P3 detail stream.  It deliberately does not use RGB, edge
    logits, deformable offsets, or a final-mask residual: the operation is a
    decoder-side recovery of semantic-consistent detail before localization.

    The residual injected into the existing localization path starts at zero,
    so enabling the branch preserves the M6 forward path at initialization.
    The separate weak-stitch logit is only an auxiliary training signal.
    """

    def __init__(
        self,
        fpn_channels: int,
        loc_channels: int,
        hidden_channels: int = 32,
        window_size: int = 5,
        residual_scale_init: float = 0.0,
        bidirectional_consistency: bool = False,
        detail_resampling: str = 'average',
        abstention_enable: bool = False,
        null_logit_bias_init: float = -2.0,
    ):
        super().__init__()
        if hidden_channels < 8:
            raise ValueError("weak-stitch recovery hidden_channels must be >= 8")
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("weak-stitch recovery window_size must be odd and >= 3")

        self.hidden_channels = int(hidden_channels)
        self.window_size = int(window_size)
        self.bidirectional_consistency = bool(bidirectional_consistency)
        self.abstention_enable = bool(abstention_enable)
        self.detail_resampling = str(detail_resampling).lower()
        if self.detail_resampling not in {'average', 'phase_preserving'}:
            raise ValueError(
                "weak-stitch recovery detail_resampling must be 'average' or "
                f"'phase_preserving', got {detail_resampling!r}"
            )
        # Keep the high-resolution P2 evidence explicit.  P2 is reduced only
        # for the local P3 interaction and is reintroduced at P2 afterwards.
        if self.detail_resampling == 'phase_preserving':
            # PixelUnshuffle preserves the four sub-pixel phases in every
            # 2x2 P2 cell.  It replaces average pooling only when the input
            # pyramid has the exact 2:1 P2/P3 relation; forward provides a
            # safe average-pooling fallback for non-standard input sizes.
            self.p2_to_p3 = nn.Sequential(
                nn.PixelUnshuffle(downscale_factor=2),
                ConvGNReLU(
                    fpn_channels * 4, hidden_channels, kernel_size=1, padding=0
                ),
            )
            self.p3_to_p2 = nn.Sequential(
                ConvGNReLU(
                    hidden_channels, hidden_channels * 4, kernel_size=1, padding=0
                ),
                nn.PixelShuffle(upscale_factor=2),
                DepthwiseSeparableRefine(hidden_channels),
            )
            # The fallback is parameter-free up to the same projection; this
            # path is used only for non-standard spatial shapes.
            self.p2_fallback_proj = ConvGNReLU(
                fpn_channels, hidden_channels, kernel_size=1, padding=0
            )
        else:
            self.p2_to_p3 = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                ConvGNReLU(fpn_channels, hidden_channels, kernel_size=1, padding=0),
            )
            self.p3_to_p2 = None
            self.p2_fallback_proj = None
        self.p3_detail = ConvGNReLU(
            fpn_channels, hidden_channels, kernel_size=1, padding=0
        )
        self.semantic_query = ConvGNReLU(
            fpn_channels, hidden_channels, kernel_size=1, padding=0
        )
        self.local_key = nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False)
        self.local_value = nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False)
        if self.abstention_enable:
            # A zero-valued null token lets the local retrieval abstain when
            # P4 semantics cannot validate any detail in the 5x5 window.  Its
            # initially negative logit keeps M30 close to M26 at startup.
            self.null_key = nn.Parameter(
                torch.zeros(1, hidden_channels, 1, 1, dtype=torch.float32)
            )
            self.null_logit_bias = nn.Parameter(
                torch.tensor(float(null_logit_bias_init), dtype=torch.float32)
            )
        else:
            self.null_key = None
            self.null_logit_bias = None
        self.p2_detail = ConvGNReLU(
            fpn_channels, hidden_channels, kernel_size=1, padding=0
        )
        self.recover = nn.Sequential(
            ConvGNReLU(hidden_channels * 2, hidden_channels, kernel_size=3),
            DepthwiseSeparableRefine(hidden_channels),
        )
        self.loc_delta = nn.Sequential(
            ConvGNReLU(hidden_channels, loc_channels, kernel_size=1, padding=0),
            nn.Conv2d(loc_channels, loc_channels, kernel_size=1, bias=True),
        )
        self.weak_pred = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )
        if self.bidirectional_consistency:
            # Reproject retrieved P2 evidence back to the original P4
            # semantic grid.  The verifier is a feature transformation, not a
            # mask gate: its validated residual is added to the recovery path
            # only through a zero-start scale below.
            self.context_p4_proj = ConvGNReLU(
                fpn_channels, hidden_channels, kernel_size=1, padding=0
            )
            self.recovery_to_p4 = nn.Sequential(
                nn.AvgPool2d(kernel_size=4, stride=4),
                ConvGNReLU(hidden_channels, hidden_channels, kernel_size=1, padding=0),
            )
            self.consistency_verifier = nn.Sequential(
                ConvGNReLU(hidden_channels * 4, hidden_channels, kernel_size=1, padding=0),
                DepthwiseSeparableRefine(hidden_channels),
                ConvGNReLU(hidden_channels, hidden_channels, kernel_size=3),
            )
            self.consistency_pred = nn.Conv2d(hidden_channels, 1, kernel_size=1)
            self.verification_to_p2 = ConvGNReLU(
                hidden_channels, hidden_channels, kernel_size=1, padding=0
            )
            self.verification_scale = nn.Parameter(torch.tensor(0.0))

    def _local_semantic_retrieve(
        self, query: torch.Tensor, detail: torch.Tensor
    ) -> torch.Tensor:
        """P3-window attention: semantic query selects nearby detail only."""
        b, channels, height, width = query.shape
        key = self.local_key(detail)
        value = self.local_value(detail)
        padding = self.window_size // 2
        key_windows = F.unfold(
            key, kernel_size=self.window_size, padding=padding
        ).view(b, channels, self.window_size * self.window_size, height, width)
        value_windows = F.unfold(
            value, kernel_size=self.window_size, padding=padding
        ).view(b, channels, self.window_size * self.window_size, height, width)
        affinity = (query.unsqueeze(2) * key_windows).sum(dim=1)
        affinity = affinity / math.sqrt(float(channels))
        if self.abstention_enable:
            null_key = self.null_key.expand(b, -1, height, width)
            null_affinity = (query * null_key).sum(dim=1, keepdim=True)
            null_affinity = (
                null_affinity / math.sqrt(float(channels)) + self.null_logit_bias
            )
            # The null token has a fixed zero value.  It participates in the
            # softmax normalization but contributes no recovered feature.
            affinity = torch.cat([affinity, null_affinity], dim=1)
        weights = torch.softmax(affinity, dim=1)
        local_weights = weights[:, :self.window_size * self.window_size]
        return (local_weights.unsqueeze(1) * value_windows).sum(dim=2)

    def forward(
        self,
        loc_features: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
        context_p4: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        exact_two_to_one = (
            p2.shape[-2] == p3.shape[-2] * 2
            and p2.shape[-1] == p3.shape[-1] * 2
        )
        can_use_phase = (
            self.detail_resampling == 'phase_preserving'
            and exact_two_to_one
            and p2.shape[-2] % 2 == 0
            and p2.shape[-1] % 2 == 0
        )
        semantic_p3 = F.interpolate(
            context_p4, size=p3.shape[-2:], mode='bilinear', align_corners=False
        )
        if can_use_phase:
            p2_detail_p3 = self.p2_to_p3(p2)
        elif self.detail_resampling == 'phase_preserving':
            # Preserve compatibility for odd / non-2x FPN grids.  The regular
            # 320x320 experiment always uses the phase-preserving route.
            p2_pooled = F.adaptive_avg_pool2d(p2, p3.shape[-2:])
            p2_detail_p3 = self.p2_fallback_proj(p2_pooled)
        else:
            p2_detail_p3 = self.p2_to_p3(p2)
            if p2_detail_p3.shape[-2:] != p3.shape[-2:]:
                p2_detail_p3 = F.interpolate(
                    p2_detail_p3, size=p3.shape[-2:], mode='bilinear',
                    align_corners=False,
                )
        detail_p3 = p2_detail_p3 + self.p3_detail(p3)
        retrieved_p3 = self._local_semantic_retrieve(
            self.semantic_query(semantic_p3), detail_p3
        )
        if can_use_phase:
            retrieved_p2 = self.p3_to_p2(retrieved_p3)
        else:
            retrieved_p2 = F.interpolate(
                retrieved_p3, size=p2.shape[-2:], mode='bilinear', align_corners=False
            )
        recovery_features = self.recover(
            torch.cat([retrieved_p2, self.p2_detail(p2)], dim=1)
        )
        consistency_logits = None
        if self.bidirectional_consistency:
            p4_semantic = self.context_p4_proj(context_p4)
            recovered_p4 = self.recovery_to_p4(recovery_features)
            verifier_input = torch.cat(
                [
                    p4_semantic,
                    recovered_p4,
                    p4_semantic * recovered_p4,
                    torch.abs(p4_semantic - recovered_p4),
                ],
                dim=1,
            )
            verified_p4 = self.consistency_verifier(verifier_input)
            consistency_logits = self.consistency_pred(verified_p4)
            verified_p2 = F.interpolate(
                self.verification_to_p2(verified_p4),
                size=p2.shape[-2:], mode='bilinear', align_corners=False,
            )
            recovery_features = recovery_features + torch.tanh(
                self.verification_scale
            ) * verified_p2
        weak_logits = self.weak_pred(recovery_features)
        loc_delta = self.loc_delta(recovery_features)
        recovered_loc = loc_features + torch.tanh(self.residual_scale) * loc_delta
        return recovered_loc, weak_logits, consistency_logits


class MobileFPNC3Decoder(nn.Module):
    """
    MobileNet-oriented C3 decoder.

    The decoder keeps fusion at P2 resolution instead of expanding branch features
    to full image resolution.  P4+P5 provide contextual localization, while
    P2+P3 provide boundary detail for the edge path.
    """

    def __init__(
        self,
        backbone_channels: List[int],
        fpn_channels: int = 192,
        edge_channels: int = 128,
        loc_channels: int = 128,
        fusion_channels: int = 128,
        num_edge_blocks: int = 1,
        num_context_blocks: int = 1,
        use_coord_att: bool = True,
        icg_params: Optional[Dict] = None,
        fusion_cfg: Optional[Dict] = None,
        high_res_cfg: Optional[Dict] = None,
        deep_context_cfg: Optional[Dict] = None,
        trajectory_cfg: Optional[Dict] = None,
        dynamic_upsample_cfg: Optional[Dict] = None,
        siad_conv_cfg: Optional[Dict] = None,
        c2p_alignment_cfg: Optional[Dict] = None,
        directional_edge_cfg: Optional[Dict] = None,
        cross_scale_align_cfg: Optional[Dict] = None,
        correlation_calibration_cfg: Optional[Dict] = None,
        semantic_detail_align_cfg: Optional[Dict] = None,
        p3_semantic_detail_reinforce_cfg: Optional[Dict] = None,
        bsde_cfg: Optional[Dict] = None,
        sdf_auxiliary_cfg: Optional[Dict] = None,
        stitch_pattern_cfg: Optional[Dict] = None,
        contour_refine_cfg: Optional[Dict] = None,
        instance_decoder_cfg: Optional[Dict] = None,
        thickness_router_cfg: Optional[Dict] = None,
        style_adapter_cfg: Optional[Dict] = None,
        texture_invariant_cfg: Optional[Dict] = None,
        texture_structure_purification_cfg: Optional[Dict] = None,
        prototype_context_cfg: Optional[Dict] = None,
        hard_negative_suppression_cfg: Optional[Dict] = None,
        horizontal_continuity_cfg: Optional[Dict] = None,
        bidirectional_reconstruction_cfg: Optional[Dict] = None,
        triple_path_cfg: Optional[Dict] = None,
        semantic_area_cfg: Optional[Dict] = None,
        texture_semantic_contradiction_cfg: Optional[Dict] = None,
        weak_stitch_recovery_cfg: Optional[Dict] = None,
        afm_coarse_supervision_cfg: Optional[Dict] = None,
        afm_aux_supervision_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        if len(backbone_channels) != 4:
            raise ValueError(
                f"MobileFPNC3Decoder expects 4 backbone stages, got {len(backbone_channels)}"
            )

        self.fpn_channels = fpn_channels
        self.edge_channels = edge_channels
        self.loc_channels = loc_channels
        afm_coarse_supervision_cfg = afm_coarse_supervision_cfg or {}
        self.use_afm_coarse_supervision = bool(
            afm_coarse_supervision_cfg.get('enable', False)
        )
        if self.use_afm_coarse_supervision:
            logger.info(
                "AFM coarse supervision enabled: exposing the native P2 AFM logits "
                "for auxiliary training only"
            )
        afm_aux_supervision_cfg = afm_aux_supervision_cfg or {}
        self.use_afm_aux_supervision = bool(
            afm_aux_supervision_cfg.get('enable', False)
        )
        if self.use_afm_aux_supervision:
            logger.info(
                "AFM auxiliary segmentation supervision enabled: exposing P2 AFM "
                "logits for standard full-resolution loss resizing"
            )

        siad_conv_cfg = siad_conv_cfg or {}
        self.use_siad_conv = siad_conv_cfg.get('enable', False)
        siad_dilations = tuple(siad_conv_cfg.get('dilations', [2, 4]))
        siad_gate_channels = int(siad_conv_cfg.get('gate_channels', 48))

        c2p_alignment_cfg = c2p_alignment_cfg or {}
        self.use_c2p_alignment = c2p_alignment_cfg.get('enable', False)
        c2p_gate_channels = int(c2p_alignment_cfg.get('gate_channels', 48))

        cross_scale_align_cfg = cross_scale_align_cfg or {}
        use_cross_scale_alignment = cross_scale_align_cfg.get('enable', False)
        self.use_p3_cross_scale_alignment = (
            use_cross_scale_alignment
            and cross_scale_align_cfg.get('p3_enable', True)
        )
        self.use_p2_cross_scale_alignment = (
            use_cross_scale_alignment
            and cross_scale_align_cfg.get('p2_enable', False)
        )
        cross_scale_alignment_type = str(
            cross_scale_align_cfg.get('alignment_type',
                                      cross_scale_align_cfg.get('type', 'standard'))
        ).lower()
        if cross_scale_alignment_type not in {'standard', 'direction_aware'}:
            raise ValueError(
                "cross_scale_deformable_alignment.alignment_type must be "
                "'standard' or 'direction_aware', got "
                f"{cross_scale_alignment_type!r}"
            )
        if cross_scale_alignment_type == 'direction_aware' and self.use_p2_cross_scale_alignment:
            raise ValueError(
                "direction-aware cross-scale alignment is restricted to P4-to-P3; "
                "disable p2_enable to avoid the known D2-style P2 disturbance."
            )
        if self.use_c2p_alignment and self.use_p3_cross_scale_alignment:
            raise ValueError(
                "c2p_alignment and cross_scale_deformable_alignment cannot both "
                "modify the P4-to-P3 fusion in one controlled experiment"
            )
        if self.use_p3_cross_scale_alignment:
            alignment_cls = (
                DirectionAwareCrossScaleDeformableAlignment
                if cross_scale_alignment_type == 'direction_aware'
                else CrossScaleDeformableAlignment
            )
            alignment_kwargs = {
                'channels': fpn_channels,
                'gate_channels': int(cross_scale_align_cfg.get('gate_channels', 48)),
                'max_offset': float(cross_scale_align_cfg.get('max_offset', 2.0)),
            }
            if cross_scale_alignment_type == 'direction_aware':
                alignment_kwargs.update({
                    'normal_scale': float(
                        cross_scale_align_cfg.get('normal_scale', 0.3)
                    ),
                    'direction_channels': cross_scale_align_cfg.get(
                        'direction_channels', [192, 96, 48]
                    ),
                })
            self.p3_cross_scale_alignment = alignment_cls(**alignment_kwargs)
            if cross_scale_alignment_type == 'direction_aware':
                logger.info(
                    "D13 DA-CDA-Lite enabled at P4-to-P3: "
                    f"gate_channels={alignment_kwargs['gate_channels']}, "
                    f"max_offset={alignment_kwargs['max_offset']}, "
                    f"normal_scale={alignment_kwargs['normal_scale']}"
                )
        else:
            self.p3_cross_scale_alignment = None
        if self.use_p2_cross_scale_alignment:
            self.p2_cross_scale_alignment = CrossScaleDeformableAlignment(
                channels=fpn_channels,
                gate_channels=int(cross_scale_align_cfg.get('gate_channels', 48)),
                max_offset=float(cross_scale_align_cfg.get('max_offset', 2.0)),
            )
        else:
            self.p2_cross_scale_alignment = None

        correlation_calibration_cfg = correlation_calibration_cfg or {}
        self.use_p3_correlation_calibration = bool(
            correlation_calibration_cfg.get('enable', False)
        )
        if self.use_p3_correlation_calibration and not self.use_p3_cross_scale_alignment:
            raise ValueError(
                "p3_correlation_calibration requires D1 P3 cross-scale "
                "deformable alignment to be enabled"
            )
        if self.use_p3_correlation_calibration:
            self.p3_correlation_calibrator = LocalCorrelationAlignmentCalibrator(
                channels=fpn_channels,
                correlation_channels=int(
                    correlation_calibration_cfg.get('correlation_channels', 32)
                ),
                kernel_size=int(correlation_calibration_cfg.get('kernel_size', 3)),
            )
            logger.info(
                "D7 P3 local-correlation alignment calibration enabled: "
                f"channels={correlation_calibration_cfg.get('correlation_channels', 32)}, "
                f"kernel={correlation_calibration_cfg.get('kernel_size', 3)}"
            )
        else:
            self.p3_correlation_calibrator = None

        semantic_detail_align_cfg = semantic_detail_align_cfg or {}
        self.use_p3_semantic_detail_alignment = bool(
            semantic_detail_align_cfg.get('enable', False)
        )
        if self.use_p3_semantic_detail_alignment and (
            self.use_p3_cross_scale_alignment or self.use_p2_cross_scale_alignment
        ):
            raise ValueError(
                "semantic_detail_decomposed_alignment replaces standalone "
                "cross_scale_deformable_alignment in a controlled experiment"
            )
        if self.use_p3_semantic_detail_alignment:
            self.p3_semantic_detail_alignment = SemanticDetailDecomposedAlignment(
                channels=fpn_channels,
                gate_channels=int(semantic_detail_align_cfg.get('gate_channels', 48)),
                max_offset=float(semantic_detail_align_cfg.get('max_offset', 2.0)),
                lowpass_kernel_size=int(
                    semantic_detail_align_cfg.get('lowpass_kernel_size', 5)
                ),
            )
        else:
            self.p3_semantic_detail_alignment = None

        p3_semantic_detail_reinforce_cfg = p3_semantic_detail_reinforce_cfg or {}
        self.use_p3_semantic_detail_reinforcement = bool(
            p3_semantic_detail_reinforce_cfg.get('enable', False)
        )
        if self.use_p3_semantic_detail_reinforcement and not self.use_p3_cross_scale_alignment:
            raise ValueError(
                "p3_semantic_detail_reinforcement is designed as a D1-side "
                "enhancement; enable cross_scale_deformable_alignment.p3_enable."
            )
        if self.use_p3_semantic_detail_reinforcement and self.use_p3_semantic_detail_alignment:
            raise ValueError(
                "p3_semantic_detail_reinforcement enhances D1 and cannot be "
                "combined with semantic_detail_decomposed_alignment, which "
                "replaces D1 alignment."
            )
        if self.use_p3_semantic_detail_reinforcement:
            self.p3_semantic_detail_reinforcer = P3SemanticDetailReinforcement(
                channels=fpn_channels,
                semantic_channels=int(
                    p3_semantic_detail_reinforce_cfg.get('semantic_channels', 48)
                ),
                detail_channels=int(
                    p3_semantic_detail_reinforce_cfg.get('detail_channels', 48)
                ),
                gate_channels=int(
                    p3_semantic_detail_reinforce_cfg.get('gate_channels', 48)
                ),
                lowpass_kernel_size=int(
                    p3_semantic_detail_reinforce_cfg.get('lowpass_kernel_size', 5)
                ),
                num_blocks=int(
                    p3_semantic_detail_reinforce_cfg.get('num_blocks', 1)
                ),
                residual_scale_init=float(
                    p3_semantic_detail_reinforce_cfg.get('residual_scale_init', 0.0)
                ),
            )
            logger.info(
                "D11 P3 semantic-detail reinforcement enabled after D1 alignment"
            )
        else:
            self.p3_semantic_detail_reinforcer = None

        bsde_cfg = bsde_cfg or {}
        self.use_bsde = bsde_cfg.get('enable', False)
        if self.use_bsde and (
            self.use_p3_cross_scale_alignment or self.use_p2_cross_scale_alignment
        ):
            raise ValueError(
                "BSDE already contains P4-to-P3 alignment; disable standalone "
                "cross_scale_deformable_alignment when BSDE is enabled"
            )
        if self.use_bsde and self.use_c2p_alignment:
            raise ValueError(
                "BSDE and c2p_alignment cannot both modify P4-to-P3 fusion"
            )
        if self.use_bsde and self.use_p3_semantic_detail_alignment:
            raise ValueError(
                "BSDE and semantic_detail_decomposed_alignment cannot both "
                "modify P4-to-P3 fusion"
            )
        if self.use_bsde:
            self.bsde = BidirectionalSemanticDetailExchange(
                channels=fpn_channels,
                gate_channels=int(bsde_cfg.get('gate_channels', 48)),
                max_offset=float(bsde_cfg.get('max_offset', 2.0)),
                num_blocks=int(bsde_cfg.get('num_blocks', 2)),
                residual_scale_init=float(bsde_cfg.get('residual_scale_init', 0.1)),
            )
        else:
            self.bsde = None

        # The original Mobile-FPN baseline uses fixed bilinear interpolation.
        # DySample is deliberately opt-in so that the M0 checkpoint/configuration
        # remains a clean, reproducible baseline for the LP ablation.
        dynamic_upsample_cfg = dynamic_upsample_cfg or {}
        self.use_dynamic_upsample = dynamic_upsample_cfg.get('enable', False)
        if self.use_dynamic_upsample:
            upsample_style = dynamic_upsample_cfg.get('style', 'lp')
            upsample_groups = int(dynamic_upsample_cfg.get('groups', 4))
            upsample_dyscope = bool(dynamic_upsample_cfg.get('dyscope', False))
            # One DySample module per top-down FPN transition.  The same
            # transition module is reused for the corresponding edge/context
            # alignment so every P-level resize is learned consistently.
            self.fpn_upsamplers = nn.ModuleList([
                DySample(
                    in_channels=fpn_channels,
                    scale=2,
                    style=upsample_style,
                    groups=upsample_groups,
                    dyscope=upsample_dyscope,
                )
                for _ in range(3)
            ])
            logger.info(
                "Mobile FPN dynamic upsampling enabled: "
                f"style={upsample_style}, groups={upsample_groups}, "
                f"dyscope={upsample_dyscope}"
            )
        else:
            self.fpn_upsamplers = None

        texture_invariant_cfg = texture_invariant_cfg or {}
        self.use_texture_invariant_encoder = bool(
            texture_invariant_cfg.get('enable', False)
        )
        texture_levels = set(int(level) for level in texture_invariant_cfg.get('levels', [0, 1]))
        self.lateral_convs = nn.ModuleList([
            (
                TextureInvariantLateralProjection(ch, fpn_channels)
                if self.use_texture_invariant_encoder and level in texture_levels
                else ConvGNReLU(ch, fpn_channels, kernel_size=1, padding=0)
            )
            for level, ch in enumerate(backbone_channels)
        ])
        if self.use_texture_invariant_encoder:
            logger.info(
                "M11 texture-invariant shallow encoder enabled at FPN levels: "
                f"{sorted(texture_levels)}"
            )

        texture_structure_purification_cfg = (
            texture_structure_purification_cfg or {}
        )
        self.use_texture_structure_purification = bool(
            texture_structure_purification_cfg.get('enable', False)
        )
        purification_levels = set(
            int(level) for level in texture_structure_purification_cfg.get(
                'levels', [1, 2]
            )
        )
        invalid_purification_levels = purification_levels.difference({1, 2})
        if invalid_purification_levels:
            raise ValueError(
                "texture_structure_purification is limited to C3/C4 lateral "
                f"features (levels 1/2), got {sorted(invalid_purification_levels)}"
            )
        self.texture_structure_purifiers = nn.ModuleDict({
            str(level): TextureStructurePurification(
                channels=fpn_channels,
                hidden_channels=int(
                    texture_structure_purification_cfg.get('hidden_channels', 64)
                ),
                texture_kernel_size=int(
                    texture_structure_purification_cfg.get('texture_kernel_size', 5)
                ),
                residual_scale_init=float(
                    texture_structure_purification_cfg.get('residual_scale_init', 0.05)
                ),
            )
            for level in sorted(purification_levels)
        }) if self.use_texture_structure_purification else nn.ModuleDict()
        if self.use_texture_structure_purification:
            logger.info(
                "M13 texture-structure purification enabled before Mobile-FPN "
                f"at C3/C4 levels: {sorted(purification_levels)}"
            )

        bidirectional_reconstruction_cfg = bidirectional_reconstruction_cfg or {}
        self.use_bidirectional_reconstruction = bool(
            bidirectional_reconstruction_cfg.get('enable', False)
        )
        if self.use_bidirectional_reconstruction:
            # Restore the RNG after constructing the added branch so that all
            # existing M6 layers retain same-seed initialization.
            rng_state = torch.get_rng_state()
            self.bidirectional_reconstruction = ThreadBidirectionalReconstructionPyramid(
                channels=fpn_channels,
                hidden_channels=int(
                    bidirectional_reconstruction_cfg.get('hidden_channels', 96)
                ),
            )
            torch.set_rng_state(rng_state)
            logger.info(
                "M15 TBRP enabled: single P2->P3->P4 structural reconstruction, hidden=%d",
                int(bidirectional_reconstruction_cfg.get('hidden_channels', 96)),
            )
        else:
            self.bidirectional_reconstruction = None
        # P2/P3/P4 retain the spatial stitch details that SIAD-Conv is meant
        # to contrast.  P5 stays lightweight because its 10x10 feature map is
        # already a coarse semantic context map.
        self.smooth_convs = nn.ModuleList([
            StitchIntervalAwareDifferenceConv(
                fpn_channels,
                dilations=siad_dilations,
                gate_channels=siad_gate_channels,
            ) if self.use_siad_conv and level < 3
            else DepthwiseSeparableRefine(fpn_channels)
            for level in range(4)
        ])
        if self.use_siad_conv:
            logger.info(
                "SIAD-Conv enabled at P2/P3/P4: "
                f"dilations={siad_dilations}, gate_channels={siad_gate_channels}"
            )
        if self.use_c2p_alignment:
            # P5→P4, P4→P3, and ICG-enhanced P4 context→P3 alignment.
            self.topdown_alignments = nn.ModuleList([
                C2PContextAlignment(fpn_channels, c2p_gate_channels)
                for _ in range(2)
            ])
            self.localization_alignment = C2PContextAlignment(
                fpn_channels, c2p_gate_channels
            )
            logger.info(
                "C2P context alignment enabled at P4/P3 and localization P3: "
                f"gate_channels={c2p_gate_channels}"
            )
        else:
            self.topdown_alignments = None
            self.localization_alignment = None

        self.use_coord_att = use_coord_att
        if use_coord_att:
            # Apply CoordAtt only to semantic stages P3-P5; P2 is kept lean.
            self.coord_att_modules = nn.ModuleList([
                CoordAtt(fpn_channels, fpn_channels, reduction=16)
                for _ in range(3)
            ])
        else:
            self.coord_att_modules = None

        deep_context_cfg = deep_context_cfg or {}
        self.use_deep_context_gate = deep_context_cfg.get('enable', False)
        if self.use_deep_context_gate:
            self.deep_context_gate = DeepContextGate(
                channels=fpn_channels,
                gate_channels=deep_context_cfg.get('gate_channels', 64),
                residual_scale_init=deep_context_cfg.get(
                    'residual_scale_init', 0.0
                ),
            )
        else:
            self.deep_context_gate = None

        trajectory_cfg = trajectory_cfg or {}
        use_mrfa_dte = trajectory_cfg.get('enable', True)
        edge_blocks: List[nn.Module] = []
        if use_mrfa_dte:
            enhancer_type = str(trajectory_cfg.get('type', 'mrfa')).lower()
            if enhancer_type not in {'mrfa', 'ccte'}:
                raise ValueError(
                    "trajectory_enhance.type must be 'mrfa' or 'ccte', got "
                    f"{enhancer_type!r}"
                )
            num_dte_blocks = max(
                1, int(trajectory_cfg.get('num_blocks', max(1, num_edge_blocks)))
            )
            strip_kernel_size = int(trajectory_cfg.get('strip_kernel_size', 7))
            dilations = tuple(trajectory_cfg.get('dilations', [2, 3]))
            gate_channels = int(trajectory_cfg.get('gate_channels', 32))
            residual_scale_init = float(
                trajectory_cfg.get('residual_scale_init', 0.2)
            )

            if enhancer_type == 'ccte':
                edge_blocks.append(
                    ContrastContextTrajectoryEnhancer(
                        fpn_channels * 2,
                        edge_channels,
                        dilations=dilations,
                        gate_channels=gate_channels,
                        residual_scale_init=residual_scale_init,
                    )
                )
            else:
                edge_blocks.append(
                    MRFADTEBlock(
                        fpn_channels * 2,
                        edge_channels,
                        strip_kernel_size=strip_kernel_size,
                        dilations=dilations,
                        gate_channels=gate_channels,
                        residual_scale_init=residual_scale_init,
                    )
                )
            for _ in range(num_dte_blocks - 1):
                if enhancer_type == 'ccte':
                    edge_blocks.append(
                        ContrastContextTrajectoryEnhancer(
                            edge_channels,
                            edge_channels,
                            dilations=dilations,
                            gate_channels=gate_channels,
                            residual_scale_init=residual_scale_init,
                        )
                    )
                else:
                    edge_blocks.append(
                        MRFADTEBlock(
                            edge_channels,
                            edge_channels,
                            strip_kernel_size=strip_kernel_size,
                            dilations=dilations,
                            gate_channels=gate_channels,
                            residual_scale_init=residual_scale_init,
                        )
                    )
            edge_prior_blocks = int(
                trajectory_cfg.get('edge_prior_blocks', max(0, num_edge_blocks))
            )
            edge_blocks.extend([
                EdgeEnhanceBlock(edge_channels) for _ in range(edge_prior_blocks)
            ])
        else:
            edge_blocks.append(
                ConvGNReLU(fpn_channels * 2, edge_channels, kernel_size=3)
            )
            edge_blocks.extend([
                EdgeEnhanceBlock(edge_channels) for _ in range(num_edge_blocks)
            ])
        self.edge_refine = nn.Sequential(*edge_blocks)

        directional_edge_cfg = directional_edge_cfg or {}
        self.use_directional_edge_refine = directional_edge_cfg.get('enable', False)
        if self.use_directional_edge_refine:
            self.directional_edge_refine = DirectionalConfidenceEdgeRefine(
                channels=edge_channels,
                kernel_size=int(directional_edge_cfg.get('kernel_size', 7)),
                gate_channels=int(directional_edge_cfg.get('gate_channels', 32)),
            )
        else:
            self.directional_edge_refine = None

        context_blocks: List[nn.Module] = [
            SemanticEnhanceBlock(fpn_channels)
            for _ in range(num_context_blocks)
        ]
        self.context_refine = nn.Sequential(*context_blocks)
        horizontal_continuity_cfg = horizontal_continuity_cfg or {}
        self.use_horizontal_continuity = bool(horizontal_continuity_cfg.get('enable', False))
        if self.use_horizontal_continuity:
            rng_state = torch.get_rng_state()
            self.horizontal_continuity = HorizontalContinuityEnhancement(
                channels=fpn_channels,
                kernel_size=int(horizontal_continuity_cfg.get('kernel_size', 15)),
            )
            torch.set_rng_state(rng_state)
            logger.info(
                "M16 horizontal P4 continuity enabled: 1x%d depthwise kernel",
                int(horizontal_continuity_cfg.get('kernel_size', 15)),
            )
        else:
            self.horizontal_continuity = None
        prototype_context_cfg = prototype_context_cfg or {}
        self.use_prototype_context = bool(prototype_context_cfg.get('enable', False))
        if self.use_prototype_context:
            # Do not let the new branch perturb initialization of any existing
            # M6 layers constructed afterwards.  This makes a same-seed M6/M14
            # comparison attributable to STPC rather than random init order.
            rng_state = torch.get_rng_state()
            self.prototype_context = CrossScaleStitchTexturePrototypeContext(
                channels=fpn_channels,
                embedding_channels=int(prototype_context_cfg.get('embedding_channels', 64)),
                foreground_prototypes=int(prototype_context_cfg.get('foreground_prototypes', 1)),
                background_prototypes=int(prototype_context_cfg.get('background_prototypes', 4)),
                temperature=float(prototype_context_cfg.get('temperature', 0.10)),
                residual_scale_init=float(prototype_context_cfg.get('residual_scale_init', 0.05)),
            )
            torch.set_rng_state(rng_state)
            logger.info(
                "M14 STPC calibrates ICG context at P3: P4 support -> P3 queries, K_fg=%d, K_bg=%d",
                self.prototype_context.foreground_prototypes,
                self.prototype_context.background_prototypes,
            )
        else:
            self.prototype_context = None
        hard_negative_suppression_cfg = hard_negative_suppression_cfg or {}
        self.use_hard_negative_suppression = bool(
            hard_negative_suppression_cfg.get('enable', False)
        )
        if self.use_hard_negative_suppression and self.use_prototype_context:
            raise ValueError(
                "prototype_context and hard_negative_suppression both alter the ICG P3 "
                "return path; enable only one in a controlled experiment"
            )
        if self.use_hard_negative_suppression:
            rng_state = torch.get_rng_state()
            self.hard_negative_suppression = HardNegativePrototypeResidualSuppression(
                channels=fpn_channels,
                embedding_channels=int(hard_negative_suppression_cfg.get('embedding_channels', 64)),
                num_prototypes=int(hard_negative_suppression_cfg.get('num_prototypes', 3)),
                threshold=float(hard_negative_suppression_cfg.get('threshold', 0.50)),
                temperature=float(hard_negative_suppression_cfg.get('temperature', 0.10)),
                safety_threshold=float(hard_negative_suppression_cfg.get('safety_threshold', 0.60)),
                safety_temperature=float(hard_negative_suppression_cfg.get('safety_temperature', 0.10)),
                max_suppression=float(hard_negative_suppression_cfg.get('max_suppression', 0.20)),
            )
            torch.set_rng_state(rng_state)
            logger.info("M16 hard-negative prototype suppression enabled: K=%d", int(
                hard_negative_suppression_cfg.get('num_prototypes', 3)
            ))
        else:
            self.hard_negative_suppression = None
        self.icg_module = ImageBasedContextGuidance(
            feature_channels=fpn_channels,
            **(icg_params or {})
        )
        self.loc_refine = nn.Sequential(
            ConvGNReLU(fpn_channels, loc_channels, kernel_size=3),
            DepthwiseSeparableRefine(loc_channels),
        )

        weak_stitch_recovery_cfg = weak_stitch_recovery_cfg or {}
        self.use_weak_stitch_recovery = bool(
            weak_stitch_recovery_cfg.get('enable', False)
        )
        if self.use_weak_stitch_recovery:
            # Creating an optional module must not change M6's seeded
            # initialization downstream (AFM / HighRes) in a fair ablation.
            rng_state = torch.get_rng_state()
            self.weak_stitch_recovery = SemanticGuidedWeakStitchRecovery(
                fpn_channels=fpn_channels,
                loc_channels=loc_channels,
                hidden_channels=int(weak_stitch_recovery_cfg.get('hidden_channels', 32)),
                window_size=int(weak_stitch_recovery_cfg.get('window_size', 5)),
                residual_scale_init=float(
                    weak_stitch_recovery_cfg.get('residual_scale_init', 0.0)
                ),
                bidirectional_consistency=bool(
                    weak_stitch_recovery_cfg.get('bidirectional_consistency', False)
                ),
                detail_resampling=str(
                    weak_stitch_recovery_cfg.get('detail_resampling', 'average')
                ),
                abstention_enable=bool(
                    weak_stitch_recovery_cfg.get('abstention_enable', False)
                ),
                null_logit_bias_init=float(
                    weak_stitch_recovery_cfg.get('null_logit_bias_init', -2.0)
                ),
            )
            torch.set_rng_state(rng_state)
            logger.info(
                "M26 semantic-guided weak-stitch recovery enabled: "
                "P4 query -> local P2/P3 evidence, hidden=%d, window=%d, "
                "bidirectional=%s, resampling=%s, abstention=%s",
                int(weak_stitch_recovery_cfg.get('hidden_channels', 32)),
                int(weak_stitch_recovery_cfg.get('window_size', 5)),
                bool(weak_stitch_recovery_cfg.get('bidirectional_consistency', False)),
                str(weak_stitch_recovery_cfg.get('detail_resampling', 'average')),
                bool(weak_stitch_recovery_cfg.get('abstention_enable', False)),
            )
        else:
            self.weak_stitch_recovery = None

        triple_path_cfg = triple_path_cfg or {}
        self.use_triple_path_decoder = bool(triple_path_cfg.get('enable', False))
        if self.use_triple_path_decoder:
            self.triple_path_adapter = BoundaryControlledTriplePathAdapter(
                fpn_channels=fpn_channels,
                edge_channels=edge_channels,
                loc_channels=loc_channels,
                control_channels=int(triple_path_cfg.get('control_channels', 64)),
            )
            logger.info(
                "M11 boundary-controlled triple-path decoder adapter enabled"
            )
        else:
            self.triple_path_adapter = None

        semantic_area_cfg = semantic_area_cfg or {}
        self.use_semantic_area_localization = bool(
            semantic_area_cfg.get('enable', False)
        )
        if self.use_semantic_area_localization:
            self.semantic_area_localizer = SemanticAreaLocalizationPyramid(
                fpn_channels=fpn_channels,
                loc_channels=loc_channels,
                hidden_channels=int(semantic_area_cfg.get('hidden_channels', 64)),
                dilations=tuple(semantic_area_cfg.get('dilations', [1, 2, 4])),
            )
            logger.info(
                "M12 semantic area localization pyramid enabled for loc branch"
            )
        else:
            self.semantic_area_localizer = None

        texture_semantic_contradiction_cfg = texture_semantic_contradiction_cfg or {}
        self.use_texture_semantic_contradiction = bool(
            texture_semantic_contradiction_cfg.get('enable', False)
        )
        if self.use_texture_semantic_contradiction:
            # Preserve the seeded M6 initialization of all following modules
            # (AFM/HighRes included).  TSCB is an optional residual branch and
            # must not alter the base model merely by consuming RNG samples.
            rng_state = torch.get_rng_state()
            self.texture_semantic_contradiction = TextureSemanticContradictionBranch(
                p2_channels=fpn_channels,
                edge_channels=edge_channels,
                loc_channels=loc_channels,
                texture_channels=int(
                    texture_semantic_contradiction_cfg.get('texture_channels', 128)
                ),
                offsets=tuple(
                    texture_semantic_contradiction_cfg.get('offsets', [1, 2, 3])
                ),
                repeat_threshold=float(
                    texture_semantic_contradiction_cfg.get('repeat_threshold', 0.55)
                ),
                repeat_temperature=float(
                    texture_semantic_contradiction_cfg.get('repeat_temperature', 0.10)
                ),
                semantic_gamma=float(
                    texture_semantic_contradiction_cfg.get('semantic_gamma', 1.5)
                ),
                max_suppression=float(
                    texture_semantic_contradiction_cfg.get('max_suppression', 0.25)
                ),
                suppression_init_logit=float(
                    texture_semantic_contradiction_cfg.get('suppression_init_logit', -8.0)
                ),
                tri_evidence_enable=bool(
                    texture_semantic_contradiction_cfg.get('tri_evidence_enable', True)
                ),
            )
            torch.set_rng_state(rng_state)
            logger.info(
                "TSCB-v2 enabled: parameter-free P2 repeat cue + detached "
                "localization safety + zero-start tri-evidence edge update"
            )
        else:
            self.texture_semantic_contradiction = None

        fusion_cfg = fusion_cfg or {}
        fusion_type = str(fusion_cfg.get('type', 'attentive')).lower()
        fusion_kwargs = {
            'edge_channels': edge_channels,
            'loc_channels': loc_channels,
            'fuse_channels': fusion_channels,
            'num_refine_blocks': fusion_cfg.get('num_refine_blocks', 1),
        }
        if fusion_type == 'attentive':
            self.fusion_head = FinalFusionHead(
                **fusion_kwargs,
                attn_gen_intermediate_channels=fusion_cfg.get(
                    'attn_gen_intermediate_channels',
                    max(16, fusion_channels // 2),
                ),
            )
        elif fusion_type == 'plain':
            self.fusion_head = PlainFusionHead(**fusion_kwargs)
        else:
            raise ValueError(
                "decoder.fusion_head.type must be 'attentive' or 'plain', "
                f"got {fusion_type!r}"
            )
        self.edge_pred_head = nn.Conv2d(edge_channels, 1, kernel_size=1)
        self.loc_pred_head = nn.Conv2d(loc_channels, 1, kernel_size=1)
        self.centerline_pred_head = nn.Conv2d(edge_channels, 1, kernel_size=1)

        thickness_router_cfg = thickness_router_cfg or {}
        self.use_thickness_router = bool(thickness_router_cfg.get('enable', False))
        if self.use_thickness_router:
            self.thickness_router = ThicknessAwareBranchRouter(
                edge_channels=edge_channels,
                loc_channels=loc_channels,
                hidden_channels=int(thickness_router_cfg.get('hidden_channels', 64)),
                image_channels=int(thickness_router_cfg.get('image_channels', 12)),
                strip_kernel_size=int(thickness_router_cfg.get('strip_kernel_size', 7)),
                dilation_rates=tuple(thickness_router_cfg.get('dilation_rates', [2, 4])),
                width_center=float(thickness_router_cfg.get('width_center', 0.45)),
                width_temperature=float(
                    thickness_router_cfg.get('width_temperature', 0.08)
                ),
            )
            logger.info(
                "M7 thickness-aware branch router enabled before AFM/HighRes"
            )
        else:
            self.thickness_router = None

        style_adapter_cfg = style_adapter_cfg or {}
        self.use_style_conditioned_adapter = bool(style_adapter_cfg.get('enable', False))
        if self.use_style_conditioned_adapter:
            if self.thickness_router is not None:
                logger.warning(
                    "Both style_conditioned_adapter and thickness_router are enabled; "
                    "for controlled ablation keep only one of them enabled."
                )
            self.style_conditioned_adapter = StyleConditionedDecoderAdapter(
                edge_channels=edge_channels,
                loc_channels=loc_channels,
                hidden_channels=int(style_adapter_cfg.get('hidden_channels', 64)),
                image_channels=int(style_adapter_cfg.get('image_channels', 12)),
                style_channels=int(style_adapter_cfg.get('style_channels', 32)),
                strip_kernel_size=int(style_adapter_cfg.get('strip_kernel_size', 7)),
                dilation_rate=int(style_adapter_cfg.get('dilation_rate', 3)),
            )
            logger.info(
                "M8 style-conditioned decoder adapter enabled before AFM/HighRes"
            )
        else:
            self.style_conditioned_adapter = None

        sdf_auxiliary_cfg = sdf_auxiliary_cfg or {}
        self.use_sdf_auxiliary = bool(sdf_auxiliary_cfg.get('enable', False))
        if self.use_sdf_auxiliary:
            self.sdf_auxiliary_head = SDFBoundaryBandHead(
                in_channels=loc_channels,
                hidden_channels=int(sdf_auxiliary_cfg.get('hidden_channels', 64)),
            )
        else:
            self.sdf_auxiliary_head = None

        stitch_pattern_cfg = stitch_pattern_cfg or {}
        self.use_stitch_pattern_relation = bool(stitch_pattern_cfg.get('enable', False))
        if self.use_stitch_pattern_relation:
            relation_channels = int(stitch_pattern_cfg.get('relation_channels', 64))
            self.stitch_pattern_decoder = StitchPatternRelationDecoder(
                edge_channels=edge_channels,
                loc_channels=loc_channels,
                relation_channels=relation_channels,
                min_spacing_norm=float(stitch_pattern_cfg.get('min_spacing_norm', 0.06)),
                max_spacing_norm=float(stitch_pattern_cfg.get('max_spacing_norm', 0.30)),
            )
            self.stitch_pattern_channels = relation_channels
        else:
            self.stitch_pattern_decoder = None
            self.stitch_pattern_channels = 0

        high_res_cfg = high_res_cfg or {}
        self.use_high_res_refine = high_res_cfg.get('enable', False)
        self.use_progressive_d1_high_res = bool(
            high_res_cfg.get('progressive_d1_enable', False)
        )
        if self.use_progressive_d1_high_res and not self.use_high_res_refine:
            raise ValueError(
                "progressive_d1_enable requires high_res_refine.enable=true"
            )
        if self.use_high_res_refine:
            if self.use_progressive_d1_high_res:
                self.high_res_refine = ProgressiveD1HighResRefineHead(
                    edge_channels=edge_channels,
                    loc_channels=loc_channels,
                    p3_channels=fpn_channels,
                    mid_channels=int(high_res_cfg.get('mid_channels', 32)),
                    p3_semantic_channels=int(
                        high_res_cfg.get('p3_semantic_channels', 24)
                    ),
                    image_channels=int(high_res_cfg.get('image_channels', 16)),
                    num_blocks=int(high_res_cfg.get('num_blocks', 1)),
                    pattern_channels=self.stitch_pattern_channels,
                )
                logger.info(
                    "D6 progressive HighRes reconstruction enabled: D1 P3 -> "
                    "160x160 semantic-boundary stage -> 320x320 detail stage"
                )
            else:
                self.high_res_refine = HighResStitchRefineHead(
                    edge_channels=edge_channels,
                    loc_channels=loc_channels,
                    refine_channels=high_res_cfg.get('refine_channels', 48),
                    image_channels=high_res_cfg.get('image_channels', 16),
                    num_blocks=high_res_cfg.get('num_blocks', 1),
                    output_stride=high_res_cfg.get('output_stride', 1),
                    pattern_channels=self.stitch_pattern_channels,
                    use_afm_prior=bool(
                        high_res_cfg.get('afm_guidance', {}).get('enable', False)
                    ),
                )
                if high_res_cfg.get('afm_guidance', {}).get('enable', False):
                    logger.info(
                        "AFM-guided HighRes enabled: AFM probability and "
                        "uncertainty are fused into HighRes refinement"
                    )
        else:
            self.high_res_refine = None

        agreement_calibration_cfg = high_res_cfg.get('agreement_calibration', {})
        self.use_agreement_calibration = bool(
            agreement_calibration_cfg.get('enable', False)
        )
        if self.use_agreement_calibration and self.high_res_refine is None:
            raise ValueError("agreement_calibration requires high_res_refine.enable=true")
        if self.use_agreement_calibration and self.use_progressive_d1_high_res:
            raise ValueError(
                "agreement_calibration is defined for the M6 HighRes head, not D6 progressive HighRes"
            )
        if self.use_agreement_calibration:
            self.agreement_calibrator = AgreementGuidedAreaBoundaryCalibrator(
                highres_channels=int(high_res_cfg.get('refine_channels', 48)),
                edge_channels=edge_channels,
                loc_channels=loc_channels,
                semantic_channels=int(
                    agreement_calibration_cfg.get('semantic_channels', 16)
                ),
                hidden_channels=int(
                    agreement_calibration_cfg.get('hidden_channels', 32)
                ),
                max_residual=float(
                    agreement_calibration_cfg.get('max_residual', 1.0)
                ),
            )
            logger.info(
                "M21 agreement-guided area-boundary calibration enabled after HighRes"
            )
        else:
            self.agreement_calibrator = None

        foreground_completion_cfg = high_res_cfg.get('foreground_completion', {})
        self.use_foreground_completion = bool(
            foreground_completion_cfg.get('enable', False)
        )
        if self.use_foreground_completion and self.high_res_refine is None:
            raise ValueError("D15 foreground_completion requires high_res_refine.enable=true")
        if self.use_foreground_completion and self.use_progressive_d1_high_res:
            raise ValueError(
                "D15 foreground_completion is defined for the original D1/M4 "
                "HighRes head, not the D6 progressive head"
            )
        if self.use_foreground_completion:
            self.foreground_completion_refiner = ForegroundCompletionResidualHead(
                highres_channels=int(high_res_cfg.get('refine_channels', 48)),
                hidden_channels=int(
                    foreground_completion_cfg.get('hidden_channels', 32)
                ),
                max_residual=float(
                    foreground_completion_cfg.get('max_residual', 2.0)
                ),
                residual_scale_init=float(
                    foreground_completion_cfg.get('residual_scale_init', -4.0)
                ),
                uncertainty_floor=float(
                    foreground_completion_cfg.get('uncertainty_floor', 0.35)
                ),
            )
            logger.info(
                "D15 foreground-completion positive residual head enabled after HighRes"
            )
        else:
            self.foreground_completion_refiner = None

        width_reconstruction_cfg = high_res_cfg.get('width_reconstruction', {})
        self.use_width_reconstruction = bool(
            width_reconstruction_cfg.get('enable', False)
        )
        if self.use_width_reconstruction and self.high_res_refine is None:
            raise ValueError("B7 width_reconstruction requires high_res_refine.enable=true")
        if self.use_width_reconstruction and self.use_progressive_d1_high_res:
            raise ValueError(
                "B7 width_reconstruction is defined for the original M4 HighRes "
                "head, not the D6 progressive head"
            )
        if self.use_width_reconstruction and self.use_foreground_completion:
            raise ValueError(
                "B7 width_reconstruction and D15 foreground_completion are separate experiments"
            )
        if self.use_width_reconstruction:
            self.width_reconstruction_decoder = WidthAwareReconstructionDecoder(
                highres_channels=int(high_res_cfg.get('refine_channels', 48)),
                edge_channels=edge_channels,
                loc_channels=loc_channels,
                hidden_channels=int(width_reconstruction_cfg.get('hidden_channels', 64)),
                prior_channels=int(width_reconstruction_cfg.get('prior_channels', 24)),
                image_channels=int(width_reconstruction_cfg.get('image_channels', 16)),
                strip_kernel_sizes=tuple(
                    width_reconstruction_cfg.get('strip_kernel_sizes', [3, 5, 7])
                ),
                dilation_rates=tuple(
                    width_reconstruction_cfg.get('dilation_rates', [2, 4])
                ),
                num_refine_blocks=int(
                    width_reconstruction_cfg.get('num_refine_blocks', 2)
                ),
                max_residual=float(
                    width_reconstruction_cfg.get('max_residual', 2.0)
                ),
                uncertainty_floor=float(
                    width_reconstruction_cfg.get('uncertainty_floor', 0.15)
                ),
            )
            logger.info(
                "B7 WRD enabled after M4 HighRes: width-aware residual "
                "reconstruction for Binary-IoU correction"
            )
        else:
            self.width_reconstruction_decoder = None

        narrow_band_resampling_cfg = high_res_cfg.get('narrow_band_resampling', {})
        self.use_narrow_band_resampling = bool(
            narrow_band_resampling_cfg.get('enable', False)
        )
        if self.use_narrow_band_resampling and self.high_res_refine is None:
            raise ValueError("M10 narrow_band_resampling requires high_res_refine.enable=true")
        if self.use_narrow_band_resampling and self.use_progressive_d1_high_res:
            raise ValueError(
                "M10 narrow_band_resampling is defined for the original M4/M6 "
                "HighRes head, not the D6 progressive head"
            )
        if self.use_narrow_band_resampling and (
            self.use_foreground_completion or self.use_width_reconstruction
        ):
            raise ValueError(
                "M10 narrow_band_resampling should be evaluated separately "
                "from D15 foreground_completion and B7 width_reconstruction"
            )
        if self.use_narrow_band_resampling:
            self.narrow_band_resampler = MaskGuidedNarrowBandResamplingHead(
                highres_channels=int(high_res_cfg.get('refine_channels', 48)),
                hidden_channels=int(
                    narrow_band_resampling_cfg.get('hidden_channels', 32)
                ),
                image_channels=int(
                    narrow_band_resampling_cfg.get('image_channels', 12)
                ),
                upsample_scale=int(
                    narrow_band_resampling_cfg.get('upsample_scale', 2)
                ),
                uncertainty_temperature=float(
                    narrow_band_resampling_cfg.get('uncertainty_temperature', 8.0)
                ),
                band_dilation=int(
                    narrow_band_resampling_cfg.get('band_dilation', 3)
                ),
                max_residual=float(
                    narrow_band_resampling_cfg.get('max_residual', 1.5)
                ),
            )
            logger.info(
                "M10 mask-guided narrow-band local resampling enabled after HighRes"
            )
        else:
            self.narrow_band_resampler = None

        boundary_residual_cfg = high_res_cfg.get('boundary_residual', {})
        self.use_boundary_residual_refine = bool(
            boundary_residual_cfg.get('enable', False)
        )
        if self.use_boundary_residual_refine and self.high_res_refine is None:
            raise ValueError(
                "D7 boundary_residual requires high_res_refine.enable=true"
            )
        if self.use_boundary_residual_refine and self.use_progressive_d1_high_res:
            raise ValueError(
                "D7 boundary_residual and D6 progressive HighRes are separate "
                "experiments and cannot be enabled together"
            )
        if self.use_boundary_residual_refine and self.use_foreground_completion:
            raise ValueError(
                "D7 boundary_residual and D15 foreground_completion are separate experiments"
            )
        if self.use_boundary_residual_refine and self.use_width_reconstruction:
            raise ValueError(
                "D7 boundary_residual and B7 width_reconstruction are separate experiments"
            )
        if self.use_boundary_residual_refine:
            self.boundary_residual_refiner = BasePreservingBoundaryResidualHead(
                edge_channels=edge_channels,
                loc_channels=loc_channels,
                p3_channels=fpn_channels,
                hidden_channels=int(boundary_residual_cfg.get('hidden_channels', 32)),
                semantic_channels=int(
                    boundary_residual_cfg.get('semantic_channels', 24)
                ),
                image_channels=int(boundary_residual_cfg.get('image_channels', 12)),
                num_blocks=int(boundary_residual_cfg.get('num_blocks', 1)),
            )
            logger.info(
                "D7 base-preserving 160x160 boundary residual HighRes refinement enabled"
            )
        else:
            self.boundary_residual_refiner = None

        geometry_ribbon_cfg = high_res_cfg.get('geometry_ribbon', {})
        self.use_geometry_ribbon = bool(geometry_ribbon_cfg.get('enable', False))
        if self.use_geometry_ribbon and self.high_res_refine is None:
            raise ValueError("G1 geometry_ribbon requires high_res_refine.enable=true")
        if self.use_geometry_ribbon and self.use_progressive_d1_high_res:
            raise ValueError(
                "G1 geometry_ribbon and D6 progressive HighRes are separate experiments"
            )
        if self.use_geometry_ribbon and self.use_boundary_residual_refine:
            raise ValueError(
                "G1 geometry_ribbon and D7 boundary residual are separate experiments"
            )
        if self.use_geometry_ribbon and self.use_foreground_completion:
            raise ValueError(
                "G1 geometry_ribbon and D15 foreground_completion are separate experiments"
            )
        if self.use_geometry_ribbon and self.use_width_reconstruction:
            raise ValueError(
                "G1 geometry_ribbon and B7 width_reconstruction are separate experiments"
            )
        if self.use_geometry_ribbon:
            self.geometry_ribbon_head = CenterlineWidthRibbonHead(
                highres_channels=int(high_res_cfg.get('refine_channels', 48)),
                p3_channels=fpn_channels,
                hidden_channels=int(geometry_ribbon_cfg.get('hidden_channels', 32)),
                semantic_channels=int(
                    geometry_ribbon_cfg.get('semantic_channels', 24)
                ),
                max_half_width=float(
                    geometry_ribbon_cfg.get('max_half_width', 5.5)
                ),
                min_half_width=float(
                    geometry_ribbon_cfg.get('min_half_width', 0.5)
                ),
                renderer_radius=int(
                    geometry_ribbon_cfg.get('renderer_radius', 5)
                ),
                renderer_temperature=float(
                    geometry_ribbon_cfg.get('renderer_temperature', 0.75)
                ),
                positive_only_residual=bool(
                    geometry_ribbon_cfg.get('positive_only_residual', False)
                ),
                detach_inputs=bool(geometry_ribbon_cfg.get('detach_inputs', False)),
            )
            logger.info(
                "G1 centerline-width ribbon reconstruction enabled: "
                "base-preserving geometric residual"
            )
        else:
            self.geometry_ribbon_head = None

        contour_refine_cfg = contour_refine_cfg or {}
        self.use_contour_refine = bool(contour_refine_cfg.get('enable', False))
        if self.use_contour_refine and self.use_progressive_d1_high_res:
            raise ValueError(
                "D6 progressive HighRes and the D5 contour refiner are separate "
                "experiments and cannot be enabled together"
            )
        if self.use_contour_refine and self.use_boundary_residual_refine:
            raise ValueError(
                "D7 boundary residual and D5 contour refinement are separate "
                "experiments and cannot be enabled together"
            )
        if self.use_contour_refine and self.use_geometry_ribbon:
            raise ValueError(
                "G1 geometry ribbon and D5 contour refinement are separate experiments"
            )
        if self.use_contour_refine and self.use_foreground_completion:
            raise ValueError(
                "D5 contour refinement and D15 foreground_completion are separate experiments"
            )
        if self.use_contour_refine and self.use_width_reconstruction:
            raise ValueError(
                "D5 contour refinement and B7 width_reconstruction are separate experiments"
            )
        if self.use_contour_refine and self.high_res_refine is None:
            raise ValueError(
                "D1-conditioned contour refinement requires high_res_refine.enable=true"
            )
        if self.use_contour_refine:
            self.contour_refiner = D1ConditionedContourRefiner(
                highres_channels=int(high_res_cfg.get('refine_channels', 48)),
                p3_channels=fpn_channels,
                image_channels=int(contour_refine_cfg.get('image_channels', 16)),
                hidden_channels=int(contour_refine_cfg.get('hidden_channels', 48)),
            )
        else:
            self.contour_refiner = None

        instance_decoder_cfg = instance_decoder_cfg or {}
        self.use_instance_conditioned_highres = bool(
            instance_decoder_cfg.get('enable', False)
        )
        if self.use_instance_conditioned_highres and self.high_res_refine is None:
            raise ValueError(
                "instance_conditioned_highres requires high_res_refine.enable=true"
            )
        if self.use_instance_conditioned_highres and not self.use_p3_cross_scale_alignment:
            raise ValueError(
                "instance_conditioned_highres is defined as a D1 P3-aligned "
                "experiment; enable cross_scale_deformable_alignment."
            )
        if self.use_instance_conditioned_highres and (
            self.use_progressive_d1_high_res
            or self.use_boundary_residual_refine
            or self.use_geometry_ribbon
            or self.use_contour_refine
            or self.use_foreground_completion
            or self.use_width_reconstruction
        ):
            raise ValueError(
                "instance_conditioned_highres should be evaluated separately "
                "from D5/D6/D7/G1/D15/B7 heads."
            )
        if self.use_instance_conditioned_highres:
            self.instance_conditioned_highres = InstanceConditionedHighResDecoder(
                p3_channels=fpn_channels,
                detail_channels=int(instance_decoder_cfg.get('detail_channels', 32)),
                semantic_channels=int(
                    instance_decoder_cfg.get('semantic_channels', 48)
                ),
                mask_channels=int(instance_decoder_cfg.get('mask_channels', 48)),
                num_instances=int(instance_decoder_cfg.get('num_instances', 6)),
                num_detail_blocks=int(
                    instance_decoder_cfg.get('num_detail_blocks', 2)
                ),
                residual_scale_init=float(
                    instance_decoder_cfg.get('residual_scale_init', 0.0)
                ),
                residual_max_scale=float(
                    instance_decoder_cfg.get('residual_max_scale', 1.0)
                ),
                uncertainty_gate=bool(
                    instance_decoder_cfg.get('uncertainty_gate', False)
                ),
                uncertainty_gate_power=float(
                    instance_decoder_cfg.get('uncertainty_gate_power', 1.0)
                ),
            )
            logger.info(
                "ICHRD enabled: RGB 160x160 detail stream + D1 P3-conditioned "
                f"{instance_decoder_cfg.get('num_instances', 6)} instance masks"
            )
        else:
            self.instance_conditioned_highres = None

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, size=ref.shape[-2:], mode='bilinear', align_corners=False
        )

    def _upsample_level(
        self, x: torch.Tensor, ref: torch.Tensor, level: int
    ) -> torch.Tensor:
        """Upsample one FPN level, preserving M0 bilinear behavior by default."""
        if self.fpn_upsamplers is None:
            return self._resize_like(x, ref)

        upsampled = self.fpn_upsamplers[level](x)
        # DySample(scale=2) is exact for the 320-pixel M0 pyramid.  Keep a
        # defensive alignment path for non-standard input shapes.
        if upsampled.shape[-2:] != ref.shape[-2:]:
            upsampled = F.interpolate(
                upsampled, size=ref.shape[-2:], mode='bilinear',
                align_corners=False,
            )
        return upsampled

    def forward(
        self,
        features: List[torch.Tensor],
        image_for_icg: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        c2, c3, c4, c5 = features

        lateral_features = [
            proj(feat) for proj, feat in zip(self.lateral_convs, (c2, c3, c4, c5))
        ]
        if self.texture_structure_purifiers:
            # L5 provides semantic context; C3/C4 are purified before any FPN
            # top-down fusion, so later branches keep the original M6 topology.
            for level_key, purifier in self.texture_structure_purifiers.items():
                level = int(level_key)
                lateral_features[level] = purifier(
                    lateral_features[level], lateral_features[3]
                )
        l2, l3, l4, l5 = lateral_features

        p5 = self.smooth_convs[3](l5)
        p5_up = self._upsample_level(p5, l4, 0)
        p4_input = (
            self.topdown_alignments[0](l4, p5_up)
            if self.topdown_alignments is not None else l4 + p5_up
        )
        p4 = self.smooth_convs[2](p4_input)
        if self.bsde is not None:
            p4, p3_input = self.bsde(l3, p4)
        else:
            p4_up = self._upsample_level(p4, l3, 1)
            if self.p3_semantic_detail_alignment is not None:
                p3_input = self.p3_semantic_detail_alignment(l3, p4_up)
            elif self.p3_cross_scale_alignment is not None:
                p3_input = self.p3_cross_scale_alignment(l3, p4_up)
            elif self.topdown_alignments is not None:
                p3_input = self.topdown_alignments[1](l3, p4_up)
            else:
                p3_input = l3 + p4_up
        p3 = self.smooth_convs[1](p3_input)
        if self.p3_correlation_calibrator is not None:
            # D7 calibrates the completed D1 P3 map against the same P4-up
            # semantic context.  Zero-initialization preserves D1 initially.
            p3 = self.p3_correlation_calibrator(p3, p4_up)
        if self.p3_semantic_detail_reinforcer is not None:
            p3 = self.p3_semantic_detail_reinforcer(p3, p4_up)
        p3_up = self._upsample_level(p3, l2, 2)
        p2_input = (
            self.p2_cross_scale_alignment(l2, p3_up)
            if self.p2_cross_scale_alignment is not None else l2 + p3_up
        )
        p2 = self.smooth_convs[0](p2_input)

        if self.bidirectional_reconstruction is not None:
            p3, p4 = self.bidirectional_reconstruction(p2, p3, p4)

        if self.coord_att_modules is not None:
            p3 = self.coord_att_modules[0](p3)
            p4 = self.coord_att_modules[1](p4)
            p5 = self.coord_att_modules[2](p5)

        edge_input = torch.cat([p2, self._upsample_level(p3, p2, 2)], dim=1)
        edge_features = self.edge_refine(edge_input)
        if self.directional_edge_refine is not None:
            edge_features = self.directional_edge_refine(edge_features, image_for_icg)

        if self.deep_context_gate is not None:
            p4 = self.deep_context_gate(p4, p5)

        context_features = self.context_refine(p4)
        if self.horizontal_continuity is not None:
            context_features = self.horizontal_continuity(context_features)
        prototype_context_logits = None
        hard_negative_logits = None
        hard_negative_safety_logits = None
        icg_image = (
            F.interpolate(
                self.icg_module.purify_rgb(image_for_icg),
                size=context_features.shape[-2:],
                mode='bilinear', align_corners=False,
            )
            if self.icg_module.uses_rgb else None
        )
        context_features = self.icg_module(context_features, icg_image)
        if self.prototype_context is not None:
            context_p3, prototype_context_logits = self.prototype_context(context_features, p3)
        elif self.hard_negative_suppression is not None:
            (
                context_p3,
                hard_negative_logits,
                hard_negative_safety_logits,
            ) = self.hard_negative_suppression(context_features, p3)
        else:
            context_p3 = self._upsample_level(context_features, p3, 1)

        # Align ICG-enhanced deep context with P3 before returning it to P2.
        if self.localization_alignment is not None:
            context_p3 = self.localization_alignment(p3, context_p3)
            loc_features = self._upsample_level(context_p3, p2, 2) + p2
        else:
            loc_features = self._upsample_level(context_p3, p2, 2) + p2
        loc_features = self.loc_refine(loc_features)

        weak_stitch_logits = None
        weak_stitch_consistency_logits = None
        if self.weak_stitch_recovery is not None:
            (
                loc_features,
                weak_stitch_logits,
                weak_stitch_consistency_logits,
            ) = self.weak_stitch_recovery(
                loc_features, p2, p3, context_features
            )

        semantic_area_logits = None
        if self.semantic_area_localizer is not None:
            loc_features, semantic_area_logits = self.semantic_area_localizer(
                loc_features, p3, p4, p5
            )

        triple_path_boundary_logits = None
        if self.triple_path_adapter is not None:
            edge_features, loc_features, triple_path_boundary_logits = self.triple_path_adapter(
                edge_features, loc_features, p2, p3
            )

        thickness_width_logits = None
        style_adapter_weights = None
        if self.style_conditioned_adapter is not None:
            edge_features, loc_features, style_adapter_weights = self.style_conditioned_adapter(
                edge_features, loc_features, image_for_icg
            )
        if self.thickness_router is not None:
            edge_features, loc_features, thickness_width_logits = self.thickness_router(
                edge_features, loc_features, image_for_icg
            )

        pattern_features = None
        pattern_center_logits = None
        pattern_direction_logits = None
        pattern_spacing_logits = None
        if self.stitch_pattern_decoder is not None:
            (
                pattern_features,
                pattern_center_logits,
                pattern_direction_logits,
                pattern_spacing_logits,
            ) = self.stitch_pattern_decoder(edge_features, loc_features)

        # Use final loc prediction only as a detached TSCB safety prior.  The
        # localization path itself remains unchanged.
        texture_confidence = None
        loc_logits = self.loc_pred_head(loc_features)
        if self.texture_semantic_contradiction is not None:
            edge_features, texture_confidence = self.texture_semantic_contradiction(
                p2, edge_features, loc_features, loc_logits
            )
        edge_logits = self.edge_pred_head(edge_features)
        centerline_logits = self.centerline_pred_head(edge_features)
        afm_logits = self.fusion_head(edge_features, loc_features)
        final_logits = afm_logits
        sdf_logits = (
            self.sdf_auxiliary_head(loc_features)
            if self.sdf_auxiliary_head is not None else None
        )

        contour_logits = None
        ribbon_logits = None
        ribbon_width = None
        wrd_residual_logits = None
        instance_aggregate_logits = None
        instance_logits = None
        highres_features = None
        if self.high_res_refine is not None:
            if self.contour_refiner is not None:
                (
                    final_logits,
                    edge_logits,
                    centerline_logits,
                    highres_features,
                ) = self.high_res_refine(
                    edge_features, loc_features, image_for_icg,
                    pattern_features=pattern_features,
                    afm_logits=afm_logits,
                    return_features=True,
                )
                final_logits, contour_logits = self.contour_refiner(
                    highres_features, p3, image_for_icg, final_logits
                )
            elif self.geometry_ribbon_head is not None:
                (
                    final_logits,
                    edge_logits,
                    _base_centerline_logits,
                    highres_features,
                ) = self.high_res_refine(
                    edge_features, loc_features, image_for_icg,
                    pattern_features=pattern_features,
                    afm_logits=afm_logits,
                    return_features=True,
                )
                (
                    final_logits,
                    ribbon_logits,
                    centerline_logits,
                    ribbon_width,
                ) = self.geometry_ribbon_head(final_logits, highres_features, p3)
            else:
                if self.use_progressive_d1_high_res:
                    final_logits, edge_logits, centerline_logits = self.high_res_refine(
                        edge_features, loc_features, image_for_icg, p3,
                        pattern_features=pattern_features,
                    )
                else:
                    if (
                        self.foreground_completion_refiner is not None
                        or self.width_reconstruction_decoder is not None
                        or self.narrow_band_resampler is not None
                        or self.agreement_calibrator is not None
                    ):
                        (
                            final_logits,
                            edge_logits,
                            centerline_logits,
                            highres_features,
                        ) = self.high_res_refine(
                            edge_features, loc_features, image_for_icg,
                            pattern_features=pattern_features,
                            afm_logits=afm_logits,
                            return_features=True,
                        )
                    else:
                        final_logits, edge_logits, centerline_logits = self.high_res_refine(
                            edge_features, loc_features, image_for_icg,
                            pattern_features=pattern_features,
                            afm_logits=afm_logits,
                        )
            if self.foreground_completion_refiner is not None:
                if highres_features is None:
                    raise RuntimeError(
                        "foreground_completion requires HighRes features but none were returned"
                    )
                final_logits = self.foreground_completion_refiner(
                    final_logits, highres_features, loc_logits, edge_logits
                )
            if self.width_reconstruction_decoder is not None:
                if highres_features is None:
                    raise RuntimeError(
                        "width_reconstruction requires HighRes features but none were returned"
                    )
                final_logits, wrd_residual_logits = self.width_reconstruction_decoder(
                    final_logits,
                    highres_features,
                    edge_features,
                    loc_features,
                    image_for_icg,
                )
            if self.boundary_residual_refiner is not None:
                final_logits = self.boundary_residual_refiner(
                    final_logits, edge_features, loc_features, p3, image_for_icg
                )
            narrow_band_residual_logits = None
            if self.narrow_band_resampler is not None:
                if highres_features is None:
                    raise RuntimeError(
                        "narrow_band_resampling requires HighRes features but none were returned"
                    )
                final_logits, narrow_band_residual_logits = self.narrow_band_resampler(
                    final_logits, highres_features, image_for_icg
                )
            if self.instance_conditioned_highres is not None:
                (
                    final_logits,
                    instance_aggregate_logits,
                    instance_logits,
                ) = self.instance_conditioned_highres(final_logits, p3, image_for_icg)
            if self.agreement_calibrator is not None:
                if highres_features is None:
                    raise RuntimeError(
                        "agreement_calibration requires HighRes features but none were returned"
                    )
                final_logits = self.agreement_calibrator(
                    final_logits,
                    highres_features,
                    edge_features,
                    loc_features,
                    afm_logits,
                    loc_logits,
                    edge_logits,
                )

        outputs = {
            'final_mask': final_logits,
            'object_mask': loc_logits,
            'edge_mask': edge_logits,
            'centerline_mask': centerline_logits,
        }
        # AFM is normally an internal coarse prediction.  Expose it only for
        # read-only evaluation diagnostics so training and ordinary inference
        # keep their existing output contract.
        if return_intermediates:
            outputs['afm_mask'] = afm_logits
        if self.use_afm_coarse_supervision:
            # Keep this native 80x80 map separate from the diagnostic-only
            # ``afm_mask``.  The loss builds its target by max-pooling the GT,
            # which preserves sub-P2 thin stitches without changing M6's
            # final prediction path.
            outputs['afm_coarse_mask'] = afm_logits
        if self.use_afm_aux_supervision:
            # Unlike ``afm_coarse_mask``, this map deliberately goes through
            # the trainer's standard bilinear resize before being supervised
            # against the exact full-resolution GT, mirroring M0's AFM loss.
            outputs['afm_aux_mask'] = afm_logits
        if weak_stitch_logits is not None:
            # Training-only auxiliary evidence map.  It is not fused as a
            # standalone mask and therefore cannot bypass AFM/HighRes.
            outputs['weak_stitch_mask'] = weak_stitch_logits
        if weak_stitch_consistency_logits is not None:
            # Auxiliary P4 consistency map.  It supervises whether recovered
            # local evidence is semantically compatible with P4; it is never
            # emitted as a final segmentation mask.
            outputs['weak_stitch_consistency_mask'] = weak_stitch_consistency_logits
        if sdf_logits is not None:
            outputs['sdf_mask'] = sdf_logits
        if pattern_center_logits is not None:
            outputs['pattern_center'] = pattern_center_logits
            outputs['pattern_direction'] = pattern_direction_logits
            outputs['pattern_spacing'] = pattern_spacing_logits
        if contour_logits is not None:
            outputs['contour_logits'] = contour_logits
        if ribbon_logits is not None:
            outputs['ribbon_mask'] = ribbon_logits
            outputs['ribbon_width'] = ribbon_width
        if wrd_residual_logits is not None:
            outputs['wrd_residual_mask'] = wrd_residual_logits
        if 'narrow_band_residual_logits' in locals() and narrow_band_residual_logits is not None:
            outputs['narrow_band_residual_mask'] = narrow_band_residual_logits
        if thickness_width_logits is not None:
            outputs['thickness_width_prior'] = thickness_width_logits
        if style_adapter_weights is not None:
            outputs['style_adapter_weights'] = style_adapter_weights
        if triple_path_boundary_logits is not None:
            outputs['triple_path_boundary_mask'] = triple_path_boundary_logits
        if semantic_area_logits is not None:
            outputs['semantic_area_mask'] = semantic_area_logits
        if texture_confidence is not None:
            # Probability map, consumed by the selected hard-background and
            # foreground-safety losses rather than as a regular mask logit.
            outputs['texture_confidence'] = texture_confidence
        if instance_aggregate_logits is not None:
            outputs['instance_aggregate_mask'] = instance_aggregate_logits
            outputs['instance_logits'] = instance_logits
        if prototype_context_logits is not None:
            outputs['prototype_context_mask'] = prototype_context_logits
        if hard_negative_logits is not None:
            outputs['hard_negative_mask'] = hard_negative_logits
            outputs['hard_negative_safety_mask'] = hard_negative_safety_logits
        return outputs


def _load_local_backbone_weights(model: nn.Module, path: str) -> None:
    """Load backbone weights from a local .safetensors or .pth file."""
    if not os.path.exists(path):
        logger.warning("Local backbone weights not found: %s; skipping.", path)
        return
    try:
        if path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(path)
        else:
            state_dict = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as e:
        logger.warning("Failed to load backbone weights from %s: %s", path, e)
        return

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.info("Backbone missing keys (expected): %d keys", len(missing))
    if unexpected:
        logger.info("Backbone unexpected keys (ignored): %d keys", len(unexpected))
    logger.info("Backbone weights loaded from local file: %s", path)


class C3Net(nn.Module):
    def __init__(self, config: Dict, backbone_type: str = 'mobilenetv4'):
        """
        Args:
            config: Model configuration dictionary.
            backbone_type: 'mobilenetv4' (student) or 'dinov2' (teacher).
                           DINOv2 mode skips CoordAtt and stem_projections.
        """
        super().__init__()
        backbone_tag = 'DINOv2-L (Teacher)' if backbone_type == 'dinov2' else 'MobileNetV4 (Student)'
        logger.info(f"--- Initializing C3Net Model ({backbone_tag} + CoordAtt) ---")

        # --- Validate and Store Basic Config ---
        if not isinstance(config, dict):
            raise TypeError("`config` must be a dictionary.")
        img_size = config['image_processing'].get('target_size', 320)
        if not isinstance(img_size, int) or img_size <= 0:
             raise ValueError("`img_size` must be a positive integer.")
        self.config = config
        self.img_size = img_size
        self.backbone_type = backbone_type

        # =====================================================================
        # Read new configuration parameters with defaults
        # =====================================================================
        # Backbone / Encoder configuration
        backbone_cfg = config.get('backbone', {})
        backbone_name = backbone_cfg.get('name', 'mobilenetv4_hybrid_medium')
        self.use_coord_att = backbone_cfg.get('use_coord_att', True)
        # Allow disabling pretrained weights (useful when HF is unreachable)
        pretrained_backbone = backbone_cfg.get('pretrained', True)
        decoder_cfg = config.get('decoder', {})
        decoder_type = decoder_cfg.get(
            'type',
            'mobile_fpn' if backbone_type == 'mobilenetv4' else 'legacy'
        )
        self.use_mobile_fpn_decoder = (
            backbone_type == 'mobilenetv4' and decoder_type == 'mobile_fpn'
        )

        # =====================================================================
        # 2. Backbone Initialization
        # =====================================================================
        if backbone_type == 'dinov2':
            from models.backbone import DinoV2WithRegistersEncoder

            # --- Teacher: DINOv2-with-registers-large via HuggingFace ---
            encoder_cfg = config.get('encoder', {})
            model_name = encoder_cfg.get('model_name', 'facebook/dinov2-with-registers-large')
            variant = encoder_cfg.get('variant', 'large')
            output_stages = encoder_cfg.get('output_stages', [1, 2, 23, 24])

            self.backbone = DinoV2WithRegistersEncoder(
                model_name=model_name,
                variant=variant,
                output_stages=output_stages,
            )
            # DINOv2-Large: all 4 stages output 1024-dim features
            backbone_feat_channels = [self.backbone.hidden_dim] * 4
            logger.info(f"DINOv2 '{variant}' backbone feature channels: {backbone_feat_channels}")
        else:
            # --- Student: MobileNetV4 via timm ---
            encoder_cfg = config.get('encoder', {})
            create_kwargs = {}
            create_kwargs.update(encoder_cfg)
            for key in ("norm_layer", "model_name", "variant", "output_stages",
                        "patch_size", "pretrained", "features_only", "out_indices"):
                create_kwargs.pop(key, None)

            # Support local pretrained file (path string 鈫?load after creation)
            local_pretrained_path = None
            if isinstance(pretrained_backbone, str) and pretrained_backbone.lower() not in ('true', 'false'):
                local_pretrained_path = pretrained_backbone
                pretrained_backbone = False

            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained_backbone,
                features_only=True,
                out_indices=(1, 2, 3, 4),
                **create_kwargs,
            )

            if local_pretrained_path:
                _load_local_backbone_weights(self.backbone, local_pretrained_path)

            # Automatically discover backbone output channel dimensions
            if hasattr(self.backbone, 'feature_info'):
                backbone_feat_channels = self.backbone.feature_info.channels()
            else:
                with torch.no_grad():
                    dummy = torch.randn(1, 3, self.img_size, self.img_size)
                    dummy_feats = self.backbone(dummy)
                    backbone_feat_channels = [f.shape[1] for f in dummy_feats]
            logger.info(f"Backbone '{backbone_name}' feature channels: {backbone_feat_channels}")

        # Store original backbone channel list
        self.backbone_feat_channels = backbone_feat_channels

        with torch.no_grad():
            dummy = torch.randn(1, 3, self.img_size, self.img_size)
            dummy_feats = self.backbone(dummy)
            self.feature_sizes = [tuple(f.shape[-2:]) for f in dummy_feats]
            self.feature_h, self.feature_w = self.feature_sizes[-1]
        logger.info(f"{backbone_tag} feature sizes for {self.img_size}x{self.img_size}: "
                    f"{self.feature_sizes}")

        if self.use_mobile_fpn_decoder:
            mobile_cfg = decoder_cfg.get('mobile_fpn', {})
            fusion_cfg = config.get('fusion_head', {})

            fpn_channels = mobile_cfg.get('fpn_channels', 192)
            edge_channels = mobile_cfg.get('edge_channels', 128)
            loc_channels = mobile_cfg.get('loc_channels', 128)
            fusion_channels = mobile_cfg.get('fusion_channels', fusion_cfg.get('fuse_channels', 128))

            self.encoder_channels = fpn_channels
            self.stem_projections = None
            self.coord_att_modules = None
            self.edge_final_channels = edge_channels
            self.loc_final_channels = loc_channels

            detail_encoder_cfg = backbone_cfg.get(
                'dual_resolution_detail_encoder', {}
            )
            self.use_dual_resolution_detail_encoder = bool(
                detail_encoder_cfg.get('enable', False)
            )
            if self.use_dual_resolution_detail_encoder:
                self.dual_resolution_detail_encoder = DualResolutionDetailSemanticEncoder(
                    c2_channels=int(backbone_feat_channels[0]),
                    c3_channels=int(backbone_feat_channels[1]),
                    detail_channels=int(detail_encoder_cfg.get('detail_channels', 32)),
                )
            else:
                self.dual_resolution_detail_encoder = None

            self.mobile_decoder = MobileFPNC3Decoder(
                backbone_channels=backbone_feat_channels,
                fpn_channels=fpn_channels,
                edge_channels=edge_channels,
                loc_channels=loc_channels,
                fusion_channels=fusion_channels,
                num_edge_blocks=mobile_cfg.get('num_edge_blocks', 1),
                num_context_blocks=mobile_cfg.get('num_context_blocks', 1),
                use_coord_att=self.use_coord_att,
                icg_params=mobile_cfg.get(
                    'icg_params',
                    decoder_cfg.get('loc_branch', {}).get('icg_params', {})
                ),
                fusion_cfg={
                    **fusion_cfg,
                    'num_refine_blocks': mobile_cfg.get(
                        'fusion_refine_blocks',
                        fusion_cfg.get('num_refine_blocks', 1),
                    ),
                    'attn_gen_intermediate_channels': mobile_cfg.get(
                        'attn_gen_intermediate_channels',
                        fusion_cfg.get('attn_gen_intermediate_channels', 64),
                    ),
                },
                high_res_cfg=mobile_cfg.get('high_res_refine', {}),
                deep_context_cfg=mobile_cfg.get('deep_context', {}),
                trajectory_cfg=mobile_cfg.get('trajectory_enhance', {}),
                dynamic_upsample_cfg=mobile_cfg.get('dynamic_upsample', {}),
                siad_conv_cfg=mobile_cfg.get('siad_conv', {}),
                c2p_alignment_cfg=mobile_cfg.get('c2p_alignment', {}),
                directional_edge_cfg=mobile_cfg.get('directional_edge_refine', {}),
                cross_scale_align_cfg=mobile_cfg.get(
                    'cross_scale_deformable_alignment', {}
                ),
                correlation_calibration_cfg=mobile_cfg.get(
                    'p3_correlation_calibration', {}
                ),
                semantic_detail_align_cfg=mobile_cfg.get(
                    'semantic_detail_decomposed_alignment', {}
                ),
                p3_semantic_detail_reinforce_cfg=mobile_cfg.get(
                    'p3_semantic_detail_reinforcement', {}
                ),
                bsde_cfg=mobile_cfg.get('bsde', {}),
                sdf_auxiliary_cfg=mobile_cfg.get('sdf_auxiliary', {}),
                stitch_pattern_cfg=mobile_cfg.get('stitch_pattern_relation', {}),
                contour_refine_cfg=mobile_cfg.get('contour_refine', {}),
                instance_decoder_cfg=mobile_cfg.get(
                    'instance_conditioned_highres', {}
                ),
                thickness_router_cfg=mobile_cfg.get('thickness_router', {}),
                style_adapter_cfg=mobile_cfg.get('style_conditioned_adapter', {}),
                texture_invariant_cfg=mobile_cfg.get('texture_invariant_encoder', {}),
                texture_structure_purification_cfg=mobile_cfg.get(
                    'texture_structure_purification', {}
                ),
                prototype_context_cfg=mobile_cfg.get('prototype_context', {}),
                hard_negative_suppression_cfg=mobile_cfg.get(
                    'hard_negative_suppression', {}
                ),
                horizontal_continuity_cfg=mobile_cfg.get(
                    'horizontal_continuity', {}
                ),
                bidirectional_reconstruction_cfg=mobile_cfg.get(
                    'bidirectional_reconstruction', {}
                ),
                triple_path_cfg=mobile_cfg.get('triple_path_decoder', {}),
                semantic_area_cfg=mobile_cfg.get('semantic_area_localization', {}),
                texture_semantic_contradiction_cfg=mobile_cfg.get(
                    'texture_semantic_contradiction', {}
                ),
                weak_stitch_recovery_cfg=mobile_cfg.get(
                    'weak_stitch_recovery', {}
                ),
                afm_coarse_supervision_cfg=mobile_cfg.get(
                    'afm_coarse_supervision', {}
                ),
                afm_aux_supervision_cfg=mobile_cfg.get(
                    'afm_aux_supervision', {}
                ),
            )

            logger.info(
                "Mobile FPN decoder enabled: "
                f"P4+P5 context, P2+P3 edge, P2 fusion, "
                f"fpn={fpn_channels}, edge={edge_channels}, loc={loc_channels}"
            )
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info(f"--- C3Net Initialization Complete ---")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            return

        # =====================================================================
        # Encoder channel dimension for decoder compatibility.
        #
        # DINOv2-Large 鈫?1024 (no stem projection needed 鈥?decoder adapters
        #   directly accept 1024-dim features).
        # MobileNetV4 鈫?512 (after stem projection to common dimension).
        # =====================================================================
        if backbone_type == 'dinov2':
            # DINOv2 outputs are already at a common dim (1024).
            # No stem projection 鈥?decoder adapters will be Conv(1024, chan_adapt, 1).
            self.encoder_channels = self.backbone.hidden_dim
            self.stem_projections = None
        else:
            common_encoder_dim = 512
            self.encoder_channels = common_encoder_dim
            # Stem projection layers: adapt each backbone feature to common_encoder_dim
            self.stem_projections = nn.ModuleList([
                nn.Conv2d(ch, common_encoder_dim, kernel_size=1, bias=False)
                if ch != common_encoder_dim else nn.Identity()
                for ch in backbone_feat_channels
            ])
            # Initialize stem projection weights
            for proj in self.stem_projections:
                if isinstance(proj, nn.Conv2d):
                    nn.init.kaiming_normal_(proj.weight, mode='fan_out', nonlinearity='relu')

        # =====================================================================
        # 3. Coordinate Attention Modules
        # Apply CoordAtt to each backbone output stage for enhanced
        # spatial-channel feature refinement.
        # Disabled for DINOv2 teacher.
        # =====================================================================
        if self.use_coord_att and backbone_type == 'mobilenetv4':
            self.coord_att_modules = nn.ModuleList([
                CoordAtt(self.encoder_channels, self.encoder_channels, reduction=16)
                for _ in range(4)
            ])
            logger.info(f"CoordAtt modules initialized for 4 backbone stages (reduction=16).")
        else:
            self.coord_att_modules = None

        # --- Compute feature map spatial dimensions ---
        # For 320x320 input, MobileNetV4 output sizes are [80, 40, 20, 10].
        # We compute these dynamically for the given img_size.
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.img_size, self.img_size)
            dummy_feats = self.backbone(dummy)
            self.feature_sizes = [tuple(f.shape[-2:]) for f in dummy_feats]
            # Use the last (deepest) feature's spatial size for ICG resizing
            self.feature_h, self.feature_w = self.feature_sizes[-1]
        logger.info(f"{backbone_tag} feature sizes for {self.img_size}x{self.img_size}: "
                    f"{self.feature_sizes}")
        logger.info(f"ICG base resolution (feature_h, feature_w): "
                    f"({self.feature_h}, {self.feature_w})")

        # --- 4. Shared Upsamplers (DySample) Initialization ---
        decoder_cfg = config.get('decoder', {})
        upsampler_cfg = decoder_cfg.get('upsampler', {})
        dysample_style = upsampler_cfg.get('style', 'lp')
        dysample_groups = upsampler_cfg.get('groups', 4)
        dysample_dyscope = upsampler_cfg.get('dyscope', False)

        # Determine the unique input channel dimensions needed for DySample
        # across both branches.
        edge_cfg = decoder_cfg.get('edge_branch', {})
        loc_cfg = decoder_cfg.get('loc_branch', {})
        edge_intermediate_channels = edge_cfg.get('intermediate_channels', [256, 256, 128])
        edge_output_channels = edge_cfg.get('output_channels', 128)
        loc_intermediate_channels = loc_cfg.get('intermediate_channels', [256, 256, 128])
        loc_output_channels = loc_cfg.get('output_channels', 128)

        upsampler_input_channels_needed: Set[int] = set()
        upsampler_input_channels_needed.add(edge_intermediate_channels[0])
        upsampler_input_channels_needed.add(edge_intermediate_channels[1])
        upsampler_input_channels_needed.add(edge_output_channels)
        upsampler_input_channels_needed.add(loc_intermediate_channels[0])
        upsampler_input_channels_needed.add(loc_intermediate_channels[1])
        upsampler_input_channels_needed.add(loc_output_channels)

        valid_channels = {ch for ch in upsampler_input_channels_needed
                         if isinstance(ch, int) and ch > 0}

        self.upsamplers = nn.ModuleDict({
            str(ch): DySample(in_channels=ch,
                              scale=2,
                              style=dysample_style,
                              groups=dysample_groups,
                              dyscope=dysample_dyscope)
            for ch in sorted(list(valid_channels))
        })

        # --- 5. Decoder Branches Initialization ---
        self.edge_final_channels = edge_output_channels
        self.loc_final_channels = loc_output_channels

        self.edge_branch = EdgeBranch(
            encoder_channels=self.encoder_channels,
            intermediate_channels=edge_intermediate_channels,
            output_channels=edge_output_channels,
            num_enhance_blocks=edge_cfg.get('num_enhance_blocks', 2),
            shared_upsamplers=self.upsamplers
        )
        self.loc_branch = ObjectLocalizationBranch(
            encoder_channels=self.encoder_channels,
            intermediate_channels=loc_intermediate_channels,
            output_channels=loc_output_channels,
            num_enhance_blocks=loc_cfg.get('num_enhance_blocks', 1),
            icg_params=loc_cfg.get('icg_params', {}),
            shared_upsamplers=self.upsamplers
        )

        # --- 6. Intermediate Prediction Heads (for deep supervision) ---
        self.edge_pred_head = nn.Conv2d(self.edge_final_channels, 1, kernel_size=1)
        self.loc_pred_head = nn.Conv2d(self.loc_final_channels, 1, kernel_size=1)

        # --- 7. Final Fusion Head Initialization ---
        fusion_cfg = config.get('fusion_head', {})
        self.fusion_head = FinalFusionHead(
            edge_channels=self.edge_final_channels,
            loc_channels=self.loc_final_channels,
            fuse_channels=fusion_cfg.get('fuse_channels', 64),
            num_refine_blocks=fusion_cfg.get('num_refine_blocks', 2),
            attn_gen_intermediate_channels=fusion_cfg.get('attn_gen_intermediate_channels', 32)
        )

        logger.info(f"--- C3Net Initialization Complete ---")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full C3Net model.

        Architecture flow:
        1. MobileNetV4 backbone (hierarchical feature extraction)
        2. Stem projection + CoordAtt (feature enhancement)
        3. EdgeBranch (edge refinement) + LocalizationBranch (context localization)
        4. Intermediate predictions (deep supervision)
        5. FinalFusionHead (attention-gated fusion) 鈫?final mask

        Args:
            x (torch.Tensor): Input image tensor (B, 3, H, W).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing output logits maps:
                - Training mode:
                    {'final_mask': logits from Final Fusion Head (B, 1, H_out, W_out),
                     'object_mask': logits from Loc Branch (B, 1, H_out, W_out),
                     'edge_mask': logits from Edge Branch (B, 1, H_out, W_out)}
                - Inference mode (not self.training):
                    {'final_mask': logits from Final Fusion Head (B, 1, H_out, W_out)}
        """
        # =====================================================================
        # Stage 1: Backbone Feature Extraction
        # =====================================================================
        # backbone_features: List of 4 tensors with varying spatial sizes
        # MobileNetV4 at 320x320: [80x80, 40x40, 20x20, 10x10]
        # DINOv2 uses a patch grid derived from its internal H/14 size.
        backbone_features: List[torch.Tensor] = self.backbone(x)

        if self.use_mobile_fpn_decoder:
            if self.dual_resolution_detail_encoder is not None:
                backbone_features = self.dual_resolution_detail_encoder(x, backbone_features)
            output_dict = self.mobile_decoder(
                backbone_features,
                x,
                return_intermediates=return_intermediates,
            )
            return output_dict

        # DINOv2 teacher: features are already at encoder_channels (1024-dim),
        # no stem projection or CoordAtt needed.
        if self.backbone_type == 'dinov2':
            enhanced_features = list(backbone_features)
        else:
            # Project each feature to common encoder dimension (512)
            projected_features = []
            for proj, feat in zip(self.stem_projections, backbone_features):
                projected_features.append(proj(feat))

            # Apply Coordinate Attention to each projected feature
            if self.use_coord_att and self.coord_att_modules is not None:
                enhanced_features = []
                for ca, feat in zip(self.coord_att_modules, projected_features):
                    enhanced_features.append(ca(feat))
            else:
                enhanced_features = projected_features

        # Assign features: [early1, early2, late1, late2]
        # Same order as original DINOv2: [S1, S2, S_N-1, S_N]
        f_s1_raw = enhanced_features[0]  # e.g., (B, 512, 98, 98)
        f_s2_raw = enhanced_features[1]  # e.g., (B, 512, 49, 49)
        f_n_minus_1_raw = enhanced_features[2]  # e.g., (B, 512, 25, 25)
        f_n_raw = enhanced_features[3]          # e.g., (B, 512, 13, 13)

        # =====================================================================
        # Feature Resizing for Branch Compatibility
        # EdgeBranch expects f_s1 and f_s2 at the same spatial resolution.
        # LocalizationBranch expects f_n_minus_1 and f_n at the same resolution.
        # We resize early features up to match and late features to the
        # deepest (smallest) resolution.
        # =====================================================================
        # EdgeBranch: both inputs at f_s2's resolution (middle-early)
        f_s1 = F.interpolate(f_s1_raw, size=f_s2_raw.shape[-2:],
                             mode='bilinear', align_corners=False)
        f_s2 = f_s2_raw

        # LocalizationBranch: both inputs at f_n's resolution (deepest)
        f_n_minus_1 = F.interpolate(f_n_minus_1_raw, size=f_n_raw.shape[-2:],
                                    mode='bilinear', align_corners=False)
        f_n = f_n_raw

        # =====================================================================
        # Prepare Resized Image for ICG Module
        # The ICG module requires the original (restored) image at the same
        # spatial resolution as f_n (the deepest feature).
        # =====================================================================
        icg_input_res: Tuple[int, int] = (self.feature_h, self.feature_w)
        try:
            x_resized_for_icg = F.interpolate(
                x, size=icg_input_res, mode='bilinear', align_corners=False)
        except Exception as e:
            logger.error(f"Failed to resize input image for ICG: {e}")
            raise e
        x_resized_map: Dict[Tuple[int, int], torch.Tensor] = {
            icg_input_res: x_resized_for_icg
        }

        # =====================================================================
        # Stage 2: Decoder Branches
        # =====================================================================
        # Edge Refinement Pathway (ERP): Processes early features (S1, S2)
        f_edge_features = self.edge_branch(f_s1, f_s2)

        # Contextual Localization Pathway (CLP): Processes late features
        f_loc_features = self.loc_branch(f_n_minus_1, f_n, x_resized_map)

        # =====================================================================
        # Spatial Alignment for FusionHead
        # With MobileNetV4 backbone, EdgeBranch (S2=49) and LocalizationBranch
        # (S_N=13) start from different resolutions, so their outputs differ
        # Resize the smaller loc features to match edge features before fusion.
        # =====================================================================
        if f_edge_features.shape[-2:] != f_loc_features.shape[-2:]:
            f_loc_features = F.interpolate(
                f_loc_features, size=f_edge_features.shape[-2:],
                mode='bilinear', align_corners=False
            )

        # =====================================================================
        # Stage 3: Intermediate Predictions (for Deep Supervision)
        # =====================================================================
        edge_logits = self.edge_pred_head(f_edge_features)
        loc_logits = self.loc_pred_head(f_loc_features)

        # =====================================================================
        # Stage 4: Final Fusion (Attention-Gated)
        # =====================================================================
        final_logits = self.fusion_head(f_edge_features, f_loc_features)

        # =====================================================================
        # Stage 5: Assemble Output Dictionary
        # Always return all outputs 鈥?the caller decides which keys to use.
        # Validation needs object_mask / edge_mask for loss computation;
        # inference typically only uses final_mask.
        # =====================================================================
        output_dict = {
            'final_mask': final_logits,
            'object_mask': loc_logits,
            'edge_mask': edge_logits,
        }

        return output_dict

    # =========================================================================
    # Teacher Model Factory
    # =========================================================================
    @classmethod
    def create_teacher(cls, config: Dict, checkpoint_path: str,
                       device: torch.device) -> 'C3Net':
        """
        Build a frozen DINOv2-based C3Net teacher from a trained checkpoint.

        The teacher uses DINOv2-with-registers-large backbone (no CoordAtt or stem_projections) and shares the same decoder structure
        as the student.  Weights are loaded with strict=False to tolerate minor
        mismatches (e.g. DySample dyscope vs no-dyscope) while still loading
        all matching decoder + backbone parameters.

        Args:
            config:      Model configuration dict (same format as student).
            checkpoint_path: Path to the teacher .pth checkpoint.
            device:      torch.device to load onto.

        Returns:
            C3Net instance with teacher weights loaded and frozen (eval mode,
            requires_grad=False for all parameters).
        """
        logger.info(f"=== Building Teacher Model from {checkpoint_path} ===")

        # 1. Instantiate C3Net with DINOv2 backbone
        teacher = cls(config, backbone_type='dinov2')
        teacher.to(device)

        # 2. Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=device,
                          weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)

        # Strip 'module.' prefix from DataParallel checkpoints
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', '', 1): v
                          for k, v in state_dict.items()}
            logger.info("Stripped 'module.' prefix from teacher state dict.")

        # Remap legacy 'encoder.*' 鈫?'backbone.*' (old DINOv2 C3Net used
        # `self.encoder`, new code uses `self.backbone` for both backbones).
        if any(k.startswith('encoder.') for k in state_dict.keys()):
            state_dict = {
                (k.replace('encoder.', 'backbone.', 1) if k.startswith('encoder.') else k): v
                for k, v in state_dict.items()
            }
            logger.info("Remapped 'encoder.*' 鈫?'backbone.*' in teacher state dict.")

        # 3. Load with strict=False 鈥?decoder adapters differ in input channels
        #    (1024 for teacher vs 512 for student) and DySample dyscope may
        #    differ between checkpoints. All matching keys load correctly.
        missing, unexpected = teacher.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info(f"Teacher missing keys (expected, will skip): "
                        f"{len(missing)} keys 鈥?first 5: {missing[:5]}")
        if unexpected:
            logger.info(f"Teacher unexpected keys (from checkpoint): "
                        f"{len(unexpected)} keys 鈥?first 5: {unexpected[:5]}")

        # 4. Freeze all parameters
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        total_params = sum(p.numel() for p in teacher.parameters())
        logger.info(f"Teacher frozen with {total_params:,} parameters.")
        return teacher


# =============================================================================
# End-to-End Test Code
# Verifies model instantiation, forward pass, and output shapes.
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("C3Net End-to-End Test (MobileNetV4 + CoordAtt)")
    print("=" * 70)
    # --- Build a minimal configuration matching the project's config structure ---
    test_config = {
        'image_processing': {
            'target_size': 320,
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225],
        },
        'encoder': {
            'model_name': 'facebook/dinov2-with-registers-large',  # no longer used
            'variant': 'large',
            'output_stages': [1, 2, 23, 24],
            'patch_size': 14,
        },
        'backbone': {
            'name': 'mobilenetv4_hybrid_large',
            'use_coord_att': True,
        },
        'decoder': {
            'upsampler': {
                'style': 'lp',
                'groups': 4,
                'dyscope': False,
            },
            'edge_branch': {
                'intermediate_channels': [256, 256, 128],
                'output_channels': 128,
                'num_enhance_blocks': 2,
            },
            'loc_branch': {
                'intermediate_channels': [256, 256, 128],
                'output_channels': 128,
                'num_enhance_blocks': 1,
                'icg_params': {
                    'image_channels': 3,
                    'ia_cnn_channels': 32,
                    'contrast_channels': 16,
                    'contrast_agg_kernel': 5,
                    'contrast_bg_thresh': 0.1,
                },
            },
        },
        'fusion_head': {
            'fuse_channels': 128,
            'num_refine_blocks': 2,
            'attn_gen_intermediate_channels': 64,
        },
    }

    # --- Device selection ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # --- Model instantiation ---
    print("\n[1] Instantiating C3Net model...")
    try:
        model = C3Net(test_config).to(device)
        print("    Model created successfully.")
    except Exception as e:
        print(f"    ERROR during model creation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # --- Parameter count ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[2] Parameter counts:")
    print(f"    Total:     {total_params:>12,}")
    print(f"    Trainable: {trainable_params:>12,}")
    # --- Test forward pass (training mode) ---
    print("\n[3] Testing forward pass (training mode)...")
    model.train()
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 320, 320).to(device)
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"    Output keys: {list(output.keys())}")
        for k, v in output.items():
            print(f"      {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
    except Exception as e:
        print(f"    ERROR during training forward pass: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Verify training mode keys
    expected_train_keys = {'final_mask', 'object_mask', 'edge_mask'}
    actual_keys = set(output.keys())
    if actual_keys == expected_train_keys:
        print(f"    鉁?Training output keys match expected: {expected_train_keys}")
    else:
        print(f"    鉁?Training output keys mismatch!")
        print(f"      Expected: {expected_train_keys}")
        print(f"      Got:      {actual_keys}")

    # --- Test forward pass (inference mode) ---
    print("\n[4] Testing forward pass (inference mode)...")
    model.eval()
    dummy_input_infer = torch.randn(1, 3, 320, 320).to(device)
    try:
        with torch.no_grad():
            output_infer = model(dummy_input_infer)
        print(f"    Output keys: {list(output_infer.keys())}")
        for k, v in output_infer.items():
            print(f"      {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
    except Exception as e:
        print(f"    ERROR during inference forward pass: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Verify inference mode keys (only 'final_mask')
    expected_infer_keys = {'final_mask'}
    actual_infer_keys = set(output_infer.keys())
    if actual_infer_keys == expected_infer_keys:
        print(f"    鉁?Inference output keys match expected: {expected_infer_keys}")
    else:
        print(f"    鉁?Inference output keys mismatch!")
        print(f"      Expected: {expected_infer_keys}")
        print(f"      Got:      {actual_infer_keys}")


    # --- Test gradient flow ---
    print("\n[6] Testing gradient flow...")
    model.train()
    dummy_input_grad = torch.randn(2, 3, 320, 320).to(device)
    try:
        output_grad = model(dummy_input_grad)
        loss = output_grad['final_mask'].sum() + output_grad['object_mask'].sum() + \
               output_grad['edge_mask'].sum()
        loss.backward()
        # Check gradients in a few key layers
        grad_flow_ok = True
        for name, param in model.named_parameters():
            if param.grad is None and param.requires_grad:
                # Skip BatchNorm running stats (no grad expected)
                if 'running_mean' not in name and 'running_var' not in name and \
                   'num_batches_tracked' not in name:
                    print(f"    WARNING: No gradient for {name}")
                    grad_flow_ok = False
        if grad_flow_ok:
            print(f"    鉁?Gradients flow through all trainable parameters.")
    except Exception as e:
        print(f"    ERROR during gradient test: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("C3Net End-to-End Test Complete!")
    print("=" * 70)

