"""MBDNet-P: a motion-biased, single-backbone network for fabric deblurring.

The data in this project has two different directional properties: the
synthetic motion blur is predominantly horizontal, while fabric, gold powder
and printed ornaments have arbitrary orientations.  MBDNet-P therefore does
not classify materials or complete line structures.  Its only specialised
operation is a *local* soft mixture of an isotropic restoration filter and a
near-horizontal motion filter.  Everywhere else it uses a compact DWPW
encoder-decoder.

The model receives RGB only at inference time.  It never consumes a seam mask,
domain label, or stored degradation parameter.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.arch_util import LayerNorm2d


class DWPWBlock(nn.Module):
    """A lightweight generic restoration block built from DWPW operators."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm1 = LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, hidden * 2, 1, bias=True)
        self.depthwise = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=True)
        self.project = nn.Conv2d(hidden, channels, 1, bias=True)

        self.norm2 = LayerNorm2d(channels)
        self.ffn_expand = nn.Conv2d(channels, hidden * 2, 1, bias=True)
        self.ffn_project = nn.Conv2d(hidden, channels, 1, bias=True)
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    @staticmethod
    def _simple_gate(features: torch.Tensor) -> torch.Tensor:
        first, second = features.chunk(2, dim=1)
        return first * second

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.project(self._simple_gate(self.depthwise(self.expand(self.norm1(features)))))
        features = features + self.beta * residual
        residual = self.ffn_project(self._simple_gate(self.ffn_expand(self.norm2(features))))
        return features + self.gamma * residual


class RepDepthwiseConv(nn.Module):
    """Training-time multi-branch depthwise convolution with exact deploy fusion.

    The three learned branches capture isotropic, horizontal and vertical local
    evidence.  At deployment their kernels and biases are algebraically summed
    into one depthwise 3x3 convolution; no branch, gate, or approximation is
    retained in the inference graph.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.depthwise_3x3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True)
        self.depthwise_1x3 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), groups=channels, bias=True)
        self.depthwise_3x1 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), groups=channels, bias=True)
        self.deploy: nn.Conv2d | None = None

    def equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        kernel = self.depthwise_3x3.weight
        bias = self.depthwise_3x3.bias
        horizontal = F.pad(self.depthwise_1x3.weight, (0, 0, 1, 1))
        vertical = F.pad(self.depthwise_3x1.weight, (1, 1, 0, 0))
        return kernel + horizontal + vertical, bias + self.depthwise_1x3.bias + self.depthwise_3x1.bias

    def switch_to_deploy(self) -> None:
        if self.deploy is not None:
            return
        kernel, bias = self.equivalent_kernel_bias()
        fused = nn.Conv2d(self.channels, self.channels, 3, padding=1, groups=self.channels, bias=True)
        fused.weight.data.copy_(kernel)
        fused.bias.data.copy_(bias)
        self.deploy = fused.to(device=kernel.device, dtype=kernel.dtype)
        del self.depthwise_3x3
        del self.depthwise_1x3
        del self.depthwise_3x1

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.deploy is not None:
            return self.deploy(features)
        return self.depthwise_3x3(features) + self.depthwise_1x3(features) + self.depthwise_3x1(features)


class RepDWPWBlock(DWPWBlock):
    """DWPW block with a re-parameterizable first depthwise operation."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__(channels, expansion)
        hidden = channels * expansion
        self.depthwise = RepDepthwiseConv(hidden * 2)


class MotionBiasedDeblurringBlock(nn.Module):
    """Locally mix isotropic and near-horizontal restoration features.

    The block deliberately has no global direction value and no hard mask.
    ``content_gate`` selects how much horizontal filtering is appropriate at
    each location, while ``scale_gate`` chooses short versus longer horizontal
    motion support.  This lets the generic branch preserve arbitrary gold and
    printed curves when the local evidence is not motion-like.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.generic = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True)
        self.horizontal_short = nn.Conv2d(
            channels, channels, (1, 5), padding=(0, 2), groups=channels, bias=True
        )
        self.horizontal_long = nn.Conv2d(
            channels, channels, (1, 5), padding=(0, 6), dilation=(1, 3), groups=channels, bias=True
        )
        self.router_context = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True)
        self.content_gate = nn.Conv2d(channels, 1, 1, bias=True)
        self.scale_gate = nn.Conv2d(channels, 2, 1, bias=True)
        self.project = nn.Conv2d(channels, channels, 1, bias=True)
        # Begin as an identity.  The router and filters receive useful gradients
        # after the residual scale moves away from zero, avoiding a brittle
        # hard directional prior at the start of training.
        self.residual_scale = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(features)
        context = self.router_context(normalized)
        generic = self.generic(normalized)
        short = self.horizontal_short(normalized)
        long = self.horizontal_long(normalized)

        support = self.scale_gate(context).softmax(dim=1)
        horizontal = support[:, 0:1] * short + support[:, 1:2] * long
        horizontal_weight = self.content_gate(context).sigmoid()
        mixed = (1.0 - horizontal_weight) * generic + horizontal_weight * horizontal
        return features + self.residual_scale * self.project(mixed)


class LowResolutionContext(nn.Module):
    """One generic multi-scale context unit, used only at quarter resolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True)
        self.dilated = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=True)
        self.fuse = nn.Conv2d(channels * 2, channels, 1, bias=True)
        self.scale = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(features)
        context = self.fuse(torch.cat((self.local(normalized), self.dilated(normalized)), dim=1))
        return features + self.scale * context


class DownsampleDWPW(nn.Module):
    """Anti-fragile depthwise downsampling followed by channel projection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels, bias=True)
        self.pointwise = nn.Conv2d(channels, channels * 2, 1, bias=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(features))


class UpsampleDWPW(nn.Module):
    """Pointwise expansion followed by PixelShuffle upsampling."""

    def __init__(self, channels: int) -> None:
        if channels % 2:
            raise ValueError(f"UpsampleDWPW requires an even channel count, got {channels}")
        super().__init__()
        self.expand = nn.Conv2d(channels, channels * 2, 1, bias=True)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.expand(features))


class MBDNetP(nn.Module):
    """Paper-oriented MBDNet-P (48 -> 96 -> 192 by default).

    There are only three MBD blocks: after half-resolution encoding, at the
    quarter-resolution bottleneck, and after half-resolution decoding.  No MBD
    block is used at full resolution, where a horizontal bias is most likely to
    over-sharpen glitter, weave and arbitrary ornament edges.
    """

    def __init__(self, img_channel: int = 3, width: int = 48) -> None:
        super().__init__()
        if img_channel != 3:
            raise ValueError("MBDNetP currently expects RGB inputs")
        if width <= 0 or width % 2:
            raise ValueError(f"MBDNetP width must be a positive even integer, got {width}")

        half = width * 2
        quarter = width * 4
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.full_encoder = nn.Sequential(DWPWBlock(width), DWPWBlock(width))
        self.down1 = DownsampleDWPW(width)
        self.half_encoder = nn.Sequential(DWPWBlock(half), DWPWBlock(half), DWPWBlock(half))
        self.half_encoder_mbd = MotionBiasedDeblurringBlock(half)
        self.down2 = DownsampleDWPW(half)
        self.quarter_encoder = nn.Sequential(*[DWPWBlock(quarter) for _ in range(6)])
        self.context = LowResolutionContext(quarter)
        self.quarter_mbd = MotionBiasedDeblurringBlock(quarter)

        self.up1 = UpsampleDWPW(quarter)
        self.half_decoder = nn.Sequential(DWPWBlock(half), DWPWBlock(half))
        self.half_decoder_mbd = MotionBiasedDeblurringBlock(half)
        self.up2 = UpsampleDWPW(half)
        self.full_decoder = nn.Sequential(DWPWBlock(width), DWPWBlock(width))
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, bias=True)
        self.padder_size = 4

    def check_image_size(self, image: torch.Tensor) -> torch.Tensor:
        _, _, height, width = image.shape
        pad_height = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_width = (self.padder_size - width % self.padder_size) % self.padder_size
        if pad_height == 0 and pad_width == 0:
            return image
        return F.pad(image, (0, pad_width, 0, pad_height), mode="replicate")

    def forward(self, blurry: torch.Tensor) -> torch.Tensor:
        _, _, height, width = blurry.shape
        padded = self.check_image_size(blurry)

        full_skip = self.full_encoder(self.intro(padded))
        half_skip = self.half_encoder_mbd(self.half_encoder(self.down1(full_skip)))
        features = self.quarter_encoder(self.down2(half_skip))
        features = self.quarter_mbd(self.context(features))

        features = self.up1(features) + half_skip
        features = self.half_decoder_mbd(self.half_decoder(features))
        features = self.up2(features) + full_skip
        features = self.full_decoder(features)
        restored = self.ending(features) + padded
        return restored[:, :, :height, :width]


class MBDNetPPlus(nn.Module):
    """Higher-capacity MBDNet-P+ for a fair accuracy-oriented experiment.

    P0 established that the MBD hypothesis is stable and considerably faster,
    but its 3.69 M parameters were far below the 17.11 M A5 reference.  This
    version increases generic restoration depth and width without adding any
    material-specific branch, mask, or new inference input.  The additional
    MBD block stays at quarter resolution, where it has broad blur context at
    modest cost.  It is not placed at full resolution deliberately.

    Default topology: 64 -> 128 -> 256; block counts 3 / 4 / 12 / 4 / 3.
    """

    def __init__(self, img_channel: int = 3, width: int = 64) -> None:
        super().__init__()
        if img_channel != 3:
            raise ValueError("MBDNetPPlus currently expects RGB inputs")
        if width <= 0 or width % 2:
            raise ValueError(f"MBDNetPPlus width must be a positive even integer, got {width}")

        half = width * 2
        quarter = width * 4
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.full_encoder = nn.Sequential(*[DWPWBlock(width) for _ in range(3)])
        self.down1 = DownsampleDWPW(width)
        self.half_encoder = nn.Sequential(*[DWPWBlock(half) for _ in range(4)])
        self.half_encoder_mbd = MotionBiasedDeblurringBlock(half)
        self.down2 = DownsampleDWPW(half)
        self.quarter_encoder = nn.Sequential(*[DWPWBlock(quarter) for _ in range(12)])
        self.context = LowResolutionContext(quarter)
        self.quarter_mbd_1 = MotionBiasedDeblurringBlock(quarter)
        self.quarter_refine = nn.Sequential(DWPWBlock(quarter), DWPWBlock(quarter))
        self.quarter_mbd_2 = MotionBiasedDeblurringBlock(quarter)

        self.up1 = UpsampleDWPW(quarter)
        self.half_decoder = nn.Sequential(*[DWPWBlock(half) for _ in range(4)])
        self.half_decoder_mbd = MotionBiasedDeblurringBlock(half)
        self.up2 = UpsampleDWPW(half)
        self.full_decoder = nn.Sequential(*[DWPWBlock(width) for _ in range(3)])
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, bias=True)
        self.padder_size = 4

    def check_image_size(self, image: torch.Tensor) -> torch.Tensor:
        _, _, height, width = image.shape
        pad_height = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_width = (self.padder_size - width % self.padder_size) % self.padder_size
        if pad_height == 0 and pad_width == 0:
            return image
        return F.pad(image, (0, pad_width, 0, pad_height), mode="replicate")

    def forward(self, blurry: torch.Tensor) -> torch.Tensor:
        _, _, height, width = blurry.shape
        padded = self.check_image_size(blurry)

        full_skip = self.full_encoder(self.intro(padded))
        half_skip = self.half_encoder_mbd(self.half_encoder(self.down1(full_skip)))
        features = self.quarter_encoder(self.down2(half_skip))
        features = self.quarter_mbd_1(self.context(features))
        features = self.quarter_mbd_2(self.quarter_refine(features))

        features = self.up1(features) + half_skip
        features = self.half_decoder_mbd(self.half_decoder(features))
        features = self.up2(features) + full_skip
        features = self.full_decoder(features)
        restored = self.ending(features) + padded
        return restored[:, :, :height, :width]


class MBDNetPPlusR(nn.Module):
    """Re-parameterizable accuracy/speed candidate derived from MBDNet-P+.

    Training topology uses RepDWPW blocks.  After ``switch_to_deploy``, every
    RepDepthwiseConv is exactly fused into one 3x3 depthwise convolution.  The
    quarter-resolution generic path has 12 blocks (10 encoder + 2 refine),
    compared with 14 in MBDNet-P+, reducing actual deployment operators while
    retaining the three MBD blocks and their local motion-biased routing.
    """

    def __init__(self, img_channel: int = 3, width: int = 64) -> None:
        super().__init__()
        if img_channel != 3:
            raise ValueError("MBDNetPPlusR currently expects RGB inputs")
        if width <= 0 or width % 2:
            raise ValueError(f"MBDNetPPlusR width must be a positive even integer, got {width}")

        half = width * 2
        quarter = width * 4
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, bias=True)
        self.full_encoder = nn.Sequential(*[RepDWPWBlock(width) for _ in range(3)])
        self.down1 = DownsampleDWPW(width)
        self.half_encoder = nn.Sequential(*[RepDWPWBlock(half) for _ in range(4)])
        self.half_encoder_mbd = MotionBiasedDeblurringBlock(half)
        self.down2 = DownsampleDWPW(half)
        self.quarter_encoder = nn.Sequential(*[RepDWPWBlock(quarter) for _ in range(10)])
        self.context = LowResolutionContext(quarter)
        self.quarter_mbd_1 = MotionBiasedDeblurringBlock(quarter)
        self.quarter_refine = nn.Sequential(RepDWPWBlock(quarter), RepDWPWBlock(quarter))
        self.quarter_mbd_2 = MotionBiasedDeblurringBlock(quarter)

        self.up1 = UpsampleDWPW(quarter)
        self.half_decoder = nn.Sequential(*[RepDWPWBlock(half) for _ in range(4)])
        self.half_decoder_mbd = MotionBiasedDeblurringBlock(half)
        self.up2 = UpsampleDWPW(half)
        self.full_decoder = nn.Sequential(*[RepDWPWBlock(width) for _ in range(3)])
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, bias=True)
        self.padder_size = 4

    def switch_to_deploy(self) -> None:
        for module in list(self.modules()):
            if isinstance(module, RepDepthwiseConv):
                module.switch_to_deploy()

    def check_image_size(self, image: torch.Tensor) -> torch.Tensor:
        _, _, height, width = image.shape
        pad_height = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_width = (self.padder_size - width % self.padder_size) % self.padder_size
        if pad_height == 0 and pad_width == 0:
            return image
        return F.pad(image, (0, pad_width, 0, pad_height), mode="replicate")

    def forward(self, blurry: torch.Tensor) -> torch.Tensor:
        _, _, height, width = blurry.shape
        padded = self.check_image_size(blurry)
        full_skip = self.full_encoder(self.intro(padded))
        half_skip = self.half_encoder_mbd(self.half_encoder(self.down1(full_skip)))
        features = self.quarter_encoder(self.down2(half_skip))
        features = self.quarter_mbd_1(self.context(features))
        features = self.quarter_mbd_2(self.quarter_refine(features))
        features = self.up1(features) + half_skip
        features = self.half_decoder_mbd(self.half_decoder(features))
        features = self.up2(features) + full_skip
        features = self.full_decoder(features)
        restored = self.ending(features) + padded
        return restored[:, :, :height, :width]
