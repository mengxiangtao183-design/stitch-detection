"""Standard semantic-segmentation baselines adapted to the C3Net engine.

Every model returns raw logits under the three keys consumed by the existing
trainer: ``final_mask``, ``object_mask`` and ``edge_mask``.  The final mask is
the architecture's primary prediction; the other two outputs are lightweight
auxiliary heads and are ignored by the evaluator.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for value in (32, 16, 8, 4, 2):
        if channels % value == 0:
            return value
    return 1


class ConvGNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1):
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                      dilation=dilation, bias=False),
            nn.GroupNorm(_groups(out_ch), out_ch),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(ConvGNAct(in_ch, out_ch), ConvGNAct(out_ch, out_ch))


def _pack(final: torch.Tensor, object_logits: torch.Tensor | None = None,
          edge_logits: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
    return {
        "final_mask": final,
        "object_mask": final if object_logits is None else object_logits,
        "edge_mask": final if edge_logits is None else edge_logits,
    }


class UNet(nn.Module):
    """Vanilla five-level U-Net with independent auxiliary prediction heads."""

    def __init__(self, config: Dict):
        super().__init__()
        base = int(config.get("base_channels", 64))
        channels = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc1 = DoubleConv(3, channels[0])
        self.enc2 = DoubleConv(channels[0], channels[1])
        self.enc3 = DoubleConv(channels[1], channels[2])
        self.enc4 = DoubleConv(channels[2], channels[3])
        self.bottleneck = DoubleConv(channels[3], channels[4])
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(channels[4], channels[3], 2, 2)
        self.dec4 = DoubleConv(channels[3] * 2, channels[3])
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, 2)
        self.dec3 = DoubleConv(channels[2] * 2, channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, 2)
        self.dec2 = DoubleConv(channels[1] * 2, channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, 2)
        self.dec1 = DoubleConv(channels[0] * 2, channels[0])
        self.final_head = nn.Conv2d(channels[0], 1, 1)
        self.edge_head = nn.Conv2d(channels[0], 1, 1)
        self.object_head = nn.Conv2d(channels[4], 1, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        size = x.shape[-2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        final = self.final_head(d1)
        edge = self.edge_head(d1)
        obj = F.interpolate(self.object_head(b), size=size, mode="bilinear", align_corners=False)
        return _pack(final, obj, edge)


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 256,
                 rates: Sequence[int] = (6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([ConvGNAct(in_ch, out_ch, 1)])
        self.branches.extend(ConvGNAct(in_ch, out_ch, 3, dilation=r) for r in rates)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvGNAct(in_ch, out_ch, 1))
        self.project = ConvGNAct(out_ch * (len(rates) + 2), out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        feats = [branch(x) for branch in self.branches]
        feats.append(F.interpolate(self.pool(x), size=size, mode="bilinear", align_corners=False))
        return self.project(torch.cat(feats, dim=1))


def _resnet50(pretrained: bool):
    from torchvision.models import ResNet50_Weights, resnet50
    return resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)


class ResNetFeatures(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        net = _resnet50(pretrained)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ with a ResNet-50 encoder and low-level decoder."""

    def __init__(self, config: Dict):
        super().__init__()
        self.backbone = ResNetFeatures(bool(config.get("pretrained", True)))
        self.aspp = ASPP(2048, 256, tuple(config.get("aspp_rates", [6, 12, 18])))
        self.low_proj = ConvGNAct(256, 48, 1)
        self.decoder = nn.Sequential(ConvGNAct(304, 256), ConvGNAct(256, 256))
        self.final_head = nn.Conv2d(256, 1, 1)
        self.object_head = nn.Conv2d(256, 1, 1)
        self.edge_head = nn.Conv2d(48, 1, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        size = x.shape[-2:]
        c2, _, _, c5 = self.backbone(x)
        high = self.aspp(c5)
        low = self.low_proj(c2)
        high_up = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        dec = self.decoder(torch.cat([high_up, low], dim=1))
        final = F.interpolate(self.final_head(dec), size=size, mode="bilinear", align_corners=False)
        obj = F.interpolate(self.object_head(high), size=size, mode="bilinear", align_corners=False)
        edge = F.interpolate(self.edge_head(low), size=size, mode="bilinear", align_corners=False)
        return _pack(final, obj, edge)


class SegFormerB0(nn.Module):
    """SegFormer-B0 encoder with the paper's all-MLP multi-scale decoder."""

    def __init__(self, config: Dict):
        super().__init__()
        from transformers import SegformerConfig, SegformerModel
        pretrained = bool(config.get("pretrained", True))
        model_id = str(config.get("pretrained_path") or config.get("pretrained_id", "nvidia/mit-b0"))
        if pretrained:
            self.backbone = SegformerModel.from_pretrained(
                model_id,
                local_files_only=bool(config.get("pretrained_path")),
            )
        else:
            sf_config = SegformerConfig(
                num_channels=3,
                hidden_sizes=[32, 64, 160, 256],
                depths=[2, 2, 2, 2],
                num_attention_heads=[1, 2, 5, 8],
                sr_ratios=[8, 4, 2, 1],
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                mlp_ratios=[4, 4, 4, 4],
            )
            self.backbone = SegformerModel(sf_config)
        hidden = list(self.backbone.config.hidden_sizes)
        decoder_ch = int(config.get("decoder_channels", 256))
        self.projections = nn.ModuleList([nn.Conv2d(ch, decoder_ch, 1) for ch in hidden])
        self.fuse = ConvGNAct(decoder_ch * 4, decoder_ch, 1)
        self.final_head = nn.Conv2d(decoder_ch, 1, 1)
        self.object_head = nn.Conv2d(hidden[-1], 1, 1)
        self.edge_head = nn.Conv2d(decoder_ch, 1, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        size = x.shape[-2:]
        outputs = self.backbone(x, output_hidden_states=True, return_dict=True)
        stages = list(outputs.hidden_states)
        target = stages[0].shape[-2:]
        projected = [
            F.interpolate(proj(feat), size=target, mode="bilinear", align_corners=False)
            for proj, feat in zip(self.projections, stages)
        ]
        fused = self.fuse(torch.cat(projected, dim=1))
        final = F.interpolate(self.final_head(fused), size=size, mode="bilinear", align_corners=False)
        obj = F.interpolate(self.object_head(stages[-1]), size=size, mode="bilinear", align_corners=False)
        edge = F.interpolate(self.edge_head(projected[0]), size=size, mode="bilinear", align_corners=False)
        return _pack(final, obj, edge)


class GatedSpatialConv(nn.Module):
    def __init__(self, channels: int, gate_channels: int):
        super().__init__()
        self.gate_proj = nn.Conv2d(gate_channels, 1, 1)
        self.alpha = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1), nn.Sigmoid(),
        )
        self.refine = ConvGNAct(channels, channels)

    def forward(self, shape: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = F.interpolate(self.gate_proj(gate), size=shape.shape[-2:],
                             mode="bilinear", align_corners=False)
        alpha = self.alpha(torch.cat([shape, gate], dim=1))
        return self.refine(shape * (1.0 + alpha))


class GatedSCNN(nn.Module):
    """Gated-SCNN-style regular/shape dual stream with ResNet-50 context."""

    def __init__(self, config: Dict):
        super().__init__()
        self.backbone = ResNetFeatures(bool(config.get("pretrained", True)))
        self.semantic = ASPP(2048, 256)
        self.low_proj = ConvGNAct(256, 48, 1)
        self.shape_init = ConvGNAct(256, 64, 1)
        self.gate2 = GatedSpatialConv(64, 512)
        self.gate3 = GatedSpatialConv(64, 1024)
        self.gate4 = GatedSpatialConv(64, 256)
        self.edge_head = nn.Conv2d(64, 1, 1)
        self.decoder = nn.Sequential(ConvGNAct(256 + 48 + 64, 256), ConvGNAct(256, 128))
        self.final_head = nn.Conv2d(128, 1, 1)
        self.object_head = nn.Conv2d(256, 1, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        size = x.shape[-2:]
        c2, c3, c4, c5 = self.backbone(x)
        semantic = self.semantic(c5)
        shape = self.shape_init(c2)
        shape = self.gate2(shape, c3)
        shape = self.gate3(shape, c4)
        shape = self.gate4(shape, semantic)
        low = self.low_proj(c2)
        semantic_up = F.interpolate(semantic, size=shape.shape[-2:], mode="bilinear", align_corners=False)
        dec = self.decoder(torch.cat([semantic_up, low, shape], dim=1))
        final = F.interpolate(self.final_head(dec), size=size, mode="bilinear", align_corners=False)
        obj = F.interpolate(self.object_head(semantic), size=size, mode="bilinear", align_corners=False)
        edge = F.interpolate(self.edge_head(shape), size=size, mode="bilinear", align_corners=False)
        return _pack(final, obj, edge)


class FPN(nn.Module):
    def __init__(self, out_ch: int = 256):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(ch, out_ch, 1) for ch in (256, 512, 1024, 2048)])
        self.smooth = nn.ModuleList([ConvGNAct(out_ch, out_ch) for _ in range(4)])

    def forward(self, features: Iterable[torch.Tensor]):
        c2, c3, c4, c5 = features
        laterals = [layer(feat) for layer, feat in zip(self.lateral, (c2, c3, c4, c5))]
        p5 = laterals[3]
        p4 = laterals[2] + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = laterals[1] + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = laterals[0] + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        return tuple(layer(feat) for layer, feat in zip(self.smooth, (p2, p3, p4, p5)))


class PointRend(nn.Module):
    """PointRend with a ResNet-50 FPN, coarse head and uncertainty points."""

    def __init__(self, config: Dict):
        super().__init__()
        self.backbone = ResNetFeatures(bool(config.get("pretrained", True)))
        self.fpn = FPN(256)
        self.coarse_fuse = ConvGNAct(256 * 4, 256, 1)
        self.coarse_head = nn.Conv2d(256, 1, 1)
        self.edge_head = nn.Conv2d(256, 1, 1)
        self.num_points = int(config.get("num_points", 8192))
        self.point_mlp = nn.Sequential(
            nn.Conv1d(257, 256, 1), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 1), nn.ReLU(inplace=True),
            nn.Conv1d(256, 1, 1),
        )

    @staticmethod
    def _sample(feat: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        grid = coords.unsqueeze(2)
        return F.grid_sample(feat, grid, mode="bilinear", align_corners=False).squeeze(3)

    def _refine(self, coarse: torch.Tensor, fine: torch.Tensor,
                target_size) -> torch.Tensor:
        dense = F.interpolate(coarse, size=target_size, mode="bilinear", align_corners=False)
        b, _, h, w = dense.shape
        count = min(self.num_points, h * w)
        uncertainty = -dense.detach().abs().flatten(2)
        indices = uncertainty.topk(count, dim=2).indices
        ys = torch.div(indices, w, rounding_mode="floor").to(dense.dtype)
        xs = (indices % w).to(dense.dtype)
        coords = torch.stack(((xs + 0.5) * 2.0 / w - 1.0,
                              (ys + 0.5) * 2.0 / h - 1.0), dim=-1).squeeze(1)
        fine_points = self._sample(fine, coords)
        coarse_points = self._sample(dense, coords)
        point_logits = self.point_mlp(torch.cat([fine_points, coarse_points], dim=1))
        refined = dense.flatten(2).scatter(2, indices, point_logits)
        return refined.view(b, 1, h, w)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        size = x.shape[-2:]
        pyramids = self.fpn(self.backbone(x))
        p2 = pyramids[0]
        multi = [p2] + [
            F.interpolate(feat, size=p2.shape[-2:], mode="bilinear", align_corners=False)
            for feat in pyramids[1:]
        ]
        coarse_feat = self.coarse_fuse(torch.cat(multi, dim=1))
        coarse = self.coarse_head(coarse_feat)
        final = self._refine(coarse, p2, size)
        obj = F.interpolate(coarse, size=size, mode="bilinear", align_corners=False)
        edge = F.interpolate(self.edge_head(p2), size=size, mode="bilinear", align_corners=False)
        return _pack(final, obj, edge)


class PyramidPooling(nn.Module):
    """PSPNet pyramid pooling with four fixed global-context scales."""

    def __init__(self, in_ch: int, reduction_ch: int = 512,
                 bins: Sequence[int] = (1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(bin_size), ConvGNAct(in_ch, reduction_ch, 1))
            for bin_size in bins
        ])
        self.fuse = nn.Sequential(
            ConvGNAct(in_ch + len(bins) * reduction_ch, reduction_ch),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        pooled = [x]
        pooled.extend(F.interpolate(stage(x), size=size, mode="bilinear", align_corners=False)
                      for stage in self.stages)
        return self.fuse(torch.cat(pooled, dim=1))


class PSPNet(nn.Module):
    """PSPNet-ResNet50 reference baseline with binary auxiliary heads."""

    def __init__(self, config: Dict):
        super().__init__()
        self.backbone = ResNetFeatures(bool(config.get("pretrained", True)))
        psp_channels = int(config.get("psp_channels", 512))
        self.ppm = PyramidPooling(2048, psp_channels)
        self.final_head = nn.Conv2d(psp_channels, 1, 1)
        self.object_head = nn.Conv2d(1024, 1, 1)
        self.edge_head = nn.Sequential(ConvGNAct(512, 128), nn.Conv2d(128, 1, 1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        size = x.shape[-2:]
        c2, c3, c4, c5 = self.backbone(x)
        context = self.ppm(c5)
        final = F.interpolate(self.final_head(context), size=size, mode="bilinear", align_corners=False)
        obj = F.interpolate(self.object_head(c4), size=size, mode="bilinear", align_corners=False)
        edge = F.interpolate(self.edge_head(c3), size=size, mode="bilinear", align_corners=False)
        return _pack(final, obj, edge)
