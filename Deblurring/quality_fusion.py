import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import utils
from MPRNet import build_mprnet_from_config


class QualityFusionGate(nn.Module):
    """Image-level quality gate for blending two restoration experts."""

    def __init__(
        self,
        in_channels=12,
        hidden=(32, 48, 64),
        mlp_hidden=16,
        use_descriptors=False,
        descriptor_hidden=16,
        roi_top_ratio=1.0 / 3.0,
        roi_bottom_ratio=2.0 / 3.0,
    ):
        super().__init__()
        c1, c2, c3 = hidden
        self.use_descriptors = bool(use_descriptors)
        self.roi_top_ratio = float(roi_top_ratio)
        self.roi_bottom_ratio = float(roi_bottom_ratio)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1),
            nn.PReLU(c1),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.PReLU(c2),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.PReLU(c3),
            nn.AdaptiveAvgPool2d(1),
        )
        classifier_channels = c3
        if self.use_descriptors:
            self.descriptor_features = nn.Sequential(
                nn.Linear(11, descriptor_hidden),
                nn.PReLU(descriptor_hidden),
            )
            classifier_channels += descriptor_hidden
        else:
            self.descriptor_features = None
        self.classifier = nn.Sequential(
            # Kept for state-dict compatibility with the V1 gate.
            nn.Flatten(),
            nn.Linear(classifier_channels, mlp_hidden),
            nn.PReLU(mlp_hidden),
            nn.Linear(mlp_hidden, 1),
        )

    @staticmethod
    def _luma(image):
        weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (image * weights).sum(dim=1, keepdim=True)

    def _roi(self, tensor):
        height = tensor.shape[2]
        top = max(0, min(height - 1, int(round(height * self.roi_top_ratio))))
        bottom = max(top + 1, min(height, int(round(height * self.roi_bottom_ratio))))
        return tensor[:, :, top:bottom, :]

    @staticmethod
    def _mean_std(tensor):
        flattened = tensor.flatten(1)
        return flattened.mean(dim=1), flattened.std(dim=1, unbiased=False)

    @staticmethod
    def _gradient_energy(luma):
        grad_x = torch.abs(luma[:, :, :, 1:] - luma[:, :, :, :-1]).mean(dim=(1, 2, 3))
        grad_y = torch.abs(luma[:, :, 1:, :] - luma[:, :, :-1, :]).mean(dim=(1, 2, 3))
        return 0.5 * (grad_x + grad_y)

    def _quality_descriptors(self, degraded, expert_a, expert_b):
        luma = self._luma(degraded)
        luma_roi = self._roi(luma)
        luma_mean, luma_std = self._mean_std(luma)
        roi_mean, roi_std = self._mean_std(luma_roi)

        disagreement = torch.abs(expert_a - expert_b)
        disagreement_roi = self._roi(disagreement)
        diff_mean, diff_std = self._mean_std(disagreement)
        roi_diff_mean, roi_diff_std = self._mean_std(disagreement_roi)
        disagreement_ratio = roi_diff_mean / diff_mean.clamp_min(1.0e-6)

        # Low-amplitude statistics are scaled into a range comparable to luma.
        return torch.stack(
            [
                luma_mean,
                luma_std,
                roi_mean,
                roi_std,
                self._gradient_energy(luma) * 4.0,
                self._gradient_energy(luma_roi) * 4.0,
                diff_mean * 10.0,
                diff_std * 10.0,
                roi_diff_mean * 10.0,
                roi_diff_std * 10.0,
                disagreement_ratio.clamp(0.0, 4.0) * 0.25,
            ],
            dim=1,
        )

    def forward(
        self,
        degraded,
        expert_a,
        expert_b,
        temperature=1.0,
        return_logits=False,
    ):
        gate_input = torch.cat(
            [degraded, expert_a, expert_b, torch.abs(expert_a - expert_b)],
            dim=1,
        )
        encoded = self.features(gate_input).flatten(1)
        if self.use_descriptors:
            descriptors = self.descriptor_features(
                self._quality_descriptors(degraded, expert_a, expert_b)
            )
            encoded = torch.cat([encoded, descriptors], dim=1)
        logits = self.classifier(encoded).view(-1, 1, 1, 1)
        temperature = max(float(temperature), 1.0e-4)
        alpha = torch.sigmoid(logits / temperature)
        fused = expert_a * (1.0 - alpha) + expert_b * alpha
        if return_logits:
            return fused, alpha, logits
        return fused, alpha


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path, base_dir=None):
    if not path:
        return ""
    path = Path(path)
    if path.is_absolute() or base_dir is None:
        return str(path)
    return str(Path(base_dir) / path)


def model_config_from_training_yaml(config_path):
    config = load_yaml(config_path)
    return config.get("MODEL", {})


def build_frozen_expert(config_path, weights_path, device, strict=True):
    model_cfg = model_config_from_training_yaml(config_path)
    model = build_mprnet_from_config(model_cfg)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Expert weights not found: {weights_path}")
    utils.load_checkpoint(model, weights_path, strict=strict)
    utils.freeze(model)
    model.to(device)
    model.eval()
    return model


def forward_padded(model, input_tensor, factor=8, use_amp=False):
    h, w = input_tensor.shape[2], input_tensor.shape[3]
    padded_h = ((h + factor - 1) // factor) * factor
    padded_w = ((w + factor - 1) // factor) * factor
    pad_h = padded_h - h
    pad_w = padded_w - w
    if pad_h > 0 or pad_w > 0:
        input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode="reflect")

    autocast_enabled = bool(use_amp and input_tensor.is_cuda)
    if autocast_enabled and hasattr(torch, "amp"):
        with torch.amp.autocast("cuda"):
            output = model(input_tensor, return_intermediate=False)
    elif autocast_enabled:
        with torch.cuda.amp.autocast():
            output = model(input_tensor, return_intermediate=False)
    else:
        output = model(input_tensor, return_intermediate=False)
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, (list, tuple)):
        output = output[0]
    output = torch.clamp(output, 0.0, 1.0)
    if pad_h > 0 or pad_w > 0:
        output = output[:, :, :h, :w]
    return output
