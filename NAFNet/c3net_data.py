"""Paired data utilities for the DB/XJ/SD/RX industrial stitch datasets."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset


DEFAULT_DOMAINS = ("DB", "XJ", "SD", "RX")


def read_manifest(path: str | Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        target, source = line.split("\t")
        pairs.append((target, source))
    if not pairs:
        raise RuntimeError(f"Manifest is empty: {path}")
    return pairs


def _to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _pad_to_size(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, height, width = image.shape
    pad_w = max(0, patch_size - width)
    pad_h = max(0, patch_size - height)
    if not pad_w and not pad_h:
        return image

    mode = "reflect" if pad_w < width and pad_h < height else "replicate"
    return F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode=mode).squeeze(0)


class PairedManifestDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        manifest: str | Path,
        patch_size: int | None,
        augment: bool,
        source_style_prob: float = 0.0,
        source_gain: float = 0.0,
        source_bias: float = 0.0,
        source_gamma: float = 0.0,
        source_noise_std: float = 0.0,
        source_band_gain: float = 0.0,
    ) -> None:
        self.data_root = Path(data_root)
        self.pairs = read_manifest(manifest)
        self.patch_size = patch_size
        self.augment = augment
        self.source_style_prob = float(source_style_prob)
        self.source_gain = float(source_gain)
        self.source_bias = float(source_bias)
        self.source_gamma = float(source_gamma)
        self.source_noise_std = float(source_noise_std)
        self.source_band_gain = float(source_band_gain)
        values = (
            self.source_style_prob,
            self.source_gain,
            self.source_bias,
            self.source_gamma,
            self.source_noise_std,
            self.source_band_gain,
        )
        if any(value < 0.0 for value in values):
            raise ValueError("source style randomization parameters must be non-negative")

    def _randomize_source_style(self, source: torch.Tensor) -> torch.Tensor:
        """Apply training-only, source-only domain randomization.

        Motion blur remains the pre-generated LineTrace SynBlur-v2 input.  This
        step only varies photometric/style statistics so that RX-like texture
        and illumination shifts are not mistaken for the restoration target.
        It is intentionally never called for validation or test samples.
        """

        if self.source_style_prob <= 0.0 or random.random() >= self.source_style_prob:
            return source

        out = source
        if self.source_gain > 0.0 or self.source_bias > 0.0:
            gain = random.uniform(1.0 - self.source_gain, 1.0 + self.source_gain)
            bias = random.uniform(-self.source_bias, self.source_bias)
            out = out * gain + bias
        if self.source_gamma > 0.0:
            gamma = random.uniform(1.0 - self.source_gamma, 1.0 + self.source_gamma)
            out = out.clamp(1e-4, 1.0).pow(gamma)
        if self.source_band_gain > 0.0:
            # Smooth row-wise illumination variation.  It is not a fixed
            # middle-band prior and is independent of GT or degradation labels.
            height = out.shape[-2]
            knots = torch.tensor(
                [
                    random.uniform(1.0 - self.source_band_gain, 1.0 + self.source_band_gain)
                    for _ in range(4)
                ],
                dtype=out.dtype,
                device=out.device,
            ).view(1, 1, 4, 1)
            row_gain = F.interpolate(knots, size=(height, 1), mode="bilinear", align_corners=True)
            out = out * row_gain.squeeze(0)
        if self.source_noise_std > 0.0:
            std = random.uniform(0.0, self.source_noise_std)
            out = out + torch.randn_like(out) * std
        return out.clamp(0.0, 1.0)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        target_rel, source_rel = self.pairs[index]
        target = Image.open(self.data_root / target_rel).convert("RGB")
        source = Image.open(self.data_root / source_rel).convert("RGB")
        if target.size != source.size:
            raise RuntimeError(f"Unaligned pair: {target_rel} / {source_rel}")

        target = _to_tensor(target)
        source = _to_tensor(source)
        if self.patch_size is not None:
            target = _pad_to_size(target, self.patch_size)
            source = _pad_to_size(source, self.patch_size)
            _, height, width = target.shape
            if self.augment:
                top = random.randint(0, height - self.patch_size)
                left = random.randint(0, width - self.patch_size)
            else:
                top = (height - self.patch_size) // 2
                left = (width - self.patch_size) // 2
            target = target[:, top : top + self.patch_size, left : left + self.patch_size]
            source = source[:, top : top + self.patch_size, left : left + self.patch_size]

        if self.augment:
            if random.random() < 0.5:
                target, source = target.flip(2), source.flip(2)
            if random.random() < 0.5:
                target, source = target.flip(1), source.flip(1)
            if random.random() < 0.5:
                target, source = torch.rot90(target, 2, (1, 2)), torch.rot90(source, 2, (1, 2))
            source = self._randomize_source_style(source)

        name = source_rel.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return target, source, name


def _make_odd(value: int, minimum: int = 5, maximum: int = 35) -> int:
    value = int(max(minimum, min(maximum, value)))
    return value if value % 2 else value + 1 if value < maximum else value - 1


def _motion_kernel(length: int, angle_degrees: float) -> np.ndarray:
    """Construct a deterministic, near-horizontal line-motion kernel."""

    length = _make_odd(length)
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    radians = np.deg2rad(angle_degrees)
    dx, dy = np.cos(radians), np.sin(radians)
    for index in range(length):
        x = int(round(center + (index - center) * dx))
        y = int(round(center + (index - center) * dy))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    if float(kernel.sum()) <= 0.0:
        kernel[center, :] = 1.0
    return kernel / kernel.sum()


def _line_evidence_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Create a training-only trace/edge evidence map from the clear target."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(smooth, 40, 120)
    grad_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    threshold = max(8.0, float(np.percentile(magnitude, 82.0)))
    evidence = cv2.bitwise_or(edges, (magnitude >= threshold).astype(np.uint8) * 255)
    evidence = cv2.morphologyEx(evidence, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)))
    evidence = cv2.dilate(evidence, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    return np.clip(cv2.GaussianBlur(evidence.astype(np.float32) / 255.0, (0, 0), 2.0), 0.0, 1.0)


def _v3_stripe_weights(height: int, width: int, stripe_count: int, rng: random.Random) -> list[np.ndarray]:
    """Generate non-fixed, feathered horizontal regions that sum to one."""

    minimum_gap = max(24, height // 10)
    candidates = list(range(minimum_gap, height - minimum_gap))
    cuts = sorted(rng.sample(candidates, stripe_count - 1))
    boundaries = [0, *cuts, height]
    feather = max(6, height // 40)
    y = np.arange(height, dtype=np.float32)[:, None]
    weights: list[np.ndarray] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        left = np.clip((y - (start - feather)) / max(1.0, feather), 0.0, 1.0)
        right = np.clip(((end + feather) - y) / max(1.0, feather), 0.0, 1.0)
        weights.append(np.repeat(np.minimum(left, right), width, axis=1))
    total = np.maximum(sum(weights), 1e-6)
    return [(weight / total).astype(np.float32) for weight in weights]


def _dynamic_linetrace_v3(image_rgb: np.ndarray, rng: random.Random, profile: str) -> np.ndarray:
    """Synthesize one fresh, strictly paired V3 input from one clear target.

    The target is never photometrically modified.  Every corruption is applied
    only while producing the input image, making ``source -> original target``
    a valid supervised restoration pair.  No test images or stored degradation
    parameters are consulted.
    """

    if profile not in {"mild", "mixed"}:
        raise ValueError(f"Unsupported dynamic V3 profile: {profile}")
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = image_bgr.shape[:2]
    if profile == "mild":
        base_length = rng.choice((9, 11, 13, 15, 17, 19, 21))
        angle_limit, stripe_choices = 14.0, (2, 3)
        line_range, background_range = (0.84, 0.96), (0.34, 0.56)
    else:
        base_length = rng.choice((9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31))
        angle_limit, stripe_choices = 22.0, (2, 3, 4)
        line_range, background_range = (0.86, 1.00), (0.36, 0.66)
    base_angle = rng.uniform(-angle_limit, angle_limit)
    weights = _v3_stripe_weights(height, width, rng.choice(stripe_choices), rng)

    strong = np.zeros_like(image_bgr, dtype=np.float32)
    for index, weight in enumerate(weights):
        length = _make_odd(base_length + rng.choice((-6, -4, -2, 0, 2, 4, 6)))
        # The middle region may be modestly harder, but its position is random
        # because the stripe boundaries are sampled per image.
        if index == len(weights) // 2:
            length = _make_odd(length + rng.choice((0, 2, 4)))
        angle = base_angle + rng.uniform(-7.0, 7.0)
        blurred = cv2.filter2D(image_bgr, -1, _motion_kernel(length, angle), borderType=cv2.BORDER_REFLECT101)
        strong += blurred.astype(np.float32) * weight[:, :, None]

    mild_length = _make_odd(base_length - rng.choice((6, 8, 10)), minimum=5, maximum=21)
    mild = cv2.filter2D(
        image_bgr, -1, _motion_kernel(mild_length, base_angle + rng.uniform(-4.0, 4.0)), borderType=cv2.BORDER_REFLECT101
    ).astype(np.float32)
    evidence = _line_evidence_mask(image_bgr)[:, :, None]
    image_f = image_bgr.astype(np.float32)
    line_strength = rng.uniform(*line_range)
    background_strength = rng.uniform(*background_range)
    line_region = image_f * (1.0 - line_strength) + strong * line_strength
    background = image_f * (1.0 - background_strength) + mild * background_strength
    degraded = evidence * line_region + (1.0 - evidence) * background

    # Exactly zero or one weak auxiliary degradation; this avoids the V2G
    # failure mode of indiscriminately stacking unrelated input-only styles.
    choice = rng.random()
    if choice < 0.08:
        sigma = rng.uniform(0.8, 3.0)
        degraded += np.random.default_rng(rng.randrange(2**32)).normal(0.0, sigma, degraded.shape)
    elif choice < 0.15:
        quality = rng.randint(84, 96)
        ok, encoded = cv2.imencode(".jpg", np.clip(degraded, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                degraded = decoded.astype(np.float32)
    elif choice < 0.20:
        gamma = rng.uniform(0.94, 1.08)
        degraded = np.power(np.clip(degraded / 255.0, 0.0, 1.0), gamma) * 255.0
    return cv2.cvtColor(np.clip(degraded, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


class DynamicLineTraceV3Dataset(Dataset):
    """Train-only DB/XJ/SD dynamic, strictly paired line-trace degradation."""

    def __init__(
        self,
        data_root: str | Path,
        manifest: str | Path,
        patch_size: int,
        repeats: int = 3,
        profile: str = "mixed",
    ) -> None:
        if repeats < 1:
            raise ValueError("Dynamic V3 repeats must be at least one")
        pairs = read_manifest(manifest)
        targets = list(dict.fromkeys(target for target, _source in pairs))
        domains = {Path(target.replace("\\", "/")).parts[0] for target in targets}
        if not domains.issubset({"DB", "XJ", "SD"}):
            raise ValueError(f"Dynamic V3 training must exclude RX, found domains: {sorted(domains)}")
        self.data_root = Path(data_root)
        self.targets = targets
        self.patch_size = patch_size
        self.repeats = repeats
        self.profile = profile

    def __len__(self) -> int:
        return len(self.targets) * self.repeats

    def __getitem__(self, index: int):
        target_rel = self.targets[index % len(self.targets)]
        image = Image.open(self.data_root / target_rel).convert("RGB")
        target = _pad_to_size(_to_tensor(image), self.patch_size)
        _, height, width = target.shape
        top = random.randint(0, height - self.patch_size)
        left = random.randint(0, width - self.patch_size)
        target = target[:, top : top + self.patch_size, left : left + self.patch_size]
        if random.random() < 0.5:
            target = target.flip(2)
        if random.random() < 0.5:
            target = target.flip(1)
        if random.random() < 0.5:
            target = torch.rot90(target, 2, (1, 2))
        target_rgb = (target.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        source_rgb = _dynamic_linetrace_v3(target_rgb, random, self.profile)
        source = _to_tensor(Image.fromarray(source_rgb, mode="RGB"))
        name = Path(target_rel).stem
        return target, source, name


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3)).clamp_min(1e-10)
    return 10.0 * torch.log10(1.0 / mse)


def middle_third(image: torch.Tensor) -> torch.Tensor:
    """Return the fixed middle horizontal third of an NCHW/CHW image."""
    height = image.shape[-2]
    top = height // 3
    bottom = (2 * height) // 3
    if bottom - top < 11:
        raise ValueError(f"Middle-third ROI is too short for 11x11 SSIM: height={height}")
    return image[..., top:bottom, :]


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Standard RGB SSIM with an 11x11 Gaussian window and data range 1.

    Returns one value per image. SSIM is averaged over the three RGB channels
    and all valid window positions, matching the common MATLAB/BasicSR setup.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"SSIM shape mismatch: {prediction.shape} != {target.shape}")
    if prediction.ndim != 4:
        raise ValueError(f"SSIM expects NCHW tensors, got shape {prediction.shape}")
    if min(prediction.shape[-2:]) < window_size:
        raise ValueError(
            f"Image is smaller than the SSIM window: {prediction.shape[-2:]} < {window_size}"
        )

    coordinates = torch.arange(
        window_size, device=prediction.device, dtype=prediction.dtype
    ) - (window_size - 1) / 2
    kernel_1d = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    channels = prediction.shape[1]
    window = kernel_2d.expand(channels, 1, window_size, window_size)

    def filter_image(image: torch.Tensor) -> torch.Tensor:
        return F.conv2d(image, window, groups=channels)

    mu_prediction = filter_image(prediction)
    mu_target = filter_image(target)
    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_prediction * mu_target

    variance_prediction = filter_image(prediction.square()) - mu_prediction_sq
    variance_target = filter_image(target.square()) - mu_target_sq
    covariance = filter_image(prediction * target) - mu_product

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    score_map = ((2 * mu_product + c1) * (2 * covariance + c2)) / (
        (mu_prediction_sq + mu_target_sq + c1)
        * (variance_prediction + variance_target + c2)
    )
    return score_map.mean(dim=(1, 2, 3))
