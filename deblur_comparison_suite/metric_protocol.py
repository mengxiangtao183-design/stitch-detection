from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3)).clamp_min(1e-10)
    return 10.0 * torch.log10(1.0 / mse)


def psnr_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    return (10.0 / math.log(10.0)) * torch.log(mse + 1e-8).mean()


def middle_third(image: torch.Tensor) -> torch.Tensor:
    height = image.shape[-2]
    top = height // 3
    bottom = (2 * height) // 3
    return image[..., top:bottom, :]


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(f"SSIM expects equal NCHW tensors, got {prediction.shape}, {target.shape}")
    coordinates = torch.arange(window_size, device=prediction.device, dtype=prediction.dtype)
    coordinates = coordinates - (window_size - 1) / 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    channels = prediction.shape[1]
    window = kernel_2d.expand(channels, 1, window_size, window_size)

    def filt(image: torch.Tensor) -> torch.Tensor:
        return F.conv2d(image, window, groups=channels)

    mu_x, mu_y = filt(prediction), filt(target)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    var_x = filt(prediction.square()) - mu_x2
    var_y = filt(target.square()) - mu_y2
    cov_xy = filt(prediction * target) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_xy + c1) * (2 * cov_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (var_x + var_y + c2)
    )
    return score.mean(dim=(1, 2, 3))
