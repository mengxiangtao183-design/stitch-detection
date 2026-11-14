import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class EdgeMapLoss(nn.Module):
    """Balanced supervision for a predicted one-channel stitch edge map."""

    def __init__(self, eps=1e-6, bce_weight=0.5):
        super(EdgeMapLoss, self).__init__()
        self.eps = eps
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        if pred is None:
            return target.new_tensor(0.0)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if target.size(1) != 1:
            target = target[:, :1]
        if pred.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=pred.shape[-2:], mode='nearest')

        pred = torch.clamp(pred.float(), self.eps, 1.0 - self.eps)
        target = (target > 0.5).float()
        # Stitch labels occupy only about 0.4% of a patch.  A plain mean BCE
        # is dominated by textile background and quickly learns an all-zero
        # edge map, so balance positive and negative pixels per image before
        # combining it with Dice.
        positive = target.sum(dim=(1, 2, 3), keepdim=True)
        negative = target[0].numel() - positive
        positive_weight = (negative / (positive + self.eps)).clamp(max=128.0)
        bce_terms = -(
            positive_weight * target * torch.log(pred)
            + (1.0 - target) * torch.log(1.0 - pred)
        )
        normalizer = (positive_weight * positive + negative).clamp_min(self.eps)
        bce = (bce_terms.sum(dim=(1, 2, 3), keepdim=True) / normalizer).mean()

        pred_flat = pred.reshape(pred.size(0), -1)
        target_flat = target.reshape(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = 1.0 - (2.0 * intersection + self.eps) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.eps
        )
        return self.bce_weight * bce + dice.mean()


class BandLimitedFrequencyLoss(nn.Module):
    """Band-limited complex-spectrum consistency loss for restoration training."""

    def __init__(
        self,
        low_start=0.05,
        low_end=0.10,
        high_start=0.30,
        high_end=0.40,
        log_magnitude_weight=0.25,
        eps=1e-3,
    ):
        super(BandLimitedFrequencyLoss, self).__init__()
        self.low_start = float(low_start)
        self.low_end = float(low_end)
        self.high_start = float(high_start)
        self.high_end = float(high_end)
        self.log_magnitude_weight = float(log_magnitude_weight)
        self.eps = float(eps)
        if not (
            0.0 <= self.low_start < self.low_end
            < self.high_start < self.high_end <= 0.5
        ):
            raise ValueError(
                'Frequency consistency band edges must satisfy '
                '0 <= low_start < low_end < high_start < high_end <= 0.5.'
            )
        if self.log_magnitude_weight < 0.0:
            raise ValueError('log_magnitude_weight must be non-negative.')

    @staticmethod
    def _smoothstep(x):
        x = x.clamp(0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _band_mask(self, height, width, device):
        fy = torch.fft.fftfreq(height, device=device, dtype=torch.float32).view(
            height, 1
        )
        fx = torch.fft.rfftfreq(width, device=device, dtype=torch.float32).view(
            1, width // 2 + 1
        )
        radius = torch.sqrt(fy.square() + fx.square())
        low_rise = self._smoothstep(
            (radius - self.low_start) / (self.low_end - self.low_start)
        )
        high_fall = 1.0 - self._smoothstep(
            (radius - self.high_start) / (self.high_end - self.high_start)
        )
        return (low_rise * high_fall).clamp_min(0.0)

    def _masked_mean(self, value, mask):
        normalizer = mask.sum().clamp_min(1e-6) * value.size(0) * value.size(1)
        return (value * mask.view(1, 1, *mask.shape)).sum() / normalizer

    def forward(self, x, y):
        if x.shape[-2:] != y.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)

        x = x.float()
        y = y.float()
        height, width = x.shape[-2:]
        mask = self._band_mask(height, width, x.device)

        spectrum_x = torch.fft.rfft2(x, norm='ortho')
        spectrum_y = torch.fft.rfft2(y, norm='ortho')
        diff = spectrum_x - spectrum_y
        complex_error = torch.sqrt(
            diff.real.square() + diff.imag.square() + self.eps * self.eps
        )
        loss = self._masked_mean(complex_error, mask)

        if self.log_magnitude_weight > 0.0:
            magnitude_error = torch.sqrt(
                (
                    torch.log1p(spectrum_x.abs())
                    - torch.log1p(spectrum_y.abs())
                ).square()
                + self.eps * self.eps
            )
            loss = loss + self.log_magnitude_weight * self._masked_mean(
                magnitude_error, mask
            )
        return loss
