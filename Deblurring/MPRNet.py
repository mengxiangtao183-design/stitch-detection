"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_function
from pdb import set_trace as stx

##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

##########################################################################
## U-Net

class TrajectoryCalibratedCSFFGate(nn.Module):
    """Calibrate cross-stage injection while preserving CSFF at initialization."""

    def __init__(self, channels, horizontal_kernel=7):
        super(TrajectoryCalibratedCSFFGate, self).__init__()
        horizontal_kernel = int(horizontal_kernel)
        if horizontal_kernel < 1 or horizontal_kernel % 2 == 0:
            raise ValueError('gated_csff_horizontal_kernel must be a positive odd integer.')

        hidden_channels = max(8, channels // 4)
        self.context = nn.Sequential(
            nn.Conv2d(channels * 4, hidden_channels, kernel_size=1, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=(1, horizontal_kernel),
                padding=(0, horizontal_kernel // 2),
                groups=hidden_channels,
                bias=True,
            ),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
        )
        # 2 * sigmoid(0) == 1, so the first forward pass is exactly the
        # original additive CSFF while gradients can immediately reach the gate.
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def forward(self, current, encoder_feature, decoder_feature):
        context = torch.cat([
            current,
            encoder_feature,
            decoder_feature,
            torch.abs(encoder_feature - decoder_feature),
        ], dim=1)
        return 2.0 * torch.sigmoid(self.context(context))


class StitchTrajectoryAlignmentUnit(nn.Module):
    """Align Stage-1 features to a Stage-2 pair using a learned motion field.

    A horizontal correlation volume provides explicit trajectory evidence.  A
    zero-initialized offset head makes the initial sampling identity, so a
    pretrained progressive backbone can be fine-tuned without an initial warp.
    """

    def __init__(self, current_channels, feature_channels, horizontal_radius=6,
                 vertical_radius=1, hidden_channels=32):
        super().__init__()
        self.horizontal_radius = max(1, int(horizontal_radius))
        self.vertical_radius = max(0, int(vertical_radius))
        hidden_channels = max(8, int(hidden_channels))

        self.current_projection = nn.Conv2d(
            current_channels, hidden_channels, kernel_size=1, bias=True
        )
        self.reference_projection = nn.Conv2d(
            feature_channels, hidden_channels, kernel_size=1, bias=True
        )
        correlation_channels = 2 * self.horizontal_radius + 1
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(hidden_channels * 2 + correlation_channels,
                      hidden_channels, kernel_size=3, padding=1, bias=True),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1, bias=True),
        )
        # Identity alignment at initialization; the correlation pathway learns
        # a spatial trajectory field during fine-tuning.
        nn.init.zeros_(self.offset_predictor[-1].weight)
        nn.init.zeros_(self.offset_predictor[-1].bias)

    def _horizontal_correlation(self, query, reference):
        radius = self.horizontal_radius
        padded = F.pad(reference, (radius, radius, 0, 0), mode='replicate')
        width = reference.size(-1)
        correlations = []
        for displacement in range(-radius, radius + 1):
            shifted = padded[:, :, :, radius + displacement:radius + displacement + width]
            correlations.append((query * shifted).mean(dim=1, keepdim=True))
        return torch.cat(correlations, dim=1)

    @staticmethod
    def _warp(feature, offsets):
        batch, _, height, width = feature.shape
        y = torch.linspace(-1.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        norm_x = 2.0 / max(width - 1, 1)
        norm_y = 2.0 / max(height - 1, 1)
        offset_grid = torch.stack((offsets[:, 0] * norm_x, offsets[:, 1] * norm_y), dim=-1)
        return F.grid_sample(
            feature,
            base_grid + offset_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )

    def forward(self, current, encoder_feature, decoder_feature):
        if current.shape[-2:] != encoder_feature.shape[-2:]:
            current = F.interpolate(
                current, size=encoder_feature.shape[-2:], mode='bilinear', align_corners=False
            )
        query = self.current_projection(current)
        reference = self.reference_projection(encoder_feature)
        # Cosine correlation prevents low-contrast fabric regions from
        # producing trajectory evidence solely because feature magnitudes vary.
        correlation = self._horizontal_correlation(
            F.normalize(query, p=2, dim=1, eps=1e-6),
            F.normalize(reference, p=2, dim=1, eps=1e-6),
        )
        offsets = self.offset_predictor(torch.cat([query, reference, correlation], dim=1))
        offsets_x = torch.tanh(offsets[:, :1]) * self.horizontal_radius
        offsets_y = torch.tanh(offsets[:, 1:]) * self.vertical_radius
        motion_field = torch.cat([offsets_x, offsets_y], dim=1)
        return (
            self._warp(encoder_feature, motion_field),
            self._warp(decoder_feature, motion_field),
        )


class StitchTrajectoryAlignmentPyramid(nn.Module):
    """Multi-scale trajectory alignment for Stage-1 to Stage-2 feature transfer."""

    def __init__(self, current_channels, feature_channels, horizontal_radius=6,
                 vertical_radius=1, hidden_channels=32):
        super().__init__()
        self.units = nn.ModuleList([
            StitchTrajectoryAlignmentUnit(
                current_channels,
                channels,
                horizontal_radius=horizontal_radius,
                vertical_radius=vertical_radius,
                hidden_channels=hidden_channels,
            )
            for channels in feature_channels
        ])

    def forward(self, current, encoder_features, decoder_features):
        aligned_encoders = []
        aligned_decoders = []
        for unit, encoder_feature, decoder_feature in zip(
            self.units, encoder_features, decoder_features
        ):
            aligned_encoder, aligned_decoder = unit(
                current, encoder_feature, decoder_feature
            )
            aligned_encoders.append(aligned_encoder)
            aligned_decoders.append(aligned_decoder)
        return aligned_encoders, aligned_decoders


class Encoder(nn.Module):
    def __init__(
        self,
        n_feat,
        kernel_size,
        reduction,
        act,
        bias,
        scale_unetfeats,
        csff,
        gated_csff=False,
        gated_csff_horizontal_kernel=7,
    ):
        super(Encoder, self).__init__()

        self.gated_csff = bool(gated_csff) and bool(csff)

        self.encoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12  = DownSample(n_feat, scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

            if self.gated_csff:
                # Keep the outer RNG stream unchanged so adding a gate does not
                # alter the initialization of SAM, edge refinement, or the tail.
                # The gate still follows the experiment seed and changes across
                # independently seeded runs.
                gate_seed = torch.initial_seed()

                def build_gate(channels, level):
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(gate_seed + 1009 * level)
                        return TrajectoryCalibratedCSFFGate(
                            channels, gated_csff_horizontal_kernel
                        )

                self.csff_gate1 = build_gate(n_feat, 1)
                self.csff_gate2 = build_gate(n_feat + scale_unetfeats, 2)
                self.csff_gate3 = build_gate(n_feat + scale_unetfeats * 2, 3)

    def _fuse_csff(self, current, encoder_out, decoder_out, level):
        encoder_feature = getattr(self, f'csff_enc{level}')(encoder_out)
        decoder_feature = getattr(self, f'csff_dec{level}')(decoder_out)
        fusion = encoder_feature + decoder_feature
        if not self.gated_csff:
            return current + fusion
        gate = getattr(self, f'csff_gate{level}')(
            current, encoder_feature, decoder_feature
        )
        return current + gate * fusion

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = self._fuse_csff(enc1, encoder_outs[0], decoder_outs[0], 1)

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = self._fuse_csff(enc2, encoder_outs[1], decoder_outs[1], 2)

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = self._fuse_csff(enc3, encoder_outs[2], decoder_outs[2], 3)
        
        return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat,                 kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21  = SkipUpSample(n_feat, scale_unetfeats)
        self.up32  = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1,dec2,dec3]

##########################################################################
##---------- Resizing Modules ----------    
class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x


##########################################################################
## Lightweight restoration blocks
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = float(eps)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Compact NAF-style residual block for low-level restoration."""

    def __init__(self, channels, dw_expand=2, ffn_expand=2, bias=True):
        super(NAFBlock, self).__init__()
        dw_channels = channels * int(dw_expand)
        ffn_channels = channels * int(ffn_expand)
        if dw_channels % 2 != 0 or ffn_channels % 2 != 0:
            raise ValueError('NAFBlock expanded channels must be even.')

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            dw_channels,
            dw_channels,
            3,
            padding=1,
            groups=dw_channels,
            bias=bias,
        )
        self.sg1 = SimpleGate()
        gated_channels = dw_channels // 2
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gated_channels, gated_channels, 1, bias=bias),
        )
        self.conv2 = nn.Conv2d(gated_channels, channels, 1, bias=bias)

        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_channels, 1, bias=bias)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_channels // 2, channels, 1, bias=bias)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg1(y)
        y = y * self.sca(y)
        y = self.conv2(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv3(y)
        y = self.sg2(y)
        y = self.conv4(y)
        return x + y * self.gamma


class SimpleGatedRestorationBlock(nn.Module):
    """Attention-free local restoration block used only by SG-ORS.

    It intentionally has no directional kernels, dilated branches, spatial
    attention, edge prediction, or frequency transform.  Its two SimpleGate
    paths perform channel-space nonlinear restoration, keeping this MPRNet
    modification separate from the trajectory/topology modelling in C3Net.
    """

    def __init__(self, channels, residual_init=0.10, bias=True):
        super(SimpleGatedRestorationBlock, self).__init__()
        channels = int(channels)
        residual_init = float(residual_init)
        if residual_init < 0.0 or residual_init > 1.0:
            raise ValueError('SG-ORS residual_init must be in [0, 1].')

        expanded = channels * 2
        self.norm1 = LayerNorm2d(channels)
        self.expand1 = nn.Conv2d(channels, expanded, 1, bias=bias)
        self.depthwise = nn.Conv2d(
            expanded, expanded, 3, padding=1, groups=expanded, bias=bias
        )
        self.gate1 = SimpleGate()
        self.project1 = nn.Conv2d(channels, channels, 1, bias=bias)

        self.norm2 = LayerNorm2d(channels)
        self.expand2 = nn.Conv2d(channels, expanded, 1, bias=bias)
        self.gate2 = SimpleGate()
        self.project2 = nn.Conv2d(channels, channels, 1, bias=bias)

        # A modest active residual lets the new Stage-3 operator learn within
        # a 25-epoch screen without injecting an uncontrolled random feature.
        self.beta = nn.Parameter(torch.full((1, channels, 1, 1), residual_init))
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), residual_init))

    def forward(self, x):
        y = self.norm1(x)
        y = self.expand1(y)
        y = self.depthwise(y)
        y = self.gate1(y)
        x = x + self.project1(y) * self.beta

        y = self.norm2(x)
        y = self.expand2(y)
        y = self.gate2(y)
        return x + self.project2(y) * self.gamma


class SimpleGatedORB(nn.Module):
    """Original-resolution reconstruction cell composed of SG restoration blocks."""

    def __init__(self, channels, num_blocks=8, residual_init=0.10, bias=True):
        super(SimpleGatedORB, self).__init__()
        self.body = nn.Sequential(*[
            SimpleGatedRestorationBlock(channels, residual_init=residual_init, bias=bias)
            for _ in range(max(1, int(num_blocks)))
        ])

    def forward(self, x):
        return self.body(x)


class SimpleGatedORSNet(nn.Module):
    """Stage-3 replacement preserving MPRNet's cross-stage injection topology."""

    def __init__(self, base_orsnet, channels, num_blocks=8, residual_init=0.10):
        super(SimpleGatedORSNet, self).__init__()
        # Keep the original stage-to-stage projections and resize operators so
        # the replacement changes reconstruction blocks, not the MPR hierarchy.
        self.up_enc1 = base_orsnet.up_enc1
        self.up_dec1 = base_orsnet.up_dec1
        self.up_enc2 = base_orsnet.up_enc2
        self.up_dec2 = base_orsnet.up_dec2
        self.conv_enc1 = base_orsnet.conv_enc1
        self.conv_enc2 = base_orsnet.conv_enc2
        self.conv_enc3 = base_orsnet.conv_enc3
        self.conv_dec1 = base_orsnet.conv_dec1
        self.conv_dec2 = base_orsnet.conv_dec2
        self.conv_dec3 = base_orsnet.conv_dec3

        self.orb1 = SimpleGatedORB(channels, num_blocks, residual_init, bias=True)
        self.orb2 = SimpleGatedORB(channels, num_blocks, residual_init, bias=True)
        self.orb3 = SimpleGatedORB(channels, num_blocks, residual_init, bias=True)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        return x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))


class SimpleGatedResidualAdapter(nn.Module):
    """Parallel SG reconstruction residual around a pretrained ORB output."""

    def __init__(
        self,
        channels,
        num_blocks=4,
        residual_init=0.05,
        adapter_scale_init=0.02,
    ):
        super(SimpleGatedResidualAdapter, self).__init__()
        self.body = SimpleGatedORB(
            channels,
            num_blocks=num_blocks,
            residual_init=residual_init,
            bias=True,
        )
        self.scale = nn.Parameter(torch.tensor(float(adapter_scale_init)))

    def forward(self, x):
        # The adapter is a residual correction, not a second full feature path:
        # body(x) starts close to x, so this preserves the pretrained ORB while
        # allowing the SimpleGate branch to learn immediately in a 25-epoch run.
        return x + self.scale * (self.body(x) - x)


class HybridSimpleGatedORSNet(nn.Module):
    """Pretrained ORSNet plus parallel attention-free SG reconstruction paths."""

    def __init__(
        self,
        base_orsnet,
        channels,
        num_blocks=4,
        residual_init=0.10,
        adapter_scale_init=0.50,
    ):
        super(HybridSimpleGatedORSNet, self).__init__()
        # Preserve every original ORS parameter and its state-dict name so a
        # task-domain MPRNet checkpoint remains fully useful for B0.
        self.orb1 = base_orsnet.orb1
        self.orb2 = base_orsnet.orb2
        self.orb3 = base_orsnet.orb3
        self.up_enc1 = base_orsnet.up_enc1
        self.up_dec1 = base_orsnet.up_dec1
        self.up_enc2 = base_orsnet.up_enc2
        self.up_dec2 = base_orsnet.up_dec2
        self.conv_enc1 = base_orsnet.conv_enc1
        self.conv_enc2 = base_orsnet.conv_enc2
        self.conv_enc3 = base_orsnet.conv_enc3
        self.conv_dec1 = base_orsnet.conv_dec1
        self.conv_dec2 = base_orsnet.conv_dec2
        self.conv_dec3 = base_orsnet.conv_dec3

        self.adapter1 = SimpleGatedResidualAdapter(
            channels, num_blocks, residual_init, adapter_scale_init
        )
        self.adapter2 = SimpleGatedResidualAdapter(
            channels, num_blocks, residual_init, adapter_scale_init
        )
        self.adapter3 = SimpleGatedResidualAdapter(
            channels, num_blocks, residual_init, adapter_scale_init
        )

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.adapter1(self.orb1(x))
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.adapter2(self.orb2(x))
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.adapter3(self.orb3(x))
        return x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))


class TrajectoryGatedRestorationBlock(nn.Module):
    """Horizontal-trajectory gated residual block for original-resolution recovery.

    This is the HSGORS-v2 restoration unit.  It keeps the attention-free
    SimpleGate behavior that worked in HSGORS, but replaces the plain local
    depthwise mixer with a line-trace prior: local texture, long horizontal
    context, and a small vertical tolerance are mixed before gating.
    """

    def __init__(
        self,
        channels,
        residual_init=0.05,
        horizontal_kernel=15,
        vertical_kernel=5,
        bias=True,
    ):
        super(TrajectoryGatedRestorationBlock, self).__init__()
        channels = int(channels)
        horizontal_kernel = int(horizontal_kernel)
        vertical_kernel = int(vertical_kernel)
        if horizontal_kernel < 1 or horizontal_kernel % 2 == 0:
            raise ValueError('hsgors_v2_horizontal_kernel must be a positive odd integer.')
        if vertical_kernel < 1 or vertical_kernel % 2 == 0:
            raise ValueError('hsgors_v2_vertical_kernel must be a positive odd integer.')

        expanded = channels * 2
        self.norm1 = LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, expanded, 1, bias=bias)
        self.local = nn.Conv2d(
            expanded, expanded, 3, padding=1, groups=expanded, bias=bias
        )
        self.horizontal = nn.Conv2d(
            expanded,
            expanded,
            (1, horizontal_kernel),
            padding=(0, horizontal_kernel // 2),
            groups=expanded,
            bias=bias,
        )
        self.vertical = nn.Conv2d(
            expanded,
            expanded,
            (vertical_kernel, 1),
            padding=(vertical_kernel // 2, 0),
            groups=expanded,
            bias=bias,
        )
        self.mix = nn.Conv2d(expanded, expanded, 1, bias=bias)
        self.gate = SimpleGate()
        self.calibration = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=bias)
        self.beta = nn.Parameter(
            torch.full((1, channels, 1, 1), float(residual_init))
        )

        self.norm2 = LayerNorm2d(channels)
        self.ffn_expand = nn.Conv2d(channels, expanded, 1, bias=bias)
        self.ffn_gate = SimpleGate()
        self.ffn_project = nn.Conv2d(channels, channels, 1, bias=bias)
        self.gamma = nn.Parameter(
            torch.full((1, channels, 1, 1), float(residual_init))
        )
        nn.init.zeros_(self.project.weight)
        if self.project.bias is not None:
            nn.init.zeros_(self.project.bias)
        nn.init.zeros_(self.ffn_project.weight)
        if self.ffn_project.bias is not None:
            nn.init.zeros_(self.ffn_project.bias)

    def forward(self, x):
        y = self.expand(self.norm1(x))
        y = self.local(y) + self.horizontal(y) + self.vertical(y)
        y = self.gate(self.mix(y))
        # 2 * sigmoid(0) starts at identity strength for the channel calibration.
        y = y * (2.0 * self.calibration(y))
        x = x + self.project(y) * self.beta

        y = self.ffn_project(self.ffn_gate(self.ffn_expand(self.norm2(x))))
        return x + y * self.gamma


class TrajectoryGatedORB(nn.Module):
    """Stack of HSGORS-v2 trajectory-gated recovery units."""

    def __init__(
        self,
        channels,
        num_blocks=4,
        residual_init=0.05,
        horizontal_kernel=15,
        vertical_kernel=5,
        bias=True,
    ):
        super(TrajectoryGatedORB, self).__init__()
        self.body = nn.Sequential(*[
            TrajectoryGatedRestorationBlock(
                channels,
                residual_init=residual_init,
                horizontal_kernel=horizontal_kernel,
                vertical_kernel=vertical_kernel,
                bias=bias,
            )
            for _ in range(max(1, int(num_blocks)))
        ])

    def forward(self, x):
        return self.body(x)


class TrajectoryGatedResidualAdapter(nn.Module):
    """Trajectory-aware residual adapter around a pretrained ORB output."""

    def __init__(
        self,
        channels,
        num_blocks=4,
        residual_init=0.05,
        adapter_scale_init=0.02,
        horizontal_kernel=15,
        vertical_kernel=5,
    ):
        super(TrajectoryGatedResidualAdapter, self).__init__()
        self.body = TrajectoryGatedORB(
            channels,
            num_blocks=num_blocks,
            residual_init=residual_init,
            horizontal_kernel=horizontal_kernel,
            vertical_kernel=vertical_kernel,
            bias=True,
        )
        self.scale = nn.Parameter(torch.tensor(float(adapter_scale_init)))

    def forward(self, x):
        return x + self.scale * (self.body(x) - x)


class HybridTrajectoryGatedORSNet(nn.Module):
    """HSGORS-v2: pretrained ORS topology plus trajectory-gated adapters."""

    def __init__(
        self,
        base_orsnet,
        channels,
        num_blocks=4,
        residual_init=0.05,
        adapter_scale_init=0.02,
        horizontal_kernel=15,
        vertical_kernel=5,
    ):
        super(HybridTrajectoryGatedORSNet, self).__init__()
        self.orb1 = base_orsnet.orb1
        self.orb2 = base_orsnet.orb2
        self.orb3 = base_orsnet.orb3
        self.up_enc1 = base_orsnet.up_enc1
        self.up_dec1 = base_orsnet.up_dec1
        self.up_enc2 = base_orsnet.up_enc2
        self.up_dec2 = base_orsnet.up_dec2
        self.conv_enc1 = base_orsnet.conv_enc1
        self.conv_enc2 = base_orsnet.conv_enc2
        self.conv_enc3 = base_orsnet.conv_enc3
        self.conv_dec1 = base_orsnet.conv_dec1
        self.conv_dec2 = base_orsnet.conv_dec2
        self.conv_dec3 = base_orsnet.conv_dec3

        self.adapter1 = TrajectoryGatedResidualAdapter(
            channels,
            num_blocks=num_blocks,
            residual_init=residual_init,
            adapter_scale_init=adapter_scale_init,
            horizontal_kernel=horizontal_kernel,
            vertical_kernel=vertical_kernel,
        )
        self.adapter2 = TrajectoryGatedResidualAdapter(
            channels,
            num_blocks=num_blocks,
            residual_init=residual_init,
            adapter_scale_init=adapter_scale_init,
            horizontal_kernel=horizontal_kernel,
            vertical_kernel=vertical_kernel,
        )
        self.adapter3 = TrajectoryGatedResidualAdapter(
            channels,
            num_blocks=num_blocks,
            residual_init=residual_init,
            adapter_scale_init=adapter_scale_init,
            horizontal_kernel=horizontal_kernel,
            vertical_kernel=vertical_kernel,
        )

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.adapter1(self.orb1(x))
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.adapter2(self.orb2(x))
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.adapter3(self.orb3(x))
        return x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))


class TrajectoryEvidencePyramidRecovery(nn.Module):
    """Stage-3 recovery without ORB/CAB or sequential ORS injections.

    All Stage-2 encoder/decoder evidence is first projected to a shared
    full-resolution state. Three trajectory state-propagation blocks then
    refine that state. Zero-initialized evidence projections make the new path
    an identity feature map at initialization, which is useful while it is
    gradually substituted for a legacy recovery path during training.
    """

    def __init__(
        self,
        channels,
        evidence_channels,
        num_blocks=3,
        horizontal_kernel=15,
        vertical_kernel=5,
        bias=True,
    ):
        super(TrajectoryEvidencePyramidRecovery, self).__init__()
        channels = int(channels)
        evidence_channels = [int(value) for value in evidence_channels]
        if len(evidence_channels) != 3:
            raise ValueError('Trajectory evidence recovery requires three Stage-2 scales.')

        self.encoder_projects = nn.ModuleList([
            nn.Conv2d(value, channels, 1, bias=bias)
            for value in evidence_channels
        ])
        self.decoder_projects = nn.ModuleList([
            nn.Conv2d(value, channels, 1, bias=bias)
            for value in evidence_channels
        ])
        self.scale_evidence = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1, bias=bias),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, 3, padding=1, bias=bias),
            )
            for _ in evidence_channels
        ])
        self.evidence_fuse = nn.Conv2d(channels * 3, channels, 1, bias=bias)
        self.evidence_project = nn.Conv2d(channels, channels, 3, padding=1, bias=bias)
        self.evidence_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=bias),
            nn.Sigmoid(),
        )
        self.state_blocks = nn.ModuleList([
            TrajectoryGatedRestorationBlock(
                channels,
                residual_init=0.05,
                horizontal_kernel=horizontal_kernel,
                vertical_kernel=vertical_kernel,
                bias=bias,
            )
            for _ in range(max(1, int(num_blocks)))
        ])
        self.state_injections = nn.ModuleList([
            nn.Conv2d(channels * 2, channels, 1, bias=bias)
            for _ in self.state_blocks
        ])

        # The new recovery stream initially leaves its input unchanged. This
        # is intentional: during structural migration, only alpha controls how
        # much of this stream reaches the reconstruction loss.
        nn.init.zeros_(self.evidence_fuse.weight)
        nn.init.zeros_(self.evidence_project.weight)
        if self.evidence_fuse.bias is not None:
            nn.init.zeros_(self.evidence_fuse.bias)
        if self.evidence_project.bias is not None:
            nn.init.zeros_(self.evidence_project.bias)
        for injection in self.state_injections:
            nn.init.zeros_(injection.weight)
            if injection.bias is not None:
                nn.init.zeros_(injection.bias)

    def forward(self, x, encoder_outs, decoder_outs):
        if len(encoder_outs) != 3 or len(decoder_outs) != 3:
            raise ValueError('Trajectory evidence recovery expects three encoder and decoder states.')
        full_size = x.shape[-2:]
        evidence_levels = []
        for encoder_feature, decoder_feature, encoder_project, decoder_project, mixer in zip(
            encoder_outs,
            decoder_outs,
            self.encoder_projects,
            self.decoder_projects,
            self.scale_evidence,
        ):
            encoder_feature = F.interpolate(
                encoder_project(encoder_feature), size=full_size,
                mode='bilinear', align_corners=False,
            )
            decoder_feature = F.interpolate(
                decoder_project(decoder_feature), size=full_size,
                mode='bilinear', align_corners=False,
            )
            agreement = 0.5 * (encoder_feature + decoder_feature)
            disagreement = (encoder_feature - decoder_feature).abs()
            evidence_levels.append(mixer(torch.cat([agreement, disagreement], dim=1)))

        evidence = self.evidence_fuse(torch.cat(evidence_levels, dim=1))
        gate = 2.0 * self.evidence_gate(torch.cat([x, evidence], dim=1))
        evidence = self.evidence_project(evidence) * gate
        state = x + evidence
        for block, injection in zip(self.state_blocks, self.state_injections):
            state = block(state)
            state = state + injection(torch.cat([state, evidence], dim=1))
        return state


class TrajectoryEvidenceSynthesisHead(nn.Module):
    """New RGB synthesis head for trajectory-evidence recovery.

    ``base_projection`` can receive a compatible legacy RGB projection at
    initialization, while the explicit horizontal/vertical correction branch
    is learned from scratch. The deployed module is not the legacy Tail.
    """

    def __init__(self, channels, out_channels=3, horizontal_kernel=15,
                 vertical_kernel=5, bias=True):
        super(TrajectoryEvidenceSynthesisHead, self).__init__()
        channels = int(channels)
        expanded = channels * 2
        self.base_projection = nn.Conv2d(channels, out_channels, 3, padding=1, bias=bias)
        self.norm = LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, expanded, 1, bias=bias)
        self.horizontal = nn.Conv2d(
            expanded, expanded, (1, horizontal_kernel),
            padding=(0, horizontal_kernel // 2), groups=expanded, bias=bias,
        )
        self.vertical = nn.Conv2d(
            expanded, expanded, (vertical_kernel, 1),
            padding=(vertical_kernel // 2, 0), groups=expanded, bias=bias,
        )
        self.mix = nn.Conv2d(expanded, expanded, 1, bias=bias)
        self.gate = SimpleGate()
        self.residual_projection = nn.Conv2d(channels, out_channels, 3, padding=1, bias=bias)
        self.residual_scale = nn.Parameter(torch.tensor(0.05))
        nn.init.zeros_(self.residual_projection.weight)
        if self.residual_projection.bias is not None:
            nn.init.zeros_(self.residual_projection.bias)

    def forward(self, x):
        residual = self.expand(self.norm(x))
        residual = self.horizontal(residual) + self.vertical(residual)
        residual = self.gate(self.mix(residual))
        residual = self.residual_projection(residual)
        return self.base_projection(x) + residual * self.residual_scale


def make_naf_stack(channels, num_blocks, bias=True):
    return nn.Sequential(*[
        NAFBlock(channels, bias=bias) for _ in range(max(1, int(num_blocks)))
    ])


class DirectionalStripContext(nn.Module):
    """Low-cost directional context mixer placed at the Stage-2 bottleneck."""

    def __init__(self, channels, horizontal_kernel=15, vertical_kernel=5, bias=True):
        super(DirectionalStripContext, self).__init__()
        horizontal_kernel = int(horizontal_kernel)
        vertical_kernel = int(vertical_kernel)
        if horizontal_kernel % 2 == 0 or vertical_kernel % 2 == 0:
            raise ValueError('Directional kernels must be odd.')

        self.norm = LayerNorm2d(channels)
        self.local = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=bias
        )
        self.horizontal = nn.Conv2d(
            channels,
            channels,
            (1, horizontal_kernel),
            padding=(0, horizontal_kernel // 2),
            groups=channels,
            bias=bias,
        )
        self.vertical = nn.Conv2d(
            channels,
            channels,
            (vertical_kernel, 1),
            padding=(vertical_kernel // 2, 0),
            groups=channels,
            bias=bias,
        )
        hidden = max(8, channels // 8)
        self.branch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.PReLU(),
            nn.Conv2d(hidden, channels * 3, 1, bias=True),
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=bias)
        self.res_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.norm(x)
        branches = torch.stack(
            [self.local(y), self.horizontal(y), self.vertical(y)], dim=1
        )
        b, _, c, _, _ = branches.shape
        weights = self.branch_gate(y).view(b, 3, c, 1, 1).softmax(dim=1)
        y = (branches * weights).sum(dim=1)
        return x + self.project(y) * self.res_scale


class RowWiseStripeTransformer(nn.Module):
    """Apply self-attention along image rows at a low-resolution stripe bottleneck."""

    def __init__(self, channels, num_heads=4, ffn_expansion=2, bias=True):
        super(RowWiseStripeTransformer, self).__init__()
        num_heads = int(num_heads)
        if num_heads < 1 or channels % num_heads != 0:
            raise ValueError('Stripe Transformer heads must divide the channel count.')

        hidden = channels * int(ffn_expansion)
        self.norm1 = LayerNorm2d(channels)
        self.local_position = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=bias
        )
        self.attention = nn.MultiheadAttention(
            channels, num_heads, dropout=0.0, bias=bias, batch_first=True
        )
        self.norm2 = LayerNorm2d(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1, bias=bias),
        )
        # Zero-gated residuals preserve the task-domain MPRNet initialization.
        self.attn_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.ffn_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = y + self.local_position(y)
        batch, channels, height, width = y.shape
        row_tokens = y.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        y, _ = self.attention(row_tokens, row_tokens, row_tokens, need_weights=False)
        y = y.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
        x = x + y * self.attn_scale
        return x + self.ffn(self.norm2(x)) * self.ffn_scale


class NAFEncoder(nn.Module):
    def __init__(
        self,
        n_feat,
        scale_unetfeats,
        csff,
        num_blocks=2,
        directional_context=False,
        horizontal_kernel=15,
        vertical_kernel=5,
        bias=True,
    ):
        super(NAFEncoder, self).__init__()
        channels = [n_feat, n_feat + scale_unetfeats, n_feat + scale_unetfeats * 2]
        self.encoder_level1 = make_naf_stack(channels[0], num_blocks, bias=bias)
        self.encoder_level2 = make_naf_stack(channels[1], num_blocks, bias=bias)
        self.encoder_level3 = make_naf_stack(channels[2], num_blocks, bias=bias)
        self.down12 = DownSample(channels[0], scale_unetfeats)
        self.down23 = DownSample(channels[1], scale_unetfeats)
        self.directional_context = (
            DirectionalStripContext(
                channels[2],
                horizontal_kernel=horizontal_kernel,
                vertical_kernel=vertical_kernel,
                bias=bias,
            )
            if directional_context
            else nn.Identity()
        )

        self.csff = bool(csff)
        if self.csff:
            self.csff_enc = nn.ModuleList([
                nn.Conv2d(c, c, 1, bias=bias) for c in channels
            ])
            self.csff_dec = nn.ModuleList([
                nn.Conv2d(c, c, 1, bias=bias) for c in channels
            ])

    def _fuse_csff(self, x, level, encoder_outs, decoder_outs):
        if not self.csff or encoder_outs is None or decoder_outs is None:
            return x
        return (
            x
            + self.csff_enc[level](encoder_outs[level])
            + self.csff_dec[level](decoder_outs[level])
        )

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        enc1 = self._fuse_csff(enc1, 0, encoder_outs, decoder_outs)
        enc2 = self.encoder_level2(self.down12(enc1))
        enc2 = self._fuse_csff(enc2, 1, encoder_outs, decoder_outs)
        enc3 = self.encoder_level3(self.down23(enc2))
        enc3 = self._fuse_csff(enc3, 2, encoder_outs, decoder_outs)
        enc3 = self.directional_context(enc3)
        return [enc1, enc2, enc3]


class NAFDecoder(nn.Module):
    def __init__(self, n_feat, scale_unetfeats, num_blocks=2, bias=True):
        super(NAFDecoder, self).__init__()
        channels = [n_feat, n_feat + scale_unetfeats, n_feat + scale_unetfeats * 2]
        self.decoder_level3 = make_naf_stack(channels[2], num_blocks, bias=bias)
        self.decoder_level2 = make_naf_stack(channels[1], num_blocks, bias=bias)
        self.decoder_level1 = make_naf_stack(channels[0], num_blocks, bias=bias)
        self.skip_attn2 = NAFBlock(channels[1], bias=bias)
        self.skip_attn1 = NAFBlock(channels[0], bias=bias)
        self.up32 = SkipUpSample(channels[1], scale_unetfeats)
        self.up21 = SkipUpSample(channels[0], scale_unetfeats)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)
        dec2 = self.decoder_level2(self.up32(dec3, self.skip_attn2(enc2)))
        dec1 = self.decoder_level1(self.up21(dec2, self.skip_attn1(enc1)))
        return [dec1, dec2, dec3]


class TrajectoryMemoryTransferUnit(nn.Module):
    """Trajectory-aware replacement for plain CSFF at one Stage-2 scale.

    The inherited encoder/decoder features are converted into an agreement
    memory, an uncertainty cue, and a horizontal/vertical strip context.  The
    last projection is zero-initialized so a pretrained TPT/BandGate checkpoint
    starts from the same restoration function, while the new memory path learns
    with its own optimizer multiplier.
    """

    def __init__(self, channels, horizontal_kernel=15, vertical_kernel=5,
                 bias=True):
        super(TrajectoryMemoryTransferUnit, self).__init__()
        horizontal_kernel = int(horizontal_kernel)
        vertical_kernel = int(vertical_kernel)
        if horizontal_kernel < 1 or horizontal_kernel % 2 == 0:
            raise ValueError('stage2_tmt_horizontal_kernel must be a positive odd integer.')
        if vertical_kernel < 1 or vertical_kernel % 2 == 0:
            raise ValueError('stage2_tmt_vertical_kernel must be a positive odd integer.')

        self.encoder_project = nn.Conv2d(channels, channels, 1, bias=bias)
        self.decoder_project = nn.Conv2d(channels, channels, 1, bias=bias)
        self.horizontal_context = nn.Conv2d(
            channels,
            channels,
            (1, horizontal_kernel),
            padding=(0, horizontal_kernel // 2),
            groups=channels,
            bias=bias,
        )
        self.vertical_context = nn.Conv2d(
            channels,
            channels,
            (vertical_kernel, 1),
            padding=(vertical_kernel // 2, 0),
            groups=channels,
            bias=bias,
        )
        hidden = max(8, channels // 4)
        self.memory_gate = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, 1, bias=True),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.out_project = nn.Conv2d(channels, channels, 3, padding=1, bias=bias)
        nn.init.zeros_(self.out_project.weight)
        if self.out_project.bias is not None:
            nn.init.zeros_(self.out_project.bias)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.10))

    def forward(self, current, encoder_feature, decoder_feature):
        encoder_feature = self.encoder_project(encoder_feature)
        decoder_feature = self.decoder_project(decoder_feature)
        agreement = 0.5 * (encoder_feature + decoder_feature)
        disagreement = (encoder_feature - decoder_feature).abs()
        strip_context = (
            self.horizontal_context(agreement)
            + self.vertical_context(agreement)
        )
        gate = self.memory_gate(torch.cat([
            current,
            agreement,
            disagreement,
            strip_context,
        ], dim=1))
        return current + self.out_project(gate * strip_context) * self.scale


class TrajectoryMemoryTransferEncoder(Encoder):
    """Stage-2 encoder with trajectory-memory transfer instead of plain CSFF."""

    def __init__(
        self,
        n_feat,
        kernel_size,
        reduction,
        act,
        bias,
        scale_unetfeats,
        gated_csff=False,
        gated_csff_horizontal_kernel=7,
        horizontal_kernel=15,
        vertical_kernel=5,
    ):
        super(TrajectoryMemoryTransferEncoder, self).__init__(
            n_feat,
            kernel_size,
            reduction,
            act,
            bias,
            scale_unetfeats,
            csff=True,
            gated_csff=gated_csff,
            gated_csff_horizontal_kernel=gated_csff_horizontal_kernel,
        )
        channels = [
            n_feat,
            n_feat + scale_unetfeats,
            n_feat + scale_unetfeats * 2,
        ]
        self.tmt_units = nn.ModuleList([
            TrajectoryMemoryTransferUnit(
                channels[0],
                horizontal_kernel=horizontal_kernel,
                vertical_kernel=vertical_kernel,
                bias=True,
            ),
            TrajectoryMemoryTransferUnit(
                channels[1],
                horizontal_kernel=horizontal_kernel,
                vertical_kernel=vertical_kernel,
                bias=True,
            ),
            TrajectoryMemoryTransferUnit(
                channels[2],
                horizontal_kernel=horizontal_kernel,
                vertical_kernel=vertical_kernel,
                bias=True,
            ),
        ])

    def _fuse_csff(self, current, encoder_out, decoder_out, level):
        current = super(TrajectoryMemoryTransferEncoder, self)._fuse_csff(
            current, encoder_out, decoder_out, level
        )
        return self.tmt_units[level - 1](current, encoder_out, decoder_out)


class TrajectoryMemoryDecoderRefiner(nn.Module):
    """Strip-context correction for decoded Stage-2 feature states."""

    def __init__(self, channels, horizontal_kernel=15, vertical_kernel=5,
                 bias=True):
        super(TrajectoryMemoryDecoderRefiner, self).__init__()
        self.norm = LayerNorm2d(channels)
        self.horizontal = nn.Conv2d(
            channels,
            channels,
            (1, horizontal_kernel),
            padding=(0, horizontal_kernel // 2),
            groups=channels,
            bias=bias,
        )
        self.vertical = nn.Conv2d(
            channels,
            channels,
            (vertical_kernel, 1),
            padding=(vertical_kernel // 2, 0),
            groups=channels,
            bias=bias,
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=bias)
        nn.init.zeros_(self.project.weight)
        if self.project.bias is not None:
            nn.init.zeros_(self.project.bias)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.10))

    def forward(self, x):
        y = self.norm(x)
        y = self.horizontal(y) + self.vertical(y)
        return x + self.project(y) * self.scale


class TrajectoryMemoryTransferDecoder(Decoder):
    """Stage-2 decoder with a trajectory-memory correction at every scale."""

    def __init__(self, n_feat, kernel_size, reduction, act, bias,
                 scale_unetfeats, horizontal_kernel=15, vertical_kernel=5):
        super(TrajectoryMemoryTransferDecoder, self).__init__(
            n_feat, kernel_size, reduction, act, bias, scale_unetfeats
        )
        channels = [
            n_feat,
            n_feat + scale_unetfeats,
            n_feat + scale_unetfeats * 2,
        ]
        self.trajectory_refiners = nn.ModuleList([
            TrajectoryMemoryDecoderRefiner(
                channels[0],
                horizontal_kernel=horizontal_kernel,
                vertical_kernel=vertical_kernel,
                bias=True,
            ),
            TrajectoryMemoryDecoderRefiner(
                channels[1],
                horizontal_kernel=horizontal_kernel,
                vertical_kernel=vertical_kernel,
                bias=True,
            ),
            TrajectoryMemoryDecoderRefiner(
                channels[2],
                horizontal_kernel=horizontal_kernel,
                vertical_kernel=vertical_kernel,
                bias=True,
            ),
        ])

    def forward(self, outs):
        dec1, dec2, dec3 = super(TrajectoryMemoryTransferDecoder, self).forward(outs)
        return [
            self.trajectory_refiners[0](dec1),
            self.trajectory_refiners[1](dec2),
            self.trajectory_refiners[2](dec3),
        ]


class TrajectoryMemorySAM(SAM):
    """SAM23 replacement with an explicit trajectory residual image branch."""

    def __init__(self, n_feat, kernel_size, bias, horizontal_kernel=15,
                 vertical_kernel=5):
        super(TrajectoryMemorySAM, self).__init__(n_feat, kernel_size, bias)
        self.feature_refiner = TrajectoryMemoryDecoderRefiner(
            n_feat,
            horizontal_kernel=horizontal_kernel,
            vertical_kernel=vertical_kernel,
            bias=True,
        )
        hidden = max(16, n_feat // 2)
        self.image_residual = nn.Sequential(
            nn.Conv2d(n_feat + 6, hidden, 1, bias=True),
            nn.PReLU(hidden),
            nn.Conv2d(
                hidden,
                hidden,
                (1, horizontal_kernel),
                padding=(0, horizontal_kernel // 2),
                groups=hidden,
                bias=True,
            ),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, 3, 3, padding=1, bias=True),
        )
        nn.init.zeros_(self.image_residual[-1].weight)
        nn.init.zeros_(self.image_residual[-1].bias)
        self.image_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x, x_img):
        features, image = super(TrajectoryMemorySAM, self).forward(x, x_img)
        features = self.feature_refiner(features)
        delta = self.image_residual(torch.cat([x, image, x_img], dim=1))
        image = image + delta * self.image_scale
        return features, image

##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class DualDomainRestorationBlock(nn.Module):
    """Local-frequency restoration block with active residual initialization."""

    def __init__(
        self,
        channels,
        residual_init=0.3,
        bias=True,
    ):
        super(DualDomainRestorationBlock, self).__init__()
        self.channels = int(channels)

        self.norm1 = LayerNorm2d(self.channels)
        self.local_expand = nn.Conv2d(
            self.channels, self.channels * 2, kernel_size=1, bias=bias
        )
        self.local_depthwise = nn.Conv2d(
            self.channels * 2,
            self.channels * 2,
            kernel_size=3,
            padding=1,
            groups=self.channels * 2,
            bias=bias,
        )
        self.local_gate = SimpleGate()
        self.local_project = nn.Conv2d(
            self.channels, self.channels, kernel_size=1, bias=bias
        )

        # Real/imaginary values are interleaved by channel. Grouped 1x1
        # convolution therefore learns a complex-valued transform per feature
        # channel without an expensive dense projection at full resolution.
        self.frequency_filter = nn.Sequential(
            nn.Conv2d(
                self.channels * 2,
                self.channels * 2,
                kernel_size=(1, 3),
                padding=(0, 1),
                groups=self.channels * 2,
                bias=True,
            ),
            nn.PReLU(self.channels * 2),
            nn.Conv2d(
                self.channels * 2,
                self.channels * 2,
                kernel_size=1,
                groups=self.channels,
                bias=True,
            ),
        )
        self.frequency_project = nn.Conv2d(
            self.channels, self.channels, kernel_size=1, bias=bias
        )

        gate_hidden = max(8, self.channels // 8)
        self.domain_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels, gate_hidden, kernel_size=1, bias=True),
            nn.PReLU(gate_hidden),
            nn.Conv2d(gate_hidden, self.channels * 2, kernel_size=1, bias=True),
        )
        self.domain_project = nn.Conv2d(
            self.channels, self.channels, kernel_size=1, bias=bias
        )
        self.domain_scale = nn.Parameter(
            torch.full((1, self.channels, 1, 1), float(residual_init))
        )

        self.norm2 = LayerNorm2d(self.channels)
        self.ffn_expand = nn.Conv2d(
            self.channels, self.channels * 2, kernel_size=1, bias=bias
        )
        self.ffn_gate = SimpleGate()
        self.ffn_project = nn.Conv2d(
            self.channels, self.channels, kernel_size=1, bias=bias
        )
        self.ffn_scale = nn.Parameter(
            torch.full((1, self.channels, 1, 1), float(residual_init))
        )

    def _frequency_branch(self, x):
        channels = x.size(1)
        width = x.size(3)
        input_dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            # A row-wise spectrum gives every trajectory row full-width context
            # without the overhead or boundary artifacts of many small windows.
            spectrum = torch.fft.rfft(x.float(), dim=-1, norm='ortho')
            real_imag = torch.stack(
                (spectrum.real, spectrum.imag), dim=2
            ).reshape(
                spectrum.size(0),
                channels * 2,
                spectrum.size(2),
                spectrum.size(3),
            )
            real_imag = self.frequency_filter(real_imag)
            real_imag = real_imag.view(
                spectrum.size(0),
                channels,
                2,
                spectrum.size(2),
                spectrum.size(3),
            )
            filtered_spectrum = torch.complex(
                real_imag[:, :, 0], real_imag[:, :, 1]
            )
            restored = torch.fft.irfft(
                filtered_spectrum,
                n=width,
                dim=-1,
                norm='ortho',
            ).to(dtype=input_dtype)
        return restored

    def forward(self, x):
        normalized = self.norm1(x)
        local = self.local_project(
            self.local_gate(self.local_depthwise(self.local_expand(normalized)))
        )
        frequency = self.frequency_project(self._frequency_branch(normalized))
        domain_weights = self.domain_gate(local + frequency).view(
            x.size(0), 2, self.channels, 1, 1
        ).softmax(dim=1)
        fused = (
            local * domain_weights[:, 0]
            + frequency * domain_weights[:, 1]
        )
        x = x + self.domain_project(fused) * self.domain_scale

        ffn = self.ffn_project(self.ffn_gate(self.ffn_expand(self.norm2(x))))
        return x + ffn * self.ffn_scale


class LocalGatedRestorationBlock(nn.Module):
    """NAF-style local restoration with an active, bounded residual start."""

    def __init__(self, channels, residual_init=0.3, bias=True):
        super(LocalGatedRestorationBlock, self).__init__()
        channels = int(channels)
        self.norm1 = LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, channels * 2, 1, bias=bias)
        self.depthwise = nn.Conv2d(
            channels * 2,
            channels * 2,
            3,
            padding=1,
            groups=channels * 2,
            bias=bias,
        )
        self.gate = SimpleGate()
        self.channel_calibration = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=True),
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=bias)
        self.local_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), float(residual_init))
        )

        self.norm2 = LayerNorm2d(channels)
        self.ffn_expand = nn.Conv2d(channels, channels * 2, 1, bias=bias)
        self.ffn_gate = SimpleGate()
        self.ffn_project = nn.Conv2d(channels, channels, 1, bias=bias)
        self.ffn_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), float(residual_init))
        )

    def forward(self, x):
        local = self.gate(self.depthwise(self.expand(self.norm1(x))))
        local = local * self.channel_calibration(local)
        x = x + self.project(local) * self.local_scale
        ffn = self.ffn_project(self.ffn_gate(self.ffn_expand(self.norm2(x))))
        return x + ffn * self.ffn_scale


class CheckpointedRestorationBlock(nn.Module):
    """Recompute a heavy restoration block during backward to cap VRAM use."""

    def __init__(self, block):
        super(CheckpointedRestorationBlock, self).__init__()
        self.block = block

    def forward(self, x):
        if self.training and torch.is_grad_enabled():
            return checkpoint_function(self.block, x, use_reentrant=False)
        return self.block(x)


class HybridDualDomainORB(nn.Module):
    """Retain the first/last pretrained CABs and replace the middle six."""

    def __init__(
        self,
        base_orb,
        channels,
        residual_init=0.3,
    ):
        super(HybridDualDomainORB, self).__init__()
        base_modules = list(base_orb.body.children())
        if len(base_modules) < 6:
            raise ValueError('HybridDualDomainORB requires at least four CABs and a tail convolution.')
        last_cab_index = len(base_modules) - 2
        replacement_slots = [1, last_cab_index // 2, last_cab_index - 2]
        if len(set(replacement_slots)) != 3:
            raise ValueError('Unable to place three dual-domain blocks in this ORB.')

        modules = []
        for index in range(last_cab_index + 1):
            if index == 0 or index == last_cab_index:
                modules.append(base_modules[index])
            elif index in replacement_slots:
                block_type = (
                    DualDomainRestorationBlock
                    if index == replacement_slots[1]
                    else LocalGatedRestorationBlock
                )
                modules.append(CheckpointedRestorationBlock(block_type(
                    channels, residual_init=residual_init, bias=True
                )))
            else:
                modules.append(nn.Identity())
        modules.append(base_modules[-1])
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return x + self.body(x)


class SelectiveORSFusion(nn.Module):
    """Per-image channel selection between Stage-2 encoder/decoder features."""

    def __init__(self, channels):
        super(SelectiveORSFusion, self).__init__()
        self.channels = int(channels)
        self.selection = nn.Conv2d(
            self.channels * 4,
            self.channels * 2,
            kernel_size=1,
            bias=True,
        )
        # 2 * softmax([0, 0]) gives [1, 1], exactly reproducing addition.
        nn.init.zeros_(self.selection.weight)
        nn.init.zeros_(self.selection.bias)

    def forward(self, current, encoder_feature, decoder_feature):
        context = torch.cat([
            F.adaptive_avg_pool2d(current, 1),
            F.adaptive_avg_pool2d(encoder_feature, 1),
            F.adaptive_avg_pool2d(decoder_feature, 1),
            F.adaptive_avg_pool2d(
                torch.abs(encoder_feature - decoder_feature), 1
            ),
        ], dim=1)
        weights = 2.0 * self.selection(context).view(
            current.size(0), 2, self.channels, 1, 1
        ).softmax(dim=1)
        return (
            current
            + encoder_feature * weights[:, 0]
            + decoder_feature * weights[:, 1]
        )


class DualDomainSelectiveORSNet(nn.Module):
    """Original-resolution refinement with hybrid ORBs and selective fusion."""

    def __init__(
        self,
        base_orsnet,
        channels,
        residual_init=0.3,
    ):
        super(DualDomainSelectiveORSNet, self).__init__()
        # Construct all new modules in an isolated RNG stream. This keeps the
        # following edge/tail initialization paired with the base experiment.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed() + 4079)
            self.orb1 = HybridDualDomainORB(
                base_orsnet.orb1, channels, residual_init
            )
            self.orb2 = HybridDualDomainORB(
                base_orsnet.orb2, channels, residual_init
            )
            self.orb3 = HybridDualDomainORB(
                base_orsnet.orb3, channels, residual_init
            )
            self.fusion1 = SelectiveORSFusion(channels)
            self.fusion2 = SelectiveORSFusion(channels)
            self.fusion3 = SelectiveORSFusion(channels)

        self.up_enc1 = base_orsnet.up_enc1
        self.up_dec1 = base_orsnet.up_dec1
        self.up_enc2 = base_orsnet.up_enc2
        self.up_dec2 = base_orsnet.up_dec2
        self.conv_enc1 = base_orsnet.conv_enc1
        self.conv_enc2 = base_orsnet.conv_enc2
        self.conv_enc3 = base_orsnet.conv_enc3
        self.conv_dec1 = base_orsnet.conv_dec1
        self.conv_dec2 = base_orsnet.conv_dec2
        self.conv_dec3 = base_orsnet.conv_dec3

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = self.fusion1(
            x,
            self.conv_enc1(encoder_outs[0]),
            self.conv_dec1(decoder_outs[0]),
        )

        x = self.orb2(x)
        x = self.fusion2(
            x,
            self.conv_enc2(self.up_enc1(encoder_outs[1])),
            self.conv_dec2(self.up_dec1(decoder_outs[1])),
        )

        x = self.orb3(x)
        return self.fusion3(
            x,
            self.conv_enc3(self.up_enc2(encoder_outs[2])),
            self.conv_dec3(self.up_dec2(decoder_outs[2])),
        )


##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x


class Stage3TrajectoryResidualAdapter(nn.Module):
    """Trajectory residual correction for one original-resolution stage.

    The adapter sees the current restoration state plus the Stage-2 encoder and
    decoder injections.  It models horizontal stitch trajectories and a small
    vertical tolerance, then predicts a residual feature correction.  The last
    projection is zero-initialized, so a pretrained ORSNet is exactly preserved
    at initialization.
    """

    def __init__(self, channels, horizontal_kernel=15, vertical_kernel=5,
                 hidden_channels=32, bias=True):
        super(Stage3TrajectoryResidualAdapter, self).__init__()
        horizontal_kernel = int(horizontal_kernel)
        vertical_kernel = int(vertical_kernel)
        if horizontal_kernel < 1 or horizontal_kernel % 2 == 0:
            raise ValueError('stage3_trr_horizontal_kernel must be a positive odd integer.')
        if vertical_kernel < 1 or vertical_kernel % 2 == 0:
            raise ValueError('stage3_trr_vertical_kernel must be a positive odd integer.')
        hidden_channels = max(8, int(hidden_channels))

        self.reduce = nn.Sequential(
            nn.Conv2d(channels * 4, hidden_channels, 1, bias=True),
            nn.PReLU(hidden_channels),
        )
        self.horizontal = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            (1, horizontal_kernel),
            padding=(0, horizontal_kernel // 2),
            groups=hidden_channels,
            bias=bias,
        )
        self.vertical = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            (vertical_kernel, 1),
            padding=(vertical_kernel // 2, 0),
            groups=hidden_channels,
            bias=bias,
        )
        self.local = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            3,
            padding=1,
            groups=hidden_channels,
            bias=bias,
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, max(8, hidden_channels // 2), 1, bias=True),
            nn.PReLU(max(8, hidden_channels // 2)),
            nn.Conv2d(max(8, hidden_channels // 2), hidden_channels * 3, 1, bias=True),
        )
        self.project = nn.Conv2d(hidden_channels, channels, 1, bias=bias)
        nn.init.zeros_(self.project.weight)
        if self.project.bias is not None:
            nn.init.zeros_(self.project.bias)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.10))

    def forward(self, current, encoder_feature, decoder_feature):
        context = torch.cat([
            current,
            encoder_feature,
            decoder_feature,
            torch.abs(encoder_feature - decoder_feature),
        ], dim=1)
        reduced = self.reduce(context)
        branches = torch.stack([
            self.horizontal(reduced),
            self.vertical(reduced),
            self.local(reduced),
        ], dim=1)
        b, _, c, _, _ = branches.shape
        weights = self.gate(reduced).view(b, 3, c, 1, 1).softmax(dim=1)
        residual = (branches * weights).sum(dim=1)
        return current + self.project(residual) * self.scale


class TrajectoryResidualRestorer(nn.Module):
    """Pretrained-compatible Stage-3 trajectory residual restorer.

    Original ORSNet parameters are kept under their original names, while the
    added adapters provide trajectory-aware residual corrections after each
    Stage-2 feature injection.
    """

    def __init__(self, base_orsnet, channels, horizontal_kernel=15,
                 vertical_kernel=5, hidden_channels=32):
        super(TrajectoryResidualRestorer, self).__init__()
        self.orb1 = base_orsnet.orb1
        self.orb2 = base_orsnet.orb2
        self.orb3 = base_orsnet.orb3
        self.up_enc1 = base_orsnet.up_enc1
        self.up_dec1 = base_orsnet.up_dec1
        self.up_enc2 = base_orsnet.up_enc2
        self.up_dec2 = base_orsnet.up_dec2
        self.conv_enc1 = base_orsnet.conv_enc1
        self.conv_enc2 = base_orsnet.conv_enc2
        self.conv_enc3 = base_orsnet.conv_enc3
        self.conv_dec1 = base_orsnet.conv_dec1
        self.conv_dec2 = base_orsnet.conv_dec2
        self.conv_dec3 = base_orsnet.conv_dec3
        self.trr1 = Stage3TrajectoryResidualAdapter(
            channels,
            horizontal_kernel=horizontal_kernel,
            vertical_kernel=vertical_kernel,
            hidden_channels=hidden_channels,
            bias=True,
        )
        self.trr2 = Stage3TrajectoryResidualAdapter(
            channels,
            horizontal_kernel=horizontal_kernel,
            vertical_kernel=vertical_kernel,
            hidden_channels=hidden_channels,
            bias=True,
        )
        self.trr3 = Stage3TrajectoryResidualAdapter(
            channels,
            horizontal_kernel=horizontal_kernel,
            vertical_kernel=vertical_kernel,
            hidden_channels=hidden_channels,
            bias=True,
        )

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        enc = self.conv_enc1(encoder_outs[0])
        dec = self.conv_dec1(decoder_outs[0])
        x = self.trr1(x + enc + dec, enc, dec)

        x = self.orb2(x)
        enc = self.conv_enc2(self.up_enc1(encoder_outs[1]))
        dec = self.conv_dec2(self.up_dec1(decoder_outs[1]))
        x = self.trr2(x + enc + dec, enc, dec)

        x = self.orb3(x)
        enc = self.conv_enc3(self.up_enc2(encoder_outs[2]))
        dec = self.conv_dec3(self.up_dec2(decoder_outs[2]))
        return self.trr3(x + enc + dec, enc, dec)


class TrajectoryResidualSynthesisHead(nn.Module):
    """Image-space residual synthesis after the pretrained RGB tail."""

    def __init__(self, feature_channels, hidden_channels=32,
                 horizontal_kernel=15, bias=True):
        super(TrajectoryResidualSynthesisHead, self).__init__()
        hidden_channels = max(8, int(hidden_channels))
        self.feature_project = nn.Conv2d(
            feature_channels, hidden_channels, 1, bias=bias
        )
        self.image_project = nn.Conv2d(12, hidden_channels, 1, bias=True)
        self.horizontal = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            (1, int(horizontal_kernel)),
            padding=(0, int(horizontal_kernel) // 2),
            groups=hidden_channels,
            bias=bias,
        )
        self.local = nn.Conv2d(
            hidden_channels, hidden_channels, 3, padding=1,
            groups=hidden_channels, bias=bias
        )
        self.out = nn.Conv2d(hidden_channels, 3, 3, padding=1, bias=bias)
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, feature, restored, input_image, stage2_image, stage1_image):
        image_context = torch.cat([
            restored,
            input_image,
            stage2_image,
            stage1_image,
        ], dim=1)
        feature_context = self.feature_project(feature)
        image_context = self.image_project(image_context)
        residual = self.horizontal(feature_context + image_context)
        residual = residual + self.local(feature_context)
        return restored + self.out(residual) * self.scale


class MultiScaleFusionNAF(nn.Module):
    """Lightweight full-resolution refinement with Stage-2 feature injection."""

    def __init__(
        self,
        n_feat,
        scale_orsnetfeats,
        scale_unetfeats,
        num_blocks=3,
        bias=True,
    ):
        super(MultiScaleFusionNAF, self).__init__()
        channels = n_feat + scale_orsnetfeats
        self.refine1 = make_naf_stack(channels, num_blocks, bias=bias)
        self.refine2 = make_naf_stack(channels, num_blocks, bias=bias)
        self.refine3 = make_naf_stack(channels, num_blocks, bias=bias)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)
        self.up_enc2 = nn.Sequential(
            UpSample(n_feat + scale_unetfeats, scale_unetfeats),
            UpSample(n_feat, scale_unetfeats),
        )
        self.up_dec2 = nn.Sequential(
            UpSample(n_feat + scale_unetfeats, scale_unetfeats),
            UpSample(n_feat, scale_unetfeats),
        )

        self.conv_enc1 = nn.Conv2d(n_feat, channels, 1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, channels, 1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, channels, 1, bias=bias)
        self.conv_dec1 = nn.Conv2d(n_feat, channels, 1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, channels, 1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, channels, 1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])
        x = self.refine1(x)

        x = (
            x
            + self.conv_enc2(self.up_enc1(encoder_outs[1]))
            + self.conv_dec2(self.up_dec1(decoder_outs[1]))
        )
        x = self.refine2(x)

        x = (
            x
            + self.conv_enc3(self.up_enc2(encoder_outs[2]))
            + self.conv_dec3(self.up_dec2(decoder_outs[2]))
        )
        return self.refine3(x)


##########################################################################
## Light Adaptive Restoration Module
class LightAdaptiveRestorationModule(nn.Module):
    """Parameter-free luminance adapter for low-light restoration inputs.

    MPRNet receives images in [0, 1]. This module estimates each image's mean
    luminance and applies bounded gamma/gain correction toward a target
    brightness. It is deliberately parameter-free, so old checkpoints remain
    compatible and the behaviour is deterministic at inference time.
    """

    def __init__(
        self,
        target_luma=0.50,
        min_gain=0.75,
        max_gain=1.80,
        min_gamma=0.65,
        max_gamma=1.35,
        blend=0.75,
        dark_threshold=0.45,
        bright_threshold=0.70,
    ):
        super(LightAdaptiveRestorationModule, self).__init__()
        self.target_luma = float(target_luma)
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)
        self.min_gamma = float(min_gamma)
        self.max_gamma = float(max_gamma)
        self.blend = float(blend)
        self.dark_threshold = float(dark_threshold)
        self.bright_threshold = float(bright_threshold)
        self.eps = 1e-6

        luma_weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer('luma_weights', luma_weights, persistent=False)
        self.last_stats = {}

    def forward(self, x):
        image = torch.clamp(x, 0.0, 1.0)
        luma_map = (image * self.luma_weights).sum(dim=1, keepdim=True)
        luma = luma_map.mean(dim=(2, 3), keepdim=True)

        gain = torch.clamp(
            self.target_luma / (luma + self.eps),
            self.min_gain,
            self.max_gain,
        )

        gamma = torch.where(
            luma < self.dark_threshold,
            torch.clamp(luma / max(self.target_luma, self.eps), self.min_gamma, 1.0),
            torch.ones_like(luma),
        )
        gamma = torch.where(
            luma > self.bright_threshold,
            torch.clamp(luma / max(self.target_luma, self.eps), 1.0, self.max_gamma),
            gamma,
        )

        adjusted = torch.clamp(torch.pow(image + self.eps, gamma) * gain, 0.0, 1.0)
        adjusted = torch.lerp(image, adjusted, self.blend)

        with torch.no_grad():
            self.last_stats = {
                'mean_luma': float(luma.mean().detach().cpu()),
                'mean_gain': float(gain.mean().detach().cpu()),
                'mean_gamma': float(gamma.mean().detach().cpu()),
            }

        return adjusted


class StitchEdgeGuidedRefinement(nn.Module):
    def __init__(self, n_feat, mid_channels=24, kernel_size=3, bias=False):
        super(StitchEdgeGuidedRefinement, self).__init__()
        mid_channels = max(8, int(mid_channels))
        self.edge_head = nn.Sequential(
            conv(n_feat, mid_channels, kernel_size, bias=bias),
            nn.PReLU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            conv(n_feat + 1, n_feat, kernel_size, bias=bias),
            nn.PReLU(),
            conv(n_feat, n_feat, kernel_size, bias=bias),
        )
        self.edge_gate = nn.Sequential(
            nn.Conv2d(1, n_feat, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        edge = self.edge_head(x)
        residual = self.refine(torch.cat([x, edge], dim=1))
        x = x + residual * self.edge_gate(edge)
        return x, edge


class TopologyContinuityBlock(nn.Module):
    """Directional, restoration-only local continuity refinement.

    It deliberately predicts no edge or semantic mask.  Parallel horizontal,
    vertical and isotropic paths let STRD repair thin trajectories regardless
    of their local orientation, while retaining the original texture path.
    """

    def __init__(self, channels, bias=True):
        super(TopologyContinuityBlock, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=bias),
            nn.PReLU(channels),
        )
        self.horizontal = nn.Conv2d(
            channels, channels, (1, 9), padding=(0, 4),
            groups=channels, bias=bias,
        )
        self.vertical = nn.Conv2d(
            channels, channels, (9, 1), padding=(4, 0),
            groups=channels, bias=bias,
        )
        self.local = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=bias,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=bias),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 1, bias=bias),
        )

    def forward(self, x):
        evidence = self.pre(x)
        fused = self.fuse(torch.cat([
            self.horizontal(evidence),
            self.vertical(evidence),
            self.local(evidence),
        ], dim=1))
        return x + fused


class StripeTopologyReconstructionDecoder(nn.Module):
    """STRD: multi-scale line-topology reconstruction before the RGB tail.

    Stage-2 decoder features contain complementary coarse-to-fine restoration
    evidence.  STRD decodes them top-down and injects a residual *feature*
    correction into the Stage-3 image-restoration stream.  It is intentionally
    separate from C3Net: no segmentation feature, edge label or mask enters
    this module.
    """

    def __init__(
        self,
        stage3_channels,
        stage2_channels,
        hidden_channels=32,
        num_blocks=2,
        bias=True,
    ):
        super(StripeTopologyReconstructionDecoder, self).__init__()
        hidden_channels = max(16, int(hidden_channels))
        num_blocks = max(1, int(num_blocks))
        self.hidden_channels = hidden_channels

        def project(in_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 1, bias=bias),
                nn.PReLU(hidden_channels),
            )

        def topology_stack():
            return nn.Sequential(*[
                TopologyContinuityBlock(hidden_channels, bias=bias)
                for _ in range(num_blocks)
            ])

        self.stage3_project = project(stage3_channels)
        self.stage2_full_project = project(stage2_channels[0])
        self.stage2_half_project = project(stage2_channels[1])
        self.stage2_quarter_project = project(stage2_channels[2])

        self.quarter_refine = topology_stack()
        self.half_fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 1, bias=bias),
            nn.PReLU(hidden_channels),
        )
        self.half_refine = topology_stack()
        self.full_fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, 1, bias=bias),
            nn.PReLU(hidden_channels),
        )
        self.full_refine = topology_stack()
        self.feature_delta = nn.Conv2d(
            hidden_channels, stage3_channels, 3, padding=1, bias=bias,
        )
        # Keep the imported baseline near its original output at step zero,
        # while preserving gradients for every STRD branch from batch one.
        self.residual_scale = nn.Parameter(torch.tensor(0.02))
        self.register_buffer('last_delta_abs', torch.zeros(()), persistent=False)

    @staticmethod
    def _resize(x, reference):
        return F.interpolate(
            x, size=reference.shape[-2:], mode='bilinear', align_corners=False
        )

    def forward(self, stage3_feature, stage2_decoder_features):
        if len(stage2_decoder_features) != 3:
            raise ValueError('STRD requires three Stage-2 decoder feature levels.')

        full = self.stage2_full_project(stage2_decoder_features[0])
        half = self.stage2_half_project(stage2_decoder_features[1])
        quarter = self.stage2_quarter_project(stage2_decoder_features[2])

        quarter = self.quarter_refine(quarter)
        half = self.half_fuse(torch.cat([
            half, self._resize(quarter, half)
        ], dim=1))
        half = self.half_refine(half)

        stage3 = self.stage3_project(stage3_feature)
        full = self.full_fuse(torch.cat([
            full,
            self._resize(half, full),
            self._resize(stage3, full),
        ], dim=1))
        full = self.full_refine(full)
        delta = self.feature_delta(self._resize(full, stage3_feature))
        delta = delta * torch.tanh(self.residual_scale)
        with torch.no_grad():
            self.last_delta_abs.copy_(delta.detach().abs().mean())
        return stage3_feature + delta


class DirectionalPyramidBlock(nn.Module):
    """Local topology block used by the fully rewritten TPT-RNet core.

    This is not an attention gate: horizontal, vertical and isotropic evidence
    is explicitly concatenated, projected and added back.  It suits unknown
    thin-trajectory orientations without introducing a segmentation edge head.
    """

    def __init__(self, channels, bias=True):
        super(DirectionalPyramidBlock, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=bias),
            nn.PReLU(channels),
        )
        self.horizontal = nn.Conv2d(
            channels, channels, (1, 7), padding=(0, 3),
            groups=channels, bias=bias,
        )
        self.vertical = nn.Conv2d(
            channels, channels, (7, 1), padding=(3, 0),
            groups=channels, bias=bias,
        )
        self.local = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=bias,
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=bias),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 1, bias=bias),
        )

    def forward(self, x):
        evidence = self.pre(x)
        return x + self.mix(torch.cat([
            self.horizontal(evidence),
            self.vertical(evidence),
            self.local(evidence),
        ], dim=1))


class HorizontalTrajectoryBlock(nn.Module):
    """Horizontal-trajectory-prior block for the stitch-line datasets.

    Two receptive fields follow the known horizontal line direction.  The
    remaining 3x1 path is deliberately local: it models line thickness and
    small slant without spending capacity on long vertical trajectories.
    """

    def __init__(self, channels, bias=True):
        super(HorizontalTrajectoryBlock, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=bias),
            nn.PReLU(channels),
        )
        self.horizontal_short = nn.Conv2d(
            channels, channels, (1, 7), padding=(0, 3),
            groups=channels, bias=bias,
        )
        self.horizontal_long = nn.Conv2d(
            channels, channels, (1, 15), padding=(0, 7),
            groups=channels, bias=bias,
        )
        self.vertical_local = nn.Conv2d(
            channels, channels, (3, 1), padding=(1, 0),
            groups=channels, bias=bias,
        )
        self.local = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=bias,
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=bias),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 1, bias=bias),
        )
        # The HMT path stacks many of these residual blocks from scratch.
        # Start each correction close to zero so the pyramid initially keeps
        # the encoder signal instead of repeatedly perturbing it.
        nn.init.normal_(self.mix[-1].weight, mean=0.0, std=1.0e-3)
        if self.mix[-1].bias is not None:
            nn.init.zeros_(self.mix[-1].bias)

    def forward(self, x):
        evidence = self.pre(x)
        return x + self.mix(torch.cat([
            self.horizontal_short(evidence),
            self.horizontal_long(evidence),
            self.vertical_local(evidence),
            self.local(evidence),
        ], dim=1))


class HorizontalStateMixer(nn.Module):
    """Broadcast a long horizontal state sequence back to every feature row."""

    def __init__(self, channels, bias=True):
        super(HorizontalStateMixer, self).__init__()
        self.short = nn.Conv1d(
            channels, channels, 15, padding=7, groups=channels, bias=bias
        )
        self.long = nn.Conv1d(
            channels, channels, 15, padding=14, dilation=2,
            groups=channels, bias=bias,
        )
        self.sequence_fuse = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 1, bias=bias),
            nn.PReLU(channels),
        )
        self.feature_fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=bias),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 1, bias=bias),
        )
        nn.init.normal_(self.feature_fuse[-1].weight, mean=0.0, std=1.0e-3)
        if self.feature_fuse[-1].bias is not None:
            nn.init.zeros_(self.feature_fuse[-1].bias)

    def forward(self, x):
        sequence = x.mean(dim=2)
        sequence = self.sequence_fuse(torch.cat([
            self.short(sequence), self.long(sequence)
        ], dim=1)).unsqueeze(2)
        sequence = sequence.expand(-1, -1, x.size(2), -1)
        return x + self.feature_fuse(torch.cat([x, sequence], dim=1))


class RowWiseTrajectoryAttention(nn.Module):
    """Global horizontal matching without collapsing row positions.

    Woven backgrounds contain repeated horizontal texture, while a damaged
    stitch is a long, locally curved trajectory.  Attention is evaluated
    independently for every row, so it follows long horizontal evidence but
    never averages one trajectory row into another.
    """

    def __init__(self, channels, heads=4, bias=True):
        super(RowWiseTrajectoryAttention, self).__init__()
        heads = max(1, min(int(heads), channels))
        while channels % heads:
            heads -= 1
        self.heads = heads
        self.head_dim = channels // heads
        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=bias)
        self.project = nn.Conv2d(channels, channels, 1, bias=bias)
        # Keep the complete global-attention path as an initially neutral
        # residual while retaining gradients for its output projection.
        nn.init.normal_(self.project.weight, mean=0.0, std=1.0e-3)
        if self.project.bias is not None:
            nn.init.zeros_(self.project.bias)

    def forward(self, x):
        batch, channels, height, width = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)

        def to_rows(tensor):
            return tensor.permute(0, 2, 3, 1).reshape(
                batch * height, width, self.heads, self.head_dim
            ).permute(0, 2, 1, 3)

        attended = F.scaled_dot_product_attention(
            to_rows(q), to_rows(k), to_rows(v), dropout_p=0.0
        )
        attended = attended.permute(0, 2, 1, 3).reshape(
            batch, height, width, channels
        ).permute(0, 3, 1, 2).contiguous()
        return x + self.project(attended)


class SlantedStripAttention(nn.Module):
    """Attend over a bank of sheared strips for small-angle stitch traces.

    Measured labels in DB/XJ/SD deviate from horizontal by up to about nine
    degrees.  Instead of restricting a query to its own row, the key/value
    bank contains five sheared views spanning that observed slope interval.
    A slanted trace can therefore become horizontal in one view while woven
    texture remains available as context in every view.
    """

    def __init__(self, channels, heads=4,
                 slopes=(-0.16, -0.08, 0.0, 0.08, 0.16), bias=True):
        super(SlantedStripAttention, self).__init__()
        heads = max(1, min(int(heads), channels))
        while channels % heads:
            heads -= 1
        self.heads = heads
        self.head_dim = channels // heads
        self.slopes = tuple(float(slope) for slope in slopes)
        self.norm = nn.GroupNorm(1, channels)
        self.query = nn.Conv2d(channels, channels, 1, bias=bias)
        self.key_value = nn.Conv2d(channels, channels * 2, 1, bias=bias)
        self.project = nn.Conv2d(channels, channels, 1, bias=bias)
        nn.init.normal_(self.project.weight, mean=0.0, std=1.0e-3)
        if self.project.bias is not None:
            nn.init.zeros_(self.project.bias)

    def _sheared_views(self, x):
        batch, _, height, width = x.shape
        dtype, device = x.dtype, x.device
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x_coord = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        base_y, base_x = torch.meshgrid(y, x_coord, indexing='ij')
        pixel_x = torch.arange(width, device=device, dtype=dtype) - (width - 1) / 2.0
        # Convert a pixel-space slope to normalized grid displacement.
        normalized_x = pixel_x * (2.0 / max(1.0, float(height - 1)))
        views = []
        for slope in self.slopes:
            grid = torch.stack([
                base_x,
                base_y + slope * normalized_x.view(1, width),
            ], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
            views.append(F.grid_sample(
                x, grid, mode='bilinear', padding_mode='border',
                align_corners=True,
            ))
        return torch.cat(views, dim=3)

    def forward(self, x):
        batch, channels, height, width = x.shape
        normalized = self.norm(x)
        query = self.query(normalized)
        key, value = self.key_value(self._sheared_views(normalized)).chunk(2, dim=1)
        key_width = key.size(3)

        def to_tokens(tensor, token_width):
            return tensor.permute(0, 2, 3, 1).reshape(
                batch * height, token_width, self.heads, self.head_dim
            ).permute(0, 2, 1, 3)

        attended = F.scaled_dot_product_attention(
            to_tokens(query, width), to_tokens(key, key_width),
            to_tokens(value, key_width), dropout_p=0.0,
        )
        attended = attended.permute(0, 2, 1, 3).reshape(
            batch, height, width, channels
        ).permute(0, 3, 1, 2).contiguous()
        return x + self.project(attended)


class CoordinateTrajectoryMixer(nn.Module):
    """Position-preserving horizontal/vertical trajectory context mixer."""

    def __init__(self, channels, bias=True):
        super(CoordinateTrajectoryMixer, self).__init__()
        self.coordinate = nn.Sequential(
            nn.Conv2d(3, channels, 1, bias=bias),
            nn.PReLU(channels),
        )
        self.horizontal_short = nn.Conv2d(
            channels, channels, (1, 9), padding=(0, 4),
            groups=channels, bias=bias,
        )
        self.horizontal_long = nn.Conv2d(
            channels, channels, (1, 15), padding=(0, 14), dilation=(1, 2),
            groups=channels, bias=bias,
        )
        self.vertical_local = nn.Conv2d(
            channels, channels, (5, 1), padding=(2, 0),
            groups=channels, bias=bias,
        )
        self.vertical_long = nn.Conv2d(
            channels, channels, (9, 1), padding=(8, 0), dilation=(2, 1),
            groups=channels, bias=bias,
        )
        self.local = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=bias,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 7, channels, 1, bias=bias),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 1, bias=bias),
        )
        nn.init.normal_(self.fuse[-1].weight, mean=0.0, std=1.0e-3)
        if self.fuse[-1].bias is not None:
            nn.init.zeros_(self.fuse[-1].bias)

    @staticmethod
    def _coordinates(x, start, full_height):
        height, width = x.shape[-2:]
        device, dtype = x.device, x.dtype
        global_y = (
            torch.arange(height, device=device, dtype=dtype) + float(start)
        ) / max(1.0, float(full_height - 1))
        local_y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        horizontal_x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        return torch.cat([
            global_y.view(1, 1, height, 1).expand(x.size(0), 1, height, width),
            local_y.view(1, 1, height, 1).expand(x.size(0), 1, height, width),
            horizontal_x.view(1, 1, 1, width).expand(x.size(0), 1, height, width),
        ], dim=1)

    def forward(self, x, start=0, full_height=None):
        if full_height is None:
            full_height = x.size(2)
        position = self.coordinate(self._coordinates(x, start, full_height))
        evidence = x + position
        return x + self.fuse(torch.cat([
            x,
            position,
            self.horizontal_short(evidence),
            self.horizontal_long(evidence),
            self.vertical_local(evidence),
            self.vertical_long(evidence),
            self.local(evidence),
        ], dim=1))


class GlobalPairContextFusion(nn.Module):
    """Fuse two stripe-pair pyramids on their shared global coordinate canvas.

    Unlike overlap-only memory, both pair branches first project onto the full
    image-height canvas.  Coordinate-aware mixing then transports trajectory
    evidence through the canvas before each branch receives its matching crop.
    """

    def __init__(self, channels, use_row_attention=False, bias=True):
        super(GlobalPairContextFusion, self).__init__()
        self.disagreement = nn.Conv2d(channels, channels, 1, bias=bias)
        self.canvas_mixer = CoordinateTrajectoryMixer(channels, bias=bias)
        self.row_attention = (
            SlantedStripAttention(channels, bias=bias)
            if use_row_attention else None
        )
        self.local_inject = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=bias),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 1, bias=bias),
        )
        nn.init.normal_(self.disagreement.weight, mean=0.0, std=1.0e-3)
        nn.init.normal_(self.local_inject[-1].weight, mean=0.0, std=1.0e-3)
        if self.disagreement.bias is not None:
            nn.init.zeros_(self.disagreement.bias)
        if self.local_inject[-1].bias is not None:
            nn.init.zeros_(self.local_inject[-1].bias)

    def forward(self, left, right, left_start, right_start, full_height, blend_fn):
        canvas = blend_fn([left, right], [left_start, right_start], full_height)
        disagreement = canvas.new_zeros(canvas.shape)
        overlap_start = max(int(left_start), int(right_start))
        overlap_end = min(
            int(left_start) + left.size(2),
            int(right_start) + right.size(2),
        )
        if overlap_end > overlap_start:
            left_top = overlap_start - int(left_start)
            right_top = overlap_start - int(right_start)
            overlap = torch.abs(left[:, :, left_top:left_top + overlap_end - overlap_start]
                                - right[:, :, right_top:right_top + overlap_end - overlap_start])
            disagreement[:, :, overlap_start:overlap_end] = overlap
        canvas = canvas + self.disagreement(disagreement)
        canvas = self.canvas_mixer(canvas, 0, full_height)
        if self.row_attention is not None:
            canvas = self.row_attention(canvas)

        def inject(branch, start):
            context = canvas[:, :, start:start + branch.size(2), :]
            return branch + self.local_inject(torch.cat([branch, context], dim=1))

        return inject(left, int(left_start)), inject(right, int(right_start))


class SharedMiddleTrajectoryMemory(nn.Module):
    """Exchange only the physical overlap observed by both adjacent pairs."""

    def __init__(self, channels, bias=True):
        super(SharedMiddleTrajectoryMemory, self).__init__()
        self.memory_fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=bias),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=bias),
        )
        self.horizontal_state = HorizontalStateMixer(channels, bias=bias)
        self.memory_residual_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, left, right, left_start, right_start):
        overlap_start = max(int(left_start), int(right_start))
        overlap_end = min(
            int(left_start) + left.size(2),
            int(right_start) + right.size(2),
        )
        if overlap_end <= overlap_start:
            return left, right

        left_top = overlap_start - int(left_start)
        right_top = overlap_start - int(right_start)
        length = overlap_end - overlap_start
        left_overlap = left[:, :, left_top:left_top + length, :]
        right_overlap = right[:, :, right_top:right_top + length, :]
        memory = self.horizontal_state(self.memory_fuse(torch.cat([
            0.5 * (left_overlap + right_overlap),
            torch.abs(left_overlap - right_overlap),
            left_overlap * right_overlap,
        ], dim=1)))

        left_delta = left.new_zeros(left.shape)
        right_delta = right.new_zeros(right.shape)
        left_delta[:, :, left_top:left_top + length, :] = memory
        right_delta[:, :, right_top:right_top + length, :] = memory
        scale = torch.tanh(self.memory_residual_scale)
        return left + scale * left_delta, right + scale * right_delta


class ResidualStateBridge(nn.Module):
    """Replace a SAM with an image-residual to feature-state bridge.

    Intermediate images remain available for progressive supervision, but no
    spatial attention map is generated.  The next stage receives a state built
    from restoration features and the visible image residual.
    """

    def __init__(self, feature_channels, state_channels, bias=True):
        super(ResidualStateBridge, self).__init__()
        self.image_delta = nn.Conv2d(feature_channels, 3, 3, padding=1, bias=bias)
        self.residual_encode = nn.Sequential(
            nn.Conv2d(3, state_channels, 3, padding=1, bias=bias),
            nn.PReLU(state_channels),
        )
        self.state = nn.Sequential(
            nn.Conv2d(feature_channels + state_channels, state_channels, 1, bias=bias),
            nn.PReLU(state_channels),
            nn.Conv2d(state_channels, state_channels, 3, padding=1, bias=bias),
        )
        # At initialization the progressive image is the input image; this is
        # stable for partial fine-tuning while all bridge paths receive signal.
        nn.init.normal_(self.image_delta.weight, mean=0.0, std=1.0e-4)
        if self.image_delta.bias is not None:
            nn.init.zeros_(self.image_delta.bias)

    def forward(self, feature, image):
        restored = image + self.image_delta(feature)
        residual_state = self.residual_encode(restored - image)
        state = self.state(torch.cat([feature, residual_state], dim=1))
        return state, restored


class FeatureMemoryBridge(nn.Module):
    """Convert a feature map into state memory without reconstructing an image.

    HMT-Net uses its early stripes and pair branches only to analyse the
    trajectory.  Returning the untouched image keeps the public multi-output
    interface compatible while ensuring that no early path becomes an
    auxiliary restoration branch.
    """

    def __init__(self, feature_channels, state_channels, bias=True):
        super(FeatureMemoryBridge, self).__init__()
        self.image_context = nn.Sequential(
            nn.Conv2d(3, state_channels, 3, padding=1, bias=bias),
            nn.PReLU(state_channels),
        )
        self.state = nn.Sequential(
            nn.Conv2d(
                feature_channels + state_channels,
                state_channels,
                1,
                bias=bias,
            ),
            nn.PReLU(state_channels),
            nn.Conv2d(state_channels, state_channels, 3, padding=1, bias=bias),
        )

    def forward(self, feature, image):
        context = self.image_context(image)
        state = self.state(torch.cat([feature, context], dim=1))
        return state, image


class DualViewTrajectoryMemory(nn.Module):
    """Preserve agreement and disagreement from both PairAligned views.

    A normal overlap blend immediately averages the two branch observations,
    which erases the uncertainty caused by a blurred stitch.  This module
    creates an aligned full-canvas memory, retains a separate disagreement
    tensor, and records branch visibility for regions observed by one view.
    """

    def __init__(self, channels, bias=True):
        super(DualViewTrajectoryMemory, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3 + 2, channels, 1, bias=bias),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=bias),
        )
        self.disagreement = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 1, bias=bias),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=bias),
        )
        nn.init.normal_(self.fuse[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.normal_(self.disagreement[-1].weight, mean=0.0, std=1.0e-3)
        for module in (self.fuse[-1], self.disagreement[-1]):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _place(feature, start, full_height):
        batch, channels, _, width = feature.shape
        canvas = feature.new_zeros(batch, channels, full_height, width)
        mask = feature.new_zeros(batch, 1, full_height, width)
        start = max(0, min(int(start), full_height))
        length = min(feature.size(2), full_height - start)
        if length > 0:
            canvas[:, :, start:start + length, :] = feature[:, :, :length, :]
            mask[:, :, start:start + length, :] = 1.0
        return canvas, mask

    def forward(self, left, right, left_start, right_start, full_height):
        left_canvas, left_mask = self._place(left, left_start, full_height)
        right_canvas, right_mask = self._place(right, right_start, full_height)
        count = (left_mask + right_mask).clamp_min(1.0)
        mean = (left_canvas + right_canvas) / count
        overlap = left_mask * right_mask
        signed_difference = left_canvas - right_canvas
        disagreement = signed_difference.abs() * overlap
        memory = mean + self.fuse(torch.cat([
            mean,
            signed_difference,
            disagreement,
            left_mask,
            right_mask,
        ], dim=1))
        uncertainty = self.disagreement(torch.cat([
            disagreement,
            overlap,
        ], dim=1))
        return memory, uncertainty


class PairwiseStripePyramidEncoder(nn.Module):
    """Four-scale encoder for one [stripe_i + stripe_(i+1)] branch."""

    def __init__(self, input_channels, channels, blocks=2,
                 horizontal_trajectory=False, bias=True):
        super(PairwiseStripePyramidEncoder, self).__init__()
        self.channels = list(channels)
        block_type = (
            HorizontalTrajectoryBlock if horizontal_trajectory
            else DirectionalPyramidBlock
        )
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], 3, padding=1, bias=bias),
            nn.PReLU(channels[0]),
        )
        self.level1 = nn.Sequential(*[
            block_type(channels[0], bias=bias) for _ in range(blocks)
        ])
        self.down12 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=bias)
        self.level2 = nn.Sequential(*[
            block_type(channels[1], bias=bias) for _ in range(blocks)
        ])
        self.down23 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=bias)
        self.level3 = nn.Sequential(*[
            block_type(channels[2], bias=bias) for _ in range(blocks)
        ])
        self.down34 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=bias)
        self.level4 = nn.Sequential(*[
            block_type(channels[3], bias=bias) for _ in range(blocks)
        ])

    def forward(self, x):
        level1 = self.level1(self.stem(x))
        level2 = self.level2(self.down12(level1))
        level3 = self.level3(self.down23(level2))
        level4 = self.level4(self.down34(level3))
        return [level1, level2, level3, level4]


class PairwiseStripeThreeScaleEncoder(nn.Module):
    """Three-scale pair encoder used by TFR-Net recovery."""

    def __init__(self, input_channels, channels, blocks=2,
                 horizontal_trajectory=True, bias=True):
        super(PairwiseStripeThreeScaleEncoder, self).__init__()
        if len(channels) != 3:
            raise ValueError('PairwiseStripeThreeScaleEncoder expects three scales.')
        block_type = (
            HorizontalTrajectoryBlock if horizontal_trajectory
            else DirectionalPyramidBlock
        )
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], 3, padding=1, bias=bias),
            nn.PReLU(channels[0]),
        )
        self.level1 = nn.Sequential(*[
            block_type(channels[0], bias=bias) for _ in range(blocks)
        ])
        self.down12 = nn.Conv2d(
            channels[0], channels[1], 3, stride=2, padding=1, bias=bias
        )
        self.level2 = nn.Sequential(*[
            block_type(channels[1], bias=bias) for _ in range(blocks)
        ])
        self.down23 = nn.Conv2d(
            channels[1], channels[2], 3, stride=2, padding=1, bias=bias
        )
        self.level3 = nn.Sequential(*[
            block_type(channels[2], bias=bias) for _ in range(blocks)
        ])

    def forward(self, x):
        level1 = self.level1(self.stem(x))
        level2 = self.level2(self.down12(level1))
        level3 = self.level3(self.down23(level2))
        return [level1, level2, level3]


class PairwisePyramidDecoder(nn.Module):
    """Top-down decoder producing Stage-2 full/half/quarter states."""

    def __init__(self, channels, blocks=2, horizontal_trajectory=False, bias=True):
        super(PairwisePyramidDecoder, self).__init__()
        c1, c2, c3, c4 = channels
        block_type = (
            HorizontalTrajectoryBlock if horizontal_trajectory
            else DirectionalPyramidBlock
        )
        self.quarter_fuse = nn.Sequential(
            nn.Conv2d(c3 + c4, c3, 1, bias=bias), nn.PReLU(c3),
        )
        self.quarter = nn.Sequential(*[
            block_type(c3, bias=bias) for _ in range(blocks)
        ])
        self.half_fuse = nn.Sequential(
            nn.Conv2d(c2 + c3, c2, 1, bias=bias), nn.PReLU(c2),
        )
        self.half_refine = nn.Sequential(*[
            block_type(c2, bias=bias) for _ in range(blocks)
        ])
        self.full_fuse = nn.Sequential(
            nn.Conv2d(c1 + c2, c1, 1, bias=bias), nn.PReLU(c1),
        )
        self.full = nn.Sequential(*[
            block_type(c1, bias=bias) for _ in range(blocks)
        ])

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, features):
        full, half, quarter, eighth = features
        quarter = self.quarter(self.quarter_fuse(torch.cat([
            quarter, self._up(eighth, quarter)
        ], dim=1)))
        half = self.half_refine(self.half_fuse(torch.cat([
            half, self._up(quarter, half)
        ], dim=1)))
        full = self.full(self.full_fuse(torch.cat([
            full, self._up(half, full)
        ], dim=1)))
        return [full, half, quarter]


class SharedStripeEvidenceEncoder(nn.Module):
    """One shared four-scale encoder, reused for all three full-width stripes."""

    def __init__(self, channels, blocks=2, bias=True):
        super(SharedStripeEvidenceEncoder, self).__init__()
        c1, c2, c3, c4 = channels
        depth = max(1, int(blocks))
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=bias), nn.PReLU(c1),
        )
        self.level1 = nn.Sequential(*[
            HorizontalTrajectoryBlock(c1, bias=bias) for _ in range(depth)
        ])
        self.down12 = nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=bias)
        self.level2 = nn.Sequential(*[
            HorizontalTrajectoryBlock(c2, bias=bias) for _ in range(depth)
        ])
        self.down23 = nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=bias)
        self.level3 = nn.Sequential(*[
            HorizontalTrajectoryBlock(c3, bias=bias) for _ in range(depth)
        ])
        self.down34 = nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=bias)
        self.level4 = nn.Sequential(*[
            HorizontalTrajectoryBlock(c4, bias=bias) for _ in range(depth)
        ])

    def forward(self, stripe):
        full = self.level1(self.stem(stripe))
        half = self.level2(self.down12(full))
        quarter = self.level3(self.down23(half))
        eighth = self.level4(self.down34(quarter))
        return [full, half, quarter, eighth]


class PairEvidenceRelation(nn.Module):
    """Mix one adjacent stripe pair without creating another feature pyramid."""

    def __init__(self, channels, bias=True):
        super(PairEvidenceRelation, self).__init__()
        self.local = HorizontalTrajectoryBlock(channels, bias=bias)
        self.row_state = HorizontalStateMixer(channels, bias=bias)

    def forward(self, pair_feature):
        return self.row_state(self.local(pair_feature))


class EvidenceCanvasDecoder(nn.Module):
    """The only decoder: evidence canvases are decoded once from 1/8 to 1/1."""

    def __init__(self, channels, blocks=2, bias=True):
        super(EvidenceCanvasDecoder, self).__init__()
        c1, c2, c3, c4 = channels
        depth = max(1, int(blocks))

        def make_stage(input_channels, output_channels):
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, bias=bias),
                nn.PReLU(output_channels),
                *[HorizontalTrajectoryBlock(output_channels, bias=bias)
                  for _ in range(depth)],
                HorizontalStateMixer(output_channels, bias=bias),
            )

        self.eighth = make_stage(c4 * 2, c4)
        self.quarter = make_stage(c3 * 2 + c4, c3)
        self.half_stage = make_stage(c2 * 2 + c3, c2)
        self.full = make_stage(c1 * 2 + c2, c1)

    @staticmethod
    def _up(feature, reference):
        return F.interpolate(
            feature,
            size=reference.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

    def forward(self, canvases, disagreements):
        full, half, quarter, eighth = canvases
        d_full, d_half, d_quarter, d_eighth = disagreements
        eighth = self.eighth(torch.cat([eighth, d_eighth], dim=1))
        quarter = self.quarter(torch.cat([
            quarter, d_quarter, self._up(eighth, quarter),
        ], dim=1))
        half = self.half_stage(torch.cat([
            half, d_half, self._up(quarter, half),
        ], dim=1))
        full = self.full(torch.cat([
            full, d_full, self._up(half, full),
        ], dim=1))
        return [full, half, quarter]


class EvidenceCanvasReconstructionCore(nn.Module):
    """Single-encoder, single-decoder restoration core for trajectory evidence.

    Three stripes share one encoder.  Two adjacent-pair views are composed at
    each scale, then assembled into a full-image canvas carrying both shared
    evidence and disagreement.  No Pair encoder, second global encoder, SAM,
    ORSNet, CAB, or intermediate image reconstruction is used.
    """

    def __init__(self, n_feat, scale_unetfeats, stripe_count=3,
                 stripe_overlap=0.20, blocks=2, edge_refine=False,
                 edge_refine_mid=24, bias=True):
        super(EvidenceCanvasReconstructionCore, self).__init__()
        if int(stripe_count) != 3:
            raise ValueError('EvidenceCanvasNet currently requires exactly three stripes.')
        self.stripe_count = int(stripe_count)
        self.stripe_overlap = float(stripe_overlap)
        self.channels = [
            n_feat,
            n_feat + scale_unetfeats,
            n_feat + scale_unetfeats * 2,
            n_feat + scale_unetfeats * 3,
        ]
        self.encoder = SharedStripeEvidenceEncoder(self.channels, blocks=blocks, bias=bias)
        self.pair_relation = nn.ModuleList([
            PairEvidenceRelation(channels, bias=bias)
            for channels in self.channels
        ])
        self.canvas_assembly = nn.ModuleList([
            DualViewTrajectoryMemory(channels, bias=bias)
            for channels in self.channels
        ])
        self.decoder = EvidenceCanvasDecoder(self.channels, blocks=blocks, bias=bias)
        self.synthesis = LaplacianPyramidSynthesisHead(self.channels[:3], bias=bias)
        self.weave_refiner = HaarDetailRefinement(self.channels[0], bias=bias)
        self.edge_refiner = (
            TrajectoryEdgeGuidedRefinement(
                self.channels[0], hidden_channels=edge_refine_mid, bias=bias
            ) if edge_refine else None
        )

    @staticmethod
    def _stripe_starts(height, overlap, align=8):
        effective = 3 - float(overlap) * 2
        stripe_height = int(math.ceil((height / effective) / align) * align)
        stripe_height = min(height, max(align, stripe_height))
        max_start = height - stripe_height
        middle = int(round((max_start / 2) / align) * align)
        return [0, middle, max_start], stripe_height

    @staticmethod
    def _pair_feature(top, bottom, top_start, bottom_start, scale):
        overlap = max(0, int(top_start) + top.size(2) * scale - int(bottom_start))
        overlap = min(bottom.size(2) * scale, overlap)
        overlap = overlap // scale
        if overlap > 0:
            bottom = bottom[:, :, overlap:, :]
        return torch.cat([top, bottom], dim=2)

    def forward(self, input_image):
        height = input_image.size(2)
        if height % 8 != 0:
            raise ValueError(
                'EvidenceCanvasNet requires image height divisible by 8; '
                f'got {height}.'
            )
        starts, stripe_height = self._stripe_starts(
            height, self.stripe_overlap, align=8
        )
        stripe_features = [
            self.encoder(input_image[:, :, start:start + stripe_height, :])
            for start in starts
        ]
        canvases, disagreements = [], []
        for level, scale in enumerate((1, 2, 4, 8)):
            left = self._pair_feature(
                stripe_features[0][level], stripe_features[1][level],
                starts[0], starts[1], scale,
            )
            right = self._pair_feature(
                stripe_features[1][level], stripe_features[2][level],
                starts[1], starts[2], scale,
            )
            left = self.pair_relation[level](left)
            right = self.pair_relation[level](right)
            canvas, disagreement = self.canvas_assembly[level](
                left, right, starts[0] // scale, starts[1] // scale,
                height // scale,
            )
            canvases.append(canvas)
            disagreements.append(disagreement)

        decoded = self.decoder(canvases, disagreements)
        restored = self.synthesis(decoded, input_image)
        restored = self.weave_refiner(decoded[0], restored)
        edge = None
        if self.edge_refiner is not None:
            restored, edge = self.edge_refiner(decoded[0], restored, input_image)
        return restored, decoded, edge


class GlobalLocalPyramidDecoder(nn.Module):
    """Full-resolution replacement for the original Stage-3 ORSNet."""

    def __init__(self, channels, blocks=2, horizontal_trajectory=False, bias=True):
        super(GlobalLocalPyramidDecoder, self).__init__()
        c1, c2, c3, c4 = channels
        block_type = (
            HorizontalTrajectoryBlock if horizontal_trajectory
            else DirectionalPyramidBlock
        )
        self.input_stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=bias), nn.PReLU(c1),
        )
        self.quarter_fuse = nn.Sequential(
            nn.Conv2d(c3 * 2 + c4, c3, 1, bias=bias), nn.PReLU(c3),
        )
        self.quarter = nn.Sequential(*[
            block_type(c3, bias=bias) for _ in range(blocks)
        ])
        self.half_fuse = nn.Sequential(
            nn.Conv2d(c2 * 2 + c3, c2, 1, bias=bias), nn.PReLU(c2),
        )
        self.half_refine = nn.Sequential(*[
            block_type(c2, bias=bias) for _ in range(blocks)
        ])
        self.full_fuse = nn.Sequential(
            nn.Conv2d(c1 * 4 + c2, c1, 1, bias=bias), nn.PReLU(c1),
        )
        self.full = nn.Sequential(*[
            block_type(c1, bias=bias) for _ in range(blocks)
        ])

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, input_image, stage2_state, encoder_features, decoder_features):
        enc_full, enc_half, enc_quarter, enc_eighth = encoder_features
        dec_full, dec_half, dec_quarter = decoder_features
        quarter = self.quarter(self.quarter_fuse(torch.cat([
            enc_quarter, dec_quarter, self._up(enc_eighth, enc_quarter)
        ], dim=1)))
        half = self.half_refine(self.half_fuse(torch.cat([
            enc_half, dec_half, self._up(quarter, enc_half)
        ], dim=1)))
        input_state = self.input_stem(input_image)
        full = self.full(self.full_fuse(torch.cat([
            input_state, stage2_state, enc_full, dec_full, self._up(half, input_state)
        ], dim=1)))
        return [full, half, quarter]


class LaplacianPyramidSynthesisHead(nn.Module):
    """Direct multi-scale image synthesis, replacing tail + single RGB skip.

    The input is represented as a Laplacian pyramid instead of being appended
    by one full-resolution residual operation.  Zero corrections reconstruct
    the input exactly, giving stable initialization and preserving low-frequency
    information on the unseen RX domain.
    """

    def __init__(self, channels, bias=True):
        super(LaplacianPyramidSynthesisHead, self).__init__()
        c1, c2, c3 = channels
        self.delta_full = nn.Conv2d(c1, 3, 3, padding=1, bias=bias)
        self.delta_half = nn.Conv2d(c2, 3, 3, padding=1, bias=bias)
        self.delta_quarter = nn.Conv2d(c3, 3, 3, padding=1, bias=bias)
        for layer in (self.delta_full, self.delta_half, self.delta_quarter):
            nn.init.normal_(layer.weight, mean=0.0, std=1.0e-4)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, features, input_image):
        full, half, quarter = features
        input_half = self._resize(input_image, half.shape[-2:])
        input_quarter = self._resize(input_image, quarter.shape[-2:])

        base = input_quarter + self.delta_quarter(quarter)
        half_image = self._resize(base, half.shape[-2:]) + (
            input_half - self._resize(input_quarter, half.shape[-2:])
        ) + self.delta_half(half)
        return self._resize(half_image, full.shape[-2:]) + (
            input_image - self._resize(input_half, full.shape[-2:])
        ) + self.delta_full(full)


class StructureTextureSynthesisHead(nn.Module):
    """Identity-preserving structure/texture residual synthesis.

    Both branches predict corrections to the degraded input.  Near-zero output
    projections therefore reconstruct the input exactly at initialization,
    while structure and high-frequency corrections remain explicitly
    supervised and separate until the final synthesis.
    """

    def __init__(self, channels, bias=True):
        super(StructureTextureSynthesisHead, self).__init__()
        c1, c2, c3 = channels
        self.structure_quarter = nn.Sequential(
            nn.Conv2d(c3, c3, 3, padding=1, bias=bias),
            nn.PReLU(c3),
            nn.Conv2d(c3, 3, 3, padding=1, bias=bias),
        )
        self.structure_half = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, bias=bias),
            nn.PReLU(c2),
            nn.Conv2d(c2, 3, 3, padding=1, bias=bias),
        )
        self.structure_full = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, bias=bias),
            nn.PReLU(c1),
            nn.Conv2d(c1, 3, 3, padding=1, bias=bias),
        )
        self.texture = nn.Sequential(
            nn.Conv2d(c1 + 3, c1, 3, padding=1, bias=bias),
            nn.PReLU(c1),
            nn.Conv2d(c1, 3, 3, padding=1, bias=bias),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(6, 12, 1, bias=bias),
            nn.PReLU(12),
            nn.Conv2d(12, 3, 1, bias=bias),
        )
        for head in (
            self.structure_quarter[-1], self.structure_half[-1],
            self.structure_full[-1], self.texture[-1], self.mix[-1],
        ):
            nn.init.normal_(head.weight, mean=0.0, std=1.0e-4)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    @staticmethod
    def _resize(x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, features, input_image):
        full, half, quarter = features
        structure_quarter = self.structure_quarter(quarter)
        structure_half = (
            self._resize(structure_quarter, half.shape[-2:])
            + self.structure_half(half)
        )
        structure_delta = (
            self._resize(structure_half, full.shape[-2:])
            + self.structure_full(full)
        )

        input_low = self._resize(
            self._resize(input_image, quarter.shape[-2:]), input_image.shape[-2:]
        )
        texture_delta = self.texture(
            torch.cat([full, input_image - input_low], dim=1)
        )
        fusion_delta = self.mix(
            torch.cat([structure_delta, texture_delta], dim=1)
        )
        restored = input_image + structure_delta + texture_delta + fusion_delta
        return restored, structure_delta, texture_delta


class TrajectoryEdgeGuidedRefinement(nn.Module):
    """Refine the provisional image with line-edge evidence, not a segmenter.

    The fixed gradients expose the faint trace against repetitive fabric
    texture.  A learned one-channel trajectory map selects restoration
    evidence from the new pyramid features; the RGB correction remains a
    near-zero residual so unseen RX colours are preserved at initialization.
    """

    def __init__(self, feature_channels, hidden_channels=24, bias=True):
        super(TrajectoryEdgeGuidedRefinement, self).__init__()
        hidden_channels = max(12, int(hidden_channels))
        self.register_buffer(
            'luma_weights', torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'sobel_x', torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            'sobel_y', torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3),
            persistent=False,
        )
        self.edge_head = nn.Sequential(
            nn.Conv2d(feature_channels + 2, hidden_channels, 3, padding=1, bias=bias),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.rgb_refine = nn.Sequential(
            nn.Conv2d(feature_channels + 4, feature_channels, 3, padding=1, bias=bias),
            nn.PReLU(feature_channels),
            nn.Conv2d(feature_channels, 3, 3, padding=1, bias=bias),
        )
        nn.init.normal_(self.rgb_refine[-1].weight, mean=0.0, std=1.0e-3)
        if self.rgb_refine[-1].bias is not None:
            nn.init.zeros_(self.rgb_refine[-1].bias)

    def _gradient(self, image):
        gray = (image * self.luma_weights.to(dtype=image.dtype)).sum(dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.sobel_x.to(dtype=image.dtype), padding=1)
        grad_y = F.conv2d(gray, self.sobel_y.to(dtype=image.dtype), padding=1)
        return torch.tanh(torch.sqrt(grad_x.square() + grad_y.square() + 1e-6) * 3.0)

    def forward(self, features, provisional, input_image):
        input_gradient = self._gradient(input_image)
        restored_gradient = self._gradient(provisional)
        edge = self.edge_head(torch.cat([
            features, input_gradient, restored_gradient
        ], dim=1))
        correction = self.rgb_refine(torch.cat([
            features, provisional, edge
        ], dim=1))
        return provisional + correction, edge


class HaarDetailRefinement(nn.Module):
    """Reconstruct periodic weave detail in a one-level Haar domain.

    The structure path retains the low-pass coefficient while this branch
    predicts only LH/HL/HH corrections.  It therefore restores thread-like
    background detail without asking the trajectory branch to reproduce the
    entire high-frequency image.
    """

    def __init__(self, feature_channels, hidden_channels=48, bias=True):
        super(HaarDetailRefinement, self).__init__()
        hidden_channels = max(24, min(int(hidden_channels), feature_channels))
        self.feature_project = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, 1, bias=bias),
            nn.PReLU(hidden_channels),
        )
        self.detail_delta = nn.Sequential(
            nn.Conv2d(hidden_channels + 9, hidden_channels, 3, padding=1, bias=bias),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, 9, 3, padding=1, bias=bias),
        )
        nn.init.normal_(self.detail_delta[-1].weight, mean=0.0, std=1.0e-3)
        if self.detail_delta[-1].bias is not None:
            nn.init.zeros_(self.detail_delta[-1].bias)

    @staticmethod
    def _haar(image):
        top_left = image[:, :, 0::2, 0::2]
        top_right = image[:, :, 0::2, 1::2]
        bottom_left = image[:, :, 1::2, 0::2]
        bottom_right = image[:, :, 1::2, 1::2]
        low = (top_left + top_right + bottom_left + bottom_right) * 0.5
        horizontal = (-top_left - top_right + bottom_left + bottom_right) * 0.5
        vertical = (-top_left + top_right - bottom_left + bottom_right) * 0.5
        diagonal = (top_left - top_right - bottom_left + bottom_right) * 0.5
        return low, horizontal, vertical, diagonal

    @staticmethod
    def _inverse_haar(low, horizontal, vertical, diagonal):
        top_left = (low - horizontal - vertical + diagonal) * 0.5
        top_right = (low - horizontal + vertical - diagonal) * 0.5
        bottom_left = (low + horizontal - vertical - diagonal) * 0.5
        bottom_right = (low + horizontal + vertical + diagonal) * 0.5
        batch, channels, height, width = low.shape
        restored = low.new_empty(batch, channels, height * 2, width * 2)
        restored[:, :, 0::2, 0::2] = top_left
        restored[:, :, 0::2, 1::2] = top_right
        restored[:, :, 1::2, 0::2] = bottom_left
        restored[:, :, 1::2, 1::2] = bottom_right
        return restored

    def forward(self, features, provisional):
        height, width = provisional.shape[-2:]
        pad_h, pad_w = height % 2, width % 2
        if pad_h or pad_w:
            provisional = F.pad(provisional, (0, pad_w, 0, pad_h), mode='reflect')
        low, horizontal, vertical, diagonal = self._haar(provisional)
        feature_half = F.interpolate(
            features, size=low.shape[-2:], mode='bilinear', align_corners=False
        )
        details = torch.cat([horizontal, vertical, diagonal], dim=1)
        delta = self.detail_delta(torch.cat([
            self.feature_project(feature_half), details
        ], dim=1))
        horizontal_delta, vertical_delta, diagonal_delta = delta.chunk(3, dim=1)
        restored = self._inverse_haar(
            low,
            horizontal + horizontal_delta,
            vertical + vertical_delta,
            diagonal + diagonal_delta,
        )
        return restored[:, :, :height, :width]


class OverlapDetailExchange(nn.Module):
    """Exchange only aligned decoded detail in the physical pair overlap.

    This is deliberately applied after independent branch decoding.  It lets
    two views correct the same stitch fragment without averaging their entire
    encoder pyramids, which would erase woven texture before reconstruction.
    """

    def __init__(self, channels, bias=True):
        super(OverlapDetailExchange, self).__init__()
        self.detail = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=bias),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=bias),
        )
        # A bounded positive value makes both decoded views converge only in
        # their observed overlap.  It starts at 0.05, reducing disagreement
        # without forcing either branch to copy the other.
        self.alignment_logit = nn.Parameter(torch.tensor(-1.3862944))
        nn.init.normal_(self.detail[-1].weight, mean=0.0, std=1.0e-3)
        if self.detail[-1].bias is not None:
            nn.init.zeros_(self.detail[-1].bias)

    def forward(self, left, right, left_start, right_start):
        overlap_start = max(int(left_start), int(right_start))
        overlap_end = min(
            int(left_start) + left.size(2),
            int(right_start) + right.size(2),
        )
        if overlap_end <= overlap_start:
            return left, right
        left_top = overlap_start - int(left_start)
        right_top = overlap_start - int(right_start)
        length = overlap_end - overlap_start
        left_overlap = left[:, :, left_top:left_top + length, :]
        right_overlap = right[:, :, right_top:right_top + length, :]
        correction = self.detail(torch.cat([
            0.5 * (left_overlap + right_overlap),
            torch.abs(left_overlap - right_overlap),
            left_overlap * right_overlap,
        ], dim=1))
        alignment = 0.25 * torch.sigmoid(self.alignment_logit) * (
            right_overlap - left_overlap
        )
        left_delta = left.new_zeros(left.shape)
        right_delta = right.new_zeros(right.shape)
        left_delta[:, :, left_top:left_top + length, :] = correction + alignment
        right_delta[:, :, right_top:right_top + length, :] = correction - alignment
        return left + left_delta, right + right_delta


class WeaveFrequencyBlock(nn.Module):
    """Feature-domain low/high decomposition for periodic textile texture."""

    def __init__(self, channels, bias=True):
        super(WeaveFrequencyBlock, self).__init__()
        self.low = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=bias),
            nn.PReLU(channels),
        )
        self.high = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=bias),
            nn.PReLU(channels),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=bias),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 1, bias=bias),
        )
        nn.init.normal_(self.fuse[-1].weight, mean=0.0, std=1.0e-3)
        if self.fuse[-1].bias is not None:
            nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        low = F.interpolate(low, size=x.shape[-2:], mode='bilinear', align_corners=False)
        high = x - low
        return self.fuse(torch.cat([self.low(low), self.high(high)], dim=1))


class TrajectoryWeaveBlock(nn.Module):
    """The main restoration block: slanted trajectory plus weave texture."""

    def __init__(self, channels, use_slanted_attention=False, bias=True):
        super(TrajectoryWeaveBlock, self).__init__()
        self.trajectory = CoordinateTrajectoryMixer(channels, bias=bias)
        self.slanted_attention = (
            SlantedStripAttention(channels, bias=bias)
            if use_slanted_attention else None
        )
        self.weave = WeaveFrequencyBlock(channels, bias=bias)

    def forward(self, x, start=0, full_height=None):
        x = self.trajectory(x, start=start, full_height=full_height)
        if self.slanted_attention is not None:
            x = self.slanted_attention(x)
        return x + self.weave(x)


class TrajectoryWeaveMainRestorer(nn.Module):
    """Full-image progressive main restoration path conditioned by pair memory."""

    def __init__(self, channels, blocks=3, bias=True):
        super(TrajectoryWeaveMainRestorer, self).__init__()
        c1, c2, c3, c4 = channels
        deep_depth = max(2, int(blocks))
        shallow_depth = 2
        self.stem = nn.Sequential(
            nn.Conv2d(3 + c1, c1, 3, padding=1, bias=bias),
            nn.PReLU(c1),
        )
        self.memory_project = nn.ModuleList([
            nn.Conv2d(channel, channel, 1, bias=bias) for channel in channels
        ])
        for projection in self.memory_project:
            nn.init.dirac_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

        self.encoder1 = nn.ModuleList([
            TrajectoryWeaveBlock(c1, use_slanted_attention=False, bias=bias)
            for _ in range(shallow_depth)
        ])
        self.down12 = nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=bias)
        self.encoder2 = nn.ModuleList([
            TrajectoryWeaveBlock(c2, use_slanted_attention=False, bias=bias)
            for _ in range(shallow_depth)
        ])
        self.down23 = nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=bias)
        self.encoder3 = nn.ModuleList([
            TrajectoryWeaveBlock(c3, use_slanted_attention=True, bias=bias)
            for _ in range(deep_depth)
        ])
        self.down34 = nn.Conv2d(c3, c4, 3, stride=2, padding=1, bias=bias)
        self.encoder4 = nn.ModuleList([
            TrajectoryWeaveBlock(c4, use_slanted_attention=True, bias=bias)
            for _ in range(deep_depth)
        ])

        self.decode3_fuse = nn.Sequential(
            nn.Conv2d(c3 + c4, c3, 1, bias=bias), nn.PReLU(c3),
        )
        self.decode3 = nn.ModuleList([
            TrajectoryWeaveBlock(c3, use_slanted_attention=True, bias=bias)
            for _ in range(deep_depth)
        ])
        self.decode2_fuse = nn.Sequential(
            nn.Conv2d(c2 + c3, c2, 1, bias=bias), nn.PReLU(c2),
        )
        self.decode2 = nn.ModuleList([
            TrajectoryWeaveBlock(c2, use_slanted_attention=False, bias=bias)
            for _ in range(shallow_depth)
        ])
        self.decode1_fuse = nn.Sequential(
            nn.Conv2d(c1 + c2, c1, 1, bias=bias), nn.PReLU(c1),
        )
        self.decode1 = nn.ModuleList([
            TrajectoryWeaveBlock(c1, use_slanted_attention=False, bias=bias)
            for _ in range(shallow_depth)
        ])

        self.input_project = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=bias), nn.PReLU(c1),
        )
        self.half_project = nn.Conv2d(c2, c1, 1, bias=bias)
        self.quarter_project = nn.Conv2d(c3, c1, 1, bias=bias)
        self.main_fuse = nn.Sequential(
            nn.Conv2d(c1 * 5, c1, 1, bias=bias), nn.PReLU(c1),
        )
        self.main_blocks = nn.ModuleList([
            TrajectoryWeaveBlock(c1, use_slanted_attention=False, bias=bias)
            for _ in range(shallow_depth)
        ])

    @staticmethod
    def _up(x, reference):
        return F.interpolate(x, size=reference.shape[-2:], mode='bilinear', align_corners=False)

    @staticmethod
    def _run(blocks, x, start, full_height):
        for block in blocks:
            x = block(x, start=start, full_height=full_height)
        return x

    def forward(self, input_image, pair_memory):
        height = input_image.size(2)
        memory1, memory2, memory3, memory4 = pair_memory

        enc1 = self.stem(torch.cat([input_image, self.memory_project[0](memory1)], dim=1))
        enc1 = self._run(self.encoder1, enc1, 0, height)
        enc2 = self.down12(enc1) + self.memory_project[1](memory2)
        enc2 = self._run(self.encoder2, enc2, 0, height // 2)
        enc3 = self.down23(enc2) + self.memory_project[2](memory3)
        enc3 = self._run(self.encoder3, enc3, 0, height // 4)
        enc4 = self.down34(enc3) + self.memory_project[3](memory4)
        enc4 = self._run(self.encoder4, enc4, 0, height // 8)

        dec3 = self.decode3_fuse(torch.cat([enc3, self._up(enc4, enc3)], dim=1))
        dec3 = self._run(self.decode3, dec3, 0, height // 4)
        dec2 = self.decode2_fuse(torch.cat([enc2, self._up(dec3, enc2)], dim=1))
        dec2 = self._run(self.decode2, dec2, 0, height // 2)
        dec1 = self.decode1_fuse(torch.cat([enc1, self._up(dec2, enc1)], dim=1))
        dec1 = self._run(self.decode1, dec1, 0, height)

        main = self.main_fuse(torch.cat([
            self.input_project(input_image), enc1, dec1,
            self._up(self.half_project(dec2), enc1),
            self._up(self.quarter_project(dec3), enc1),
        ], dim=1))
        main = self._run(self.main_blocks, main, 0, height)
        return main, [enc1, enc2, enc3, enc4], [dec1, dec2, dec3]


class TrajectoryPyramidRecoveryHead(nn.Module):
    """Single full-image recovery trunk driven by analysed multi-scale memory.

    This is the terminal feature recovery network in HMT-Net.  Unlike the
    original ORSNet, it has no repeated original-resolution stage blocks or
    cross-stage image restoration inputs: it synchronizes the main decoder,
    the feature-state bridge, and PairAligned memory once, then performs
    trajectory-aware recovery on the full canvas.
    """

    def __init__(self, channels, blocks=3, bias=True):
        super(TrajectoryPyramidRecoveryHead, self).__init__()
        c1, c2, c3, _ = channels
        self.main_project = nn.Conv2d(c1, c1, 1, bias=bias)
        self.state_project = nn.Conv2d(c1, c1, 1, bias=bias)
        self.full_project = nn.Conv2d(c1, c1, 1, bias=bias)
        self.half_project = nn.Conv2d(c2, c1, 1, bias=bias)
        self.quarter_project = nn.Conv2d(c3, c1, 1, bias=bias)
        self.memory_project = nn.Conv2d(c1, c1, 1, bias=bias)
        self.disagreement_project = nn.Conv2d(c1, c1, 1, bias=bias)
        self.fuse = nn.Sequential(
            nn.Conv2d(c1 * 7, c1, 1, bias=bias),
            nn.PReLU(c1),
            nn.Conv2d(c1, c1, 3, padding=1, bias=bias),
            nn.PReLU(c1),
        )
        self.blocks = nn.ModuleList([
            TrajectoryWeaveBlock(c1, use_slanted_attention=False, bias=bias)
            for _ in range(max(2, int(blocks)))
        ])

    @staticmethod
    def _up(feature, reference):
        return F.interpolate(
            feature,
            size=reference.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

    def forward(self, main_feature, state_feature, decoder_features, pair_memory,
                pair_disagreement):
        full, half, quarter = decoder_features
        reference = main_feature
        fused = self.fuse(torch.cat([
            self.main_project(main_feature),
            self.state_project(state_feature),
            self.full_project(full),
            self._up(self.half_project(half), reference),
            self._up(self.quarter_project(quarter), reference),
            self.memory_project(pair_memory[0]),
            self.disagreement_project(pair_disagreement[0]),
        ], dim=1))
        height = fused.size(2)
        for block in self.blocks:
            fused = block(fused, start=0, full_height=height)
        return fused


class TerminalTrajectoryResidualReconstructor(nn.Module):
    """Use dual-view uncertainty to correct only the final image residual.

    It is intentionally a terminal module: no intermediate image is emitted.
    The predicted correction is conditioned on the provisional restoration,
    its explicit input residual, and the evidence on which PairAligned views
    disagreed.
    """

    def __init__(self, feature_channels, bias=True):
        super(TerminalTrajectoryResidualReconstructor, self).__init__()
        self.evidence_project = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 1, bias=bias),
            nn.PReLU(feature_channels),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(feature_channels * 2 + 9, feature_channels, 3, padding=1, bias=bias),
            nn.PReLU(feature_channels),
            nn.Conv2d(feature_channels, feature_channels // 2, 3, padding=1, bias=bias),
            nn.PReLU(feature_channels // 2),
            nn.Conv2d(feature_channels // 2, 3, 3, padding=1, bias=bias),
        )
        nn.init.normal_(self.residual[-1].weight, mean=0.0, std=1.0e-3)
        if self.residual[-1].bias is not None:
            nn.init.zeros_(self.residual[-1].bias)

    def forward(self, feature, provisional, input_image, disagreement):
        if disagreement.shape[-2:] != feature.shape[-2:]:
            disagreement = F.interpolate(
                disagreement,
                size=feature.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        evidence = self.evidence_project(disagreement)
        input_residual = provisional - input_image
        correction = self.residual(torch.cat([
            feature,
            evidence,
            input_image,
            provisional,
            input_residual,
        ], dim=1))
        return provisional + correction


class SharedMemoryReconstructionCore(nn.Module):
    """HMT analysis-memory backend with one terminal image-recovery path."""

    def __init__(self, n_feat, scale_unetfeats, blocks=3,
                 tcfm_horizontal_kernel=15, edge_refine=False,
                 edge_refine_mid=24, bias=True):
        super(SharedMemoryReconstructionCore, self).__init__()
        channels = [
            n_feat,
            n_feat + scale_unetfeats,
            n_feat + scale_unetfeats * 2,
            n_feat + scale_unetfeats * 3,
        ]
        self.pair_encoder = PairwiseStripePyramidEncoder(
            n_feat + 3,
            channels,
            blocks=blocks,
            horizontal_trajectory=True,
            bias=bias,
        )
        self.pair_memory_context = nn.ModuleList([
            CoordinateTrajectoryMixer(channels[level], bias=bias)
            for level in range(4)
        ])
        self.pair_memory_attention = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            SlantedStripAttention(channels[2], bias=bias),
            SlantedStripAttention(channels[3], bias=bias),
        ])
        self.pair_memory_assembly = nn.ModuleList([
            DualViewTrajectoryMemory(channels[level], bias=bias)
            for level in range(4)
        ])
        self.stage1_encoder_project = nn.ModuleList([
            nn.Conv2d(channels[level], channels[level], 1, bias=bias)
            for level in range(3)
        ])
        self.stage1_decoder_project = nn.ModuleList([
            nn.Conv2d(channels[level], channels[level], 1, bias=bias)
            for level in range(3)
        ])
        self.tcfm = nn.ModuleList([
            TrajectoryCalibratedCSFFGate(
                channels[level], horizontal_kernel=tcfm_horizontal_kernel
            )
            for level in range(3)
        ])
        for projections in (
            self.stage1_encoder_project, self.stage1_decoder_project
        ):
            for projection in projections:
                nn.init.dirac_(projection.weight)
                if projection.bias is not None:
                    nn.init.zeros_(projection.bias)
        self.main_restorer = TrajectoryWeaveMainRestorer(
            channels, blocks=blocks, bias=bias
        )
        self.stage2_bridge = FeatureMemoryBridge(channels[0], channels[0], bias=bias)
        self.recovery_head = TrajectoryPyramidRecoveryHead(
            # Full-resolution trajectory mixing is intentionally shallow for
            # stable Fast100 batch-2 training; capacity stays in the lower
            # resolution main pyramid where it is substantially cheaper.
            channels, blocks=2, bias=bias
        )
        self.synthesis = StructureTextureSynthesisHead(channels[:3], bias=bias)
        self.terminal_residual = TerminalTrajectoryResidualReconstructor(
            channels[0], bias=bias
        )
        # Fast100 contains only about 850 optimizer steps.  Keep synthesis
        # residuals small, but large enough for the main restorer to receive
        # useful gradients from the first epoch.
        for head in (
            self.synthesis.structure_quarter[-1],
            self.synthesis.structure_half[-1],
            self.synthesis.structure_full[-1],
            self.synthesis.texture[-1],
            self.synthesis.mix[-1],
        ):
            nn.init.normal_(head.weight, mean=0.0, std=1.0e-3)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
        self.weave_refiner = HaarDetailRefinement(channels[0], bias=bias)
        self.edge_refiner = (
            TrajectoryEdgeGuidedRefinement(
                channels[0], hidden_channels=edge_refine_mid, bias=bias
            )
            if edge_refine else None
        )

    def _inject_stage1(self, features, encoder_features, decoder_features):
        injected = list(features)
        for level in range(3):
            current = features[level]
            encoded = self.stage1_encoder_project[level](encoder_features[level])
            decoded = self.stage1_decoder_project[level](decoder_features[level])
            gate = self.tcfm[level](current, encoded, decoded)
            injected[level] = current + 0.5 * gate * (encoded + decoded)
        return injected

    def forward(self, input_image, pair_inputs, pair_starts, blend_fn,
                stage1_encoder_pairs, stage1_decoder_pairs):
        if len(pair_inputs) != 2:
            raise ValueError('HMT-Net requires exactly two adjacent stripe pairs.')
        pair_features = [
            self._inject_stage1(
                self.pair_encoder(pair_input),
                stage1_encoder_pairs[index],
                stage1_decoder_pairs[index],
            )
            for index, pair_input in enumerate(pair_inputs)
        ]
        # PairAligned branches provide multi-view memory only.  A single full
        # image main path below owns restoration, avoiding competing branch
        # decoders and making the progressive recovery hierarchy explicit.
        height = input_image.size(2)
        pair_memory = []
        pair_disagreement = []
        for level, scale in enumerate((1, 2, 4, 8)):
            memory, disagreement = self.pair_memory_assembly[level](
                pair_features[0][level],
                pair_features[1][level],
                pair_starts[0] // scale,
                pair_starts[1] // scale,
                height // scale,
            )
            # Inject uncertainty at its native scale before pyramid encoding;
            # the full-resolution copy is also retained for terminal residual
            # reconstruction below.
            memory = memory + disagreement
            memory = self.pair_memory_context[level](
                memory, start=0, full_height=height // scale
            )
            memory = self.pair_memory_attention[level](memory)
            pair_memory.append(memory)
            pair_disagreement.append(disagreement)

        main_feature, encoder_features, decoder_features = self.main_restorer(
            input_image, pair_memory
        )
        stage2_state, stage2_image = self.stage2_bridge(main_feature, input_image)
        final_full = self.recovery_head(
            main_feature,
            stage2_state,
            decoder_features,
            pair_memory,
            pair_disagreement,
        )
        # Only the terminal recovery trunk feeds an image synthesis head.  The
        # half/quarter decoder maps remain feature evidence, never images.
        final_features = [
            final_full, decoder_features[1], decoder_features[2]
        ]
        restored, structure, texture = self.synthesis(final_features, input_image)
        restored = self.terminal_residual(
            final_features[0], restored, input_image, pair_disagreement[0]
        )
        restored = self.weave_refiner(final_features[0], restored)
        edge = None
        if self.edge_refiner is not None:
            restored, edge = self.edge_refiner(
                final_features[0], restored, input_image
            )
        return restored, stage2_image, encoder_features, decoder_features, {
            'structure': structure,
            'texture': texture,
            'edge': edge,
        }


class MotionAdaptiveSeparableResidualFilter(nn.Module):
    """Feature residual filter with horizontal-first motion adaptation."""

    def __init__(self, channels, horizontal_kernel=15, vertical_kernel=5,
                 bias=True):
        super(MotionAdaptiveSeparableResidualFilter, self).__init__()
        channels = int(channels)
        horizontal_kernel = int(horizontal_kernel)
        vertical_kernel = int(vertical_kernel)
        if horizontal_kernel < 3 or horizontal_kernel % 2 == 0:
            raise ValueError('TFR horizontal kernel must be an odd integer >= 3.')
        if vertical_kernel < 3 or vertical_kernel % 2 == 0:
            raise ValueError('TFR vertical kernel must be an odd integer >= 3.')
        self.norm = LayerNorm2d(channels)
        self.horizontal = nn.Conv2d(
            channels, channels, (1, horizontal_kernel),
            padding=(0, horizontal_kernel // 2), groups=channels, bias=bias,
        )
        self.vertical = nn.Conv2d(
            channels, channels, (vertical_kernel, 1),
            padding=(vertical_kernel // 2, 0), groups=channels, bias=bias,
        )
        self.diagonal_a = nn.Conv2d(
            channels, channels, 3, padding=2, dilation=2,
            groups=channels, bias=bias,
        )
        self.diagonal_b = nn.Conv2d(
            channels, channels, 3, padding=3, dilation=3,
            groups=channels, bias=bias,
        )
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=bias),
            nn.PReLU(channels),
            nn.Conv2d(channels, 4, 1, bias=bias),
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=bias)
        self.residual_scale = nn.Parameter(torch.tensor(0.05))
        nn.init.zeros_(self.project.weight)
        if self.project.bias is not None:
            nn.init.zeros_(self.project.bias)

    def forward(self, x, motion_memory):
        memory = F.interpolate(
            motion_memory, size=x.shape[-2:], mode='bilinear',
            align_corners=False,
        )
        weights = torch.softmax(
            self.weight_predictor(torch.cat([self.norm(x), memory], dim=1)),
            dim=1,
        )
        filtered = (
            self.horizontal(x) * weights[:, 0:1]
            + self.vertical(x) * weights[:, 1:2]
            + self.diagonal_a(x) * weights[:, 2:3]
            + self.diagonal_b(x) * weights[:, 3:4]
        )
        return x + torch.tanh(self.residual_scale) * self.project(filtered)


class TFRRestorationBlock(nn.Module):
    """NAF-style local restoration plus trajectory and frequency interaction."""

    def __init__(self, channels, horizontal_kernel=15, vertical_kernel=5,
                 bias=True):
        super(TFRRestorationBlock, self).__init__()
        self.local = NAFBlock(channels, bias=bias)
        self.trajectory = CoordinateTrajectoryMixer(channels, bias=bias)
        self.motion_filter = MotionAdaptiveSeparableResidualFilter(
            channels, horizontal_kernel=horizontal_kernel,
            vertical_kernel=vertical_kernel, bias=bias,
        )
        self.frequency = WeaveFrequencyBlock(channels, bias=bias)
        self.freq_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x, memory, start=0, full_height=None):
        x = self.local(x)
        x = self.trajectory(x, start=start, full_height=full_height)
        x = self.motion_filter(x, memory)
        return x + torch.tanh(self.freq_scale) * self.frequency(x)


class DualPairMemoryAssembler(nn.Module):
    """Keep upper/middle and middle/lower pair memories separate until recovery."""

    def __init__(self, channels, bias=True):
        super(DualPairMemoryAssembler, self).__init__()
        self.memory_fuse = nn.Sequential(
            nn.Conv2d(channels * 5, channels, 1, bias=bias),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=bias),
        )
        self.difference_project = nn.Conv2d(channels, channels, 1, bias=bias)
        nn.init.normal_(self.memory_fuse[-1].weight, mean=0.0, std=1.0e-3)
        nn.init.normal_(self.difference_project.weight, mean=0.0, std=1.0e-3)
        if self.memory_fuse[-1].bias is not None:
            nn.init.zeros_(self.memory_fuse[-1].bias)
        if self.difference_project.bias is not None:
            nn.init.zeros_(self.difference_project.bias)

    @staticmethod
    def _place_on_canvas(feature, start, full_height):
        start = int(start)
        canvas = feature.new_zeros(
            feature.size(0), feature.size(1), full_height, feature.size(3)
        )
        end = min(full_height, start + feature.size(2))
        if end > start:
            canvas[:, :, start:end, :] = feature[:, :, :end - start, :]
        return canvas

    def forward(self, left, right, left_start, right_start, full_height,
                blend_fn):
        canvas_left = self._place_on_canvas(left, left_start, full_height)
        canvas_right = self._place_on_canvas(right, right_start, full_height)
        mean_canvas = 0.5 * (canvas_left + canvas_right)
        abs_diff = torch.abs(canvas_left - canvas_right)
        signed_diff = canvas_left - canvas_right
        product = canvas_left * canvas_right
        memory = self.memory_fuse(torch.cat([
            canvas_left, canvas_right, mean_canvas, abs_diff, product,
        ], dim=1))
        difference = self.difference_project(signed_diff)
        return memory, difference


class TFRPyramidRestorer(nn.Module):
    """Strong full-image residual restorer for TFR-Net."""

    def __init__(self, channels, blocks=8, horizontal_kernel=15,
                 vertical_kernel=5, bias=True):
        super(TFRPyramidRestorer, self).__init__()
        c1, c2, c3 = channels
        self.input_project = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=bias),
            nn.PReLU(c1),
        )
        self.memory_project = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel * 2, channel, 1, bias=bias),
                nn.PReLU(channel),
                nn.Conv2d(channel, channel, 3, padding=1, bias=bias),
            )
            for channel in channels
        ])
        self.down12 = nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=bias)
        self.down23 = nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=bias)
        total_blocks = max(6, int(blocks))
        full_blocks = min(4, max(3, total_blocks // 2))
        remaining_blocks = max(2, total_blocks - full_blocks)
        half_blocks = max(1, remaining_blocks // 2)
        quarter_blocks = max(1, remaining_blocks - half_blocks)
        # Keep early high-resolution analysis cheap.  The expensive
        # motion-adaptive filters are reserved for the lower-resolution states
        # and the final 3-4 full-resolution reconstruction blocks.
        self.encoder1 = nn.ModuleList([
            NAFBlock(c1, bias=bias) for _ in range(1)
        ])
        self.encoder2 = nn.ModuleList([
            NAFBlock(c2, bias=bias) for _ in range(1)
        ])
        self.bottleneck = nn.ModuleList([
            TFRRestorationBlock(c3, horizontal_kernel, vertical_kernel, bias)
            for _ in range(quarter_blocks)
        ])
        self.decode2_fuse = nn.Sequential(
            nn.Conv2d(c2 + c3, c2, 1, bias=bias), nn.PReLU(c2),
        )
        self.decoder2 = nn.ModuleList([
            TFRRestorationBlock(c2, horizontal_kernel, vertical_kernel, bias)
            for _ in range(half_blocks)
        ])
        self.decode1_fuse = nn.Sequential(
            nn.Conv2d(c1 + c2, c1, 1, bias=bias), nn.PReLU(c1),
        )
        self.decoder1 = nn.ModuleList([
            TFRRestorationBlock(c1, horizontal_kernel, vertical_kernel, bias)
            for _ in range(full_blocks)
        ])
        self.stage1_inject = nn.ModuleList([
            nn.Conv2d(channel * 3, channel, 1, bias=bias)
            for channel in channels
        ])
        for injection in self.stage1_inject:
            nn.init.normal_(injection.weight, mean=0.0, std=1.0e-3)
            if injection.bias is not None:
                nn.init.zeros_(injection.bias)

    @staticmethod
    def _up(x, reference):
        return F.interpolate(
            x, size=reference.shape[-2:], mode='bilinear',
            align_corners=False,
        )

    @staticmethod
    def _run(blocks, x, memory):
        for block in blocks:
            x = block(x, memory)
        return x

    @staticmethod
    def _run_plain(blocks, x):
        for block in blocks:
            x = block(x)
        return x

    def _inject_stage1(self, x, memory, encoder_state, decoder_state, level):
        evidence = torch.cat([memory, encoder_state, decoder_state], dim=1)
        return x + self.stage1_inject[level](evidence)

    def forward(self, input_image, pair_memory, pair_difference,
                stage1_encoder, stage1_decoder):
        memory = [
            self.memory_project[level](torch.cat([
                pair_memory[level], pair_difference[level]
            ], dim=1))
            for level in range(3)
        ]
        x1 = self.input_project(input_image) + memory[0]
        x1 = self._inject_stage1(
            x1, memory[0], stage1_encoder[0], stage1_decoder[0], 0
        )
        x1 = self._run_plain(self.encoder1, x1)
        x2 = self.down12(x1) + memory[1]
        x2 = self._inject_stage1(
            x2, memory[1], stage1_encoder[1], stage1_decoder[1], 1
        )
        x2 = self._run_plain(self.encoder2, x2)
        x3 = self.down23(x2) + memory[2]
        x3 = self._inject_stage1(
            x3, memory[2], stage1_encoder[2], stage1_decoder[2], 2
        )
        x3 = self._run(self.bottleneck, x3, memory[2])
        y2 = self.decode2_fuse(torch.cat([x2, self._up(x3, x2)], dim=1))
        y2 = self._run(self.decoder2, y2, memory[1])
        y1 = self.decode1_fuse(torch.cat([x1, self._up(y2, x1)], dim=1))
        y1 = self._run(self.decoder1, y1, memory[0])
        return y1, y2, x3


class ExplicitResidualReconstructionHead(nn.Module):
    """Predict clear-minus-blur residual from full, half, and quarter states."""

    def __init__(self, channels, bias=True):
        super(ExplicitResidualReconstructionHead, self).__init__()
        c1, c2, c3 = channels
        self.half_project = nn.Conv2d(c2, c1, 1, bias=bias)
        self.quarter_project = nn.Conv2d(c3, c1, 1, bias=bias)
        self.fuse = nn.Sequential(
            nn.Conv2d(c1 * 3 + 3, c1, 1, bias=bias),
            nn.PReLU(c1),
            NAFBlock(c1, bias=bias),
            nn.Conv2d(c1, c1, 3, padding=1, bias=bias),
            nn.PReLU(c1),
            nn.Conv2d(c1, 3, 3, padding=1, bias=bias),
        )
        nn.init.normal_(self.fuse[-1].weight, mean=0.0, std=1.0e-3)
        if self.fuse[-1].bias is not None:
            nn.init.zeros_(self.fuse[-1].bias)

    @staticmethod
    def _up(x, reference):
        return F.interpolate(
            x, size=reference.shape[-2:], mode='bilinear',
            align_corners=False,
        )

    def forward(self, full, half, quarter, input_image):
        quarter_size = quarter.shape[-2:]
        half = self._up(self.half_project(half), full)
        quarter = self._up(self.quarter_project(quarter), full)
        input_low = self._up(
            F.interpolate(
                input_image, size=quarter_size, mode='bilinear',
                align_corners=False,
            ),
            full,
        )
        delta = self.fuse(torch.cat([
            full, half, quarter, input_image - input_low,
        ], dim=1))
        return input_image + delta


class PairCanvasResidualHead(nn.Module):
    """Image-level supervision head for the assembled dual-pair memory."""

    def __init__(self, channels, bias=True):
        super(PairCanvasResidualHead, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2 + 3, channels, 1, bias=bias),
            nn.PReLU(channels),
            NAFBlock(channels, bias=bias),
            nn.Conv2d(channels, channels, 3, padding=1, bias=bias),
            nn.PReLU(channels),
            nn.Conv2d(channels, 3, 3, padding=1, bias=bias),
        )
        nn.init.normal_(self.fuse[-1].weight, mean=0.0, std=1.0e-3)
        if self.fuse[-1].bias is not None:
            nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, pair_memory, pair_difference, input_image):
        delta = self.fuse(torch.cat([
            pair_memory, pair_difference, input_image,
        ], dim=1))
        return input_image + delta


class RecursiveResidualRefiner(nn.Module):
    """Shared image-space residual correction repeated at the terminal stage."""

    def __init__(self, channels, steps=3, bias=True):
        super(RecursiveResidualRefiner, self).__init__()
        self.steps = max(1, int(steps))
        self.body = nn.Sequential(
            nn.Conv2d(channels + 6, channels, 1, bias=bias),
            nn.PReLU(channels),
            NAFBlock(channels, bias=bias),
            nn.Conv2d(channels, channels, 3, padding=1, bias=bias),
            nn.PReLU(channels),
            nn.Conv2d(channels, 3, 3, padding=1, bias=bias),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.05))
        nn.init.normal_(self.body[-1].weight, mean=0.0, std=1.0e-3)
        if self.body[-1].bias is not None:
            nn.init.zeros_(self.body[-1].bias)

    def forward(self, features, coarse, input_image):
        current = coarse
        scale = torch.tanh(self.residual_scale)
        for _ in range(self.steps):
            delta = self.body(torch.cat([
                features, current, input_image,
            ], dim=1))
            current = current + scale * delta
        return current


class TriStripeFrequencyResidualCore(nn.Module):
    """TFR-Net core: dual pair memory, motion filtering, and residual recovery."""

    def __init__(self, n_feat, scale_unetfeats, blocks=8,
                 horizontal_kernel=15, vertical_kernel=5, edge_refine=False,
                 edge_refine_mid=24, refiner_steps=3, bias=True):
        super(TriStripeFrequencyResidualCore, self).__init__()
        channels = [
            n_feat,
            n_feat + scale_unetfeats,
            n_feat + scale_unetfeats * 2,
        ]
        self.pair_encoder = PairwiseStripeThreeScaleEncoder(
            n_feat + 3,
            channels,
            blocks=max(2, int(blocks) // 4),
            horizontal_trajectory=True,
            bias=bias,
        )
        self.memory_assembler = nn.ModuleList([
            DualPairMemoryAssembler(channel, bias=bias)
            for channel in channels
        ])
        self.stage1_encoder_project = nn.ModuleList([
            nn.Conv2d(channel, channel, 1, bias=bias)
            for channel in channels
        ])
        self.stage1_decoder_project = nn.ModuleList([
            nn.Conv2d(channel, channel, 1, bias=bias)
            for channel in channels
        ])
        for projections in (
            self.stage1_encoder_project, self.stage1_decoder_project
        ):
            for projection in projections:
                nn.init.dirac_(projection.weight)
                if projection.bias is not None:
                    nn.init.zeros_(projection.bias)
        self.restorer = TFRPyramidRestorer(
            channels,
            blocks=blocks,
            horizontal_kernel=horizontal_kernel,
            vertical_kernel=vertical_kernel,
            bias=bias,
        )
        self.pair_head = PairCanvasResidualHead(channels[0], bias=bias)
        self.head = ExplicitResidualReconstructionHead(channels, bias=bias)
        self.refiner = RecursiveResidualRefiner(
            channels[0], steps=refiner_steps, bias=bias
        )
        self.edge_refiner = (
            TrajectoryEdgeGuidedRefinement(
                channels[0], hidden_channels=edge_refine_mid, bias=bias
            )
            if edge_refine else None
        )

    def _stage1_canvas(self, features, starts, full_height, blend_fn, level):
        scale = 2 ** level
        scaled_starts = [int(start) // scale for start in starts]
        return blend_fn(
            [feature[level] for feature in features],
            scaled_starts,
            full_height // scale,
        )

    def forward(self, input_image, pair_inputs, pair_starts, blend_fn,
                stage1_encoder_pairs, stage1_decoder_pairs):
        if len(pair_inputs) != 2:
            raise ValueError('TFR-Net requires exactly two adjacent stripe pairs.')
        if input_image.size(2) % 4 != 0:
            raise ValueError('TFR-Net requires image height divisible by 4.')
        pair_features = [self.pair_encoder(pair_input) for pair_input in pair_inputs]
        height = input_image.size(2)
        pair_memory = []
        pair_difference = []
        stage1_encoder = []
        stage1_decoder = []
        for level, scale in enumerate((1, 2, 4)):
            memory, difference = self.memory_assembler[level](
                pair_features[0][level],
                pair_features[1][level],
                pair_starts[0] // scale,
                pair_starts[1] // scale,
                height // scale,
                blend_fn,
            )
            pair_memory.append(memory)
            pair_difference.append(difference)
            stage1_encoder.append(self.stage1_encoder_project[level](
                self._stage1_canvas(
                    stage1_encoder_pairs, pair_starts,
                    height, blend_fn, level,
                )
            ))
            stage1_decoder.append(self.stage1_decoder_project[level](
                self._stage1_canvas(
                    stage1_decoder_pairs, pair_starts,
                    height, blend_fn, level,
                )
            ))
        full, half, quarter = self.restorer(
            input_image,
            pair_memory,
            pair_difference,
            stage1_encoder,
            stage1_decoder,
        )
        pair_restored = self.pair_head(
            pair_memory[0], pair_difference[0], input_image
        )
        coarse_restored = self.head(full, half, quarter, input_image)
        restored = self.refiner(full, coarse_restored, input_image)
        edge = None
        if self.edge_refiner is not None:
            restored, edge = self.edge_refiner(full, restored, input_image)
        return restored, coarse_restored, pair_restored, full, {
            'edge': edge,
            'tfr_pair_difference_abs': torch.stack([
                difference.detach().abs().mean()
                for difference in pair_difference
            ]).mean(),
        }


class TrajectoryPairReconstructionCore(nn.Module):
    """New TPT-RNet backend with three-scale TCFM state injection."""

    def __init__(self, n_feat, scale_unetfeats, blocks=2, tcfm=True,
                 tcfm_horizontal_kernel=7, horizontal_trajectory=False,
                 bias=True):
        super(TrajectoryPairReconstructionCore, self).__init__()
        channels = [
            n_feat,
            n_feat + scale_unetfeats,
            n_feat + scale_unetfeats * 2,
            n_feat + scale_unetfeats * 3,
        ]
        self.pair_encoder = PairwiseStripePyramidEncoder(
            n_feat + 3,
            channels,
            blocks=blocks,
            horizontal_trajectory=horizontal_trajectory,
            bias=bias,
        )
        self.tcfm_enabled = bool(tcfm)
        self.stage1_encoder_project = nn.ModuleList([
            nn.Conv2d(channels[level], channels[level], 1, bias=bias)
            for level in range(3)
        ])
        self.stage1_decoder_project = nn.ModuleList([
            nn.Conv2d(channels[level], channels[level], 1, bias=bias)
            for level in range(3)
        ])
        self.tcfm = nn.ModuleList([
            TrajectoryCalibratedCSFFGate(
                channels[level], horizontal_kernel=tcfm_horizontal_kernel
            )
            for level in range(3)
        ])
        # Preserve the scale of pre-trained Stage-1 features at initialization:
        # encoder and decoder evidence enter as their mean, then TCFM learns
        # trajectory-dependent modulation rather than a hard replacement.
        for projections in (
            self.stage1_encoder_project, self.stage1_decoder_project
        ):
            for projection in projections:
                nn.init.dirac_(projection.weight)
                if projection.bias is not None:
                    nn.init.zeros_(projection.bias)
        self.pair_decoder = PairwisePyramidDecoder(
            channels,
            blocks=blocks,
            horizontal_trajectory=horizontal_trajectory,
            bias=bias,
        )
        self.stage2_bridge = ResidualStateBridge(
            channels[0], channels[0], bias=bias
        )
        self.final_decoder = GlobalLocalPyramidDecoder(
            channels,
            blocks=blocks,
            horizontal_trajectory=horizontal_trajectory,
            bias=bias,
        )
        self.synthesis = LaplacianPyramidSynthesisHead(
            channels[:3], bias=bias
        )

    def _inject_stage1(self, pair_features, stage1_encoder, stage1_decoder):
        if not self.tcfm_enabled:
            return pair_features
        injected = list(pair_features)
        for level in range(3):
            current = pair_features[level]
            encoder_feature = self.stage1_encoder_project[level](stage1_encoder[level])
            decoder_feature = self.stage1_decoder_project[level](stage1_decoder[level])
            modulation = self.tcfm[level](
                current, encoder_feature, decoder_feature
            )
            injected[level] = current + 0.5 * modulation * (
                encoder_feature + decoder_feature
            )
        return injected

    def forward(self, input_image, pair_inputs, pair_starts, blend_fn,
                stage1_encoder_pairs=None, stage1_decoder_pairs=None):
        if self.tcfm_enabled and (
            stage1_encoder_pairs is None or stage1_decoder_pairs is None
        ):
            raise ValueError('TCFM requires Stage-1 encoder and decoder pyramids.')
        pair_features = []
        for index, pair_input in enumerate(pair_inputs):
            features = self.pair_encoder(pair_input)
            if self.tcfm_enabled:
                features = self._inject_stage1(
                    features,
                    stage1_encoder_pairs[index],
                    stage1_decoder_pairs[index],
                )
            pair_features.append(features)
        height = input_image.size(2)
        encoder_features = []
        for level, scale in enumerate((1, 2, 4, 8)):
            encoder_features.append(blend_fn(
                [features[level] for features in pair_features],
                [start // scale for start in pair_starts],
                height // scale,
            ))
        decoder_features = self.pair_decoder(encoder_features)
        stage2_state, stage2_image = self.stage2_bridge(
            decoder_features[0], input_image
        )
        final_features = self.final_decoder(
            input_image, stage2_state, encoder_features, decoder_features
        )
        restored = self.synthesis(final_features, input_image)
        return restored, stage2_image, encoder_features, decoder_features


class TriStripeContextPyramid(nn.Module):
    """Stage-1 encoder/decoder designed for full-width TriStripe inputs.

    Unlike the inherited quadrant U-Net, this shared pyramid receives both a
    global vertical coordinate and a stripe-local coordinate.  It returns the
    same three-scale encoder/decoder interface required by downstream TCFM.
    """

    def __init__(self, n_feat, scale_unetfeats, blocks=2,
                 horizontal_trajectory=False, bias=True):
        super(TriStripeContextPyramid, self).__init__()
        self.channels = [
            n_feat,
            n_feat + scale_unetfeats,
            n_feat + scale_unetfeats * 2,
        ]
        c1, c2, c3 = self.channels
        block_type = (
            HorizontalTrajectoryBlock if horizontal_trajectory
            else DirectionalPyramidBlock
        )
        # RGB + global vertical coordinate + stripe-local vertical coordinate.
        self.stem = nn.Sequential(
            nn.Conv2d(5, c1, 3, padding=1, bias=bias), nn.PReLU(c1),
        )
        self.encoder_full = nn.Sequential(*[
            block_type(c1, bias=bias) for _ in range(blocks)
        ])
        self.down_full_half = nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=bias)
        self.encoder_half = nn.Sequential(*[
            block_type(c2, bias=bias) for _ in range(blocks)
        ])
        self.down_half_quarter = nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=bias)
        self.encoder_quarter = nn.Sequential(*[
            block_type(c3, bias=bias) for _ in range(blocks)
        ])

        self.decoder_quarter = nn.Sequential(*[
            block_type(c3, bias=bias) for _ in range(blocks)
        ])
        self.half_fuse = nn.Sequential(
            nn.Conv2d(c2 + c3, c2, 1, bias=bias), nn.PReLU(c2),
        )
        self.decoder_half = nn.Sequential(*[
            block_type(c2, bias=bias) for _ in range(blocks)
        ])
        self.full_fuse = nn.Sequential(
            nn.Conv2d(c1 + c2, c1, 1, bias=bias), nn.PReLU(c1),
        )
        self.decoder_full = nn.Sequential(*[
            block_type(c1, bias=bias) for _ in range(blocks)
        ])

    @staticmethod
    def _resize(x, reference):
        return F.interpolate(x, size=reference.shape[-2:], mode='bilinear', align_corners=False)

    @staticmethod
    def _coordinates(image, global_start, full_height):
        batch, _, height, width = image.shape
        denom = max(1, int(full_height) - 1)
        global_y = (
            torch.arange(height, device=image.device, dtype=image.dtype)
            + float(global_start)
        ) / float(denom)
        global_y = global_y.mul(2.0).sub(1.0)
        local_y = torch.linspace(
            -1.0, 1.0, height, device=image.device, dtype=image.dtype
        )
        coordinates = torch.stack([global_y, local_y], dim=0).view(1, 2, height, 1)
        return coordinates.expand(batch, -1, -1, width)

    def encode(self, stripe_image, global_start, full_height):
        x = torch.cat([
            stripe_image,
            self._coordinates(stripe_image, global_start, full_height),
        ], dim=1)
        full = self.encoder_full(self.stem(x))
        half = self.encoder_half(self.down_full_half(full))
        quarter = self.encoder_quarter(self.down_half_quarter(half))
        return [full, half, quarter]

    def decode(self, encoder_features):
        full, half, quarter = encoder_features
        quarter = self.decoder_quarter(quarter)
        half = self.decoder_half(self.half_fuse(torch.cat([
            half, self._resize(quarter, half)
        ], dim=1)))
        full = self.decoder_full(self.full_fuse(torch.cat([
            full, self._resize(half, full)
        ], dim=1)))
        return [full, half, quarter]


class InterStageResidualReassembly(nn.Module):
    """Reassemble residual evidence from the three MPRNet image predictions.

    This module is deliberately image-restoration-only: it predicts neither an
    edge map nor a segmentation map.  The three progressive predictions expose
    where each restoration stage is still changing the image.  Their signed
    residuals are encoded at full, half, and quarter resolution with
    pixel-unshuffle/pixel-shuffle reassembly.  Its residual head starts at a
    deliberately tiny magnitude so it is stable yet can learn in a short run.
    """

    def __init__(
        self,
        hidden_channels=24,
        num_blocks=1,
        delta_init=1.0e-4,
        bias=True,
    ):
        super(InterStageResidualReassembly, self).__init__()
        hidden_channels = max(8, int(hidden_channels))
        num_blocks = max(1, int(num_blocks))
        self.hidden_channels = hidden_channels
        delta_init = float(delta_init)
        if delta_init < 0:
            raise ValueError('IRR delta_init must be non-negative.')

        def refine_stack(channels):
            blocks = []
            for _ in range(num_blocks):
                blocks.extend([
                    nn.Conv2d(
                        channels, channels, 3, padding=1,
                        groups=channels, bias=bias,
                    ),
                    nn.PReLU(channels),
                    nn.Conv2d(channels, channels, 1, bias=bias),
                    nn.PReLU(channels),
                ])
            return nn.Sequential(*blocks)

        # [base, stage1-input, stage2-stage1, base-stage2] = 12 RGB channels.
        self.full_encode = nn.Sequential(
            nn.Conv2d(12, hidden_channels, 3, padding=1, bias=bias),
            nn.PReLU(hidden_channels),
            refine_stack(hidden_channels),
        )
        self.half_encode = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels, 1, bias=bias),
            nn.PReLU(hidden_channels),
            refine_stack(hidden_channels),
        )
        self.quarter_encode = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels, 1, bias=bias),
            nn.PReLU(hidden_channels),
            refine_stack(hidden_channels),
        )
        self.quarter_up = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 4, 1, bias=bias),
            nn.PReLU(hidden_channels * 4),
            nn.PixelShuffle(2),
        )
        self.half_fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 1, bias=bias),
            nn.PReLU(hidden_channels),
            refine_stack(hidden_channels),
        )
        self.half_up = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 4, 1, bias=bias),
            nn.PReLU(hidden_channels * 4),
            nn.PixelShuffle(2),
        )
        self.full_fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 1, bias=bias),
            nn.PReLU(hidden_channels),
            refine_stack(hidden_channels),
        )
        self.delta = nn.Conv2d(hidden_channels, 3, 3, padding=1, bias=bias)

        # A tiny non-zero head keeps the initial image perturbation negligible
        # while allowing the entire reassembly pyramid to receive gradients on
        # the first batch.  A zero head is safer but starts the deep IRR path
        # one optimizer step late, which is undesirable for a 25-epoch screen.
        nn.init.normal_(self.delta.weight, mean=0.0, std=delta_init)
        if self.delta.bias is not None:
            nn.init.zeros_(self.delta.bias)
        self.register_buffer('last_delta_abs', torch.zeros(()), persistent=False)

    @staticmethod
    def _pad_to_factor(x, factor=4):
        height, width = x.shape[-2:]
        pad_h = (-height) % factor
        pad_w = (-width) % factor
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        return x, height, width

    def forward(self, base_image, input_image, stage1_image, stage2_image):
        evidence = torch.cat([
            base_image,
            stage1_image - input_image,
            stage2_image - stage1_image,
            base_image - stage2_image,
        ], dim=1)
        evidence, original_height, original_width = self._pad_to_factor(evidence)

        full = self.full_encode(evidence)
        half = self.half_encode(F.pixel_unshuffle(full, 2))
        quarter = self.quarter_encode(F.pixel_unshuffle(half, 2))

        half = self.half_fuse(torch.cat([half, self.quarter_up(quarter)], dim=1))
        full = self.full_fuse(torch.cat([full, self.half_up(half)], dim=1))
        delta = self.delta(full)[..., :original_height, :original_width]
        with torch.no_grad():
            self.last_delta_abs.copy_(delta.detach().abs().mean())
        return base_image + delta


class FrequencyResidualRefiner(nn.Module):
    """Low-channel Fourier residual refinement for the final restored RGB image."""

    def __init__(
        self,
        channels=3,
        hidden_channels=16,
        band_gate=False,
        band_low_start=0.05,
        band_low_end=0.10,
        band_high_start=0.30,
        band_high_end=0.40,
        disagreement_gate=False,
        disagreement_threshold=0.0120,
        disagreement_temperature=0.0020,
        disagreement_min_strength=0.10,
        bias=True,
    ):
        super(FrequencyResidualRefiner, self).__init__()
        hidden_channels = max(4, int(hidden_channels))
        self.band_gate_enabled = bool(band_gate)
        self.band_low_start = float(band_low_start)
        self.band_low_end = float(band_low_end)
        self.band_high_start = float(band_high_start)
        self.band_high_end = float(band_high_end)
        self.disagreement_gate_enabled = bool(disagreement_gate)
        self.disagreement_threshold = float(disagreement_threshold)
        self.disagreement_temperature = float(disagreement_temperature)
        self.disagreement_min_strength = float(disagreement_min_strength)
        if not (
            0.0 <= self.band_low_start < self.band_low_end
            < self.band_high_start < self.band_high_end <= 0.5
        ):
            raise ValueError('Frequency band edges must satisfy 0 <= low_start < low_end < high_start < high_end <= 0.5.')
        if self.band_gate_enabled and self.disagreement_gate_enabled:
            raise ValueError('Use either the learned band gate or the stage-disagreement gate, not both.')
        if self.disagreement_temperature <= 0:
            raise ValueError('Frequency disagreement temperature must be positive.')
        if not 0.0 <= self.disagreement_min_strength <= 1.0:
            raise ValueError('Frequency disagreement minimum strength must be in [0, 1].')
        self.pre = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1, bias=bias),
            nn.GELU(),
        )
        frequency_channels = hidden_channels * 2
        self.frequency_mixer = nn.Sequential(
            nn.Conv2d(
                frequency_channels, frequency_channels, 1, bias=bias
            ),
            nn.GELU(),
            nn.Conv2d(
                frequency_channels, frequency_channels, 1, bias=bias
            ),
        )
        self.post = nn.Conv2d(
            hidden_channels, channels, 3, padding=1, bias=bias
        )
        # Zero-conv keeps the initial output identical while allowing gradients
        # to reach the output projection from the first optimization step.
        nn.init.zeros_(self.post.weight)
        if self.post.bias is not None:
            nn.init.zeros_(self.post.bias)
        self.residual_scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        if self.band_gate_enabled:
            self.strength_gate = nn.Sequential(
                nn.Linear(4, 8, bias=True),
                nn.SiLU(),
                nn.Linear(8, 1, bias=True),
            )
            # Gate starts at one, so the zero-conv remains an exact identity.
            nn.init.zeros_(self.strength_gate[-1].weight)
            nn.init.zeros_(self.strength_gate[-1].bias)
        else:
            self.strength_gate = None
        self.frequency_band_enabled = (
            self.band_gate_enabled or self.disagreement_gate_enabled
        )

    @staticmethod
    def _smoothstep(x):
        x = x.clamp(0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _radial_masks(self, height, width, device, dtype):
        fy = torch.fft.fftfreq(height, device=device, dtype=dtype).view(height, 1)
        fx = torch.fft.rfftfreq(width, device=device, dtype=dtype).view(1, width // 2 + 1)
        radius = torch.sqrt(fy.square() + fx.square())
        low_rise = self._smoothstep(
            (radius - self.band_low_start) / (self.band_low_end - self.band_low_start)
        )
        high_fall = 1.0 - self._smoothstep(
            (radius - self.band_high_start) / (self.band_high_end - self.band_high_start)
        )
        band = low_rise * high_fall
        low = 1.0 - low_rise
        high = 1.0 - high_fall
        return band, low, high

    @staticmethod
    def _masked_mean(power, mask):
        return (power * mask).sum(dim=(1, 2)) / mask.sum().clamp_min(1e-6)

    def forward(self, x, reference=None):
        input_dtype = x.dtype
        features = self.pre(x)
        spatial_size = features.shape[-2:]

        # cuFFT half precision does not support every image size (including 320).
        spectrum = torch.fft.rfft2(features.float(), norm='ortho')
        frequency_features = torch.cat([spectrum.real, spectrum.imag], dim=1)
        frequency_features = self.frequency_mixer(frequency_features)
        real, imag = frequency_features.chunk(2, dim=1)
        mixed_spectrum = torch.complex(real.float(), imag.float())
        strength = 1.0
        if self.frequency_band_enabled:
            band, low, high = self._radial_masks(
                spatial_size[0], spatial_size[1], x.device, torch.float32
            )
            band = band.unsqueeze(0).unsqueeze(0)
            mixed_spectrum = mixed_spectrum * band

            if self.band_gate_enabled:
                base_spectrum = torch.fft.rfft2(x.float(), norm='ortho')
                power = base_spectrum.abs().square().mean(dim=1)
                stats = torch.stack([
                    torch.log1p(self._masked_mean(power, low)),
                    torch.log1p(self._masked_mean(power, band.squeeze(0).squeeze(0))),
                    torch.log1p(self._masked_mean(power, high)),
                    x.float().std(dim=(1, 2, 3)),
                ], dim=1)
                strength = 1.0 + torch.tanh(self.strength_gate(stats.float()))
                strength = strength.view(-1, 1, 1, 1).to(dtype=input_dtype)
            elif self.disagreement_gate_enabled:
                if reference is None:
                    raise ValueError('Stage-disagreement frequency gate requires a Stage-2 reference image.')
                disagreement = (x.float() - reference.float()).abs().mean(
                    dim=(1, 2, 3), keepdim=True
                )
                strength = self.disagreement_min_strength + (
                    1.0 - self.disagreement_min_strength
                ) * torch.sigmoid(
                    (disagreement - self.disagreement_threshold)
                    / self.disagreement_temperature
                )
                strength = strength.to(dtype=input_dtype)
        residual = torch.fft.irfft2(
            mixed_spectrum, s=spatial_size, norm='ortho'
        ).to(dtype=input_dtype)
        residual = self.post(residual)
        return x + residual * torch.tanh(self.residual_scale) * strength


##########################################################################
class MPRNet(nn.Module):
    def __init__(
        self,
        in_c=3,
        out_c=3,
        n_feat=96,
        scale_unetfeats=48,
        scale_orsnetfeats=32,
        num_cab=8,
        kernel_size=3,
        reduction=4,
        bias=False,
        edge_refine=False,
        edge_refine_mid=24,
        frequency_refine=False,
        frequency_refine_hidden=16,
        frequency_band_gate=False,
        frequency_band_low_start=0.05,
        frequency_band_low_end=0.10,
        frequency_band_high_start=0.30,
        frequency_band_high_end=0.40,
        frequency_disagreement_gate=False,
        frequency_disagreement_threshold=0.0120,
        frequency_disagreement_temperature=0.0020,
        frequency_disagreement_min_strength=0.10,
        light_adapt=False,
        light_target_luma=0.50,
        light_min_gain=0.75,
        light_max_gain=1.80,
        light_min_gamma=0.65,
        light_max_gamma=1.35,
        light_blend=0.75,
        light_dark_threshold=0.45,
        light_bright_threshold=0.70,
        trajectory_lightweight=False,
        stage1_tristripe=False,
        stage1_stripes=3,
        stage1_overlap=0.25,
        stage1_stripe_transformer=False,
        stage1_stripe_transformer_heads=4,
        stage2_stripe_transformer=False,
        stage2_stripe_transformer_heads=4,
        stage2_overlap=0.0,
        stage2_pair_aligned=False,
        stage2_tmt=False,
        stage2_tmt_horizontal_kernel=15,
        stage2_tmt_vertical_kernel=5,
        gated_csff=False,
        gated_csff_horizontal_kernel=7,
        stitch_trajectory_alignment=False,
        stam_horizontal_radius=6,
        stam_vertical_radius=1,
        stam_hidden_channels=32,
        dual_domain_orsnet=False,
        ddrb_residual_init=0.3,
        simple_gated_orsnet=False,
        sgors_blocks=8,
        sgors_residual_init=0.10,
        hybrid_simple_gated_orsnet=False,
        hsgors_blocks=4,
        hsgors_residual_init=0.05,
        hsgors_adapter_scale_init=0.02,
        hsgors_v2=False,
        hsgors_v2_horizontal_kernel=15,
        hsgors_v2_vertical_kernel=5,
        stage3_tepr=False,
        stage3_tepr_transition=False,
        stage3_tepr_freeze_legacy=True,
        stage3_tepr_blocks=3,
        stage3_tepr_horizontal_kernel=15,
        stage3_tepr_vertical_kernel=5,
        stage3_trr=False,
        stage3_trr_horizontal_kernel=15,
        stage3_trr_vertical_kernel=5,
        stage3_trr_hidden_channels=32,
        stage2_naf_blocks=2,
        stage2_horizontal_kernel=15,
        stage2_vertical_kernel=5,
        stage3_naf_blocks=3,
        tpt_rnet_core=False,
        tptr_blocks=2,
        tptr_tcfm=False,
        tptr_tcfm_horizontal_kernel=7,
        tristripe_context_pyramid=False,
        tristripe_context_blocks=2,
        horizontal_trajectory_blocks=False,
        hmt_net_core=False,
        hmt_blocks=3,
        evidence_canvas_net=False,
        evidence_canvas_blocks=2,
        tfr_net_core=False,
        tfr_blocks=8,
        tfr_horizontal_kernel=15,
        tfr_vertical_kernel=5,
        tfr_refiner_steps=3,
        stripe_topology_reconstruction=False,
        strd_hidden=32,
        strd_num_blocks=2,
        interstage_residual_reassembly=False,
        irr_hidden=24,
        irr_num_blocks=1,
        irr_delta_init=1.0e-4,
    ):
        super(MPRNet, self).__init__()

        act=nn.PReLU()
        self.edge_refine_enabled = bool(edge_refine)
        self.frequency_refine_enabled = bool(frequency_refine)
        self.light_adapt_enabled = bool(light_adapt)
        self.trajectory_lightweight = bool(trajectory_lightweight)
        self.stage1_tristripe = bool(stage1_tristripe)
        self.stage1_stripes = max(1, int(stage1_stripes))
        self.stage1_overlap = float(stage1_overlap)
        if not 0.0 <= self.stage1_overlap < 0.5:
            raise ValueError('stage1_overlap must be in [0.0, 0.5).')
        self.stage1_stripe_transformer_enabled = bool(stage1_stripe_transformer)
        self.stage2_stripe_transformer_enabled = bool(stage2_stripe_transformer)
        self.stage2_overlap = float(stage2_overlap)
        if not 0.0 <= self.stage2_overlap < 0.5:
            raise ValueError('stage2_overlap must be in [0.0, 0.5).')
        self.stage2_pair_aligned = bool(stage2_pair_aligned)
        self.stage2_tmt_enabled = bool(stage2_tmt)
        self.stitch_trajectory_alignment_enabled = bool(stitch_trajectory_alignment)
        self.interstage_residual_reassembly_enabled = bool(
            interstage_residual_reassembly
        )
        self.tpt_rnet_core_enabled = bool(tpt_rnet_core)
        self.tristripe_context_pyramid_enabled = bool(
            tristripe_context_pyramid
        )
        self.horizontal_trajectory_blocks = bool(horizontal_trajectory_blocks)
        self.hmt_net_core_enabled = bool(hmt_net_core)
        self.evidence_canvas_net_enabled = bool(evidence_canvas_net)
        self.tfr_net_core_enabled = bool(tfr_net_core)
        self.stripe_topology_reconstruction_enabled = bool(
            stripe_topology_reconstruction
        )
        self.gated_csff_enabled = bool(gated_csff)
        self.dual_domain_orsnet_enabled = bool(dual_domain_orsnet)
        self.simple_gated_orsnet_enabled = bool(simple_gated_orsnet)
        self.hybrid_simple_gated_orsnet_enabled = bool(hybrid_simple_gated_orsnet)
        self.hsgors_v2_enabled = bool(hsgors_v2)
        self.stage3_tepr_enabled = bool(stage3_tepr)
        self.stage3_tepr_transition_enabled = bool(stage3_tepr_transition)
        self.stage3_tepr_freeze_legacy = bool(stage3_tepr_freeze_legacy)
        self.stage3_trr_enabled = bool(stage3_trr)
        if self.stage3_tepr_transition_enabled and not self.stage3_tepr_enabled:
            raise ValueError('stage3_tepr_transition requires stage3_tepr.')
        if self.stitch_trajectory_alignment_enabled and (
            not self.stage1_tristripe or not self.stage2_pair_aligned
        ):
            raise ValueError(
                'stitch_trajectory_alignment requires Stage-1 TriStripe and Stage-2 PairAligned.'
            )
        if self.stitch_trajectory_alignment_enabled and self.trajectory_lightweight:
            raise ValueError(
                'stitch_trajectory_alignment is designed for the original Stage-2 Encoder; '
                'disable trajectory_lightweight to use it.'
            )
        if self.tpt_rnet_core_enabled and (
            not self.stage1_tristripe or not self.stage2_pair_aligned
        ):
            raise ValueError(
                'tpt_rnet_core requires Stage-1 TriStripe and Stage-2 PairAligned.'
            )
        if self.tpt_rnet_core_enabled and self.trajectory_lightweight:
            raise ValueError(
                'tpt_rnet_core replaces the original Stage-2/3 backbone; '
                'disable trajectory_lightweight.'
            )
        if self.tpt_rnet_core_enabled and self.stitch_trajectory_alignment_enabled:
            raise ValueError('tpt_rnet_core does not use STAM; disable it.')
        if self.tristripe_context_pyramid_enabled and (
            not (
                self.tpt_rnet_core_enabled
                or self.hmt_net_core_enabled
                or self.tfr_net_core_enabled
            )
            or not self.stage1_tristripe
            or not self.stage2_pair_aligned
        ):
            raise ValueError(
                'tristripe_context_pyramid requires the TPTR, HMT, or TFR '
                'TriStripe and PairAligned analysis path.'
            )
        if self.hmt_net_core_enabled and (
            not self.tristripe_context_pyramid_enabled
            or not self.stage1_tristripe
            or not self.stage2_pair_aligned
        ):
            raise ValueError(
                'hmt_net_core requires the new TriStripe Context Pyramid and '
                'the PairAligned analysis path.'
            )
        if self.evidence_canvas_net_enabled and (
            not self.stage1_tristripe
            or not self.stage2_pair_aligned
            or self.tpt_rnet_core_enabled
            or self.hmt_net_core_enabled
            or self.tfr_net_core_enabled
        ):
            raise ValueError(
                'evidence_canvas_net requires TriStripe + PairAligned and '
                'cannot be combined with TPTR, HMT, or TFR cores.'
            )
        if self.tfr_net_core_enabled and (
            not self.stage1_tristripe
            or not self.stage2_pair_aligned
            or self.tpt_rnet_core_enabled
            or self.hmt_net_core_enabled
            or self.evidence_canvas_net_enabled
        ):
            raise ValueError(
                'tfr_net_core requires TriStripe + PairAligned and cannot be '
                'combined with TPTR, HMT, or EvidenceCanvas cores.'
            )
        if self.stage2_tmt_enabled and (
            self.trajectory_lightweight
            or self.tpt_rnet_core_enabled
            or self.hmt_net_core_enabled
            or self.evidence_canvas_net_enabled
            or self.tfr_net_core_enabled
        ):
            raise ValueError(
                'stage2_tmt replaces the original Stage-2 path; disable '
                'trajectory_lightweight, TPTR, HMT, EvidenceCanvas, and TFR cores.'
            )
        if self.gated_csff_enabled and self.trajectory_lightweight:
            raise ValueError(
                'gated_csff is designed for the original Stage-2 Encoder; '
                'disable trajectory_lightweight to use it.'
            )
        if self.dual_domain_orsnet_enabled and self.trajectory_lightweight:
            raise ValueError(
                'dual_domain_orsnet replaces the original ORSNet; '
                'disable trajectory_lightweight to use it.'
            )
        if self.simple_gated_orsnet_enabled and self.trajectory_lightweight:
            raise ValueError(
                'simple_gated_orsnet replaces the original ORSNet; '
                'disable trajectory_lightweight to use it.'
            )
        if self.simple_gated_orsnet_enabled and self.dual_domain_orsnet_enabled:
            raise ValueError(
                'Use either simple_gated_orsnet or dual_domain_orsnet, not both.'
            )
        if self.hybrid_simple_gated_orsnet_enabled and self.trajectory_lightweight:
            raise ValueError(
                'hybrid_simple_gated_orsnet augments the original ORSNet; '
                'disable trajectory_lightweight to use it.'
            )
        if self.hsgors_v2_enabled and self.trajectory_lightweight:
            raise ValueError(
                'hsgors_v2 augments the original ORSNet; '
                'disable trajectory_lightweight to use it.'
            )
        if self.hybrid_simple_gated_orsnet_enabled and (
            self.simple_gated_orsnet_enabled
            or self.dual_domain_orsnet_enabled
            or self.hsgors_v2_enabled
        ):
            raise ValueError(
                'Use only one Stage-3 ORS variant: hybrid SG-ORS, HSGORS-v2, '
                'simple SG-ORS, or dual-domain ORS.'
            )
        if self.hsgors_v2_enabled and (
            self.simple_gated_orsnet_enabled or self.dual_domain_orsnet_enabled
        ):
            raise ValueError(
                'Use only one Stage-3 ORS variant: HSGORS-v2, simple SG-ORS, '
                'or dual-domain ORS.'
            )
        if self.stage3_tepr_enabled and (
            self.simple_gated_orsnet_enabled
            or self.hybrid_simple_gated_orsnet_enabled
            or self.hsgors_v2_enabled
            or self.dual_domain_orsnet_enabled
            or self.trajectory_lightweight
            or self.tpt_rnet_core_enabled
            or self.hmt_net_core_enabled
            or self.evidence_canvas_net_enabled
            or self.tfr_net_core_enabled
            or self.stage3_trr_enabled
        ):
            raise ValueError(
                'stage3_tepr is a standalone Stage-3 replacement; disable '
                'all other Stage-3 and replacement cores.'
            )
        if self.stage3_trr_enabled and (
            self.simple_gated_orsnet_enabled
            or self.hybrid_simple_gated_orsnet_enabled
            or self.hsgors_v2_enabled
            or self.dual_domain_orsnet_enabled
            or self.trajectory_lightweight
            or self.tpt_rnet_core_enabled
            or self.hmt_net_core_enabled
            or self.evidence_canvas_net_enabled
        ):
            raise ValueError(
                'stage3_trr replaces the original Stage-3 recovery path; '
                'disable other Stage-3/core replacements.'
            )
        if self.light_adapt_enabled:
            self.light_adapter = LightAdaptiveRestorationModule(
                target_luma=light_target_luma,
                min_gain=light_min_gain,
                max_gain=light_max_gain,
                min_gamma=light_min_gamma,
                max_gamma=light_max_gamma,
                blend=light_blend,
                dark_threshold=light_dark_threshold,
                bright_threshold=light_bright_threshold,
            )
        else:
            self.light_adapter = None
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        stage1_bottleneck_channels = n_feat + scale_unetfeats * 2
        self.stage1_stripe_transformer = (
            RowWiseStripeTransformer(
                stage1_bottleneck_channels,
                num_heads=stage1_stripe_transformer_heads,
                bias=bias,
            )
            if self.stage1_stripe_transformer_enabled
            else nn.Identity()
        )

        if self.trajectory_lightweight:
            self.stage2_encoder = NAFEncoder(
                n_feat,
                scale_unetfeats,
                csff=True,
                num_blocks=stage2_naf_blocks,
                directional_context=True,
                horizontal_kernel=stage2_horizontal_kernel,
                vertical_kernel=stage2_vertical_kernel,
                bias=True,
            )
            self.stage2_decoder = NAFDecoder(
                n_feat,
                scale_unetfeats,
                num_blocks=stage2_naf_blocks,
                bias=True,
            )
            self.stage3_orsnet = MultiScaleFusionNAF(
                n_feat,
                scale_orsnetfeats,
                scale_unetfeats,
                num_blocks=stage3_naf_blocks,
                bias=True,
            )
        else:
            self.stage2_encoder = Encoder(
                n_feat,
                kernel_size,
                reduction,
                act,
                bias,
                scale_unetfeats,
                csff=True,
                gated_csff=self.gated_csff_enabled,
                gated_csff_horizontal_kernel=gated_csff_horizontal_kernel,
            )
            self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
            base_stage3_orsnet = None
            if (not self.stage3_tepr_enabled) or self.stage3_tepr_transition_enabled:
                base_stage3_orsnet = ORSNet(
                    n_feat,
                    scale_orsnetfeats,
                    kernel_size,
                    reduction,
                    act,
                    bias,
                    scale_unetfeats,
                    num_cab,
                )
            if self.stage3_tepr_transition_enabled:
                # The legacy path exists only during structural migration. It
                # is retained under the original checkpoint names so a proven
                # HSGORS-v2 checkpoint can be loaded without key rewriting.
                self.stage3_orsnet = HybridTrajectoryGatedORSNet(
                    base_stage3_orsnet,
                    n_feat + scale_orsnetfeats,
                    num_blocks=hsgors_blocks,
                    residual_init=hsgors_residual_init,
                    adapter_scale_init=hsgors_adapter_scale_init,
                    horizontal_kernel=hsgors_v2_horizontal_kernel,
                    vertical_kernel=hsgors_v2_vertical_kernel,
                )
            elif self.stage3_tepr_enabled:
                self.stage3_orsnet = None
            elif self.hsgors_v2_enabled:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(torch.initial_seed() + 6209)
                    self.stage3_orsnet = HybridTrajectoryGatedORSNet(
                        base_stage3_orsnet,
                        n_feat + scale_orsnetfeats,
                        num_blocks=hsgors_blocks,
                        residual_init=hsgors_residual_init,
                        adapter_scale_init=hsgors_adapter_scale_init,
                        horizontal_kernel=hsgors_v2_horizontal_kernel,
                        vertical_kernel=hsgors_v2_vertical_kernel,
                    )
            elif self.hybrid_simple_gated_orsnet_enabled:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(torch.initial_seed() + 6191)
                    self.stage3_orsnet = HybridSimpleGatedORSNet(
                        base_stage3_orsnet,
                        n_feat + scale_orsnetfeats,
                        num_blocks=hsgors_blocks,
                        residual_init=hsgors_residual_init,
                        adapter_scale_init=hsgors_adapter_scale_init,
                    )
            elif self.simple_gated_orsnet_enabled:
                # Construct in an isolated RNG stream.  The base ORSNet is
                # still instantiated first, so the edge/tail initialization
                # remains paired with M0 when only SG-ORS is enabled.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(torch.initial_seed() + 6173)
                    self.stage3_orsnet = SimpleGatedORSNet(
                        base_stage3_orsnet,
                        n_feat + scale_orsnetfeats,
                        num_blocks=sgors_blocks,
                        residual_init=sgors_residual_init,
                    )
            elif self.dual_domain_orsnet_enabled:
                self.stage3_orsnet = DualDomainSelectiveORSNet(
                    base_stage3_orsnet,
                    n_feat + scale_orsnetfeats,
                    residual_init=ddrb_residual_init,
                )
            elif self.stage3_trr_enabled:
                self.stage3_orsnet = TrajectoryResidualRestorer(
                    base_stage3_orsnet,
                    n_feat + scale_orsnetfeats,
                    horizontal_kernel=stage3_trr_horizontal_kernel,
                    vertical_kernel=stage3_trr_vertical_kernel,
                    hidden_channels=stage3_trr_hidden_channels,
                )
            else:
                self.stage3_orsnet = base_stage3_orsnet

        stage2_bottleneck_channels = n_feat + scale_unetfeats * 2
        self.stitch_trajectory_alignment = (
            StitchTrajectoryAlignmentPyramid(
                n_feat,
                [n_feat, n_feat + scale_unetfeats, n_feat + scale_unetfeats * 2],
                horizontal_radius=stam_horizontal_radius,
                vertical_radius=stam_vertical_radius,
                hidden_channels=stam_hidden_channels,
            )
            if self.stitch_trajectory_alignment_enabled
            else None
        )
        self.stage2_stripe_transformer = (
            RowWiseStripeTransformer(
                stage2_bottleneck_channels,
                num_heads=stage2_stripe_transformer_heads,
                bias=bias,
            )
            if self.stage2_stripe_transformer_enabled
            else nn.Identity()
        )

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)
        
        self.concat12  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        self.concat23  = conv(n_feat*2, n_feat+scale_orsnetfeats, kernel_size, bias=bias)
        if self.edge_refine_enabled:
            self.edge_refine = StitchEdgeGuidedRefinement(
                n_feat + scale_orsnetfeats,
                mid_channels=edge_refine_mid,
                kernel_size=kernel_size,
                bias=bias,
            )
        else:
            self.edge_refine = None
        self.stage3_tepr = (
            TrajectoryEvidencePyramidRecovery(
                n_feat + scale_orsnetfeats,
                [n_feat, n_feat + scale_unetfeats, n_feat + scale_unetfeats * 2],
                num_blocks=stage3_tepr_blocks,
                horizontal_kernel=stage3_tepr_horizontal_kernel,
                vertical_kernel=stage3_tepr_vertical_kernel,
                bias=True,
            )
            if self.stage3_tepr_enabled
            else None
        )
        self.tepr_head = (
            TrajectoryEvidenceSynthesisHead(
                n_feat + scale_orsnetfeats,
                out_channels=out_c,
                horizontal_kernel=stage3_tepr_horizontal_kernel,
                vertical_kernel=stage3_tepr_vertical_kernel,
                bias=True,
            )
            if self.stage3_tepr_enabled
            else None
        )
        # In transition mode ``tail`` is the frozen legacy RGB projection. In
        # pure TEPR mode it is absent from the module tree and checkpoint.
        self.tail = (
            conv(n_feat+scale_orsnetfeats, out_c, kernel_size, bias=bias)
            if (not self.stage3_tepr_enabled or self.stage3_tepr_transition_enabled)
            else None
        )
        if self.stage3_tepr_enabled:
            self.register_buffer(
                'stage3_tepr_transition_alpha',
                torch.tensor(0.0 if self.stage3_tepr_transition_enabled else 1.0),
                persistent=True,
            )
            if self.stage3_tepr_transition_enabled and self.stage3_tepr_freeze_legacy:
                for legacy_module in (self.stage3_orsnet, self.tail):
                    for parameter in legacy_module.parameters():
                        parameter.requires_grad_(False)
        self.stage3_synthesis = (
            TrajectoryResidualSynthesisHead(
                n_feat + scale_orsnetfeats,
                hidden_channels=stage3_trr_hidden_channels,
                horizontal_kernel=stage3_trr_horizontal_kernel,
                bias=True,
            )
            if self.stage3_trr_enabled and not self.stage3_tepr_enabled
            else None
        )
        self.stripe_topology_reconstruction = (
            StripeTopologyReconstructionDecoder(
                n_feat + scale_orsnetfeats,
                [n_feat, n_feat + scale_unetfeats, n_feat + scale_unetfeats * 2],
                hidden_channels=strd_hidden,
                num_blocks=strd_num_blocks,
                bias=True,
            )
            if self.stripe_topology_reconstruction_enabled
            else None
        )
        self.interstage_residual_reassembly = (
            InterStageResidualReassembly(
                hidden_channels=irr_hidden,
                num_blocks=irr_num_blocks,
                delta_init=irr_delta_init,
                bias=True,
            )
            if self.interstage_residual_reassembly_enabled
            else nn.Identity()
        )
        self.frequency_refine = (
            FrequencyResidualRefiner(
                channels=out_c,
                hidden_channels=frequency_refine_hidden,
                band_gate=frequency_band_gate,
                band_low_start=frequency_band_low_start,
                band_low_end=frequency_band_low_end,
                band_high_start=frequency_band_high_start,
                band_high_end=frequency_band_high_end,
                disagreement_gate=frequency_disagreement_gate,
                disagreement_threshold=frequency_disagreement_threshold,
                disagreement_temperature=frequency_disagreement_temperature,
                disagreement_min_strength=frequency_disagreement_min_strength,
                bias=True,
            )
            if self.frequency_refine_enabled
            else nn.Identity()
        )

        if self.stage2_tmt_enabled:
            self.stage2_encoder = TrajectoryMemoryTransferEncoder(
                n_feat,
                kernel_size,
                reduction,
                act,
                bias,
                scale_unetfeats,
                gated_csff=self.gated_csff_enabled,
                gated_csff_horizontal_kernel=gated_csff_horizontal_kernel,
                horizontal_kernel=stage2_tmt_horizontal_kernel,
                vertical_kernel=stage2_tmt_vertical_kernel,
            )
            self.stage2_decoder = TrajectoryMemoryTransferDecoder(
                n_feat,
                kernel_size,
                reduction,
                act,
                bias,
                scale_unetfeats,
                horizontal_kernel=stage2_tmt_horizontal_kernel,
                vertical_kernel=stage2_tmt_vertical_kernel,
            )
            self.sam23 = TrajectoryMemorySAM(
                n_feat,
                kernel_size=1,
                bias=bias,
                horizontal_kernel=stage2_tmt_horizontal_kernel,
                vertical_kernel=stage2_tmt_vertical_kernel,
            )

        # TPT-RNet keeps the proven TriStripe Stage-1 context encoder but
        # removes all inherited Stage-2/SAM/Stage-3/tail modules.  Registering
        # None removes their parameters from this model rather than merely
        # bypassing them in forward.
        self.tpt_rnet_core = None
        self.hmt_net_core = None
        self.evidence_canvas_core = None
        self.tfr_net_core = None
        self.stage1_bridge = None
        self.tristripe_context_pyramid = None
        if self.tpt_rnet_core_enabled and not self.hmt_net_core_enabled:
            self.tpt_rnet_core = TrajectoryPairReconstructionCore(
                n_feat,
                scale_unetfeats,
                blocks=tptr_blocks,
                tcfm=tptr_tcfm,
                tcfm_horizontal_kernel=tptr_tcfm_horizontal_kernel,
                horizontal_trajectory=self.horizontal_trajectory_blocks,
                bias=True,
            )
            self.shallow_feat2 = None
            self.shallow_feat3 = None
            self.stage2_encoder = None
            self.stage2_decoder = None
            self.stage2_stripe_transformer = None
            self.stitch_trajectory_alignment = None
            self.sam23 = None
            self.concat12 = None
            self.concat23 = None
            self.stage3_orsnet = None
            self.edge_refine = None
            self.tail = None
            self.stage3_synthesis = None
            self.stripe_topology_reconstruction = None
            self.interstage_residual_reassembly = nn.Identity()
        if self.tristripe_context_pyramid_enabled:
            self.tristripe_context_pyramid = TriStripeContextPyramid(
                n_feat,
                scale_unetfeats,
                blocks=tristripe_context_blocks,
                horizontal_trajectory=self.horizontal_trajectory_blocks,
                bias=True,
            )
            self.stage1_bridge = (
                FeatureMemoryBridge(n_feat, n_feat, bias=True)
                if (self.hmt_net_core_enabled or self.tfr_net_core_enabled)
                else ResidualStateBridge(n_feat, n_feat, bias=True)
            )
            self.shallow_feat1 = None
            self.stage1_encoder = None
            self.stage1_decoder = None
            self.stage1_stripe_transformer = None
            self.sam12 = None
        if self.hmt_net_core_enabled:
            self.hmt_net_core = SharedMemoryReconstructionCore(
                n_feat,
                scale_unetfeats,
                blocks=hmt_blocks,
                tcfm_horizontal_kernel=tptr_tcfm_horizontal_kernel,
                edge_refine=self.edge_refine_enabled,
                edge_refine_mid=edge_refine_mid,
                bias=True,
            )
            self.tpt_rnet_core = None
            # HMT-Net uses the new TriStripe context pyramid and its shared
            # memory reconstruction core exclusively.  Remove every legacy
            # Stage-2/3 reconstruction module so that parameter counts,
            # checkpoint reports, and architecture claims all describe the
            # network that is actually evaluated.
            self.shallow_feat2 = None
            self.shallow_feat3 = None
            self.stage2_encoder = None
            self.stage2_decoder = None
            self.stage2_stripe_transformer = None
            self.stitch_trajectory_alignment = None
            self.sam23 = None
            self.concat12 = None
            self.concat23 = None
            self.stage3_orsnet = None
            self.edge_refine = None
            self.tail = None
            self.stage3_synthesis = None
            self.stripe_topology_reconstruction = None
            self.interstage_residual_reassembly = nn.Identity()
        if self.tfr_net_core_enabled:
            self.tfr_net_core = TriStripeFrequencyResidualCore(
                n_feat,
                scale_unetfeats,
                blocks=tfr_blocks,
                horizontal_kernel=tfr_horizontal_kernel,
                vertical_kernel=tfr_vertical_kernel,
                edge_refine=self.edge_refine_enabled,
                edge_refine_mid=edge_refine_mid,
                refiner_steps=tfr_refiner_steps,
                bias=True,
            )
            self.shallow_feat2 = None
            self.shallow_feat3 = None
            self.stage2_encoder = None
            self.stage2_decoder = None
            self.stage2_stripe_transformer = None
            self.stitch_trajectory_alignment = None
            self.sam23 = None
            self.concat12 = None
            self.concat23 = None
            self.stage3_orsnet = None
            self.edge_refine = None
            self.tail = None
            self.stage3_synthesis = None
            self.stripe_topology_reconstruction = None
            self.interstage_residual_reassembly = nn.Identity()
            self.tpt_rnet_core = None
            self.hmt_net_core = None
        if self.evidence_canvas_net_enabled:
            self.evidence_canvas_core = EvidenceCanvasReconstructionCore(
                n_feat,
                scale_unetfeats,
                stripe_count=self.stage1_stripes,
                stripe_overlap=self.stage1_overlap,
                blocks=evidence_canvas_blocks,
                edge_refine=self.edge_refine_enabled,
                edge_refine_mid=edge_refine_mid,
                bias=True,
            )
            # The independent evidence-canvas path owns every feature and
            # recovery operation.  Explicitly unregister inherited modules so
            # neither parameter counts nor checkpoints imply MPRNet reuse.
            self.shallow_feat1 = None
            self.shallow_feat2 = None
            self.shallow_feat3 = None
            self.stage1_encoder = None
            self.stage1_decoder = None
            self.stage1_stripe_transformer = None
            self.stage2_encoder = None
            self.stage2_decoder = None
            self.stage2_stripe_transformer = None
            self.stitch_trajectory_alignment = None
            self.sam12 = None
            self.sam23 = None
            self.concat12 = None
            self.concat23 = None
            self.stage3_orsnet = None
            self.edge_refine = None
            self.tail = None
            self.stage3_synthesis = None
            self.stripe_topology_reconstruction = None
            self.interstage_residual_reassembly = nn.Identity()
            self.tpt_rnet_core = None
            self.hmt_net_core = None
            self.tfr_net_core = None
            self.stage1_bridge = None
            self.tristripe_context_pyramid = None

    @staticmethod
    def _stripe_starts(height, count, overlap, align=4):
        if height % align != 0:
            raise ValueError(
                'Trajectory lightweight mode requires image height divisible by 4; '
                f'got {height}.'
            )
        if count <= 1:
            return [0], height

        effective_stripes = count - overlap * (count - 1)
        stripe_height = int(math.ceil((height / effective_stripes) / align) * align)
        stripe_height = min(height, max(align, stripe_height))
        if stripe_height == height:
            return [0], height

        max_start = height - stripe_height
        starts = []
        for index in range(count):
            raw_start = max_start * index / (count - 1)
            start = int(round(raw_start / align) * align)
            starts.append(min(max_start, max(0, start)))
        starts[0] = 0
        starts[-1] = max_start
        return starts, stripe_height

    @staticmethod
    def _blend_stripes(stripes, starts, full_height):
        if len(stripes) == 1:
            return stripes[0]

        output = stripes[0].new_zeros(
            stripes[0].size(0),
            stripes[0].size(1),
            full_height,
            stripes[0].size(3),
        )
        normalizer = stripes[0].new_zeros(1, 1, full_height, 1)

        for index, (stripe, start) in enumerate(zip(stripes, starts)):
            stripe_height = stripe.size(2)
            end = start + stripe_height
            weight = stripe.new_ones(1, 1, stripe_height, 1)

            if index > 0:
                overlap_top = starts[index - 1] + stripes[index - 1].size(2) - start
                if overlap_top > 0:
                    weight[:, :, :overlap_top] = torch.linspace(
                        0.0,
                        1.0,
                        overlap_top + 2,
                        device=stripe.device,
                        dtype=stripe.dtype,
                    )[1:-1].view(1, 1, -1, 1)

            if index + 1 < len(stripes):
                overlap_bottom = end - starts[index + 1]
                if overlap_bottom > 0:
                    weight[:, :, -overlap_bottom:] = torch.linspace(
                        1.0,
                        0.0,
                        overlap_bottom + 2,
                        device=stripe.device,
                        dtype=stripe.dtype,
                    )[1:-1].view(1, 1, -1, 1)

            output[:, :, start:end] += stripe * weight
            normalizer[:, :, start:end] += weight

        return output / normalizer.clamp_min(1e-6)

    def set_stage3_tepr_transition_alpha(self, alpha):
        """Set the training-only legacy/new recovery mixing coefficient."""
        if not self.stage3_tepr_enabled:
            raise RuntimeError('Stage-3 TEPR is not enabled for this model.')
        if not self.stage3_tepr_transition_enabled:
            if float(alpha) != 1.0:
                raise ValueError('Pure TEPR has no legacy path; alpha must remain 1.0.')
            return
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError('stage3_tepr transition alpha must be in [0, 1].')
        self.stage3_tepr_transition_alpha.fill_(alpha)

    def initialize_stage3_tepr_from_legacy(self):
        """Copy a loaded legacy RGB projection into the new synthesis head."""
        if not self.stage3_tepr_transition_enabled:
            return
        if self.tail is None or self.tepr_head is None:
            raise RuntimeError('TEPR transition modules are incomplete.')
        with torch.no_grad():
            self.tepr_head.base_projection.weight.copy_(self.tail.weight)
            if self.tail.bias is not None and self.tepr_head.base_projection.bias is not None:
                self.tepr_head.base_projection.bias.copy_(self.tail.bias)

    @property
    def stage3_tepr_is_pure(self):
        return self.stage3_tepr_enabled and not self.stage3_tepr_transition_enabled

    def _finalize(self, x3_img, x3_samfeats, feat2, res2, stage2_img, stage1_img,
                  return_edges, return_intermediate, stripe_consistency=None):
        x3 = self.shallow_feat3(x3_img)
        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], dim=1))
        edge_pred = None
        if self.stage3_tepr_enabled:
            tepr_features = self.stage3_tepr(x3_cat, feat2, res2)
            if self.edge_refine is not None:
                tepr_features, edge_pred = self.edge_refine(tepr_features)
            if self.stripe_topology_reconstruction is not None:
                tepr_features = self.stripe_topology_reconstruction(tepr_features, res2)
            tepr_restored = self.tepr_head(tepr_features) + x3_img
            if self.stage3_tepr_transition_enabled:
                legacy_features = self.stage3_orsnet(x3_cat, feat2, res2)
                legacy_restored = self.tail(legacy_features) + x3_img
                alpha = self.stage3_tepr_transition_alpha.to(
                    device=tepr_restored.device, dtype=tepr_restored.dtype
                )
                restored = legacy_restored * (1.0 - alpha) + tepr_restored * alpha
            else:
                restored = tepr_restored
        else:
            x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)
            if self.edge_refine is not None:
                x3_cat, edge_pred = self.edge_refine(x3_cat)
            if self.stripe_topology_reconstruction is not None:
                x3_cat = self.stripe_topology_reconstruction(x3_cat, res2)
            restored = self.tail(x3_cat) + x3_img
        if self.stage3_synthesis is not None:
            restored = self.stage3_synthesis(
                x3_cat,
                restored,
                x3_img,
                stage2_img,
                stage1_img,
            )
        if self.interstage_residual_reassembly_enabled:
            restored = self.interstage_residual_reassembly(
                restored,
                x3_img,
                stage1_img,
                stage2_img,
            )
        if self.frequency_refine_enabled:
            restored = self.frequency_refine(restored, reference=stage2_img)
        else:
            restored = self.frequency_refine(restored)
        outputs = [restored, stage2_img, stage1_img] if return_intermediate else [restored]
        if return_edges:
            extras = {'edge': edge_pred}
            if stripe_consistency is not None:
                extras['stripe_consistency'] = stripe_consistency
            if self.interstage_residual_reassembly_enabled:
                extras['irr_delta_abs'] = (
                    self.interstage_residual_reassembly.last_delta_abs
                )
            if self.stripe_topology_reconstruction is not None:
                extras['strd_delta_abs'] = (
                    self.stripe_topology_reconstruction.last_delta_abs
                )
            return outputs, extras
        return outputs

    def _forward_tristripe_stage1(self, x3_img):
        """Run the shared-pair three-stripe Stage 1 and blend it to full size."""
        height = x3_img.size(2)
        starts, stripe_height = self._stripe_starts(
            height,
            self.stage1_stripes,
            self.stage1_overlap,
        )
        stripe_images = [
            x3_img[:, :, start:start + stripe_height, :] for start in starts
        ]

        stage1_features = []
        for stripe_image, start in zip(stripe_images, starts):
            if self.tristripe_context_pyramid is not None:
                features = self.tristripe_context_pyramid.encode(
                    stripe_image, start, height
                )
            else:
                features = self.stage1_encoder(self.shallow_feat1(stripe_image))
                features[-1] = self.stage1_stripe_transformer(features[-1])
            stage1_features.append(features)

        pair_results = []
        pair_sam_features = []
        pair_images = []
        pair_starts = []
        pair_encoder_features = []
        if len(stage1_features) == 1:
            if self.tristripe_context_pyramid is not None:
                results = self.tristripe_context_pyramid.decode(stage1_features[0])
            else:
                results = self.stage1_decoder(stage1_features[0])
            if (
                self.tpt_rnet_core_enabled
                or self.hmt_net_core_enabled
                or self.tfr_net_core_enabled
            ) and self.stage1_bridge is not None:
                sam_features, restored_pair = self.stage1_bridge(
                    results[0], stripe_images[0]
                )
            else:
                sam_features, restored_pair = self.sam12(
                    results[0], stripe_images[0]
                )
            pair_results.append(results)
            pair_sam_features.append(sam_features)
            pair_images.append(restored_pair)
            pair_starts.append(starts[0])
            pair_encoder_features.append(stage1_features[0])
        else:
            for index in range(len(stage1_features) - 1):
                pair_start = starts[index]
                pair_end = starts[index + 1] + stripe_height
                overlap = max(0, starts[index] + stripe_height - starts[index + 1])

                pair_features = []
                for level, scale in enumerate((1, 2, 4)):
                    overlap_scaled = overlap // scale
                    next_feature = stage1_features[index + 1][level]
                    if overlap_scaled > 0:
                        next_feature = next_feature[:, :, overlap_scaled:, :]
                    pair_features.append(torch.cat([
                        stage1_features[index][level],
                        next_feature,
                    ], dim=2))

                if self.tristripe_context_pyramid is not None:
                    results = self.tristripe_context_pyramid.decode(pair_features)
                else:
                    results = self.stage1_decoder(pair_features)
                pair_image = x3_img[:, :, pair_start:pair_end, :]
                if (
                    self.tpt_rnet_core_enabled
                    or self.hmt_net_core_enabled
                    or self.tfr_net_core_enabled
                ) and self.stage1_bridge is not None:
                    sam_features, restored_pair = self.stage1_bridge(
                        results[0], pair_image
                    )
                else:
                    sam_features, restored_pair = self.sam12(
                        results[0], pair_image
                    )
                pair_results.append(results)
                pair_sam_features.append(sam_features)
                pair_images.append(restored_pair)
                pair_starts.append(pair_start)
                pair_encoder_features.append(pair_features)

        feat1 = []
        res1 = []
        for level, scale in enumerate((1, 2, 4)):
            scaled_starts = [start // scale for start in starts]
            feat1.append(self._blend_stripes(
                [features[level] for features in stage1_features],
                scaled_starts,
                height // scale,
            ))
            pair_scaled_starts = [start // scale for start in pair_starts]
            res1.append(self._blend_stripes(
                [results[level] for results in pair_results],
                pair_scaled_starts,
                height // scale,
            ))

        stripe_consistency = x3_img.new_zeros(())
        consistency_pairs = 0
        for index in range(len(pair_images) - 1):
            left_start = pair_starts[index]
            right_start = pair_starts[index + 1]
            overlap_start = max(left_start, right_start)
            overlap_end = min(
                left_start + pair_images[index].size(2),
                right_start + pair_images[index + 1].size(2),
            )
            if overlap_end > overlap_start:
                left = pair_images[index][:, :, overlap_start - left_start:overlap_end - left_start, :]
                right = pair_images[index + 1][:, :, overlap_start - right_start:overlap_end - right_start, :]
                stripe_consistency = stripe_consistency + (left - right).abs().mean()
                consistency_pairs += 1
        if consistency_pairs:
            stripe_consistency = stripe_consistency / consistency_pairs

        stage1_sam = self._blend_stripes(pair_sam_features, pair_starts, height)
        stage1_img = self._blend_stripes(pair_images, pair_starts, height)
        pair_context = {
            'starts': pair_starts,
            'encoder_features': pair_encoder_features,
            'decoder_features': pair_results,
            'sam_features': pair_sam_features,
            'images': pair_images,
        }
        return feat1, res1, stage1_sam, stage1_img, stripe_consistency, pair_context

    def _forward_trajectory(self, x3_img, return_edges, return_intermediate):
        """Legacy all-lightweight trajectory path retained for old checkpoints."""
        feat1, res1, stage1_sam, stage1_img, stripe_consistency, _ = self._forward_tristripe_stage1(x3_img)

        stage2_shallow = self.shallow_feat2(x3_img)
        stage2_input = self.concat12(torch.cat([stage2_shallow, stage1_sam], dim=1))
        feat2 = self.stage2_encoder(stage2_input, feat1, res1)
        feat2[-1] = self.stage2_stripe_transformer(feat2[-1])
        res2 = self.stage2_decoder(feat2)
        x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)
        return self._finalize(
            x3_img,
            x3_samfeats,
            feat2,
            res2,
            stage2_img,
            stage1_img,
            return_edges,
            return_intermediate,
            stripe_consistency=stripe_consistency,
        )

    def _stage2_windows(self, height):
        """Return overlapping upper/lower Stage-2 windows aligned to the U-Net scale."""
        if height % 8 != 0:
            raise ValueError(
                'Tri-stripe + original Stage-2 mode requires image height divisible '
                f'by 8; got {height}.'
            )
        mid = height // 2
        # The overlap is split equally around the centre seam. Alignment by four
        # keeps all feature crops compatible with the two encoder downsamplings.
        half_overlap = int(round((height * self.stage2_overlap / 2.0) / 4.0) * 4)
        half_overlap = min(max(0, half_overlap), max(0, mid - 4))
        return [(0, mid + half_overlap), (mid - half_overlap, height)]

    def _forward_tristripe_original(self, x3_img, return_edges, return_intermediate):
        """Tri-stripe Stage 1 with original MPRNet Stage 2 and Stage 3.

        Stage 1 preserves full-width horizontal trajectory context. Stage 2 uses
        the original upper/lower MPRNet decomposition, optionally blending a
        narrow overlap at the centre seam, and Stage 3 remains the original
        full-resolution ORSNet.
        """
        height = x3_img.size(2)
        feat1, res1, stage1_sam, stage1_img, stripe_consistency, _ = self._forward_tristripe_stage1(x3_img)
        stage2_shallow = self.shallow_feat2(x3_img)
        stage2_input = self.concat12(torch.cat([stage2_shallow, stage1_sam], dim=1))

        windows = self._stage2_windows(height)
        stage2_features = []
        for start, end in windows:
            stage2_crop = stage2_input[:, :, start:end, :]
            feat1_crop = [
                feature[:, :, start // scale:end // scale, :]
                for feature, scale in zip(feat1, (1, 2, 4))
            ]
            res1_crop = [
                feature[:, :, start // scale:end // scale, :]
                for feature, scale in zip(res1, (1, 2, 4))
            ]
            features = self.stage2_encoder(stage2_crop, feat1_crop, res1_crop)
            features[-1] = self.stage2_stripe_transformer(features[-1])
            stage2_features.append(features)

        # With zero overlap this is identical to the original top/bottom
        # concatenation. With overlap, linear blending removes a centre seam.
        feat2 = []
        for level, scale in enumerate((1, 2, 4)):
            scaled_starts = [start // scale for start, _ in windows]
            feat2.append(self._blend_stripes(
                [features[level] for features in stage2_features],
                scaled_starts,
                height // scale,
            ))

        res2 = self.stage2_decoder(feat2)
        x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)
        return self._finalize(
            x3_img,
            x3_samfeats,
            feat2,
            res2,
            stage2_img,
            stage1_img,
            return_edges,
            return_intermediate,
            stripe_consistency=stripe_consistency,
        )

    def _forward_tristripe_pair_aligned(self, x3_img, return_edges, return_intermediate):
        """Use [stripe1+stripe2] and [stripe2+stripe3] as Stage-2 branches.

        Each branch keeps its own raw degraded crop and the matching shared-pair
        Stage-1 features. Their complete middle-stripe overlap is then blended
        before the original global Stage-2 decoder and Stage-3 ORSNet.
        """
        height = x3_img.size(2)
        _, _, _, stage1_img, stripe_consistency, pair_context = self._forward_tristripe_stage1(x3_img)
        pair_starts = pair_context['starts']
        pair_images = pair_context['images']
        pair_sam_features = pair_context['sam_features']
        pair_encoder_features = pair_context['encoder_features']
        pair_decoder_features = pair_context['decoder_features']

        stage2_features = []
        for start, pair_image, sam_features, enc_features, dec_features in zip(
            pair_starts,
            pair_images,
            pair_sam_features,
            pair_encoder_features,
            pair_decoder_features,
        ):
            end = start + pair_image.size(2)
            raw_pair = x3_img[:, :, start:end, :]
            stage2_input = self.concat12(torch.cat([
                self.shallow_feat2(raw_pair), sam_features
            ], dim=1))
            if self.stitch_trajectory_alignment is not None:
                enc_features, dec_features = self.stitch_trajectory_alignment(
                    stage2_input, enc_features, dec_features
                )
            features = self.stage2_encoder(stage2_input, enc_features, dec_features)
            features[-1] = self.stage2_stripe_transformer(features[-1])
            stage2_features.append(features)

        feat2 = []
        for level, scale in enumerate((1, 2, 4)):
            scaled_starts = [start // scale for start in pair_starts]
            feat2.append(self._blend_stripes(
                [features[level] for features in stage2_features],
                scaled_starts,
                height // scale,
            ))

        res2 = self.stage2_decoder(feat2)
        x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)
        return self._finalize(
            x3_img,
            x3_samfeats,
            feat2,
            res2,
            stage2_img,
            stage1_img,
            return_edges,
            return_intermediate,
            stripe_consistency=stripe_consistency,
        )

    def _forward_tpt_rnet(self, x3_img, return_edges, return_intermediate):
        """Run the rewritten TPT-RNet backend after shared TriStripe context."""
        if self.hmt_net_core is not None and x3_img.size(2) % 8 != 0:
            raise ValueError(
                'HMT-Net requires image height divisible by 8 for its '
                f'four-scale shared-memory pyramid; got {x3_img.size(2)}.'
            )
        _, _, _, stage1_img, stripe_consistency, pair_context = (
            self._forward_tristripe_stage1(x3_img)
        )
        pair_inputs = []
        for start, pair_image, state_feature in zip(
            pair_context['starts'],
            pair_context['images'],
            pair_context['sam_features'],
        ):
            end = start + pair_image.size(2)
            raw_pair = x3_img[:, :, start:end, :]
            pair_inputs.append(torch.cat([raw_pair, state_feature], dim=1))

        reconstruction_core = (
            self.hmt_net_core if self.hmt_net_core is not None
            else self.tpt_rnet_core
        )
        reconstruction_output = reconstruction_core(
            x3_img,
            pair_inputs,
            pair_context['starts'],
            self._blend_stripes,
            pair_context['encoder_features'],
            pair_context['decoder_features'],
        )
        if self.hmt_net_core is not None:
            restored, stage2_img, _, _, hmt_extras = reconstruction_output
        else:
            restored, stage2_img, _, _ = reconstruction_output
            hmt_extras = {}
        if self.frequency_refine_enabled:
            restored = self.frequency_refine(restored, reference=stage2_img)
        else:
            restored = self.frequency_refine(restored)
        outputs = [restored, stage2_img, stage1_img] if return_intermediate else [restored]
        if return_edges:
            extras = {
                'edge': None,
                'stripe_consistency': stripe_consistency,
            }
            extras.update(hmt_extras)
            return outputs, extras
        return outputs

    def _forward_tfr_net(self, x3_img, return_edges, return_intermediate):
        """Run TFR-Net after shared TriStripe evidence extraction."""
        _, _, _, stage1_img, stripe_consistency, pair_context = (
            self._forward_tristripe_stage1(x3_img)
        )
        pair_inputs = []
        for start, pair_image, state_feature in zip(
            pair_context['starts'],
            pair_context['images'],
            pair_context['sam_features'],
        ):
            end = start + pair_image.size(2)
            raw_pair = x3_img[:, :, start:end, :]
            pair_inputs.append(torch.cat([raw_pair, state_feature], dim=1))

        restored, stage2_img, pair_restored, _, tfr_extras = self.tfr_net_core(
            x3_img,
            pair_inputs,
            pair_context['starts'],
            self._blend_stripes,
            pair_context['encoder_features'],
            pair_context['decoder_features'],
        )
        if self.frequency_refine_enabled:
            restored = self.frequency_refine(restored, reference=stage2_img)
        else:
            restored = self.frequency_refine(restored)
        outputs = (
            [restored, stage2_img, pair_restored]
            if return_intermediate else [restored]
        )
        if return_edges:
            extras = {
                'edge': None,
                'stripe_consistency': stripe_consistency,
            }
            extras.update(tfr_extras)
            return outputs, extras
        return outputs

    def _forward_evidence_canvas(self, x3_img, return_edges, return_intermediate):
        restored, _, edge = self.evidence_canvas_core(x3_img)
        if self.frequency_refine_enabled:
            restored = self.frequency_refine(restored, reference=x3_img)
        else:
            restored = self.frequency_refine(restored)
        # Keep the common multi-output API without introducing intermediate
        # reconstructed images.  Both analysis-stage placeholders are raw input.
        outputs = [restored, x3_img, x3_img] if return_intermediate else [restored]
        if return_edges:
            return outputs, {
                'edge': edge,
                'stripe_consistency': x3_img.new_zeros(()),
            }
        return outputs

    def _forward_original(self, x3_img, return_edges, return_intermediate):
        height = x3_img.size(2)
        width = x3_img.size(3)
        x2top_img = x3_img[:, :, :height // 2, :]
        x2bot_img = x3_img[:, :, height // 2:, :]

        x1ltop_img = x2top_img[:, :, :, :width // 2]
        x1rtop_img = x2top_img[:, :, :, width // 2:]
        x1lbot_img = x2bot_img[:, :, :, :width // 2]
        x1rbot_img = x2bot_img[:, :, :, width // 2:]

        feat1_ltop = self.stage1_encoder(self.shallow_feat1(x1ltop_img))
        feat1_rtop = self.stage1_encoder(self.shallow_feat1(x1rtop_img))
        feat1_lbot = self.stage1_encoder(self.shallow_feat1(x1lbot_img))
        feat1_rbot = self.stage1_encoder(self.shallow_feat1(x1rbot_img))

        feat1_top = [torch.cat((left, right), 3) for left, right in zip(feat1_ltop, feat1_rtop)]
        feat1_bot = [torch.cat((left, right), 3) for left, right in zip(feat1_lbot, feat1_rbot)]
        res1_top = self.stage1_decoder(feat1_top)
        res1_bot = self.stage1_decoder(feat1_bot)
        x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x2top_img)
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x2bot_img)
        stage1_img = torch.cat([stage1_img_top, stage1_img_bot], dim=2)

        x2top = self.concat12(torch.cat([
            self.shallow_feat2(x2top_img), x2top_samfeats
        ], dim=1))
        x2bot = self.concat12(torch.cat([
            self.shallow_feat2(x2bot_img), x2bot_samfeats
        ], dim=1))
        feat2_top = self.stage2_encoder(x2top, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(x2bot, feat1_bot, res1_bot)
        feat2_top[-1] = self.stage2_stripe_transformer(feat2_top[-1])
        feat2_bot[-1] = self.stage2_stripe_transformer(feat2_bot[-1])
        feat2 = [torch.cat((top, bottom), 2) for top, bottom in zip(feat2_top, feat2_bot)]
        res2 = self.stage2_decoder(feat2)
        x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)
        return self._finalize(
            x3_img,
            x3_samfeats,
            feat2,
            res2,
            stage2_img,
            stage1_img,
            return_edges,
            return_intermediate,
        )

    def forward(self, x3_img, return_edges=False, return_intermediate=True):
        if self.light_adapter is not None:
            x3_img = self.light_adapter(x3_img)
        if self.evidence_canvas_net_enabled:
            return self._forward_evidence_canvas(
                x3_img, return_edges, return_intermediate
            )
        if self.tfr_net_core_enabled:
            return self._forward_tfr_net(
                x3_img, return_edges, return_intermediate
            )
        if self.tpt_rnet_core_enabled or self.hmt_net_core_enabled:
            return self._forward_tpt_rnet(
                x3_img, return_edges, return_intermediate
            )
        if self.trajectory_lightweight:
            return self._forward_trajectory(x3_img, return_edges, return_intermediate)
        if self.stage1_tristripe:
            if self.stage2_pair_aligned:
                return self._forward_tristripe_pair_aligned(
                    x3_img, return_edges, return_intermediate
                )
            return self._forward_tristripe_original(
                x3_img, return_edges, return_intermediate
            )
        return self._forward_original(x3_img, return_edges, return_intermediate)


def build_mprnet_from_config(model_cfg=None, **overrides):
    if model_cfg is None:
        model_cfg = {}

    # Accept either a MODEL sub-node or a full root Config/CfgNode.
    # This makes the builder resilient to callers passing Config directly.
    if hasattr(model_cfg, 'MODEL') and not hasattr(model_cfg, 'N_FEAT'):
        model_cfg = model_cfg.MODEL

    def cfg_get(name, default):
        if name in overrides:
            return overrides[name]
        if isinstance(model_cfg, dict):
            return model_cfg.get(name, default)
        return getattr(model_cfg, name, default)

    return MPRNet(
        n_feat=int(cfg_get('N_FEAT', 96)),
        scale_unetfeats=int(cfg_get('SCALE_UNETFEATS', 48)),
        scale_orsnetfeats=int(cfg_get('SCALE_ORSNETFEATS', 32)),
        num_cab=int(cfg_get('NUM_CAB', 8)),
        edge_refine=bool(cfg_get('EDGE_REFINE', False)),
        edge_refine_mid=int(cfg_get('EDGE_REFINE_MID', 24)),
        frequency_refine=bool(cfg_get('FREQUENCY_REFINE', False)),
        frequency_refine_hidden=int(cfg_get('FREQUENCY_REFINE_HIDDEN', 16)),
        frequency_band_gate=bool(cfg_get('FREQUENCY_BAND_GATE', False)),
        frequency_band_low_start=float(cfg_get('FREQUENCY_BAND_LOW_START', 0.05)),
        frequency_band_low_end=float(cfg_get('FREQUENCY_BAND_LOW_END', 0.10)),
        frequency_band_high_start=float(cfg_get('FREQUENCY_BAND_HIGH_START', 0.30)),
        frequency_band_high_end=float(cfg_get('FREQUENCY_BAND_HIGH_END', 0.40)),
        frequency_disagreement_gate=bool(cfg_get('FREQUENCY_DISAGREEMENT_GATE', False)),
        frequency_disagreement_threshold=float(cfg_get('FREQUENCY_DISAGREEMENT_THRESHOLD', 0.0120)),
        frequency_disagreement_temperature=float(cfg_get('FREQUENCY_DISAGREEMENT_TEMPERATURE', 0.0020)),
        frequency_disagreement_min_strength=float(cfg_get('FREQUENCY_DISAGREEMENT_MIN_STRENGTH', 0.10)),
        light_adapt=bool(cfg_get('LIGHT_ADAPT', False)),
        light_target_luma=float(cfg_get('LIGHT_TARGET_LUMA', 0.50)),
        light_min_gain=float(cfg_get('LIGHT_MIN_GAIN', 0.75)),
        light_max_gain=float(cfg_get('LIGHT_MAX_GAIN', 1.80)),
        light_min_gamma=float(cfg_get('LIGHT_MIN_GAMMA', 0.65)),
        light_max_gamma=float(cfg_get('LIGHT_MAX_GAMMA', 1.35)),
        light_blend=float(cfg_get('LIGHT_BLEND', 0.75)),
        light_dark_threshold=float(cfg_get('LIGHT_DARK_THRESHOLD', 0.45)),
        light_bright_threshold=float(cfg_get('LIGHT_BRIGHT_THRESHOLD', 0.70)),
        trajectory_lightweight=bool(cfg_get('TRAJECTORY_LIGHTWEIGHT', False)),
        stage1_tristripe=bool(cfg_get('STAGE1_TRISTRIPE', False)),
        stage1_stripes=int(cfg_get('STAGE1_STRIPES', 3)),
        stage1_overlap=float(cfg_get('STAGE1_OVERLAP', 0.25)),
        stage1_stripe_transformer=bool(cfg_get('STAGE1_STRIPE_TRANSFORMER', False)),
        stage1_stripe_transformer_heads=int(cfg_get('STAGE1_STRIPE_TRANSFORMER_HEADS', 4)),
        stage2_stripe_transformer=bool(cfg_get('STAGE2_STRIPE_TRANSFORMER', False)),
        stage2_stripe_transformer_heads=int(cfg_get('STAGE2_STRIPE_TRANSFORMER_HEADS', 4)),
        stage2_overlap=float(cfg_get('STAGE2_OVERLAP', 0.0)),
        stage2_pair_aligned=bool(cfg_get('STAGE2_PAIR_ALIGNED', False)),
        stage2_tmt=bool(cfg_get('STAGE2_TMT', False)),
        stage2_tmt_horizontal_kernel=int(
            cfg_get('STAGE2_TMT_HORIZONTAL_KERNEL', 15)
        ),
        stage2_tmt_vertical_kernel=int(
            cfg_get('STAGE2_TMT_VERTICAL_KERNEL', 5)
        ),
        gated_csff=bool(cfg_get('GATED_CSFF', False)),
        gated_csff_horizontal_kernel=int(cfg_get('GATED_CSFF_HORIZONTAL_KERNEL', 7)),
        stitch_trajectory_alignment=bool(
            cfg_get('STITCH_TRAJECTORY_ALIGNMENT', False)
        ),
        stam_horizontal_radius=int(cfg_get('STAM_HORIZONTAL_RADIUS', 6)),
        stam_vertical_radius=int(cfg_get('STAM_VERTICAL_RADIUS', 1)),
        stam_hidden_channels=int(cfg_get('STAM_HIDDEN_CHANNELS', 32)),
        dual_domain_orsnet=bool(cfg_get('DUAL_DOMAIN_ORSNET', False)),
        ddrb_residual_init=float(cfg_get('DDRB_RESIDUAL_INIT', 0.3)),
        simple_gated_orsnet=bool(cfg_get('SIMPLE_GATED_ORSNET', False)),
        sgors_blocks=int(cfg_get('SGORS_BLOCKS', 8)),
        sgors_residual_init=float(cfg_get('SGORS_RESIDUAL_INIT', 0.10)),
        hybrid_simple_gated_orsnet=bool(
            cfg_get('HYBRID_SIMPLE_GATED_ORSNET', False)
        ),
        hsgors_blocks=int(cfg_get('HSGORS_BLOCKS', 4)),
        hsgors_residual_init=float(cfg_get('HSGORS_RESIDUAL_INIT', 0.05)),
        hsgors_adapter_scale_init=float(
            cfg_get('HSGORS_ADAPTER_SCALE_INIT', 0.02)
        ),
        hsgors_v2=bool(cfg_get('HSGORS_V2', False)),
        hsgors_v2_horizontal_kernel=int(
            cfg_get('HSGORS_V2_HORIZONTAL_KERNEL', 15)
        ),
        hsgors_v2_vertical_kernel=int(
            cfg_get('HSGORS_V2_VERTICAL_KERNEL', 5)
        ),
        stage3_tepr=bool(cfg_get('STAGE3_TEPR', False)),
        stage3_tepr_transition=bool(cfg_get('STAGE3_TEPR_TRANSITION', False)),
        stage3_tepr_freeze_legacy=bool(
            cfg_get('STAGE3_TEPR_FREEZE_LEGACY', True)
        ),
        stage3_tepr_blocks=int(cfg_get('STAGE3_TEPR_BLOCKS', 3)),
        stage3_tepr_horizontal_kernel=int(
            cfg_get('STAGE3_TEPR_HORIZONTAL_KERNEL', 15)
        ),
        stage3_tepr_vertical_kernel=int(
            cfg_get('STAGE3_TEPR_VERTICAL_KERNEL', 5)
        ),
        stage2_naf_blocks=int(cfg_get('STAGE2_NAF_BLOCKS', 2)),
        stage2_horizontal_kernel=int(cfg_get('STAGE2_HORIZONTAL_KERNEL', 15)),
        stage2_vertical_kernel=int(cfg_get('STAGE2_VERTICAL_KERNEL', 5)),
        stage3_naf_blocks=int(cfg_get('STAGE3_NAF_BLOCKS', 3)),
        stage3_trr=bool(cfg_get('STAGE3_TRR', False)),
        stage3_trr_horizontal_kernel=int(
            cfg_get('STAGE3_TRR_HORIZONTAL_KERNEL', 15)
        ),
        stage3_trr_vertical_kernel=int(
            cfg_get('STAGE3_TRR_VERTICAL_KERNEL', 5)
        ),
        stage3_trr_hidden_channels=int(
            cfg_get('STAGE3_TRR_HIDDEN_CHANNELS', 32)
        ),
        tpt_rnet_core=bool(cfg_get('TPTRNET_CORE', False)),
        tptr_blocks=int(cfg_get('TPTR_BLOCKS', 2)),
        tptr_tcfm=bool(cfg_get('TPTR_TCFM', False)),
        tptr_tcfm_horizontal_kernel=int(
            cfg_get('TPTR_TCFM_HORIZONTAL_KERNEL', 7)
        ),
        tristripe_context_pyramid=bool(
            cfg_get(
                'TRISTRIPE_CONTEXT_PYRAMID',
                cfg_get('TRISTRIP_CONTEXT_PYRAMID', False),
            )
        ),
        tristripe_context_blocks=int(
            cfg_get(
                'TRISTRIPE_CONTEXT_BLOCKS',
                cfg_get('TRISTRIP_CONTEXT_BLOCKS', 2),
            )
        ),
        horizontal_trajectory_blocks=bool(
            cfg_get('HORIZONTAL_TRAJECTORY_BLOCKS', False)
        ),
        hmt_net_core=bool(cfg_get('HMT_NET_CORE', False)),
        hmt_blocks=int(cfg_get('HMT_BLOCKS', 3)),
        evidence_canvas_net=bool(cfg_get('EVIDENCE_CANVAS_NET', False)),
        evidence_canvas_blocks=int(cfg_get('EVIDENCE_CANVAS_BLOCKS', 2)),
        tfr_net_core=bool(cfg_get('TFR_NET_CORE', False)),
        tfr_blocks=int(cfg_get('TFR_BLOCKS', 8)),
        tfr_horizontal_kernel=int(cfg_get('TFR_HORIZONTAL_KERNEL', 15)),
        tfr_vertical_kernel=int(cfg_get('TFR_VERTICAL_KERNEL', 5)),
        tfr_refiner_steps=int(cfg_get('TFR_REFINER_STEPS', 3)),
        stripe_topology_reconstruction=bool(
            cfg_get('STRIPE_TOPOLOGY_RECONSTRUCTION', False)
        ),
        strd_hidden=int(cfg_get('STRD_HIDDEN', 32)),
        strd_num_blocks=int(cfg_get('STRD_NUM_BLOCKS', 2)),
        interstage_residual_reassembly=bool(
            cfg_get('INTERSTAGE_RESIDUAL_REASSEMBLY', False)
        ),
        irr_hidden=int(cfg_get('IRR_HIDDEN', 24)),
        irr_num_blocks=int(cfg_get('IRR_NUM_BLOCKS', 1)),
        irr_delta_init=float(cfg_get('IRR_DELTA_INIT', 1.0e-4)),
    )
