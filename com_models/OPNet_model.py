"""
Modified OPNet for C3Net Framework Compatibility
Includes resolution-safe handling and a dictionary output wrapper.
"""
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import numpy as np

# --- 基础组件 (保持不变) ---
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1); attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x); x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), out_features=None,
                 skip=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_features, act_layer=act_layer, drop=drop)
        self.skip = skip
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.skip: x = x + self.drop_path(self.mlp(self.norm2(x)))
        else: x = self.drop_path(self.mlp(self.norm2(x)))
        return x

class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()
        expansion = 4
        med_planes = outplanes // expansion
        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes); self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes); self.act2 = act_layer(inplace=True)
        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes); self.act3 = act_layer(inplace=True)
        self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False) if res_conv else None
        self.residual_bn = norm_layer(outplanes) if res_conv else None
        self.res_conv = res_conv; self.drop_block = drop_block; self.drop_path = drop_path
    def forward(self, x, x_t=None, return_x_2=True):
        residual = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x2 = self.act2(self.bn2(x))
        x = self.bn3(self.conv3(x2))
        if self.res_conv:
            residual = self.residual_bn(self.residual_conv(residual))
        x += residual; x = self.act3(x)
        return (x, x2) if return_x_2 else x

class FCUDown(nn.Module):
    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)
        self.ln = norm_layer(outplanes); self.act = act_layer()
    def forward(self, x, x_t):
        x = self.conv_project(x)
        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.act(self.ln(x))
        return torch.cat([x_t[:, 0][:, None, :], x], dim=1)

class FCUUp(nn.Module):
    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU, norm_layer=partial(nn.BatchNorm2d, eps=1e-6)):
        super(FCUUp, self).__init__()
        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes); self.act = act_layer()
    def forward(self, x, H, W):
        B, _, C = x.shape
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, int(H), int(W))
        x_r = self.act(self.bn(self.conv_project(x_r)))
        return F.interpolate(x_r, size=(int(self.up_stride * H), int(self.up_stride * W)))

class Med_ConvBlock(nn.Module):
    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6)):
        super(Med_ConvBlock, self).__init__()
        expansion = 4; med_planes = inplanes // expansion
        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes); self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes); self.act2 = act_layer(inplace=True)
        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes); self.act3 = act_layer(inplace=True)
    def forward(self, x):
        residual = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += residual; return self.act3(x)

class ConvTransBlock(nn.Module):
    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):
        super(ConvTransBlock, self).__init__()
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups)
        self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups) if last_fusion else ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)
        self.med_block = nn.ModuleList([Med_ConvBlock(inplanes=outplanes, groups=groups) for _ in range(num_med_block)])
        self.squeeze_block = FCUDown(inplanes=outplanes // 4, outplanes=embed_dim, dw_stride=dw_stride)
        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // 4, up_stride=dw_stride)
        self.trans_block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)
        self.dw_stride = dw_stride; self.num_med_block = num_med_block
    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)
        H, W = x2.shape[2:]
        x_t = self.trans_block(self.squeeze_block(x2, x_t) + x_t)
        for m in self.med_block: x = m(x)
        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False)
        return x, x_t

class Conformer(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64); self.act1 = nn.ReLU(inplace=True); self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_1_channel = int(base_channel * channel_ratio); trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0])
        init_stage = 2; fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i), ConvTransBlock(stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i - 1], num_med_block=num_med_block))
        stage_2_channel = int(base_channel * channel_ratio * 2)
        init_stage = fin_stage; fin_stage = fin_stage + depth // 3
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1; in_channel = stage_1_channel if i == init_stage else stage_2_channel
            self.add_module('conv_trans_' + str(i), ConvTransBlock(in_channel, stage_2_channel, True if i == init_stage else False, s, dw_stride=trans_dw_stride // 2, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i - 1], num_med_block=num_med_block))
        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        init_stage = fin_stage; fin_stage = fin_stage + depth // 3
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1; in_channel = stage_2_channel if i == init_stage else stage_3_channel
            self.add_module('conv_trans_' + str(i), ConvTransBlock(in_channel, stage_3_channel, True if i == init_stage else False, s, dw_stride=trans_dw_stride // 4, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i - 1], num_med_block=num_med_block, last_fusion=True if i == depth else False))
        self.fin_stage = fin_stage; trunc_normal_(self.cls_token, std=.02); self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear): trunc_normal_(m.weight, std=.02)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)): nn.init.constant_(m.weight, 1.0); nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def forward(self, x):
        conv_features = []; tran_features = []; B = x.shape[0]; cls_tokens = self.cls_token.expand(B, -1, -1)
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        x = self.conv_1(x_base, return_x_2=False)
        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        x_t = self.trans_1(torch.cat([cls_tokens, x_t], dim=1))
        conv_features.append(x); tran_features.append(x_t)
        for i in range(2, self.fin_stage):
            x, x_t = getattr(self, 'conv_trans_' + str(i))(x, x_t)
            conv_features.append(x); tran_features.append(x_t)
        return conv_features, tran_features

# --- 模块化预测与解码 (保持不变) ---
class Conv_Down(nn.Module):
    def __init__(self, conv_channel, tran_channel, scale):
        super(Conv_Down, self).__init__()
        self.conv_project = nn.Conv2d(conv_channel, tran_channel, 1, 1, 0)
        self.pooling = nn.AdaptiveAvgPool2d(scale); self.norm = nn.LayerNorm(tran_channel); self.act = nn.GELU()
    def forward(self, x, t_all):
        pooling = self.pooling(self.conv_project(x)).flatten(2).transpose(1, 2)
        return torch.cat([t_all[:, 0, :][:, None, :], self.act(self.norm(pooling))], 1)

class Tran_Down(nn.Module):
    def __init__(self, scale):
        super(Tran_Down, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(scale)
    def forward(self, t_all):
        t_seg = t_all[:, 0, :][:, None, :]; t = t_all[:, 1:, :]
        B, N, C = t.shape; side = int(np.sqrt(N))
        t_pool = self.pooling(t.transpose(1, 2).reshape(B, C, side, side)).reshape(B, C, -1).transpose(1, 2)
        return torch.cat([t_seg, t_pool], 1)

class Conv_Up(nn.Module):
    def __init__(self, tran_scale_channel, conv_scale_channel):
        super(Conv_Up, self).__init__()
        self.conv_project = nn.Conv2d(tran_scale_channel, conv_scale_channel, 1, 1, 0)
        self.bn = nn.BatchNorm2d(conv_scale_channel); self.relu = nn.ReLU()
    def forward(self, t_all, side_length):
        t = t_all[:, 1:, :]; B, N, C = t.shape; side = int(np.sqrt(N))
        x = self.relu(self.bn(self.conv_project(t.transpose(1, 2).reshape(B, C, side, side))))
        return F.interpolate(x, size=(side_length, side_length), mode='bilinear')

class Tran_Up(nn.Module):
    def __init__(self):
        super(Tran_Up, self).__init__()
    def forward(self, t_all, Num):
        t_seg = t_all[:, 0, :][:, None, :]; t = t_all[:, 1:, :]
        B, N, C = t.shape; side = int(np.sqrt(N)); target_side = int(np.sqrt(Num - 1))
        t_up = F.interpolate(t.transpose(1, 2).reshape(B, C, side, side), size=(target_side, target_side), mode='bilinear').reshape(B, C, -1).transpose(1, 2)
        return torch.cat([t_seg, t_up], 1)

class PPM(nn.Module):
    def __init__(self, conv_channel, tran_channel, scale=(5, 7, 9, 11), num_heads=9):
        super(PPM, self).__init__()
        self.scales = scale
        for i in range(4):
            self.add_module(f'conv_down_{i}', Conv_Down(conv_channel, tran_channel, scale[i]))
            self.add_module(f'tran_down_{i}', Tran_Down(scale[i]))
            self.add_module(f'mhsa_{i}', Block(tran_channel, num_heads=num_heads, mlp_ratio=1, out_features=tran_channel // 4, skip=False))
            self.add_module(f'conv_up_{i}', Conv_Up(tran_channel // 4, conv_channel // 4))
            self.add_module(f'tran_up_{i}', Tran_Up())
        self.conv_pos = nn.Sequential(nn.Conv2d(2 * conv_channel, conv_channel, 1, 1, 0), nn.BatchNorm2d(conv_channel), nn.ReLU())
        self.tran_pos = Mlp(in_features=int(2 * tran_channel), hidden_features=tran_channel, out_features=tran_channel)
    def forward(self, x, x_t):
        side, Num = x.shape[2], x_t.shape[1]
        c_ups, t_ups = [], []
        for i in range(4):
            f_down = getattr(self, f'conv_down_{i}')(x, x_t) + getattr(self, f'tran_down_{i}')(x_t)
            scale_f = getattr(self, f'mhsa_{i}')(f_down)
            c_ups.append(getattr(self, f'conv_up_{i}')(scale_f, side))
            t_ups.append(getattr(self, f'tran_up_{i}')(scale_f, Num))
        return self.conv_pos(torch.cat([x] + c_ups, 1)), self.tran_pos(torch.cat([x_t] + t_ups, 2))

class Decoder(nn.Module):
    def __init__(self, conv_channel, embed_dim=576, num_heads=9, qkv_bias=True, mlp_ratio=4., if_tran=True):
        super(Decoder, self).__init__()
        self.if_tran = if_tran
        self.v_conv = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(conv_channel, conv_channel // 4, 1), nn.ReLU())
        self.v_tran = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(embed_dim, conv_channel // 4, 1), nn.ReLU())
        self.c_attn_c = nn.Sequential(nn.Conv2d(conv_channel // 2, conv_channel // 2, 1), nn.Sigmoid())
        self.attn_map = nn.Sequential(nn.Conv2d(2, 1, 7, 1, 3), nn.Sigmoid())
        self.conv_up = nn.Sequential(nn.Conv2d(conv_channel, conv_channel // 2, 3, 1, 1), nn.BatchNorm2d(conv_channel // 2), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.conv_f = nn.Sequential(nn.Conv2d(conv_channel // 2, conv_channel // 2, 3, 1, 1), nn.BatchNorm2d(conv_channel // 2), nn.ReLU())
        if if_tran:
            self.c_attn_t = nn.Sequential(nn.Conv2d(conv_channel // 2, embed_dim, 1), nn.Sigmoid())
            self.tran_f = Block(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, mlp_ratio=mlp_ratio, out_features=embed_dim, skip=True)
    def forward(self, x, x_t, x_enc, x_t_enc, m_x, m_xt):
        v = torch.cat([self.v_conv(x), self.v_tran(x_t[:, 1:, :].transpose(1, 2).unsqueeze(-1))], 1)
        c_add = self.conv_up(x) + x_enc
        m = self.attn_map(torch.cat([m_x, m_xt], 1))
        c_f = self.conv_f(c_add * self.c_attn_c(v) * F.interpolate(m, scale_factor=2, mode='bilinear'))
        if self.if_tran:
            t_add = x_t
            t_ca = torch.cat([t_add[:, 0:1, :], t_add[:, 1:, :] * self.c_attn_t(v).squeeze(-1).transpose(1, 2)], 1)
            side = int(np.sqrt(x_t.shape[1] - 1))
            t_sa = torch.cat([t_ca[:, 0:1, :], t_ca[:, 1:, :] * F.interpolate(m, size=(side, side)).flatten(2).transpose(1, 2)], 1)
            return c_f, self.tran_f(t_sa)
        return c_f

class Predict(nn.Module):
    def __init__(self, conv_channel, embed_dim=576):
        super(Predict, self).__init__()
        self.map_x = nn.Conv2d(conv_channel, 1, 7, 1, 3)
        self.token_inf = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim)) 
        self.map_xt = nn.Linear(embed_dim, 1)

    def forward(self, x, x_t):
        B, _, h, w = x.shape
        # [修改点1]：减去 1 个 CLS token 后再计算边长 side
        side = int(np.sqrt(x_t.shape[1] - 1))
        
        m_x = self.map_x(x)
        
        # [修改点2]：通过切片 x_t[:, 1:, :] 丢弃第 0 个位置的 CLS token，只保留后面用于组成图像的 patch token
        spatial_tokens = x_t[:, 1:, :] 
        
        # 此时 spatial_tokens 大小完美契合 side * side，正常 reshape
        m_xt = self.map_xt(spatial_tokens).transpose(1, 2).reshape(B, 1, side, side)
        
        # 插值回原始空间大小
        m_xt = F.interpolate(m_xt, size=(h, w), mode='bilinear', align_corners=False)
        
        return m_x, m_xt

# --- OPNet 主网络 ---
class OPNet(nn.Module):
    def __init__(self, backbone_path=None):
        super(OPNet, self).__init__()
        self.conformer = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12, num_heads=9, mlp_ratio=4, qkv_bias=True)
        if backbone_path: self.conformer.load_state_dict(torch.load(backbone_path)); print(f"Loaded {backbone_path}")
        self.cr4 = nn.Sequential(nn.Conv2d(1536, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.cr3 = nn.Sequential(nn.Conv2d(1536, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv2d(768, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.cr1 = nn.Sequential(nn.Conv2d(384, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.ppm = PPM(512, 576); self.predict4 = Predict(512, 576); self.predict3 = Predict(256, 576)
        self.predict2 = Predict(128, 576); self.predict1 = nn.Conv2d(64, 1, 7, 1, 3)
        self.decoder43 = Decoder(512, 576); self.decoder32 = Decoder(256, 576); self.decoder21 = Decoder(128, 576, if_tran=False)

    def forward(self, x):
        conv_f, tran_f = self.conformer(x)
        cr4, cr3, cr2, cr1 = self.cr4(conv_f[11]), self.cr3(conv_f[10]), self.cr2(conv_f[7]), self.cr1(conv_f[3])
        c_ppm, t_ppm = self.ppm(cr4, tran_f[11])
        m4c, m4t = self.predict4(c_ppm, t_ppm)
        c43, t43 = self.decoder43(c_ppm, t_ppm, cr3, tran_f[10], m4c, m4t)
        m3c, m3t = self.predict3(c43, t43)
        c32, t32 = self.decoder32(c43, t43, cr2, tran_f[7], m3c, m3t)
        m2c, m2t = self.predict2(c32, t32)
        m1 = self.predict1(self.decoder21(c32, t32, cr1, tran_f[3], m2c, m2t))
        # Rescale & Return Logits
        out = [F.interpolate(f, size=x.shape[2:], mode='bilinear') for f in [m4c, m3c, m2c, m4t, m3t, m2t, m1]]
        return tuple(out)

# --- C3Net 适配器 ---
class OPNet_Wrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = OPNet(backbone_path=config.get('backbone_path'))
        self.input_divisor = 16
    def forward(self, x):
        orig_size = x.shape[2:]
        H, W = orig_size
        safe_h = ((H + self.input_divisor - 1) // self.input_divisor) * self.input_divisor
        safe_w = ((W + self.input_divisor - 1) // self.input_divisor) * self.input_divisor
        if (safe_h, safe_w) != orig_size:
            x = F.interpolate(x, size=(safe_h, safe_w), mode='bilinear', align_corners=False)
        preds = self.model(x)
        # OPNet has multiple progressively supervised predictions.  Limit only
        # the adapted logits (not its internal features), so an occasional
        # extreme value cannot poison the shared focal/Tversky loss.
        def safe_logits(prediction):
            prediction = F.interpolate(
                prediction, size=orig_size, mode='bilinear', align_corners=False
            ).float()
            return torch.nan_to_num(
                prediction, nan=0.0, posinf=20.0, neginf=-20.0
            ).clamp_(-20.0, 20.0)

        # 映射到 C3Net 的深监督键值
        return {
            'final_mask': safe_logits(preds[6]),
            'object_mask': safe_logits(preds[2]),
            'edge_mask': safe_logits(preds[1])
        }
