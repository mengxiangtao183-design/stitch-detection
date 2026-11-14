import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Dict

# 导入你原来的 Hitnet 模型
from .pvt import Hitnet   # 请确保 pvt.py 在 com_models/ 文件夹下


class Model(nn.Module):
    """
    PVTNet (Hitnet) 适配器
    - 输入尺寸与 C3Net 一致（320x320）
    - 仅在输入不是 32 倍数时补齐到安全尺寸
    - 输出格式与 C3Net 完全一致
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.target_size = config['image_processing'].get('target_size', 320)
        self.input_divisor = 32

        # 初始化 Hitnet
        self.model = Hitnet()

        # 加载预训练权重（可选）
        pretrained_path = config.get('pretrained_pvt')
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"[PVTNet] 加载预训练权重: {pretrained_path}")
            # 如果你希望使用 config 中的路径，可在 pvt.py 中修改加载逻辑
        else:
            print("[PVTNet] 未使用预训练权重（随机初始化）")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        输入: (B, 3, 320, 320)
        输出: 与 C3Net 完全一致的字典
        """
        B, C, H, W = x.shape

        # 1. Keep 320x320 direct; only pad/resize non-32x inputs.
        safe_h = ((H + self.input_divisor - 1) // self.input_divisor) * self.input_divisor
        safe_w = ((W + self.input_divisor - 1) // self.input_divisor) * self.input_divisor
        if H != safe_h or W != safe_w:
            x_resized = F.interpolate(
                x, size=(safe_h, safe_w),
                mode='bilinear', align_corners=False
            )
        else:
            x_resized = x

        # 2. 前向传播（Hitnet 返回 stage_loss 和 final prediction）
        stage_loss, pred = self.model(x_resized)   # pred 是 prediction2_8

        # 3. Resize 回原始输入尺寸 320x320
        final_pred = F.interpolate(
            pred, size=(H, W), mode='bilinear', align_corners=False
        )

        # 4. 返回与 C3Net 完全一致的格式
        return {
            'final_mask': final_pred,
            'object_mask': final_pred,   # 对比试验统一使用 final_mask
            'edge_mask': final_pred
        }
