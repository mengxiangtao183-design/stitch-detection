import sys
import os

# --- 核心黑科技 1：强行把当前目录加入环境变量 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# -----------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.build_sam import build_sam2

# ==== 核心黑科技 2：终极 Hydra 路径修复 ====
from hydra.core.global_hydra import GlobalHydra
import hydra

# 1. 清除任何由于 SAM2 内部写错而导致的错误初始化
GlobalHydra.instance().clear()

# 2. 获取你项目里真实的 configs 绝对路径 (E:\mxt\C3Net\sam2\configs)
config_dir = os.path.abspath(os.path.join(current_dir, "..", "sam2", "configs"))

# 3. 强行接管 Hydra，让它只能在这个绝对路径下找文件！
hydra.initialize_config_dir(config_dir=config_dir, version_base="1.2")
# ==========================================

class SAM2_Wrapper(nn.Module):
    """
    用于将 SAM2 桥接到 C3Net 训练框架的包装器。
    Handles resolution adaptation for the project input size and prompt-free segmentation.
    """
    def __init__(self, config=None):
        super(SAM2_Wrapper, self).__init__()
        
        # ---------------------------------------------------------
        # 【请根据你本地实际的 SAM2 配置文件和权重路径修改这里】
        # 默认使用 sam2_hiera_small 或 base，因为 large 太大了容易显存溢出
        # ---------------------------------------------------------
        sam2_cfg = "sam2/sam2_hiera_s.yaml" 
        sam2_ckpt = r"E:\mxt\C3Net\com_models\sam2_hiera_small.pt" 
        
        print(f"Loading SAM2 model from {sam2_ckpt}...")
        self.model = build_sam2(sam2_cfg, sam2_ckpt, device='cpu', apply_postprocessing=False)
        
        # 补充轻量级预测头 (Prediction Heads)
        # 因为 C3Net 的深监督机制需要 3 个输出，我们用 3x3 卷积基于 SAM2 的输出生成 object 和 edge
        self.head_loc = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.head_edge = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # 1. 记录原始输入尺寸 (例如 C3Net 的 320x320)
        orig_size = x.shape[2:]
        
        # 2. 【关键修复】：动态获取 SAM2 内部写死的期望尺寸 (通常是 1024)
        # 先拉伸到 SAM2 喜欢的尺寸，避免触发内部特征大小的 AssertionError
        sam_size = self.model.image_size
        x_resized = F.interpolate(x, size=(sam_size, sam_size), mode='bilinear', align_corners=False)
        
        # 3. Image Encoder (提取图像特征)
        image_embed = self.model.forward_image(x_resized)
        backbone_out, vision_feats, vision_pos_embeds, feat_sizes = self.model._prepare_backbone_features(image_embed)
        
        # 4. 构造无提示符 (Empty Prompt) 实现全自动分割
        B = x.shape[0]
        device = x.device
        point_coords = torch.zeros(B, 1, 2, device=device)
        point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device) # -1 代表没有点输入
        point_inputs = {"point_coords": point_coords, "point_labels": point_labels}
        
        # 准备高分辨率特征供 SAM2 Mask Decoder 使用
        high_res_features = [
            feat.permute(1, 2, 0).view(feat.size(1), feat.size(2), *s)
            for feat, s in zip(vision_feats[:-1], feat_sizes[:-1])
        ]
        
        top_feat = vision_feats[-1].permute(1, 2, 0).view(B, self.model.hidden_dim, *feat_sizes[-1])
        
        # 5. Mask Decoder (前向传播)
        sam_outputs = self.model._forward_sam_heads(
            backbone_features=top_feat,
            point_inputs=point_inputs,
            mask_inputs=None,
            high_res_features=high_res_features,
            multimask_output=False 
        )
        
        # sam_outputs 返回的是一个 Tuple，索引 4 是 high_res_masks 的 Logits
        high_res_masks = sam_outputs[4] 
        
        # 6. 【关键修复】：SAM2 算完后，插值拉伸回你的原始尺寸 (320x320)
        final_mask = F.interpolate(high_res_masks, size=orig_size, mode='bilinear', align_corners=False)
        
        # 使用轻量级头生成中间监督需要的 object_mask 和 edge_mask
        object_mask = self.head_loc(final_mask)
        edge_mask = self.head_edge(final_mask)
        
        # 组装成 C3Net 引擎期望的字典格式
        output_dict = {
            'final_mask': final_mask,
            'object_mask': object_mask,
            'edge_mask': edge_mask
        }
        
        return output_dict
