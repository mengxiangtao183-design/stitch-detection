"""
DINOv2 Hierarchical Vision Transformer Encoder with Register Tokens

This encoder implements an enhanced version of the DINOv2 architecture that incorporates register 
tokens for improved feature extraction. The architecture combines the hierarchical processing 
capabilities of vision transformers with a novel register token mechanism that maintains clean 
feature maps while preserving spatial information.
该编码器实现了增强版的 DINOv2 架构，集成了注册令牌（register tokens）以提
升特征提取能力。
该架构将视觉 Transformer 的分层处理能力与一种新颖的注册令牌机制相结合，在保留空间信息的同时，维持了清晰的特征图。

Architecture Details:细节
--------------------
The encoder processes images through several sophisticated stages:

1. Patch Embedding Layer:
   The input image is first divided into non-overlapping patches, each processed independently:
   - 224x224 images → 16x16 grid of 14x14 pixel patches (256 total patches)
   - 512x512 images → 36x36 grid of 14x14 pixel patches (1296 total patches)
   Each patch undergoes linear projection to create high-dimensional token embeddings.
1. 图像块嵌入层（Patch Embedding Layer）
输入图像首先被划分为互不重叠的图像块，每个图像块被独立处理：
224×224 尺寸图像 → 划分为 16×16 的网格，每个网格是 14×14 像素的图像块（总计 256 个图像块）
512×512 尺寸图像 → 划分为 36×36 的网格，每个网格是 14×14 像素的图像块（总计 1296 个图像块）
每个图像块都会经过线性投影，生成高维的令牌嵌入（token embeddings）。

2. Token Structure and Organization:
   The sequence combines three types of tokens, each serving a distinct purpose:
   - [CLS] Token: A learnable token that aggregates global image information
   - Register Tokens (N=4): Dedicated computational memory tokens that participate in 
     self-attention but are discarded at output. These prevent the model from repurposing
     patch tokens for global computation, maintaining cleaner feature maps.
   - Patch Tokens: Carry spatial information from the image patches through the network
2. 令牌结构与组织（Token Structure and Organization）
该序列融合了三种类型的令牌，每种令牌都承担独特的功能：
[CLS] 令牌：一种可学习的令牌，用于聚合图像的全局信息。
注册令牌（Register Tokens，数量 N=4）：专用的计算内存令牌，参与自注意力计算但会在输出阶段被舍弃。这类令牌可防止模型将图像块令牌挪用至全局计算，从而维持更清晰的特征图。
图像块令牌（Patch Tokens）：在网络中传递来自图像块的空间信息。

3. Transformer Processing:
   Each transformer block processes the token sequence through:
   a) Layer Normalization
   b) Multi-head Self-attention (MSA):
      - Varying attention heads by model size enable parallel relationship processing
      - Register tokens participate in attention but maintain separation of global/local info
      - Attention combines information across all tokens: patches, registers, and [CLS]
   c) Feed-forward Network (FFN):
      - Two-layer perceptron with GELU activation
      - Processes each token independently
   d) Residual connections around both MSA and FFN
3. Transformer 处理流程
每个 Transformer 模块通过以下组件处理令牌序列：
a) 层归一化（Layer Normalization）
b) 多头自注意力（Multi-head Self-attention, MSA）：
不同模型尺寸对应不同的注意力头数，可并行处理特征间的关联关系
注册令牌参与注意力计算，但保持全局 / 局部信息的分离
注意力机制会融合所有令牌（图像块令牌、注册令牌和 [CLS] 令牌）的信息
c) 前馈网络（Feed-forward Network, FFN）：
两层感知机，使用 GELU 激活函数
独立处理每个令牌
d) 在 MSA 和 FFN 周围均使用残差连接（Residual connections）

4. Model Variants and Dimensions:
   The architecture scales across three sizes:
   Giant (g/14):
   - Hidden dimension: 1536
   - Attention heads: 24
   - Transformer layers: 40
   - Best for complex tasks requiring fine-grained understanding

   Large (L/14):
   - Hidden dimension: 1024
   - Attention heads: 16
   - Transformer layers: 24
   - Balanced performance and computational cost

   Base (B/14):
   - Hidden dimension: 768
   - Attention heads: 12
   - Transformer layers: 18
   - Efficient while maintaining good performance
4. 模型变体与维度
该架构支持三种尺寸的模型扩展：
巨型（Giant, g/14）
隐藏维度：1536
注意力头数：24
Transformer 层数：40
最适合需要细粒度理解的复杂任务
大型（Large, L/14）
隐藏维度：1024
注意力头数：16
Transformer 层数：24
在性能与计算成本之间取得平衡
基础（Base, B/14）
隐藏维度：768
注意力头数：12
Transformer 层数：18
高效且保持良好性能

5. Feature Map Generation:
   The encoder outputs feature maps at different stages, maintaining spatial layout:
   - Each stage produces features of shape [B, D, H', W']
     where: B = batch size
            D = hidden dimension (varies by model variant)
            H' = input_height / patch_size
            W' = input_width / patch_size
   - Features maintain both local spatial information and global context
   - Register tokens' influence creates cleaner, artifact-free feature maps
   5. 特征图生成
编码器在不同阶段输出特征图，并保持空间布局：
每个阶段输出的特征形状为 [B, D, H', W']
其中：
B = 批次大小
D = 隐藏维度（随模型变体变化）
H' = 输入高度 /patch 大小
W' = 输入宽度 /patch 大小
特征同时保留局部空间信息和全局上下文
注册令牌的作用使特征图更清晰、无伪影

6. Register Token Innovation:
   The register token mechanism provides several key benefits:
   - Separates global computation from spatial representation
   - Prevents high-norm outlier tokens in feature maps
   - Improves performance on dense prediction tasks
   - Enables better object discovery and segmentation
   - Maintains cleaner attention maps throughout the network
6. 注册令牌创新
注册令牌机制提供了以下关键优势：
将全局计算与空间表示分离
防止特征图中出现高范数的异常令牌
提升密集预测任务的性能
实现更好的目标发现与分割效果
在整个网络中保持更清晰的注意力图
Usage Notes:
------------
    - Input images must have dimensions divisible by the patch size (14)
    - Register tokens are automatically handled internally
    - Output features maintain spatial layout for dense prediction tasks
    - Feature maps from any stage can be extracted by specifying stage indices
    - Each stage's features contain both local and global information thanks to register tokens
使用说明
输入图像尺寸必须能被 patch 大小（14）整除
注册令牌由内部自动处理
输出特征保持空间布局，适用于密集预测任务
可通过指定阶段索引提取任意阶段的特征图
得益于注册令牌，每个阶段的特征同时包含局部和全局信息

References:
-----------
    DINOv2 Paper: https://arxiv.org/abs/2304.07193
    Vision Transformers Need Registers: https://arxiv.org/abs/2309.16588
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from transformers import AutoBackbone
import logging

logger = logging.getLogger(__name__)

class DinoV2WithRegistersEncoder(nn.Module):
    """
    DINOv2 encoder with register tokens for hierarchical feature extraction.
    
    This implementation allows extracting features from specific transformer stages,
    maintaining the register token concept for clean feature maps. Each stage 
    processes tokens through self-attention while leveraging register tokens for
    global computation.

    Feature Dimensions:
        224x224 input → 16x16 patch grid → 256 patch tokens
        512x512 input → 36x36 patch grid → 1296 patch tokens
        
        Output features per stage maintain spatial layout:
        - Giant:  [B, 1536, H/14, W/14]
        - Large:  [B, 1024, H/14, W/14]
        - Base:   [B, 768, H/14, W/14]
功能定位：带注册令牌的 DINOv2 编码器，支持分层特征提取（即从指定的 Transformer 阶段提取特征）。
核心机制：保留注册令牌的设计，让其承担全局计算任务，保证输出特征图 “干净无伪影”；每个 Transformer 阶段通过自注意力处理令牌时，都会借助注册令牌完成全局信息整合。
特征维度关键规则
Patch 令牌数量计算：输入图像尺寸 ÷ 14（patch 大小）得到网格数，网格数的平方即为 patch 令牌数（如 224÷14=16 → 16×16=256 个令牌）。
输出特征形状：所有模型变体的输出特征都保持 [B, D, H/14, W/14] 的空间布局：
B：批次大小；
D：隐藏维度（随模型变体不同：Giant=1536、Large=1024、Base=768）；
H/14/W/14：特征图尺寸（输入尺寸 ÷ patch 大小 14），完美呼应 “输入尺寸必须是 14 的倍数” 的校验逻辑。
    """
    
    def __init__(
        self, 
        model_name: str = "facebook/dinov2-with-registers-large",#指定加载模型
        variant: str = "large",
        output_stages: Optional[List[int]] = [1,2,23,24]
    ):
        """
        Initialize the DINOv2 encoder with register tokens.

        Args:
            model_name: HuggingFace model identifier
            variant: Model size variant ('base', 'large', 'giant')
            output_stages: List of stage indices to output (1-indexed)
                         If None, outputs all stages
        """
        super().__init__()
        
        # Map variant names to hidden dimensions
        self.hidden_dims = {
            'small': 384,
            'base': 768,
            'large': 1024,
            'giant': 1536
        }
        
        if variant not in self.hidden_dims:
            raise ValueError(f"Invalid variant. Choose from: {list(self.hidden_dims.keys())}")
        
        self.variant = variant
        self.hidden_dim = self.hidden_dims[variant]
        self.patch_size = 14
        
        # Setup output stages (1-indexed for clarity)
        self.output_stages = output_stages or list(range(1, 13))
        stage_features = [f'stage{i}' for i in self.output_stages]
        
        # Initialize backbone with register tokens初始化注册令牌dino骨干
        self.model = AutoBackbone.from_pretrained(
            model_name,
            out_features=stage_features
        )
        
        logger.info(
            f"Initialized {variant} encoder with {self.param_count:,} parameters. "
            f"Output stages: {self.output_stages}"
        )

    @property
    def param_count(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())#返回模型中可训练参数的总数（属性）

    def get_patch_grid_size(self, height: int, width: int) -> Tuple[int, int]:
        """
        Calculate patch grid dimensions for given input size.
        
        Args:
            height: Input image height
            width: Input image width
            
        Returns:
            Tuple of (grid_height, grid_width)
        """
        return height // 14, width // 14

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass generating features with register token attention.

        The input image is divided into 14x14 patches, each processed through the
        transformer while interacting with register tokens. Register tokens participate
        in self-attention but are discarded in the final output.

        Args:
            x: Input images [B, 3, H, W]
               Supported sizes:
               - 224x224 → 16x16 patch grid
               - 512x512 → 36x36 patch grid
               Height and width must be divisible by 14

        Returns:
            List of feature tensors for requested stages
            Each tensor shape: [B, hidden_dim, H/14, W/14]
            - Base:   hidden_dim = 768
            - Large:  hidden_dim = 1024
            - Giant:  hidden_dim = 1536
        """
        B, C, H, W = x.shape
        target_h = ((H + self.patch_size - 1) // self.patch_size) * self.patch_size
        target_w = ((W + self.patch_size - 1) // self.patch_size) * self.patch_size
        if target_h != H or target_w != W:
            x = F.interpolate(
                x,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False,
            )
            
        # Process through backbone with register tokens
        outputs = self.model(x)#输入图像 x 传入已加载的 DINOv2 骨干网络 self.model（即 AutoBackbone 加载的模型）
        
        return outputs.feature_maps

    def __repr__(self) -> str:
        """Detailed string representation with architecture info."""
        return (
            f"DinoV2WithRegisterEncoder(\n"
            f"  variant={self.variant}\n"
            f"  hidden_dim={self.hidden_dim}\n"
            f"  output_stages={self.output_stages}\n"
            f"  params={self.param_count:,}\n"
            f"  model={self.model}\n"
            f")"
        )
