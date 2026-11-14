# 统一去模糊对比实验套件

本目录将现有 **MPRNet**、**NAFNet**、本文 **MBDNet-P+** 与 8 个开源方法置于同一训练、验证和测试协议下。它用于论文中的横向对比；网络内部模块消融仍应以 MBDNet-P+ 的模块开关实验单独呈现。

## 纳入模型

| 类别 | 模型 |
|---|---|
| 已有基线 | MPRNet、NAFNet |
| 注意力/Transformer 代表 | Restormer、Stripformer、LoFormer-S |
| 多尺度卷积与运动恢复代表 | BANet、MIMO-UNet+、HINet、UFPNet、SFNet |
| 本文方法 | MBDNet-P+ |

各第三方项目保留在 `third_party/` 中，未改写其网络层实现；本套件仅提供输入输出适配、统一训练和测试驱动。

## 固定公平协议

- 数据根目录：`H:/DATASET`
- 清单目录：`H:/C3Netqing/MISCFilter/manifests/c3net_multisource_linetrace_synblur_v2`
- 训练/验证来源：DB、XJ、SD、RX；训练集 `train.tsv`、验证集 `val.tsv`。
- 外部纯测试：RXHF、TJML；两者不参与训练、验证、模型选择或退化参数调整。
- 统一预算：从头训练 30 epochs，随机种子 42，裁剪尺寸 320，余弦退火和 PSNRLoss；验证与默认测试保持 FP32。
- 有效批量大小固定为 4。除 SFNet（其选择频率层的 BatchNorm 要求 `batch≥2`）外，10 个对比网络使用低显存配置 `batch=1, accumulation=4`；SFNet 和 MBDNet-P+ 使用 `batch=2, accumulation=2`。这不会改变每次参数更新所对应的有效样本数。
- 统一测试输出 Full / Middle-third 的 PSNR、SSIM 与每张图平均推理时间。测试时不读取退化参数、边缘标签或缺陷标签。
- 除了官方训练明确采用低学习率 Adam/FP32 的 BANet、Stripformer、MIMO-UNet+、HINet 与 SFNet 外，模型使用协议默认的 AdamW、`lr=1e-3`。该例外仅用于防止已验证的数值发散；所有方法的训练数据、验证数据、裁剪、增强、轮数与有效批量大小不变。

## 特殊实现约定

- MIMO-UNet+、SFNet 的低分辨率辅助输出和 UFPNet 的核图不参与统一损失；套件只以与输入同尺寸的恢复图计算损失和指标，避免为某一网络单独插值 GT 而改变损失定义。
- UFPNet 的核先验支路按照公开实现保留；该模型在本协议中只从图像本身估计核，不读取退化参数或测试标签。
- 所有模型均以本套件产生的检查点参与测试。旧项目检查点不会被静默混用，因为其数据协议或网络配置可能不同。

## 常用命令

在 `c3net` 环境中执行：

```powershell
cd H:\C3Netqing\deblur_comparison_suite

# 示例：训练 Restormer；其它模型仅替换 -Model 参数
.\scripts\train\run_model.ps1 -Model restormer -Epochs 30

# 示例：测试最佳权重
.\scripts\test\run_model_test.ps1 `
  -Model restormer `
  -Checkpoint .\checkpoints\restormer_multisource_v2_s30_s42\model_best.pth

# 汇总已完成模型的测试结果
.\scripts\evaluation\run_summarize.ps1
```

可用 `-Model`：`mbdnet_p_plus`、`mprnet`、`nafnet`、`restormer`、`stripformer`、`banet`、`loformer`、`mimo_unet_plus`、`hinet`、`ufpnet`、`sfnet`。训练脚本默认断点自动续训；若需禁用，追加 `-NoAutoResume`。

## 官方代码与论文来源

- MPRNet: https://github.com/swz30/MPRNet
- NAFNet: https://github.com/megvii-research/NAFNet
- Restormer: https://github.com/swz30/Restormer
- Stripformer / BANet: https://github.com/pp00704831/Stripformer ，https://github.com/pp00704831/banet
- LoFormer: https://github.com/INVOKERer/LoFormer
- MIMO-UNet+: https://github.com/chosj95/MIMO-UNet
- HINet: https://github.com/megvii-model/HINet
- UFPNet: https://github.com/Fangzhenxuan/UFPDeblur
- SFNet: https://github.com/c-yn/SFNet
