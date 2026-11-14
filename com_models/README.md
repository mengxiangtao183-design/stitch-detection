# 去模糊图像条件下的分割对比实验套件

本目录用于运行**外部方法对比实验**，不属于 C3Net 内部模块的消融实验。
所有模型使用同一输入协议：`320×320` 的 `clear` 图像（去模糊/清晰图像）、DB+XJ+SD 训练、DB+XJ+SD+RX 测试，其中 RX 始终不参与训练，作为外部泛化集。

## 已接入的模型

| 类别 | 配置文件 | 实现状态 | 说明 |
|---|---|---:|---|
| U-Net | `configs/unet_full150.yaml` | 可训练 | 经典编码器—解码器基线。 |
| DeepLabV3+ | `configs/deeplabv3plus_full150.yaml` | 可训练 | ResNet-50 + ASPP，采用 ImageNet 预训练。 |
| SegFormer-B0 | `configs/segformer_b0_full150.yaml` | 可训练 | MiT-B0 + MLP 解码器；本地预训练权重位于 `pretrained/mit-b0/`。 |
| Gated-SCNN-style | `configs/gated_scnn_full150.yaml` | 可训练 | 规则流/形状流双分支的项目内复现，适合边界敏感对比。 |
| PointRend-R50 | `configs/pointrend_r50_full150.yaml` | 可训练 | 粗分割加不确定点细化的项目内复现。 |
| PIDNet-S | `configs/pidnet_s_full150.yaml` | 可训练 | CVPR 2023 的 P/I/D 三分支语义分割架构；核心源码来自官方实现，项目仅添加输出接口适配。 |
| PSPNet-R50 | `configs/pspnet_r50_full150.yaml` | 可训练 | ResNet-50 + 金字塔池化的多尺度上下文基线。 |
| OPNet | `configs/opnet_full150.yaml` | 可训练 | 目录内已有的 COD 方法实现；采用 FP32、logit 限幅与 0.5 梯度裁剪的稳定配置。 |
| FSPNet | `configs/fspnet_full150.yaml` | 可训练 | 目录内已有的 COD 方法实现；Sigmoid 输出在 FP32 下安全转回 logits，采用 batch=1 的稳定配置。 |
| FSELNet | `configs/fselnet_full150.yaml` | 可训练 | 目录内已有的频率—空间 COD 方法实现；为避免 10×10 FFT 的半精度数值问题，本配置固定 `use_amp: false`。 |

已对八个模型完成统一输出接口、随机张量前向和反向传播检查；训练输出均为 `final_mask`、`object_mask` 与 `edge_mask`，可直接接入现有训练、评估和预测流程。

## 训练

请先激活 `c3net` 环境，再在 `H:\C3Netqing` 下执行。每条命令为单行：

```powershell
python main.py train --config com_models\configs\unet_full150.yaml
python main.py train --config com_models\configs\deeplabv3plus_full150.yaml
python main.py train --config com_models\configs\segformer_b0_full150.yaml
python main.py train --config com_models\configs\gated_scnn_full150.yaml
python main.py train --config com_models\configs\pointrend_r50_full150.yaml
python main.py train --config com_models\configs\pidnet_s_full150.yaml
python main.py train --config com_models\configs\pspnet_r50_full150.yaml
python main.py train --config com_models\configs\opnet_full150.yaml
python main.py train --config com_models\configs\fspnet_full150.yaml
python main.py train --config com_models\configs\fselnet_full150.yaml
```

全部配置固定：随机种子 42、最多 150 epoch、全量训练集、以验证集 `binary_iou` 保存最佳权重；若该指标连续 25 轮未提升则早停。若 OPNet 或 FSPNet 出现显存不足，只需在各自 YAML 中将 `training.batch_size` 改为 `1`；不要改变训练数据划分和输入分辨率。

## 评估与报告

训练结束后，将下面 `<checkpoint>` 替换为该次 `results/training/runs/.../checkpoints/model_best.pth` 的真实路径：

```powershell
python main.py evaluate --config com_models\configs\unet_full150.yaml --model <checkpoint>
```

对其他模型只需替换配置文件与对应最佳权重。若需报告工业在线场景的单张检测时间，请使用同名的 `*_timing_bs1.yaml` 配置；其将评估 batch 固定为 1，日志末尾的 `Average inference time` 即单张模型前向时间。论文表格建议统一报告：Binary IoU、Binary F1、Binary Recall、Boundary F1@1px 与 batch=1 单张平均推理时间。不要将不同网络的“去模糊输入条件对比”误写为内部模块消融。

## 预训练与排除项

- DeepLabV3+、SegFormer-B0、Gated-SCNN-style 和 PointRend-R50 已启用相应的通用主干预训练策略。
- OPNet、FSPNet、FSELNet 的目录内实现可运行，但若要在论文中声明“复现原论文设置”，仍应补充其官方发布的预训练权重或严格遵循原论文训练策略；否则应表述为“在统一本项目协议下重新训练”。
- `PVTNet_model.py` 依赖缺失的 `pretrained_pvt/pvt_v2_b2.pth`；`sam2_model.py` 是提示式 SAM2 路线，既缺少稳定的本地初始化链路，也不属于公平的自动全监督语义分割对比，因此未纳入正式表格。

## 代码入口

- 统一模型分发：`registry.py`
- 五个标准对比模型：`standard_models.py`
- 本项目既有 COD 模型：`OPNet_model.py`、`FSPNet_model.py`、`fselnet_model.py`
- 训练/评估/预测已统一调用 registry：`../engine/trainer.py`、`../engine/evaluator.py`、`../engine/predictor.py`、`../main.py`
