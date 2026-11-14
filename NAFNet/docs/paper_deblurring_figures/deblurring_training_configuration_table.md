# 表 X　去模糊模型的实际训练配置

本表对应十一模型对比结果中的 MBDNet-P+（实际实验标识为 MBDNetPPlusR12）及十个对比模型。配置依据实际运行记录、训练入口和测试结果中记录的权重路径整理。基线使用官方开源网络实现，在本项目的数据协议下重新训练；下列设置属于本项目的训练配置，不能整体称为“官方推荐训练配置”。

| 模型 | 初始化／预训练来源 | 训练数据 | 优化器；β；权重衰减 | 初始学习率 | 学习率调度 | 轮数 | 物理批量 × 梯度累积 | 裁剪与增强 | 训练精度 | 权重选择 |
|---|---|---|---|---|---|---|---|---|---|---|
| MBDNet-P+ | 随机初始化；未指定外部预训练权重 | D | AdamW；(0.9, 0.9)；1×10⁻³ | 1×10⁻³ | C30 | 30 | 2 × 2 | A | AMP | V |
| LoFormer-S | 官方代码构建；随机初始化，未加载官方预训练权重 | D | AdamW；(0.9, 0.9)；1×10⁻³ | 1×10⁻³ | C30 | 30 | 1 × 4 | A | AMP | V |
| Restormer | 官方代码构建；随机初始化；记录中的 resume 为同一实验续训 | D | AdamW；(0.9, 0.9)；1×10⁻³ | 1×10⁻³ | C30 | 30 | 2 × 2 | A | AMP | V |
| MPRNet | 原始 MPRNet 结构；未指定外部预训练权重；允许同实验断点续训 | D | Adam；(0.9, 0.999)；0 | 2×10⁻⁴ | 3 轮预热＋27 轮余弦衰减至 1×10⁻⁷ | 30 | 2 × 2 | A-MPR | AMP | V |
| NAFNet | 官方 NAFNet 结构；随机初始化，init_checkpoint 为空 | D | AdamW；(0.9, 0.9)；1×10⁻³ | 1×10⁻³ | C30 | 30 | 2 × 2 | A | AMP | V |
| HINet | 官方代码构建；随机初始化，未加载官方预训练权重 | D | Adam；(0.9, 0.99)；0 | 2×10⁻⁴ | C30 | 30 | 1 × 4 | A | FP32 | V |
| MIMO-UNet+ | 官方代码构建；随机初始化，未加载官方预训练权重 | D | Adam；(0.9, 0.999)；0 | 1×10⁻⁴ | C30 | 30 | 1 × 4 | A | FP32 | V |
| BANet | 官方代码构建；随机初始化，未加载官方预训练权重 | D | Adam；(0.9, 0.9)；0 | 1×10⁻⁴ | C30 | 30 | 1 × 4 | A | FP32 | V |
| Stripformer | 官方代码构建；随机初始化，未加载官方预训练权重 | D | Adam；(0.9, 0.9)；0 | 1×10⁻⁴ | C30 | 30 | 1 × 4 | A | FP32 | V |
| SFNet | 官方代码构建；随机初始化，未加载官方预训练权重 | D | Adam；(0.9, 0.9)；0 | 1×10⁻⁴ | C30 | 30 | 2 × 2 | A | FP32 | V |
| UFPNet* | 官方代码构建；未加载官方预训练权重或核先验权重* | D | AdamW；(0.9, 0.9)；1×10⁻³ | 2×10⁻⁴ | C30 | 30 | 1 × 4 | A | FP32 | V |

**表注。** D：MultiSource SynBlur-v2 的 DB、XJ、SD、RX 四域配对数据；训练集 4320 对，验证集 160 对。训练清晰图身份共 1440 张，每张对应三种离线退化。RXHF、TJML 为外部测试域，不在训练／验证清单中。全部运行随机种子为 42。

C30：按轮更新的 CosineAnnealingLR，T_max=30，eta_min=1×10⁻⁷，无预热。eta_min 为调度器设定的衰减下限，不等于第 30 轮开始时的学习率。

A：输入与目标同步随机裁剪至 320×320，独立以 0.5 概率进行水平翻转、垂直翻转和 180° 旋转；训练时不额外施加颜色、亮度或噪声增强。三项变换的组合仍保持水平运动方向。离线退化生成过程中的扰动属于数据构造，和这里的在线增强不同。

A-MPR：同步随机裁剪至 320×320，采用保持水平运动方向的恒等、水平／垂直翻转及 180° 旋转。现有实现从 (0,1,2,4,8) 五种编号等概率抽取，其中 0、8 均为恒等操作，因此恒等概率为 2/5，其余三种各为 1/5；不应写成与 A 的抽样概率完全相同。颜色、gamma、输入噪声及低照度增强概率均为 0。

V：每轮验证，按配对验证集的平均 RGB PSNR 最高值保存 model_best.pth；不是按测试集 PSNR 或中带指标选择权重。对比套件的八个基线结果文件明确记录使用 model_best.pth。MBDNet-P+、NAFNet 和 MPRNet 的训练／测试入口也采用该标准；本表记录的是当前源码、配置及日志支持的选择规则，不额外假定未保存的早期运行行为。

**损失函数差异。** MBDNet-P+、NAFNet 及对比套件中的八个基线使用 PSNR 型重建损失。套件对网络返回的全分辨率恢复输出分别计算该损失并取均值；MIMO-UNet+、SFNet 的低分辨率输出未纳入多尺度监督，不能表述为保留了它们的完整官方损失。MPRNet 使用三阶段加权 Charbonnier 重建损失与 0.05 倍 EdgeLoss，阶段权重为 (1, 0.5, 0.25)。配置中的 EDGE_WEIGHT=0 对应额外边缘图监督开关，当前 train.py 中的标准 EdgeLoss 项仍保留。

**配置来源与调参说明。** 所有模型共享配对数据和 30 轮最终训练预算，但优化器、学习率、训练精度及损失监督有所不同。BANet、Stripformer、MIMO-UNet+ 使用低学习率 FP32 稳定配置；历史日志存在高学习率 AMP 下非有限损失的记录。HINet、SFNet、UFPNet 使用各自当前记录的 FP32 配置。现有记录支持“为训练稳定性作了模型相关调整”，不支持“对所有模型进行了同等预算的充分超参数搜索”或“完整复现各模型官方训练流程”。

**UFPNet 的必要说明。** 当前工厂仅构建 UFPNet 及其核先验模块，未加载相关预训练文件；核编码器和流模型在 forward 中位于 torch.no_grad() 内，因而没有通过当前图像重建损失学习。该行应作为本项目实现结果披露，不能据此宣称完整复现官方 UFPNet。正式使用该对比支撑模型优劣前，应补齐核先验的初始化／训练流程并重新评估。

## 可放在论文中的训练设置说明

所有去模糊模型均在 MultiSource SynBlur-v2 的 DB、XJ、SD 和 RX 四域配对训练集上训练，使用相同的训练／验证清单；RXHF 和 TJML 仅用于外部测试。各模型最终训练 30 轮，输入裁剪尺寸为 320×320，等效批量为 4，随机种子为 42。本文采用官方开源网络实现，并在本项目训练框架下设置优化器、学习率与训练精度，具体配置见表 X。其中，部分模型为提高训练稳定性采用较低学习率和 FP32 训练。MPRNet 保留三阶段重建及边缘损失监督，其他模型采用本项目 PSNR 型重建目标，因此本实验属于指定数据和训练预算下的重训练比较，而非各模型完整官方训练流程的复现。所有模型均依据验证集平均 PSNR 选择最佳权重，测试集指标不用于权重选择。

上述正文必须与 UFPNet 表注同时保留；在补齐其核先验流程前，不能把它无条件纳入“完整官方复现”的表述。MBDNet-P+ 行对应 R12 的 30 轮运行，不能套用另一版模型的“30＋10＋5 轮微调”配置。

## 核对依据（供作者复查，不必进入正文）

- 主模型：H:/C3Netqing/NAFNet/checkpoints/MultiSourceV2_MBDNetPPlusR12_S30_S42/run_config.json；同目录 metrics.jsonl。
- NAFNet：H:/C3Netqing/NAFNet/checkpoints/MultiSourceV2_A0_NAFNet_Fair_S30_S42/run_config.json；同目录 metrics.jsonl。
- NAFNet／主模型训练逻辑：H:/C3Netqing/NAFNet/train_nafnet_c3.py；H:/C3Netqing/NAFNet/c3net_data.py。
- MPRNet：H:/C3Netqing/Deblurring/training_mprnet_original_multisource_synblur_v2_s30.yml；H:/C3Netqing/Deblurring/train.py；H:/C3Netqing/Deblurring/dataset_RGB.py。
- 八个基线：H:/C3Netqing/deblur_comparison_suite/checkpoints/ 下与 results/*/summary.json 中 checkpoint 字段对应的 run_config.json、history.jsonl。
- 公共训练入口：H:/C3Netqing/deblur_comparison_suite/train_comparison.py；data_protocol.py；model_factory.py；metric_protocol.py。
- UFPNet 核先验：H:/C3Netqing/deblur_comparison_suite/third_party/UFPDeblur/basicsr/models/archs/UFPNet_code_uncertainty_arch.py、Flow_arch.py、my_module.py。
- 数据协议：H:/C3Netqing/MISCFilter/manifests/c3net_multisource_linetrace_synblur_v2/metadata.json、train.tsv、val.tsv。

本次补表使用 nature-writing 的证据约束：把可核对的运行设置与官方默认配置区分，并显式披露影响对比解释的差异。
