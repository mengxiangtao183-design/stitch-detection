# 近期工作总结 - 组会汇报

日期：2026-07-13

## 1. 当前项目任务

本项目面向工业缝纫线迹检测，整体流程为：

```text
退化/模糊图像
  -> MPRNet 图像修复
  -> C3Net 线迹轨迹检测
  -> 后续工程规则判断断线、歪线、抛线、跳线等缺陷
```

当前阶段的重点不是缺陷分类，而是先稳定检测出线迹轨迹。

## 2. MPRNet 修复网络近期工作

原本效果较好的修复模型是：

```text
MPRNet_EdgeFT
```

对应修复图目录为：

```text
Imgs1_MPRNet_EdgeFT
```

该版本目前仍是修复效果最稳定的主线模型。

近期尝试构建了一个面向横向缝纫线迹的新 MPRNet 版本：

```text
LMPRNet_TrajectoryPair_NAF2_OV020_KD020
```

模型位置：

```text
H:\C3Netqing\Deblurring\checkpoints\train\Deblurring\models\LMPRNet_TrajectoryPair_NAF2_OV020_KD020
```

该版本的主要改动包括：

```text
1. 加入光照自适应模块，用于适应工厂场景中偏暗、曝光变化的图像。
2. Stage1 从原始 MPRNet 的四块切分，改为三条横向 full-width 条带。
3. 引入共享中间条带的成对解码：
   stripe1 + stripe2 -> decoder
   stripe2 + stripe3 -> decoder
4. Stage2 使用 Directional-NAF，用横向/竖向方向卷积增强线迹方向信息。
5. Stage3 使用 MultiScaleFusionNAF 替代原 ORSNet。
6. 保留边缘引导细化模块 Edge-Guided Refinement。
7. 采用两阶段训练：
   先 NoKD 训练 20 轮
   再引入原始 MPRNet teacher，KD0.20 微调 40 轮。
```

该模型已经完成训练，最终 checkpoint 正常：

```text
model_latest.pth epoch=40
model_epoch_40.pth epoch=40
```

## 3. 新 MPRNet 批量修复结果

使用 `LMPRNet_TrajectoryPair_NAF2_OV020_KD020` 生成了新的修复图，输出目录为：

```text
Imgs1_MPRNet_TrajectoryPair_KD020
```

生成数量如下：

```text
DB/train: 400 张
XJ/train: 400 张
SD/train: 400 张

DB/test: 100 张
XJ/test: 100 张
SD/test: 100 张
RX/test: 500 张
```

但是实验结果不理想。和 `MPRNet_EdgeFT` 相比，新模型 PSNR 明显下降：

```text
DB/test: 28.71 -> 22.73
XJ/test: 32.96 -> 26.75
SD/test: 30.16 -> 20.61
RX/test: 22.88 -> 20.79
```

同时，肉眼观察也能明显看到修复图质量下降。

因此当前结论为：

```text
LMPRNet_TrajectoryPair_NAF2_OV020_KD020 不作为最终修复网络。
MPRNet_EdgeFT 仍作为当前主线修复网络。
```

该实验说明：

```text
对 MPRNet 主体结构进行大幅替换会破坏像素级修复质量。
当前数据量不足以支撑这种大结构重构。
后续 MPRNet 改进应以 MPRNet_EdgeFT 为基础，只做保守增强。
```

## 4. MPRNet 训练流程澄清

近期还澄清了第二阶段训练是否使用 teacher。

实际训练流程如下：

```text
第一阶段：
LMPRNet_TrajectoryPair_NAF2_OV020_NoKD
无 teacher，训练 20 轮。

第二阶段：
LMPRNet_TrajectoryPair_NAF2_OV020_KD020
从 Pair_NAF2_OV020_NoKD 权重初始化。
引入 MPRNet teacher。
KD 权重为 0.20。
训练 40 轮。
```

需要注意：

```text
KD020 不是接着 LMPRNet_TrajectoryNAF3_NoKD 训练的。
它接的是 LMPRNet_TrajectoryPair_NAF2_OV020_NoKD。
```

准确实验记录应写为：

```text
先训练 TrajectoryPair_NAF2_OV020_NoKD 20 轮，
再以其权重初始化，引入 MPRNet teacher 进行 KD0.20 微调 40 轮。
```

## 5. C3Net 检测网络近期梳理

当前 C3Net 是主要创新方向，结构为：

```text
C3Net-MobileNetV4-Medium-FPN-KD
```

整体流程为：

```text
输入修复图像
  -> MobileNetV4 backbone
  -> Mobile FPN 多尺度融合
  -> 边缘轨迹分支 Edge / MRFA-DTE
  -> 上下文定位分支 Loc / ICG
  -> FinalFusionHead 融合
  -> High-Resolution Stitch Refine Head
  -> 输出 final_mask / object_mask / edge_mask / centerline_mask
```

当前重点模块包括：

```text
1. MobileNetV4 backbone
   用于替换原始 DINOv2 推理骨干，实现轻量化。

2. DINOv2 teacher 蒸馏
   DINOv2 只作为训练 teacher，不参与推理。

3. MRFA-DTE Block
   多感受野方向轨迹增强模块。
   使用 3x3、1xK、Kx1、空洞卷积分支增强线迹轨迹特征。

4. Centerline Head
   增加中心线预测分支，不需要人工绘制中心线。
   中心线标签由 GT mask 自动骨架化生成。

5. Topology Loss / Boundary Loss
   用于增强线迹连续性和边界几何稳定性。

6. High-Resolution Stitch Refine Head
   在高分辨率下细化 final_mask、edge_mask、centerline_mask。
```

当前 C3Net 的创新重点应放在：

```text
MRFA-DTE
centerline head
topology/boundary loss
high-resolution stitch refinement
MobileNetV4 轻量化蒸馏
```

而不是继续大改 MPRNet。

## 6. MRFA-DTE 模块作用

MRFA-DTE 的作用是增强线迹轨迹特征。

其结构包括：

```text
1. 3x3 局部分支
2. 1xK 横向条带分支
3. Kx1 竖向条带分支
4. dilation=2 空洞卷积分支
5. dilation=3 空洞卷积分支
6. softmax gate 自适应融合
```

它主要解决：

```text
线迹细、长、连续，容易模糊和断裂；
普通卷积难以同时捕获局部边缘和长距离方向信息。
```

MRFA-DTE 的意义是：

```text
让边缘分支不仅检测边界，还能感知线迹轨迹方向和断裂上下文。
```

## 7. 近期踩过的坑

近期明确踩过以下几个坑：

```text
1. MPRNet 不能大改主结构。
   大改 Stage1/Stage2/Stage3 后，PSNR 和肉眼质量都明显下降。

2. KD teacher 不能拯救一个结构破坏太大的 student。

3. 不应把 TrajectoryPair_KD020 作为 C3Net 默认输入。

4. 后续 MPRNet 改进应从 MPRNet_EdgeFT 出发，只做小模块增强。

5. 论文主创新应放在 C3Net 检测网络，而不是把失败的 MPRNet 大改作为主模型。
```

## 8. 下一步计划

后续建议按以下方向推进：

```text
1. 使用 MPRNet_EdgeFT 作为修复网络主线。
2. 用 Imgs1_MPRNet_EdgeFT 训练当前改进版 C3Net。
3. 和原始 C3Net 做主对比。
4. 做 C3Net 消融实验：
   原版 C3Net
   + MobileNetV4
   + MRFA-DTE
   + centerline head
   + topology/boundary loss
5. 评价指标重点放在：
   Weighted-F
   Binary F1
   Trajectory F1@1px
   Trajectory F1@2px
   Centerline F1
   RX 单独结果
   推理时间
   参数量
```

## 9. 汇报结论

可以在组会上这样总结：

```text
近期尝试了面向横向线迹的 MPRNet 结构重构，但实验表明大幅修改修复网络会显著损伤复原质量。因此当前将修复网络回退到表现稳定的 MPRNet_EdgeFT，把主要创新集中在 C3Net 检测阶段，通过轻量化 MobileNetV4、MRFA-DTE 轨迹增强、中心线辅助监督和拓扑约束提升线迹检测效果。
```
