# C3Net 后端六单元线迹缺陷检测头

## 结论与边界

本模块命名为 **MG-SixDefect Head**（Mask-Guided Six-Unit Defect Head）。它接在已经完成的“复原 -> C3Net 分割”之后，输入 C3Net 的 `final_mask`、`centerline_mask` 与同一张复原 RGB 图，输出每张图中从左至右六个线迹单元的缺陷概率。

它不是论文中 SPMM 的直接复刻。SPMM 的像素距离转毫米、阈值和测量误差验证建立在显微镜标定图像上；本数据是 48 张 `2448 x 2048` 的普通清晰布料图，布纹/花纹强且没有显微标尺。把 SPMM 的绝对阈值迁移过来会产生不可解释的误报。因此，本模块只使用**图内归一化**的几何量，并把最终判别交给轻量可学习模型。

## 与现有 C3Net 的接口

```python
from models.stitch_defect_head import StitchDefectHead

segmenter_output = c3net(restored_rgb)  # existing C3Net forward pass
defect_output = defect_head(
    segmenter_output["final_mask"],
    segmenter_output.get("centerline_mask"),
    restored_rgb,
)
# defect_output["unit_logits"]: [B, 6, 4]
# class order: skip, break, skew, loose
```

`final_mask` 和 `centerline_mask` 均为 logits；检测头内部才执行 sigmoid，故训练时可端到端求梯度。实际首轮训练建议冻结 C3Net：先避免 288 个单元级样本反向破坏已验证的分割性能。RGB 是可选输入，但正式版本应保留，因为仅靠二值掩码无法可靠区分“线迹变宽/浮起”和分割误差。

## 结构

```text
C3Net final mask + centerline + restored RGB
        │
        ├── mask gate: RGB x mask, RGB x centerline
        │
        ├── DW-CNN: 抑制布料花纹后的局部视觉证据
        │
        ├── Adaptive pool -> 6 个从左到右软 ROI token
        │                         │
        ├── 几何 token: 面积、中心线占比、y 中心、宽度、
        │               左右位移、横向变化、纵向占据率
        │                         │
        └── 2-layer Transformer: 相邻 6 段的相对关系
                                  │
                                  └── 4 个 unit-level sigmoid logits
```

采用固定六 ROI 而不先做连通域分割：跳线或断线恰好会让连通域数量不可靠；自适应池化使输入可为 C3Net 的 `320 x 320` 输出或被插值回原尺寸的输出。当前采集中的线迹方向近似水平，这是该顺序编码成立的前提。若以后有竖向或大角度缝线，采集端先旋转到缝线方向，或在前面加入轨迹整正模块。

## 标签与判定定义

输出采用**多标签**，而不是五分类/五选一。一个单元可同时“歪斜 + 松线”；硬互斥会把真实复合缺陷压成单类。`loose` 暂合并“浮线/松线”，因为在当前单视角宏观图上二者的可靠边界取决于工艺定义。若你能给出分别可复核的实例，再把最后一层扩成五维即可。

| 输出 | 标注定义（以六单元的正常邻域为参照） | 首要证据 |
|---|---|---|
| `skip` | 中间缺少一个应有针迹，但其左右仍恢复到正常节奏 | 低占据率 + 单个异常大间距/区域 |
| `break` | 线迹本体中断或末端消失，不能被视为单个漏针 | 持续低中心线连通性 + 断裂边界 |
| `skew` | 单元中心线相对相邻单元显著偏移或局部倾角异常 | 左右中心高度差、相邻 token 关系 |
| `loose` | 线迹浮起、松弛、局部宽度/面积或双边缘异常 | 掩码宽度/面积 + 被掩码 RGB 纹理 |

“正常”不单独设 logits；四个缺陷均低于阈值即正常。部署阈值必须在验证集按每类 PR 曲线选择，不使用 `0.5` 作为论文最终阈值。输出时同时保存 `geometry`，便于审计某一次判定的形状证据。

## 数据准备与训练协议

48 张图只能产生最多 288 个单元标签，适合作为试验集而不是足以宣称泛化的正式大样本数据集。

1. 对每张图把 C3Net 预测叠加到原图；按从左到右写 `unit_id=0..5`，每个单元填写四个 0/1 标签。裁剪、遮挡或不确定单元标为 `valid=0`，不要伪造负样本。
2. 至少让每类缺陷有 30--40 个已复核正单元；若做不到，先把对应类别保留为规则报警，不训练该类。测试前固定“跳线”和“断线”的工艺定义，避免标签自相矛盾。
3. 以**整张图、最好以布料/花纹批次**做 Group 5-fold 交叉验证；严禁把同一图的六段拆到训练和测试，后者会高估结果。
4. 冻结 C3Net，训练 `StitchDefectHead`，采用带训练集内 `pos_weight` 的 BCE（实现为 `multilabel_defect_loss`），报告每类 AP、F1、recall 及 image-level false alarm。使用轻微亮度、噪声、模糊、压缩和小角度旋转增强；不做会改变线迹几何的随机透视或大幅弹性变形。
5. 只在交叉验证稳定后尝试端到端微调；此时 C3Net 学习率至少低一个数量级，并同时保留分割 Dice/topology loss。

建议的单元标签 JSONL：

```json
{"image":"P_2026_07_27_213043_1.jpeg","units":[
  {"unit_id":0,"valid":1,"skip":0,"break":0,"skew":0,"loose":0},
  {"unit_id":1,"valid":1,"skip":1,"break":0,"skew":0,"loose":0}
]}
```

## 为什么这个设计适合当前数据

- 数据集内布料图案、颜色和纹理剧烈变化，直接把全图交给分类器很容易学到背景；mask gate 强制模型先看 C3Net 已定位的线迹区域。
- 一图固定六单元使序列关系可学：判定依据是“相对该图其他五段的异常”，而非不可迁移的显微像素尺度。
- 既保留几何可解释性，也保留 RGB 对浮/松线的辅助证据；只用 SPMM 几何规则会丢失后一类信息，只用分类 CNN 又丢掉断线/跳线的连通性先验。

## 相关研究如何影响设计

- Hu & Zhang（2024）先做 U-Net 线迹特征图，再依邻近轮廓距离判别断线，支持“分割后再做拓扑/间距分析”的主链路；本设计将其固定阈值替换为六单元的可学习相对关系。[论文页](https://journals.sagepub.com/doi/10.1177/00405175241259204)
- Kim et al.（2022）以 CNN feature map 结合图像处理检测断线，说明单独的分类/检测框不足以处理细线结构；因此本模块保留掩码和中心线输入，而没有替换 C3Net。[论文信息](https://doi.org/10.1016/j.eswa.2021.116014)
- Jung et al.（2025）的 StitchingNet 覆盖正常与十类缺陷并强调小样本迁移训练，说明应保持 unit-level 多类别标注和严格的按图分组验证；其数据分布不同，不能把其权重直接当作本宏观图的 ground truth。[论文页](https://academic.oup.com/jcde/article/12/4/140/8100483)

## 已提供的工程文件

- `models/stitch_defect_head.py`：可训练检测头、四类常量和带有效单元掩码的损失。
- `defect_detection/test_stitch_defect_head.py`：CPU forward/backward smoke test；运行 `python defect_detection/test_stitch_defect_head.py`。

这一步只提供结构和训练接口，未对 48 张无标注图虚构准确率或生成虚假缺陷标签。
