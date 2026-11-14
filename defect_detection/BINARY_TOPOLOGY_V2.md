# 二值拓扑缺陷单元第二版

第二版在 v1 的 33 维单元表示上增加 8 维有序空间特征，共 41 维。新增量包括左、中、右三个区域的前景占用率、最大内部缺口的位置与两侧前景平衡、前景中心偏移，以及左右末端分别缺失的比例。设计目标是区分“内部缺口后重新出现”的跳线与“只剩短残留”的断线。输入仍为 ACHNet 预测的二值掩码；外层四折、内层三折、受控增强来源隔离和正常留出划分均保持不变。

两个运行点：

1. `configs/nested_fourfold_joint_clear_deblur_112_binary_topology_v2.yaml`：沿用 v1 的内层准确率目标和 0.85 告警召回下限。
2. `configs/nested_fourfold_joint_clear_deblur_112_binary_topology_v2_alarm95.yaml`：内层以宏平均 F1 为目标，并要求缺陷告警召回率至少 0.95；约束仅使用内层折外来源确定。没有用外层测试预测重新设阈值。

| 版本与路径 | 图像级宏平均 F1 | 图像级准确率 | 缺陷告警召回率 |
|---|---:|---:|---:|
| 概率轨迹，清晰 | 89.52% | 87.50% | 95.54% |
| v1 二值拓扑，清晰 | 87.08% | 85.71% | 97.32% |
| v2 二值拓扑，清晰 | 87.09% | 83.93% | 92.86% |
| v2 告警约束，清晰 | 88.60% | 86.61% | 95.54% |
| 概率轨迹，去模糊后 | 86.72% | 83.93% | 93.75% |
| v1 二值拓扑，去模糊后 | 87.87% | 85.71% | 95.54% |
| v2 二值拓扑，去模糊后 | 88.22% | 85.71% | 94.64% |
| v2 告警约束，去模糊后 | 87.54% | 85.71% | 96.43% |

在 340 张正常留出图像上，v2 普通运行点的误报率为 0%；v2 告警约束运行点也是 0%。直接模糊路径只作为必要性实验的冻结测试条件；其逐类别指标见 `H:\C3Netqing\results\defect_detection\tsr_binary_topology_comparison_112_v2\multi_variant_metrics.csv`。

以实际经 MBDNet 恢复的输入为准，若优先考虑缺陷分类 F1，可使用普通 v2；若优先考虑漏检，同时不增加本次正常留出集误报，可使用 v2 告警约束运行点。两组与概率版相比的图像级准确率都净增加 2/112 张。五种配置已经在同一批 112 张来源上反复比较，所以这些改进属于开发阶段结果；独立来源验证之前不宜作为普适优势写入论文。

从 `H:\C3Netqing` 运行：

```powershell
conda run -n c3net python defect_detection/run_nested_fourfold_tsr_defect.py --config defect_detection/configs/nested_fourfold_joint_clear_deblur_112_binary_topology_v2.yaml
conda run -n c3net python defect_detection/run_nested_fourfold_tsr_defect.py --config defect_detection/configs/nested_fourfold_joint_clear_deblur_112_binary_topology_v2_alarm95.yaml
```

当前输出已存在，程序默认拒绝覆盖；复跑时需在配置中指定新的输出目录。逐图预测位于 `H:\C3Netqing\results\defect_detection\tsr_binary_topology_comparison_112_v2\multi_variant_paired_predictions.csv`。
