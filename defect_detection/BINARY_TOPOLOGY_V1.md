# 二值拓扑版 TSR 缺陷检测单元

输入为 ACHNet 预测的单通道二值 PNG（320×320，像素值 0/255）。六段定位及原有轨迹特征沿用 TSR；新提取器增加 12 维二值拓扑特征，形成 33 维单元表示。新增量包括前景连续段的长度与数量、内部空段及空段后再现、最大八连通分量的跨度与面积占比，以及内部单元与左右相邻线段的长度比。首尾拍摄截断单元不计算邻段短缺证据。局部跳线、断线和松线继续由 ExtraTrees 分类；图像级歪斜沿用全局中心斜率分支。

实验配置：`configs/nested_fourfold_joint_clear_deblur_112_binary_topology_v1.yaml`。仅使用 ACHNet 二值预测作为真实条件输入；受控增强掩码仅在训练来源内生成。清晰与 MBDNet 恢复条件参与内层训练和阈值校准，直接模糊条件只参与测试。四个外层测试折与现有概率版相同，每类每折 7 张。正常留出图像只检查误报警。

运行命令（在 `H:\C3Netqing` 下）：

```powershell
conda run -n c3net python defect_detection/run_nested_fourfold_tsr_defect.py --config defect_detection/configs/nested_fourfold_joint_clear_deblur_112_binary_topology_v1.yaml
```

程序拒绝覆盖已有输出。当前结果目录：`H:\C3Netqing\results\defect_detection\tsr_defect_nested4fold_joint_clear_deblur_112_binary_topology_v1`。

| 条件 | 概率轨迹宏平均 F1 | 二值轨迹宏平均 F1 | 二值拓扑宏平均 F1 | 二值拓扑图像级准确率 | 二值拓扑缺陷告警召回率 |
|---|---:|---:|---:|---:|---:|
| 清晰 | 89.52% | 87.30% | 87.08% | 85.71% | 97.32% |
| 直接模糊 | 71.31% | 74.34% | 73.48% | 68.75% | 88.39% |
| MBDNet 恢复 | 86.72% | 85.79% | 87.87% | 85.71% | 95.54% |

去模糊路径相对概率轨迹版提升 1.15 个百分点的宏平均 F1，但清晰路径下降 2.44 个百分点。去模糊路径的 112 张配对图像中，两版均正确 93 张，仅概率版正确 1 张，仅二值拓扑版正确 3 张，均错误 15 张。因此提升主要来自净增加 2 张正确图像，不能据此称二值拓扑版全面优于概率版。340 张正常留出图像的误报警率为 0.29%（概率版为 0%）。

逐条件、逐类别指标及逐图预测见 `H:\C3Netqing\results\defect_detection\tsr_binary_topology_comparison_112_v1`。嵌套来源分组检查全部通过；但在同一批 112 张图像上探索多个方法仍属于开发过程，正式对外泛化结论需要新增独立来源。
