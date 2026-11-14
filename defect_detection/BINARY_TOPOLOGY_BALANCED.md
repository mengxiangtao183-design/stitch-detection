# TSR 二值拓扑折中版（当前推荐）

输入为 ACHNet 输出的 320×320 二值 PNG，不再读取概率 `.npy`。先按同图轨迹切分六个逻辑单元，提取 41 维轨迹与二值拓扑特征；局部跳线、断线、松线用 ExtraTrees 判别，整体歪斜用轨迹斜率判别。图像级按相对阈值最高的类别给出一个判定，否则为正常。首末拍摄截断单元不使用相邻短缺证据。

与此前二值拓扑 V2 的特征、训练样本和四折划分完全一致。唯一变化是内层阈值校准：要求内层缺陷告警召回率至少 0.90，在满足约束的候选阈值中最大化 `(图像级宏平均 F1 + 缺陷告警召回率)/2`；正常误报率为次级排序条件。四个外层测试折和直接模糊路径未参与调阈值。

| 输入路径／版本 | 宏平均 F1 | 图像级准确率 | 缺陷告警召回率 |
|---|---:|---:|---:|
| 清晰／原概率掩码 | 89.52% | 87.50% | 95.54% |
| 清晰／二值拓扑折中版 | 90.01% | 88.39% | 96.43% |
| 直接模糊／二值拓扑折中版 | 75.30% | 71.43% | 90.18% |
| 去模糊后／原概率掩码 | 86.72% | 83.93% | 93.75% |
| 去模糊后／二值拓扑 V2 | 88.22% | 85.71% | 94.64% |
| 去模糊后／二值拓扑 V2 告警优先 | 87.54% | 85.71% | 96.43% |
| 去模糊后／二值拓扑折中版 | **89.22%** | **87.50%** | **96.43%** |

去模糊后的四类 F1：跳线 87.27%、断线 82.76%、歪斜 94.55%、松线 92.31%。340 张正常留出图的误报数为 0。去模糊路径相对概率版多正确判定 4 张（概率版独有正确 1 张，折中版独有正确 5 张）。所有折外来源隔离检查均通过。

四折结果：`H:\C3Netqing\results\defect_detection\tsr_defect_nested4fold_joint_clear_deblur_112_binary_topology_v2_balanced`；六版本配对对照：`H:\C3Netqing\results\defect_detection\tsr_binary_topology_comparison_112_balanced`。当前推荐以折中版作为后续二值输入主方案，保留概率版作为历史比较。由于同一批 112 张来源已用于多版方法开发，这属于开发阶段的四折估计，不能表述为独立外部验证。

最终部署模型单独用全部 112 张缺陷来源和正常训练来源拟合；其阈值由全数据的来源分组三折折外预测校准，340 张正常留出来源仍未参与拟合。最终模型对训练来源的推理仅为接口自检，不能代替上述四折成绩。

从 `H:\C3Netqing` 运行（已有模型无须重训）：

```powershell
conda run --no-capture-output -n c3net python defect_detection/deploy_binary_topology_v2_balanced.py predict --model H:/C3Netqing/results/defect_detection/tsr_binary_topology_v2_balanced_final112/model_final.joblib --detector-config H:/C3Netqing/configs/defect_head/tsr_defect_rules.yaml --input-dir <ACHNet二值PNG目录> --output <推理结果.json>
```

评估配置为 `configs/nested_fourfold_joint_clear_deblur_112_binary_topology_v2_balanced.yaml`；独立重训部署模型可使用同一脚本的 `train --config ... --output-dir ...`，但程序拒绝覆盖已有输出。论文图表应引用四折目录的 `nested_oof_summary.csv`、`summary.json` 和 `nested_oof_confusion_matrices.png`，不应引用最终模型的训练来源推理作为准确率。
