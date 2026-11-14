# 全流程单张延迟实验

这是连续推理的部署计时，不是三个网络联合训练。输入 RGB → 已冻结的清晰度评分与阈值路由 → 必要时去模糊 → M6 分割 → 0.5 阈值二值化 → 二值拓扑折中版 BTSR → 图像级类型和告警。两个 GPU 网络与 CPU 缺陷模型仅加载一次；中间结果在内存传递，不通过 PNG 文件转交。

已通过 Python/PowerShell 语法、文件路径、112 对文件名和冻结阈值检查，并完成 CUDA 两张图试运行（跳过和执行去模糊各一张）。两张图结果仅用于验证流程可运行，不应作为论文正式平均耗时；正式统计请运行完整工作负载。

## 使用

先运行少量输入，验证模型和设备兼容性：

```powershell
& "H:\C3Netqing\defect_detection\full_pipeline_latency\run_benchmark.ps1" -Smoke
```

正式测量实际路由：

```powershell
& "H:\C3Netqing\defect_detection\full_pipeline_latency\run_benchmark.ps1"
```

另行实测“强制去模糊完整路径”及“不去模糊路径”：

```powershell
& "H:\C3Netqing\defect_detection\full_pipeline_latency\run_benchmark.ps1" -AllPaths
```

若 PowerShell 执行策略阻止脚本，可直接使用 Python（不修改系统执行策略）：

```powershell
& "D:\professional_software\anaconda3\envs\c3net\python.exe" "H:\C3Netqing\defect_detection\full_pipeline_latency\benchmark_full_pipeline.py" --policy routed --workload mixed
```

## 配置与计时口径

- 沿用 112 张必要性实验的 MBDNet-P+ FTDirGradSSIM5 权重，不使用旧计时脚本中的另一份 PPlusR12 权重。
- M6 从 checkpoint 中读取模型结构和归一化参数，严格加载权重。
- 路由阈值和七个特征系数来自既有校准结果，当前阈值约 0.48371047；不在这 112 张计时输入上重选阈值。
- BTSR 使用现有 final112 部署模型，固定其局部类别阈值与整体歪斜阈值。输出不构成新的折外准确率结果。
- 输入大小 320×320、batch=1、FP32，禁用 AMP 与 TF32。CPU/PyTorch/ExtraTrees 单线程。两条路径都预热，总计 10 次，不纳入平均。
- 默认 112 张清晰＋112 张对应合成模糊输入，重复三轮，轮内固定种子打乱顺序；共 672 次测量，不是 672 个独立来源。禁止把翻转图当成新来源。
- 混合平均仅代表清晰和模糊各占 50% 的设定；同时报告 clear、blurred、deblur 和 bypass 分组。不将该比例当成实际工业产线分布。
- 主要指标 `total_decode_to_decision_ms`：从读取/解码输入到最终缺陷类别和告警在 CPU 内存中可用。包括路由、预处理、GPU 传输、去模糊、分割、二值化和完整 BTSR 聚合。
- `total_compute_ms`：上述总耗时减去文件读取/解码。适合说明已获得 RGB 帧后的处理延迟。
- 各 GPU 边界使用同步后的墙钟计时，避免异步返回低估。属于串行单张延迟，不代表多流并行/流水线吞吐。
- 不包括模型加载、预热、日志、摄像头曝光/采集、网络传输或 PLC 通信。操作系统缓存不清空；每轮统计保留。首次模型加载单独记录。
- 去模糊输出先转 uint8，保持与原必要性实验 PNG 交接的数值路径一致，但不写中间文件。二值化严格为 `sigmoid(final_mask) >= 0.5`，不是彩色叠加图输入。
- 可额外传 `--save-images`，单独记录 PNG 保存耗时；不要把保存可视化的时间混入主要检测延迟。

## 输出

每次创建独立时间戳目录，默认在 `H:\C3Netqing\results\full_pipeline_latency` 下，不覆盖历史结果：

- `latency_summary.json`：硬件、协议、冻结阈值、各组 mean/median/P95 与串行处理速率。
- `stage_summary.csv`：各组路由、去模糊、分割、BTSR 和总耗时均值及总耗时 P95。
- `per_image_latency.csv`：逐图、逐轮、路由选择与各阶段耗时。
- `predictions.json`：本次连续推理实际输出，不用于重新报告训练集准确率。

论文建议报告：全流程平均延迟＋P95，并说明 batch=1、320×320、硬件、精度和输入工作负载；去模糊执行分支与跳过分支分别列出。不能简单相加以前各自运行脚本的计时，也不能把不运行 MBDNet 的 0 ms 当成其真实前向耗时。
