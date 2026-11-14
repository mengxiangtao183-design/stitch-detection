# C3Netqing：图像复原、分割与缺陷检测研究代码

本仓库整理了一个多阶段视觉研究工作区：图像去模糊、目标分割、缺陷检测，以及各阶段的对比实验、消融实验和论文图表生成脚本。它以 [C3Net 原项目](https://github.com/Baber-Jan/C3Net)为分割起点，包含后续的项目改动和其他开源方法的适配。原项目的 README 保存在 [docs/UPSTREAM_C3NET_README.md](docs/UPSTREAM_C3NET_README.md)。

> 本仓库是研究代码归档。不同实验使用不同数据划分、环境和权重；运行前请阅读对应子目录的 README 与配置。原 C3Net 论文中的指标不应视为本仓库所有改动的验证结果。

## 内容导航

| 路径 | 内容 |
| --- | --- |
| `main.py`, `models/`, `engine/`, `utils/`, `configs/` | C3Net 分割主程序、模型、训练/评估流程及配置 |
| [`com_models/`](com_models/README.md) | 分割方法的统一对比实验与模型适配 |
| [`Deblurring/`](Deblurring/README.md) | 基于 MPRNet 的去模糊研究与训练脚本 |
| [`NAFNet/`](NAFNet/readme.md), [`MISCFilter/`](MISCFilter/README.md) | 相关复原方法及项目内适配 |
| [`deblur_comparison_suite/`](deblur_comparison_suite/README.md) | 去模糊模型的统一训练和测试协议 |
| `defect_detection/` | 缺陷检测、二值拓扑特征与全流程延迟实验 |
| `C3Netqing-baseline-dino/` | 独立的 DINO 主干分割基线实验 |
| `sam2/` | SAM2 相关模型接口代码 |
| `scripts/`, `methods/`, `tools/` | 数据准备、辅助分析和论文图表脚本 |
| `docs/`, `paper_draft_work/` | 技术记录、图表和论文草稿源文件 |

`deblur_comparison_suite/third_party/` 和 `com_models/third_party/` 存放第三方方法。来源、许可和本地改动应按 [THIRD_PARTY.md](THIRD_PARTY.md) 核对；各项目的许可证仍分别适用。

## 环境

主要分割流程使用 Python、PyTorch 和 CUDA。原项目提供 `setup/environment.yml`；根目录 `requirements.txt` 记录了当前工作区使用的 Python 包版本。去模糊、缺陷检测及第三方方法可能还有单独依赖，请以子目录说明为准。

```bash
conda env create -f setup/environment.yml
conda activate c3net
python main.py --help
```

`setup/setup.sh` 面向 Bash 环境；Windows 用户可先按环境文件安装，再执行各子项目中的 PowerShell 脚本。配置文件中保留的 `H:/C3Netqing`、`H:/DATASET` 等路径是原实验机器路径，迁移时须改为本机路径。

克隆时需要取得各独立项目的固定版本，并恢复本工作区对它们所做的接口适配：

```bash
git clone --recurse-submodules <本仓库地址>
cd C3Netqing
python tools/vendor_overrides.py restore
```

恢复脚本会将 `vendor_overrides/` 中的补丁与新增文件复制到子模块工作目录。它会使相关子模块显示为本地修改；这是预期状态。若只阅读主项目代码，可跳过这一步。

## 主要入口

```bash
# 分割训练
python main.py train --config configs/default.yaml

# 使用指定权重评估
python main.py evaluate --config configs/default.yaml --model checkpoints/model_best.pth

# 单张图片预测
python main.py predict --config configs/default.yaml --model checkpoints/model_best.pth --input path/to/image
```

分割对比配置与操作见 [`com_models/README.md`](com_models/README.md)；去模糊统一比较见 [`deblur_comparison_suite/README.md`](deblur_comparison_suite/README.md)；缺陷检测的设计和运行口径见 `defect_detection/` 中的 Markdown 文档。实验复现时应同时记录所用配置、数据清单、随机种子及权重来源。

当前论文实验的配置入口：分割主模型 M6 使用 [`configs/ablations/m6_afm_guided_highres_full_150_seed42.yaml`](configs/ablations/m6_afm_guided_highres_full_150_seed42.yaml)，分割对比与消融分别见 `com_models/configs/` 和 `configs/ablations/`；去模糊主模型的实际运行参数快照见 [`NAFNet/experiment_configs/mbdnet_p_plus_r12_s30_s42_run_config.json`](NAFNet/experiment_configs/mbdnet_p_plus_r12_s30_s42_run_config.json)，MPRNet 对比配置见 [`Deblurring/training_mprnet_original_multisource_synblur_v2_s30.yml`](Deblurring/training_mprnet_original_multisource_synblur_v2_s30.yml)。去模糊旧探索 YAML 的清理口径见 [`Deblurring/PAPER_YAML_AUDIT.md`](Deblurring/PAPER_YAML_AUDIT.md)。

## 数据、权重与结果

数据集、预训练权重、训练检查点和逐次运行结果通常体积很大，并受各自的数据使用条款约束，因此不作为普通 Git 源文件提交。`.gitignore` 排除了这些生成或下载的目录与文件，**本地文件不会因此被删除**。发布代码不等于已发布可直接复现全部数值的资源包。

- 原 C3Net 的 COD 数据和预训练权重下载信息见 [原项目说明](docs/UPSTREAM_C3NET_README.md#dataset-preparation)。
- 本工作区其他实验的数据清单与路径约定请查看对应子项目的 README、`configs/` 及 `MISCFilter/manifests/`。数据根目录需要在本机重新配置。
- 自训练权重如需共享，宜单独发布并标注对应配置、数据版本、训练轮次和校验值，再在这里添加链接；不要把数 GB 的检查点直接放进 Git 历史。

## 来源、许可与引用

根目录 [`LICENSE`](LICENSE) 是原 C3Net 项目的 MIT 许可证，原作者版权声明保留。这里包含的其他项目不自动继承根目录许可证；使用前请查看各自的 `LICENSE` 和 [THIRD_PARTY.md](THIRD_PARTY.md)。对于未明确授予再分发许可的第三方代码，应通过上游仓库获取。

引用原 C3Net 工作时，请使用其 [README 中的文献信息](docs/UPSTREAM_C3NET_README.md#citation)。本仓库的实验改动、比较协议与论文草稿应按相应成果独立注明，不应将第三方方法或原论文结果归为本仓库原创。
