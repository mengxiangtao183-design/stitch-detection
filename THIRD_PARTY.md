# 第三方代码与来源

本项目从 C3Net 原仓库发展而来，根目录 `LICENSE` 保留原作者的 MIT 声明。以下目录是各自独立的上游项目或其本地适配，使用时应遵守对应仓库的许可证。研究代码中的模型名称、论文指标与方法归属以原论文和上游仓库为准。

| 本地目录 | 上游仓库 | 本地状态 |
| --- | --- | --- |
| `C3Netqing-baseline-dino/` | [Baber-Jan/C3Net](https://github.com/Baber-Jan/C3Net) | 独立 Git 仓库，含本地修改与新增配置 |
| `MISCFilter/` | [ChengxuLiu/MISCFilter](https://github.com/ChengxuLiu/MISCFilter) | 独立 Git 仓库，含本地适配和数据清单 |
| `NAFNet/` | [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) | 本地代码与项目适配，保留其 `LICENSE` |
| `Deblurring/` | [swz30/MPRNet](https://github.com/swz30/MPRNet) | 基于 MPRNet 的研究代码；已附上其学术用途许可证 `Deblurring/LICENSE.md` |
| `sam2/` | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | SAM2 模型代码；附上 Apache 2.0 及 `cc_torch` 相关许可证 |
| `com_models/third_party/PIDNet/` | [XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet) | 独立 Git 仓库，含接口适配 |
| `deblur_comparison_suite/third_party/AdaRevD/` | [INVOKERer/AdaRevD](https://github.com/INVOKERer/AdaRevD) | 独立 Git 仓库 |
| `deblur_comparison_suite/third_party/BANet/` | [pp00704831/banet](https://github.com/pp00704831/banet) | 独立 Git 仓库，含本地修改；仓库内未发现顶层许可证 |
| `deblur_comparison_suite/third_party/CascadedGaze/` | [Ascend-Research/CascadedGaze](https://github.com/Ascend-Research/CascadedGaze) | 独立 Git 仓库 |
| `deblur_comparison_suite/third_party/DarkIR/` | [cidautai/DarkIR](https://github.com/cidautai/DarkIR) | 独立 Git 仓库 |
| `deblur_comparison_suite/third_party/FFTformer/` | [kkkls/FFTformer](https://github.com/kkkls/FFTformer) | 独立 Git 仓库；本地变动主要包含编译缓存 |
| `deblur_comparison_suite/third_party/HINet_official/` | [megvii-model/HINet](https://github.com/megvii-model/HINet) | 独立 Git 仓库 |
| `deblur_comparison_suite/third_party/LoFormer/` | [INVOKERer/LoFormer](https://github.com/INVOKERer/LoFormer) | 独立 Git 仓库，含本地修改 |
| `deblur_comparison_suite/third_party/LoFormer_index_only/` | [DeepMed-Lab-ECNU/Single-Image-Deblur](https://github.com/DeepMed-Lab-ECNU/Single-Image-Deblur) | 独立 Git 仓库；仓库内未发现顶层许可证 |
| `deblur_comparison_suite/third_party/MIMO-UNet/` | [chosj95/MIMO-UNet](https://github.com/chosj95/MIMO-UNet) | 独立 Git 仓库；仓库内未发现顶层许可证 |
| `deblur_comparison_suite/third_party/Restormer/` | [swz30/Restormer](https://github.com/swz30/Restormer) | 独立 Git 仓库 |
| `deblur_comparison_suite/third_party/SFNet_official/` | [c-yn/SFNet](https://github.com/c-yn/SFNet) | 独立 Git 仓库 |
| `deblur_comparison_suite/third_party/Stripformer/` | [pp00704831/Stripformer](https://github.com/pp00704831/Stripformer) | 独立 Git 仓库 |
| `deblur_comparison_suite/third_party/UFPDeblur/` | [Fangzhenxuan/UFPDeblur](https://github.com/Fangzhenxuan/UFPDeblur) | 独立 Git 仓库；仓库内未发现顶层许可证 |

上述独立 Git 仓库以固定提交的子模块形式记录在 `.gitmodules` 中；本地改动和新增文件保存在 `vendor_overrides/`。克隆后运行 `python tools/vendor_overrides.py restore`，即可恢复这些接口适配与配置。对于未明确授予再分发许可的上游代码，主仓库只保存引用，不复制其源码。`vendor_overrides/` 中的修改仍需按各上游项目的条款使用。

MPRNet 的上游许可证限制商业用途，不能以根目录的 MIT 声明替代。`sam2/` 的源代码同样受其独立许可证约束。
