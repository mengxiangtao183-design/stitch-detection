# StitchNAFNet design record

## Recent open methods surveyed

| Year | Method and open code | Transferable idea | Decision for this dataset |
|---|---|---|---|
| 2023 | [FFTformer](https://github.com/kkkls/FFTformer), CVPR 2023 | Frequency-domain correlation for motion blur | Keep only one bottleneck FFT gate; full Transformer is unnecessary for 24 GB/fast inference |
| 2023 | [FSNet](https://github.com/c-yn/FSNet), *Image Restoration Via Frequency Selection* | Content-aware local frequency selection and receptive-field modulation | Motivates high-frequency selection for periodic stitch texture |
| 2024 | [MISCFilter](https://github.com/ChengxuLiu/MISCFilter), CVPR 2024 | Motion-adaptive separable collaborative filters | Use a stable mean/variance degradation gate; avoid CuPy dynamic kernels because this project already saw Windows CUDA crashes |
| 2024 | [AdaRevD](https://github.com/INVOKERer/AdaRevD), CVPR 2024 | Adaptive patch exiting and reversible decoder | Useful efficiency principle, but its multi-stage replacement is too risky for the small four-domain dataset |
| 2024 | [LoFormer](https://github.com/INVOKERer/LoFormer), ACM MM 2024 | Local frequency transformer | Use local frequency prior without importing a heavy attention backbone |
| 2024 | [CascadedGaze](https://github.com/Ascend-Research/CascadedGaze), arXiv:2401.15235 | Efficient global context extraction | Supports adding context only at low resolution |
| 2024/2025 | [MambaIR/MambaIRv2](https://github.com/csguoh/MambaIR), ECCV 2024/CVPR 2025 | Long-range state-space context | Not selected as the main baseline because custom CUDA/scan dependencies and higher engineering risk conflict with fast stable training |
| 2025 | [EVSSM](https://github.com/kkkls/EVSSM), CVPR 2025 | Efficient visual state-space modeling for deblurring | Reference for future long-range ablation, not in the first controlled change |
| 2025 | [DarkIR](https://github.com/cidautai/DarkIR), CVPR 2025 | Robust low-light restoration | Motivates illumination/degradation statistics for RX domain shift |

## Project-specific change

`StitchNAFNet` keeps the validated NAFNet-width32 topology and adds only zero-initialized residual paths:

1. Four deepest encoder blocks and the bottleneck receive depthwise `1x7`/`7x1` axis mixing plus a dilated `3x3` oblique branch for long, curved, and diagonal stitch structures.
2. A single bottleneck FFT high-frequency gate targets periodic thread edges without applying FFT to every resolution.
3. A local mean/variance gate adapts to RX illumination and texture shifts, borrowing the domain-adaptation motivation of MISCFilter without its CUDA dynamic filtering.
4. PSNR remains the dominant objective; a small Sobel edge term and a central-band weighting protect line continuity.

The new paths start at zero, so loading `C3NetAll_NAFNet_width32/model_best.pth` preserves the baseline at initialization. This is deliberate: prior MPRNet experiments that replaced Stage 1/2/3 jointly lost 6--10 dB on DB/XJ/SD/RX.

The local images confirm the prior: DB/XJ contain a central near-horizontal stitch over strongly periodic fabric; SD has dense woven texture; RX changes to a dark background with gold, diagonal and multi-line structures. The ROI term is therefore deliberately weak and the degradation gate is spatial, rather than assigning a fixed domain label.

## Cost check on the project machine

Final architecture, RTX 4090, `1x3x320x320`, five warm-up and 30 timed iterations:

| Model | Parameters | Time | Peak allocated memory |
|---|---:|---:|---:|
| NAFNet-width32 | 17.112 M | 22.09 ms | 302 MB |
| StitchNAFNet | 18.993 M | 23.28 ms | 302 MB |

The added parameters are 11.0%, while measured latency increases by about 5.4%. Loading the NAFNet checkpoint before fine-tuning gives an exact initial output match (`max_abs_diff=0`).

## Five-epoch pilot

The pilot starts from the validated NAFNet checkpoint and uses the same manifests. Its full test-set result is:

| Model | DB | XJ | SD | RX | Macro | Weighted |
|---|---:|---:|---:|---:|---:|---:|
| NAFNet-width32, 30 epochs | 26.8879 | 30.3732 | 28.6819 | 23.6755 | 27.4046 | 25.5370 |
| StitchNAFNet, 5-epoch pilot | 27.2590 | 30.5178 | 28.9021 | 23.8905 | 27.6423 | 25.7664 |
| Change | +0.3711 | +0.1446 | +0.2202 | +0.2150 | +0.2377 | +0.2294 |

This pilot is evidence for proceeding to the controlled 30-epoch run, not the final paper result. The full experiment must keep the same split, seed, patch size and evaluation script.
