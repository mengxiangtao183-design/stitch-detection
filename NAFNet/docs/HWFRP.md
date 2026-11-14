# Hierarchical Wavelet-Frequency Reconstruction Path

HWFRP is an independently switchable third reconstruction path for TADRNet-P26.
It targets long stitch blur and cross-domain frequency shifts without adding a
full-resolution attention backbone.

## Structure

1. Two-level orthonormal Haar analysis separates RGB low-frequency content and
   horizontal, vertical, and diagonal detail coefficients.
2. Local 8x8 FFT selection blocks operate at 1/2 and 1/4 resolution. Learned
   real frequency responses are followed by depthwise local and strip filters.
3. The P26 decoder feature is projected into both frequency scales.
4. A coarse-to-fine inverse Haar decoder predicts coefficient corrections and
   synthesizes one RGB residual.
5. Both coefficient heads use zero initialization, so an existing P26
   checkpoint is exactly preserved before fine-tuning.

The structural references are FFTformer (CVPR 2023), AdaRevD (CVPR 2024), and
LoFormer (ACM MM 2024):

- https://github.com/kkkls/FFTformer
- https://github.com/INVOKERer/AdaRevD
- https://github.com/INVOKERer/LoFormer

This implementation is project-specific code, not copied source from those
repositories.

## Matched Ablation

Train the P26 control and P26+HWFRP from the same checkpoint:

```powershell
.\run_tadrnet_p26_matched_train.ps1
.\run_tadrnet_hwfrp_train.ps1
```

Both runs use 15 epochs, batch size 4, base learning rate 1e-5, and select the
best checkpoint using validation PSNR on the middle horizontal third weighted
by the DB/XJ/SD/RX test-domain proportions. HWFRP uses two branch-only warmup
epochs and a 5x branch learning rate because its zero-initialized path is new.

Test the trained HWFRP model:

```powershell
.\run_tadrnet_hwfrp_test.ps1
```

Test the matched P26 control:

```powershell
.\run_tadrnet_test.ps1 `
  -Checkpoint .\checkpoints\TADRNet_P26_MatchedFT\model_best.pth `
  -OutputDir .\results\TADRNet_P26_MatchedFT_midroi `
  -P26
```
