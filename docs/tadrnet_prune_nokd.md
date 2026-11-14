# TADRNet no-distillation speed/accuracy trial

This trial does not use a teacher network and does not overwrite the validated
`checkpoints/C3NetAll_TADRNet` model.

## Stages

1. `TADRNet_NoKD_Polish`: five low-learning-rate epochs on the complete model.
2. `TADRNet_NoKD_Fade075`, `Fade040`, `Fade000`: blocks 0 and 1 of the
   deepest appearance stage are gradually changed from residual strength
   `0.75` to `0.40` to `0.0`.
3. `TADRNet_P26_NoKD`: those two blocks are physically removed. Their weights
   are remapped from original blocks 2--27, then the compact model is trained
   with GT losses for ten epochs.

The measured structural check before training is exact: the full model with
the two blocks set to identity and the physically compact P26 model have
`max_abs_diff=0.0` on the same input. The compact model has 18.59M parameters
versus 19.53M and measured latency was 27.60 ms versus 28.73 ms on RTX 4090
for a `320x320` image.

## Run

```powershell
conda activate c3net
cd H:\C3Netqing\NAFNet
.\run_tadrnet_prune_nokd.ps1
```

The process auto-resumes each stage independently. It saves stage outputs
under `checkpoints/TADRNet_NoKD_*` and `checkpoints/TADRNet_P26_NoKD`.

## Accept/reject

Evaluate the compact result:

```powershell
.\run_tadrnet_test.ps1 `
  -Checkpoint .\checkpoints\TADRNet_P26_NoKD\model_best.pth `
  -P26
```

Keep P26 only if all of the following hold relative to the current full model:

```text
macro PSNR >= 28.3958
weighted PSNR >= 26.5738
RX PSNR >= 24.7518
inference time < 28.73 ms
```

If any accuracy criterion fails, use the existing full model checkpoint:

```text
checkpoints/C3NetAll_TADRNet/model_best.pth
```
