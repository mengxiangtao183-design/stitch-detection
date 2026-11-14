# MPRNet / C3Net Handoff - 2026-07-13

## Project

Workspace:

```text
H:\C3Netqing
```

Task:

```text
Industrial sewing-stitch detection.
Pipeline: degraded/blurred image -> MPRNet restoration -> C3Net stitch trajectory detection -> later engineering rules for broken stitch, skewed stitch, thrown stitch, skipped stitch, etc.
```

Current priority:

```text
Detect the stitch trajectory reliably first.
Defect classification comes later.
```

## Recent Work

Recent work focused on improving the MPRNet restoration network.

The previously strong restoration model is:

```text
H:\C3Netqing\Deblurring\checkpoints\train\Deblurring\models\MPRNet_EdgeFT
```

It has already been used to generate restored images:

```text
Imgs1_MPRNet_EdgeFT
```

We then tried a more horizontal-stitch-oriented MPRNet variant:

```text
LMPRNet_TrajectoryPair_NAF2_OV020_KD020
```

Checkpoint path:

```text
H:\C3Netqing\Deblurring\checkpoints\train\Deblurring\models\LMPRNet_TrajectoryPair_NAF2_OV020_KD020
```

Main design:

```text
1. LightAdaptiveRestorationModule
2. Stage 1: three horizontal full-width stripes
3. Shared-middle pair decoding:
   stripe1 + stripe2 -> decoder
   stripe2 + stripe3 -> decoder
4. Stage 2: Directional-NAF
5. Stage 3: MultiScaleFusionNAF
6. Edge-Guided Refinement
7. Two-stage training:
   NoKD 20 epochs
   KD0.20 fine-tuning 40 epochs
```

Training completed successfully. Checkpoints are valid:

```text
model_latest.pth epoch=40
model_epoch_40.pth epoch=40
```

## Generated Restored Images

The new model generated restored images under:

```text
Imgs1_MPRNet_TrajectoryPair_KD020
```

Generated counts:

```text
DB/train: 400 images
XJ/train: 400 images
SD/train: 400 images

DB/test: 100 images
XJ/test: 100 images
SD/test: 100 images
RX/test: 500 images
```

Example paths:

```text
H:\DATASET\DB\train\Imgs1_MPRNet_TrajectoryPair_KD020
H:\DATASET\RX\test\Imgs1_MPRNet_TrajectoryPair_KD020
```

## Current Blocker

The new MPRNet variant performs poorly.

Compared with `MPRNet_EdgeFT`, PSNR dropped heavily:

```text
test:
DB: 28.7110 -> 22.7286   -5.9824 dB
XJ: 32.9600 -> 26.7523   -6.2076 dB
SD: 30.1633 -> 20.6077   -9.5555 dB
RX: 22.8797 -> 20.7862   -2.0936 dB

train:
DB: 30.7856 -> 23.1209   -7.6648 dB
XJ: 34.0420 -> 27.3199   -6.7222 dB
SD: 31.9203 -> 21.3439   -10.5764 dB
```

The user also visually inspected the results and confirmed:

```text
TrajectoryPair_KD020 restored images are visibly worse than MPRNet_EdgeFT.
```

Current conclusion:

```text
Do not use LMPRNet_TrajectoryPair_NAF2_OV020_KD020 as the final restoration network.
```

Keep it only as an exploratory or failed ablation version.

## Pitfalls

1. Do not keep heavily modifying the MPRNet main structure.

This attempt changed too much:

```text
Stage1 horizontal stripes
Stage2 Directional-NAF
Stage3 ORSNet replaced by MultiScaleFusionNAF
```

The restoration quality degraded significantly. The current dataset size is only DB/XJ/SD with about 400 train images each, which is not enough to support this large architectural replacement.

2. Do not trust structure motivation alone.

The horizontal trajectory motivation is reasonable, but restoration quality must remain acceptable. A PSNR drop of 6 to 10 dB plus visible artifacts is too large for a model that will feed C3Net.

3. Do not train a heavily modified MPRNet from scratch.

The NoKD 20 epochs + KD 40 epochs schedule was not enough. `MPRNet_EdgeFT` is much more stable because it was fine-tuned from a strong MPRNet checkpoint.

4. Do not set the new restored directory as the C3Net default input.

Prefer:

```yaml
image_subdir: "Imgs1_MPRNet_EdgeFT"
```

Do not use as the main path:

```yaml
image_subdir: "Imgs1_MPRNet_TrajectoryPair_KD020"
```

5. KD cannot rescue an over-disrupted restoration architecture.

KD0.20 helped only weakly and did not recover the original MPRNet restoration quality.

## Recommended Main Route

Return the main restoration network to:

```text
MPRNet_EdgeFT
```

Recommended pipeline:

```text
MPRNet_EdgeFT restored images
  -> improved C3Net
  -> full DB/XJ/SD/RX evaluation
```

Keep the current C3Net innovations:

```text
1. MobileNetV4 backbone lightweight student
2. MRFA-DTE Block
3. centerline head
4. topology loss
5. boundary loss
6. high-resolution stitch refine head
```

Working interpretation:

```text
MPRNet should provide stable restoration.
C3Net should carry the main stitch-trajectory detection innovation.
```

## If MPRNet Is Modified Again

Do not alter the Stage1/Stage2/Stage3 core structure again.

Only make conservative changes around `MPRNet_EdgeFT`, for example:

```text
MPRNet_EdgeFT
+ LightAdaptiveRestorationModule
+ Edge-Guided Refinement
+ light edge supervision
```

Suggested experiment name:

```text
MPRNet_EdgeFT_LightAdapt
```

Rules:

```text
Fine-tune from MPRNet_EdgeFT weights.
Do not train from scratch.
Do not replace ORSNet.
Do not alter original Stage1 four-block / Stage2 two-half structure.
```

Goal:

```text
Improve RX lighting/texture generalization without damaging DB/XJ/SD PSNR and visual quality.
```

## Next Steps

The next session should:

```text
1. Avoid using TrajectoryPair_KD020 as the main model.
2. Confirm configs/default.yaml uses image_subdir: "Imgs1_MPRNet_EdgeFT".
3. Train/evaluate the current improved C3Net using MPRNet_EdgeFT restored images.
4. Compare against the original C3Net.
5. Run C3Net ablations:
   original C3Net
   + MobileNetV4
   + DTE
   + centerline head
   + topology/boundary loss
```

Core metrics:

```text
Weighted-F
Binary F1
Trajectory F1@1px
Trajectory F1@2px
Centerline F1
RX-specific results
Inference time
Parameter count
```

Recommended thesis direction:

```text
Restoration network: lightweight edge-guided restoration for input quality.
Detection network: lightweight dual-branch trajectory-aware C3Net, with main innovation in DTE, centerline prediction and topology constraints.
```

Most important current decision:

```text
TrajectoryPair_KD020 is not the final model. Do not continue large MPRNet architectural changes in that direction.
```
