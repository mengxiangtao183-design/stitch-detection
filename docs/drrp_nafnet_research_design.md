# DRRP-NAFNet research design

## Evidence from this project

The shared split contains 1,530 training and 170 validation pairs: DB/XJ/SD
contribute 360/40 each and RX contributes 450/50.  The test set contains
DB/XJ/SD/RX = 100/100/100/500 images.  RX is therefore a shifted domain, not
an unseen zero-shot domain.  All images used by this protocol are 320x320.

Measured on all test inputs, RX has the lowest mean luminance (0.102) but the
largest mean absolute Laplacian response (0.0334).  It mixes low illumination
with strong fabric/noise frequencies.  Blindly amplifying high frequencies is
therefore unsafe.  DB also has strong directional anisotropy: its mean x/y
gradient ratio is 0.282, versus 0.501/0.621/0.477 for XJ/SD/RX.

The local MPRNet experiments establish three constraints:

1. Replacing Stage 1/2/3 together caused 6--10 dB losses on DB/XJ/SD.  The
   1,700-pair training set does not support a large backbone replacement.
2. Frequency band/disagreement refiners were among the few variants that
   improved RX, but remained below the stable MPRNet-EdgeFT overall result.
3. Knowledge distillation did not rescue an over-disrupted architecture.
   Pretraining and extra fine-tuning must be controlled rather than described
   as architectural gains.

The controlled NAFNet experiment reached weighted Full PSNR 26.0873 after an
additional 15 epochs.  TADR reached only 26.1020 (+0.0148 dB) while latency
increased from 22.32 ms to 32.51 ms (+45.7%).  Its full-resolution trajectory
stream is therefore rejected.

## Recent literature and transferable mechanisms

| Work | Venue | Official source | Relevant evidence |
|---|---|---|---|
| FFTformer | CVPR 2023 | https://github.com/kkkls/FFTformer | Frequency-domain correlation and discriminative frequency FFN show that blur restoration benefits from selective, not uniform, frequency processing. |
| Image Deblurring with Image Blurring | IEEE TIP 2023 | https://doi.org/10.1109/TIP.2023.3321515 | Supports using a forward reblurring process as physical/data-consistency evidence rather than predicting only an unconstrained image residual. |
| MISCFilter | CVPR 2024 | https://github.com/ChengxuLiu/MISCFilter | Spatially varying motion should receive differentiated image-space filtering.  The project does not reuse its unstable CuPy kernel. |
| AdaRevD | CVPR 2024 | https://github.com/INVOKERer/AdaRevD | Blur difficulty is spatially varying; adaptive decoding should spend capacity where restoration remains difficult. |
| LoFormer | ACM MM 2024 | https://github.com/INVOKERer/LoFormer | Local frequency processing helps separate locally useful detail from global periodic texture. |
| EVSSM | CVPR 2025 | https://github.com/kkkls/EVSSM | Efficient non-local modeling and a discriminative frequency FFN motivate placing context modeling at low resolution. |
| DarkIR | CVPR 2025 | https://github.com/cidautai/DarkIR | Low-light blur needs degradation-aware processing; this directly matches the darker RX distribution. |
| Residual Guidance Strategy | ICCV 2025 | https://github.com/Heyan-Liu/ICCV-2025-Blind-Noisy-Image-Deblurring-Using-Residual-Guidance-Strategy | Adjacent coarser-scale residuals can guide finer-scale blur estimation and reduce contamination by noise. |

The listed mechanisms are research precedents, not copied modules.  DRRP uses
their common principles to solve the measured stitch-data failure modes.

## Proposed structural contribution

**Directional Reblur Residual Pyramid (DRRP)** replaces two redundant deepest
NAF blocks with a two-level physical reconstruction path at 1/8 and 1/4 image
resolution:

1. Each level predicts a low-resolution sharp proxy from decoder features.
2. A constrained basis bank reblurs the proxy with identity, horizontal,
   vertical, and two diagonal motion kernels.
3. The difference between the observed blurry image and its predicted reblur
   is decomposed into low- and high-band residuals.  Local luminance mean and
   deviation are added to distinguish dark RX content from actual blur.
4. Horizontal, vertical, and oblique depthwise context paths encode this
   degradation evidence.  Coarse guidance is propagated to the next finer
   level and injected into the NAF decoder.

This is a structural inverse-model path, not a scalar gate or loss-only change.
It explicitly closes the physical loop:

`blurry -> sharp proxy -> constrained reblur -> fidelity residual -> reconstruction`

The two output injections start at zero.  Before pruning, loading the NAFNet
checkpoint gives exact output equivalence (maximum absolute difference 0).
The final budget-reallocated model removes deep blocks 0 and 1 and initializes
all remaining NAF blocks from the same NAFNet epoch-30 checkpoint.

## Cost and controlled experiment

RTX 4090, 1x3x320x320, 30 warmups and 200 timed iterations:

| Model | Parameters | Inference |
|---|---:|---:|
| NAFNet-width32 | 17.112 M | 21.83 ms |
| DRRP-NAFNet (two deep blocks replaced) | 16.257 M | 24.86 ms |

DRRP has 5.0% fewer parameters and is about 13.9% slower than NAFNet, but is
about 23.5% faster than the rejected TADR experiment (32.51 ms).  The latency
increase must be reported; parameter reduction must not be presented as a
measured speedup.

The primary controlled comparison is:

| Run | Initial checkpoint | Extra epochs | Loss |
|---|---|---:|---|
| A0-FT15 | NAFNet epoch 30 | 15 | PSNR |
| A1-DRRP-FT15 | same NAFNet epoch 30, compatible weights only | 15 | PSNR + 0.05 coarse PSNR + 0.02 reblur consistency |

This A1 tests one combined structural unit: compute is reallocated from two
generic deep NAF blocks to the direction-aware physical residual pyramid.  A
positive result should satisfy weighted Full/Mid PSNR >= +0.10 dB, SSIM must
not fall, RX should improve, and XJ should not lose more than 0.05 dB.  If it
passes, a second loss ablation with auxiliary weights set to zero separates
the structural benefit from its necessary physical supervision.

Pretraining is not an innovation claim.  It is used only as a shared controlled
initialization.  The final paper model should additionally be trained from
scratch against a scratch NAFNet with the same total optimization budget.

