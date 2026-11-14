# TPT-Net Architecture

## Canonical names

- **TPT-Net**: **Tri-Pair Trajectory Network**. This is the structural baseline used for all subsequent ablations.
- **TCFM**: **Trajectory-Calibrated Feature Modulation**. This is the Stage-2 cross-stage feature-modulation module.

The current baseline contains a three-stripe front end, adjacent-pair Stage-2 reconstruction, TCFM, and the unchanged full-resolution Stage-3 restoration path. It has no edge refinement or frequency refinement.

## Detailed data flow

```mermaid
flowchart TB
    I[Input degraded image I] --> S1Split

    subgraph S1[Stage 1: Tri-Stripe Shared-Pair Reconstruction]
        S1Split[Split I into three full-width horizontal stripes<br/>s1, s2, s3 with 0.20 overlap]
        S1Split --> F11[Shared shallow feature 1 + Encoder 1<br/>F1_1]
        S1Split --> F12[Shared shallow feature 1 + Encoder 1<br/>F1_2]
        S1Split --> F13[Shared shallow feature 1 + Encoder 1<br/>F1_3]
        F11 --> P12[Pair feature assembly: F1_1 + F1_2]
        F12 --> P12
        F12 --> P23[Pair feature assembly: F1_2 + F1_3]
        F13 --> P23
        P12 --> D12[Shared Decoder 1 + SAM12<br/>stage-1 pair output r1_12]
        P23 --> D23[Shared Decoder 1 + SAM12<br/>stage-1 pair output r1_23]
        D12 --> B1[Overlap blending]
        D23 --> B1
        B1 --> O1[Stage-1 image O1<br/>and pair-specific features]
    end

    I --> Raw12[Raw adjacent crop I_12]
    I --> Raw23[Raw adjacent crop I_23]
    O1 --> S2

    subgraph S2[Stage 2: PairAligned Reconstruction with TCFM]
        Raw12 --> SF12[Shallow feature 2]
        Raw23 --> SF23[Shallow feature 2]
        O1 --> SAM12a[Matching SAM12 feature for pair 12]
        O1 --> SAM12b[Matching SAM12 feature for pair 23]
        SF12 --> C12[Concatenate shallow feature and SAM12]
        SAM12a --> C12
        SF23 --> C23[Concatenate shallow feature and SAM12]
        SAM12b --> C23
        C12 --> E12[Stage-2 Encoder, pair 12]
        C23 --> E23[Stage-2 Encoder, pair 23]
        P12 -. Stage-1 encoder / decoder features .-> E12
        P23 -. Stage-1 encoder / decoder features .-> E23
        E12 --> G12[TCFM at each encoder scale]
        E23 --> G23[TCFM at each encoder scale]
        G12 --> B2[Blend aligned pair features<br/>at scales 1, 1/2, 1/4]
        G23 --> B2
        B2 --> D2[Original Stage-2 Decoder + SAM23]
        D2 --> O2[Stage-2 image O2 and SAM23 feature]
    end

    I --> SF3[Shallow feature 3]
    O2 --> SAM23[Stage-2 SAM23 feature]
    subgraph S3[Stage 3: Full-Resolution Restoration]
        SF3 --> C3[Concatenate shallow feature 3 and SAM23]
        SAM23 --> C3
        C3 --> ORS[Original full-resolution restoration path<br/>with Stage-2 encoder and decoder features]
        ORS --> R[RGB tail + residual addition to I]
    end

    R --> BaseOut[TPT-Net restored image]
    BaseOut -. proposed spatial innovation .-> Edge[Optional edge-guided refinement]
    BaseOut -. proposed frequency innovation .-> Band[Optional adaptive frequency BandGate]
    Edge -. final model .-> Final[Final restored image]
    Band -. final model .-> Final
```

## TCFM operation

At every Stage-2 encoder scale, TCFM replaces unconditional cross-stage additive injection. Let `c` be the current Stage-2 feature and `e` / `d` be the aligned Stage-1 encoder / decoder features after their 1x1 projections.

```text
u = concat(c, e, d, abs(e - d))
g = 2 * sigmoid(phi_1x1 -> PReLU -> DWConv(1x7) -> PReLU -> phi_1x1(u))
fused = c + g * (e + d)
```

The final TCFM convolution is zero initialized. Therefore `g = 1` at initialization and the first forward pass is equivalent to unmodulated additive feature injection. Training learns where horizontal trajectory context should attenuate or strengthen the cross-stage contribution.

## Ablation protocol

Use TPT-Net as the fixed baseline. Do not include edge refinement or frequency refinement in its checkpoint.

| ID | Model | Purpose |
|---|---|---|
| B | TPT-Net | Structural baseline |
| E | B + edge-guided refinement | Spatial-boundary innovation |
| F | B + adaptive frequency BandGate | Frequency-detail innovation |
| EF | B + edge-guided refinement + BandGate | Final model |

The existing implementation class and checkpoint paths retain their historical names for compatibility. Use **TPT-Net** and **TCFM** in figures, tables, and writing.

## Innovation A: STAM

**STAM** (Stitch-Trajectory Alignment Module) replaces direct Stage-1 to Stage-2 pair injection with explicit, multi-scale feature alignment.

```text
Stage-2 pair feature + matching Stage-1 encoder feature
    -> horizontal correlation volume over offsets [-6, +6]
    -> learned 2-D trajectory field (horizontal radius 6, vertical radius 1)
    -> differentiable feature resampling
    -> aligned Stage-1 encoder and decoder features
    -> TCFM and Stage-2 encoder
```

STAM is initialized with an identity trajectory field, allowing it to inherit a checkpoint safely. The corresponding experiment uses `training_tpt_net_stam_fast100.yml` and inherits the validated AF-BandGate checkpoint, so its reported gain measures the final-path contribution of trajectory alignment.
