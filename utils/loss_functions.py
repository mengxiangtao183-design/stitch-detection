"""
Implements tailored loss functions for the C3Net architecture, optimized based on
experimental findings and analysis.

This module provides the necessary loss components for training C3Net, focusing on
addressing challenges specific to Camouflaged Object Detection (COD). It includes:
- C3NetLoss: The main loss class orchestrating deep supervision and combining
  specialized loss terms for different model outputs (edge, localization, final).
- Helper functions for:
    - Focal Loss: To handle class imbalance between foreground and background.
    - Tversky Loss: To provide a tunable balance between penalizing False Positives
      and False Negatives, with different settings applied to intermediate vs. final outputs.
    - Dice Loss on Edges: A term added to the final loss to specifically improve
      boundary alignment using ground truth edges with a controlled weight.
    - Total Variation Loss: Used as a regularization term to encourage smoothness
      in predicted background regions of the intermediate edge map.

Key features of the finalized loss strategy:
- Deep Supervision: Standard application with weights w_edge, w_loc, w_final.
- Instance Weighting: Applied to localization and final segmentation losses based on object size.
- Specialized Intermediate Edge Supervision: Unchanged (Weighted Focal + TV Reg).
- Tunable Precision/Recall Balance:
    - Intermediate Localization Loss (`loss_loc`): Uses Tversky loss prioritizing *Precision*
      (alpha=0.6, beta=0.4) to strongly penalize salient false positives potentially
      missed by the ICG module operating just before this loss point (Position P1).
    - Final Segmentation Loss (`loss_final`): Uses Tversky loss prioritizing *Recall*
      (alpha=0.4, beta=0.6) to ensure the complete object is captured after fusion,
      complementing the precision focus of the intermediate loss.
- Final Edge Guidance: Augments the final segmentation loss with a low-weighted Dice
  loss calculated only on GT edge pixels to promote sharper boundaries without
  overpowering the region segmentation.

Author  : Baber Jan
Date : 09-05-2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

# Initialize logger for this module
logger = logging.getLogger(__name__)

# --- Helper Loss Functions ---

def focal_loss_with_logits(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Computes the mean Focal Loss between predicted logits and binary targets.
    Handles optional spatial weighting internally. Always returns mean loss.
    Ensures inputs are 4D (B, C, H, W) for calculation.
    """
    # --- Input Dimension Handling & Type Conversion ---
    input_dim = inputs.dim()
    if input_dim == 2: inputs = inputs.unsqueeze(0).unsqueeze(0) # (H, W) -> (1, 1, H, W)
    elif input_dim == 3: inputs = inputs.unsqueeze(0)           # (C, H, W) -> (1, C, H, W)

    target_dim = targets.dim()
    if target_dim == 2: targets = targets.unsqueeze(0).unsqueeze(0) # (H, W) -> (1, 1, H, W)
    elif target_dim == 3: targets = targets.unsqueeze(0)           # (C, H, W) -> (1, C, H, W)
    targets = targets.float()

    # --- Channel / Shape Checks ---
    if inputs.shape[1] != 1 or targets.shape[1] != 1:
         logger.warning(f"Focal loss expects C=1. Got Input {inputs.shape}, Target {targets.shape}. Using C=1.")
         inputs = inputs[:,:1,:,:]
         targets = targets[:,:1,:,:]
    if inputs.shape[-2:] != targets.shape[-2:]:
         raise ValueError(f"Focal loss spatial dimension mismatch: Input {inputs.shape}, Target {targets.shape}")


    # --- Calculate Base BCE Loss (Per Element) ---
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') # (B, 1, H, W)

    # --- Calculate Focal Loss Components (Per Element) ---
    probs = torch.sigmoid(inputs)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    # Add epsilon inside power for stability, especially with gamma
    modulating_factor = torch.pow(1.0 - p_t + eps, gamma)
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    focal_loss_per_pixel = alpha_factor * modulating_factor * bce_loss # (B, 1, H, W)

    # --- Apply Spatial Weights and Reduce to Mean ---
    if weight is not None:
        try:
            if weight.dim() == 2: weight = weight.unsqueeze(0).unsqueeze(0) # H, W -> 1, 1, H, W
            elif weight.dim() == 3: weight = weight.unsqueeze(0)           # C, H, W -> 1, C, H, W (assume C=1)
            weight = weight.expand_as(focal_loss_per_pixel) # Ensure broadcastable
            # Compute weighted mean: sum(loss*w) / sum(w)
            weighted_loss_per_pixel = focal_loss_per_pixel * weight
            # Clamp weight sum to avoid division by zero if weights are all zero
            weight_sum = weight.sum().clamp(min=eps)
            mean_loss = weighted_loss_per_pixel.sum() / weight_sum
        except Exception as e:
            logger.error(f"Failed spatial weight application in focal loss: {e}. Returning unweighted mean.", exc_info=True)
            mean_loss = focal_loss_per_pixel.mean()
    else:
        mean_loss = focal_loss_per_pixel.mean()

    return mean_loss


def tversky_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float, # Weight for FP - Passed as argument
    beta: float,  # Weight for FN - Passed as argument
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Computes the Tversky loss with specified alpha and beta.
    Assumes inputs are probabilities [0, 1]. Averages loss across batch dim.
    """
    inputs = torch.clamp(inputs, min=0.0, max=1.0)
    targets = targets.float()

    if inputs.shape != targets.shape:
        raise ValueError(f"Shape mismatch: inputs {inputs.shape}, targets {targets.shape}")

    # Flatten spatial dimensions but keep batch dimension
    inputs_flat = inputs.view(inputs.shape[0], -1)
    targets_flat = targets.view(targets.shape[0], -1)

    # Calculate TP, FP, FN per sample
    TP = (inputs_flat * targets_flat).sum(1)
    FP = (inputs_flat * (1 - targets_flat)).sum(1)
    FN = ((1 - inputs_flat) * targets_flat).sum(1)

    denominator = TP + alpha * FP + beta * FN + smooth
    tversky_index = (TP + smooth) / denominator # Per sample Tversky Index

    # Clamp index for stability
    tversky_index = torch.clamp(tversky_index, min=0.0, max=1.0)
    loss = 1.0 - tversky_index # Per sample loss

    return loss.mean() # Average loss across batch


def dice_loss_on_edges(
    probs: torch.Tensor,
    gt_mask: torch.Tensor,
    gt_edges: torch.Tensor,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Computes Dice loss between predicted probabilities and the GT mask,
    considering only pixels marked by gt_edges. Averages loss across batch.
    Expects inputs shape (B, 1, H, W).
    """
    probs = probs.float()
    gt_mask = gt_mask.float()
    gt_edges = gt_edges.float() # Ensure edges are float {0., 1.}

    # --- Input Dimension Handling & Type Conversion ---
    input_dim = probs.dim()
    if input_dim == 2: probs = probs.unsqueeze(0).unsqueeze(0) # (H, W) -> (1, 1, H, W)
    elif input_dim == 3: probs = probs.unsqueeze(0)           # (C, H, W) -> (1, C, H, W)

    target_dim = gt_mask.dim()
    if target_dim == 2: gt_mask = gt_mask.unsqueeze(0).unsqueeze(0) # (H, W) -> (1, 1, H, W)
    elif target_dim == 3: gt_mask = gt_mask.unsqueeze(0)           # (C, H, W) -> (1, C, H, W)
    gt_mask = gt_mask.float()

    target_dim = gt_edges.dim()
    if target_dim == 2: gt_edges = gt_edges.unsqueeze(0).unsqueeze(0) # (H, W) -> (1, 1, H, W)
    elif target_dim == 3: gt_edges = gt_edges.unsqueeze(0)           # (C, H, W) -> (1, C, H, W)
    gt_edges = gt_edges.float()

    if not (probs.shape == gt_mask.shape == gt_edges.shape and probs.shape[1] == 1):
         raise ValueError(f"Inputs must have shape (B, 1, H, W) and match. Got P:{probs.shape}, G:{gt_mask.shape}, E:{gt_edges.shape}")

    # Apply edge mask
    P_edge = probs * gt_edges
    G_edge = gt_mask * gt_edges

    # Calculate sums per sample in batch (sum over H, W)
    intersection = (P_edge * G_edge).sum(dim=[2, 3]) # Shape: (B, 1)
    denominator = P_edge.sum(dim=[2, 3]) + G_edge.sum(dim=[2, 3]) # Shape: (B, 1)

    # Calculate Dice score per sample
    # Add smooth to numerator AND denominator
    dice_score = (2. * intersection + smooth) / (denominator + smooth) # Shape: (B, 1)
    dice_loss = 1.0 - dice_score # Shape: (B, 1)

    # Average loss over the batch
    mean_dice_loss = dice_loss.mean()

    return mean_dice_loss


def total_variation_loss(
    img: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Computes the mean anisotropic Total Variation (TV) Loss.
    Operates on the last two dimensions (H, W). Calculates mean loss across batch/channels.
    """
    if img.dim() < 4: # Expect B, C, H, W
        logger.debug(f"TV Loss input dim < 4 ({img.shape}), returning 0.")
        return torch.tensor(0.0, device=img.device, dtype=img.dtype)

    dh = torch.abs(img[..., 1:, :] - img[..., :-1, :])
    dw = torch.abs(img[..., :, 1:] - img[..., :, :-1])

    # Mean per pixel difference
    # Sum over all dimensions, divide by total number of difference elements computed
    total_diff_elements = dh.numel() + dw.numel()
    if total_diff_elements < eps:
        return torch.tensor(0.0, device=img.device, dtype=img.dtype)

    loss = (dh.sum() + dw.sum()) / total_diff_elements
    return loss


def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    """Convert HxW or CxHxW tensors to BxCxHxW."""
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3:
        return x.unsqueeze(0)
    return x


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    erode_h = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    erode_w = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(erode_h, erode_w)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iterations: int = 8) -> torch.Tensor:
    """
    Differentiable skeletonization used by soft-clDice.

    This approximates the centerline of thin stitch masks without leaving the
    PyTorch graph, so it can be used as a topology-preserving loss.
    """
    x = torch.clamp(x, 0.0, 1.0)
    skel = F.relu(x - _soft_open(x))
    for _ in range(max(0, iterations)):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return torch.clamp(skel, 0.0, 1.0)


def soft_cldice_loss(
    pred_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    iterations: int = 8,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Topology-preserving loss for thin stitch segmentation.

    Penalizes broken or missing centerlines more directly than region Dice.
    """
    pred = torch.sigmoid(_ensure_bchw(pred_logits)).float()
    target = _ensure_bchw(gt_mask).float()
    if pred.shape[-2:] != target.shape[-2:]:
        raise ValueError(f"soft-clDice spatial mismatch: pred={pred.shape}, target={target.shape}")

    pred_skel = soft_skeletonize(pred, iterations)
    target_skel = soft_skeletonize(target, iterations)

    tprec = (pred_skel * target).sum(dim=[2, 3]) / (pred_skel.sum(dim=[2, 3]) + smooth)
    tsens = (target_skel * pred).sum(dim=[2, 3]) / (target_skel.sum(dim=[2, 3]) + smooth)
    cldice = (2.0 * tprec * tsens + smooth) / (tprec + tsens + smooth)
    return (1.0 - cldice).mean()


def boundary_dice_loss(
    pred_logits: torch.Tensor,
    gt_edges: torch.Tensor,
    kernel_size: int = 3,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Dice loss on a soft morphological boundary map.

    This stabilizes stitch width/boundary geometry for downstream engineering
    measurements such as width spikes, thread throwing and line deviation.
    """
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError(f"boundary kernel_size must be odd and >= 3, got {kernel_size}")

    pred = torch.sigmoid(_ensure_bchw(pred_logits)).float()
    target_edges = _ensure_bchw(gt_edges).float()
    if pred.shape[-2:] != target_edges.shape[-2:]:
        raise ValueError(
            f"boundary loss spatial mismatch: pred={pred.shape}, edges={target_edges.shape}"
        )

    pad = kernel_size // 2
    pred_dilate = F.max_pool2d(pred, kernel_size=kernel_size, stride=1, padding=pad)
    pred_erode = -F.max_pool2d(-pred, kernel_size=kernel_size, stride=1, padding=pad)
    pred_boundary = torch.clamp(pred_dilate - pred_erode, 0.0, 1.0)
    target_edges = torch.clamp(target_edges, 0.0, 1.0)

    intersection = (pred_boundary * target_edges).sum(dim=[2, 3])
    denominator = pred_boundary.sum(dim=[2, 3]) + target_edges.sum(dim=[2, 3])
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return (1.0 - dice).mean()


@torch.no_grad()
def centerline_target_from_mask(
    gt_mask: torch.Tensor,
    iterations: int = 8,
    dilation_kernel_size: int = 3,
) -> torch.Tensor:
    """
    Generate a soft centerline target from an existing stitch mask.

    The target is derived from the mask by differentiable-style morphological
    skeletonization, but it is used as a fixed label. A small dilation makes the
    supervision less brittle than a strict one-pixel skeleton.
    """
    target = _ensure_bchw(gt_mask).float().clamp(0.0, 1.0)
    centerline = soft_skeletonize(target, iterations=iterations)
    if dilation_kernel_size > 1:
        if dilation_kernel_size % 2 == 0:
            raise ValueError("centerline_dilation_kernel_size must be odd")
        pad = dilation_kernel_size // 2
        centerline = F.max_pool2d(
            centerline,
            kernel_size=dilation_kernel_size,
            stride=1,
            padding=pad,
        )
    return centerline.clamp(0.0, 1.0)


def centerline_loss_with_logits(
    pred_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    iterations: int = 8,
    dilation_kernel_size: int = 3,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    focal_weight: float = 1.0,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Focal + Dice loss for the auxiliary stitch centerline prediction."""
    pred_logits = _ensure_bchw(pred_logits)
    target = centerline_target_from_mask(
        gt_mask,
        iterations=iterations,
        dilation_kernel_size=dilation_kernel_size,
    ).to(device=pred_logits.device, dtype=pred_logits.dtype)

    if pred_logits.shape[-2:] != target.shape[-2:]:
        raise ValueError(
            f"centerline loss spatial mismatch: pred={pred_logits.shape}, target={target.shape}"
        )

    focal = focal_loss_with_logits(
        inputs=pred_logits,
        targets=target,
        alpha=focal_alpha,
        gamma=focal_gamma,
        eps=smooth,
    )
    probs = torch.sigmoid(pred_logits)
    intersection = (probs * target).sum(dim=[2, 3])
    denominator = probs.sum(dim=[2, 3]) + target.sum(dim=[2, 3])
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return focal_weight * focal + (1.0 - dice).mean()


@torch.no_grad()
def truncated_signed_distance_map(
    gt_mask: torch.Tensor,
    truncation: float,
) -> torch.Tensor:
    """Create a normalized truncated SDF from a binary mask.

    Positive values are inside the stitch and negative values are outside.  The
    output is clipped to ``[-truncation, truncation]`` and normalized to
    ``[-1, 1]``.  OpenCV's Euclidean distance transform is used only for the
    non-differentiable ground-truth target; gradients flow solely through the
    SDF head prediction.
    """
    if truncation <= 0:
        raise ValueError("sdf_truncation must be > 0")
    if gt_mask.dim() == 3 and gt_mask.shape[0] == 1:
        mask_2d = gt_mask.squeeze(0)
    elif gt_mask.dim() == 2:
        mask_2d = gt_mask
    else:
        raise ValueError(
            "SDF target expects a single binary mask with shape (1, H, W) or (H, W), "
            f"got {tuple(gt_mask.shape)}"
        )

    mask_u8 = (mask_2d.detach().float().cpu().numpy() > 0.5).astype(np.uint8)
    if mask_u8.max() == 0:
        sdf = -np.full(mask_u8.shape, truncation, dtype=np.float32)
    elif mask_u8.min() == 1:
        sdf = np.full(mask_u8.shape, truncation, dtype=np.float32)
    else:
        inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
        outside = cv2.distanceTransform(1 - mask_u8, cv2.DIST_L2, 3)
        sdf = inside - outside
        sdf = np.clip(sdf, -truncation, truncation).astype(np.float32)

    return torch.from_numpy(sdf / float(truncation)).unsqueeze(0).to(
        device=gt_mask.device, dtype=gt_mask.dtype
    )


@torch.no_grad()
def ribbon_half_width_target(
    gt_mask: torch.Tensor,
    max_half_width: float,
    skeleton_iterations: int = 8,
    centerline_dilation_kernel_size: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return local half-width and valid centerline locations for the G1 head."""
    if max_half_width <= 0:
        raise ValueError("ribbon max_half_width must be positive")
    if gt_mask.dim() == 3 and gt_mask.shape[0] == 1:
        mask_2d = gt_mask.squeeze(0)
    elif gt_mask.dim() == 2:
        mask_2d = gt_mask
    else:
        raise ValueError(
            "Ribbon width target expects (1,H,W) or (H,W), got "
            f"{tuple(gt_mask.shape)}"
        )
    mask_u8 = (mask_2d.detach().float().cpu().numpy() > 0.5).astype(np.uint8)
    distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3).astype(np.float32)
    width_target = torch.from_numpy(
        np.clip(distance, 0.0, float(max_half_width))
    ).unsqueeze(0).to(device=gt_mask.device, dtype=gt_mask.dtype)
    centerline = centerline_target_from_mask(
        gt_mask,
        iterations=skeleton_iterations,
        dilation_kernel_size=centerline_dilation_kernel_size,
    )
    return width_target, centerline


def ribbon_width_loss(
    pred_half_width: torch.Tensor,
    gt_mask: torch.Tensor,
    max_half_width: float,
    skeleton_iterations: int = 8,
    centerline_dilation_kernel_size: int = 3,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Robust half-width regression restricted to GT stitch centerlines."""
    prediction = _ensure_bchw(pred_half_width)
    target, valid = ribbon_half_width_target(
        gt_mask,
        max_half_width=max_half_width,
        skeleton_iterations=skeleton_iterations,
        centerline_dilation_kernel_size=centerline_dilation_kernel_size,
    )
    target = _ensure_bchw(target).to(device=prediction.device, dtype=prediction.dtype)
    valid = _ensure_bchw(valid).to(device=prediction.device, dtype=prediction.dtype)
    if prediction.shape[-2:] != target.shape[-2:]:
        raise ValueError(
            "Ribbon width spatial mismatch: "
            f"prediction={prediction.shape}, target={target.shape}"
        )
    error = F.smooth_l1_loss(prediction, target, reduction='none')
    return (error * valid).sum() / (valid.sum() + eps)


def sample_thickness_prior_loss(
    pred_width_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    max_half_width: float,
    skeleton_iterations: int = 8,
    centerline_dilation_kernel_size: int = 3,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Supervise a sample-level normalized stitch half-width prior.

    The router predicts one normalized half-width value per image.  The target
    is derived from the existing GT mask by measuring the distance transform at
    the stitch centerline and normalizing by ``max_half_width``.
    """
    prediction = torch.sigmoid(_ensure_bchw(pred_width_logits)).mean()
    target_width, valid = ribbon_half_width_target(
        gt_mask,
        max_half_width=max_half_width,
        skeleton_iterations=skeleton_iterations,
        centerline_dilation_kernel_size=centerline_dilation_kernel_size,
    )
    target_width = target_width.to(device=prediction.device, dtype=prediction.dtype)
    valid = valid.to(device=prediction.device, dtype=prediction.dtype)
    target = (
        (target_width * valid).sum() /
        (valid.sum() * float(max_half_width) + eps)
    ).clamp(0.0, 1.0)
    return F.smooth_l1_loss(prediction, target, reduction='mean')


@torch.no_grad()
def connected_component_instance_targets(
    gt_mask: torch.Tensor,
    min_area: int = 4,
    max_instances: Optional[int] = None,
) -> torch.Tensor:
    """Split one binary mask into connected-component instance targets."""
    if gt_mask.dim() == 3 and gt_mask.shape[0] == 1:
        mask_2d = gt_mask.squeeze(0)
    elif gt_mask.dim() == 2:
        mask_2d = gt_mask
    else:
        raise ValueError(
            "Instance targets expect one mask with shape (1,H,W) or (H,W), "
            f"got {tuple(gt_mask.shape)}"
        )
    mask_u8 = (mask_2d.detach().float().cpu().numpy() > 0.5).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    components = []
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        components.append((area, labels == label))

    if max_instances is not None and max_instances > 0:
        components = sorted(components, key=lambda item: item[0], reverse=True)
        components = components[:max_instances]

    if not components:
        height, width = mask_u8.shape
        return torch.zeros((0, height, width), dtype=torch.float32, device=gt_mask.device)

    targets = np.stack([component.astype(np.float32) for _, component in components], axis=0)
    return torch.from_numpy(targets).to(device=gt_mask.device, dtype=torch.float32)


def trajectory_instance_hungarian_loss(
    instance_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    unmatched_weight: float = 0.15,
    min_area: int = 4,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Hungarian-matched per-trajectory BCE+Dice loss for multi-slot masks."""
    logits = instance_logits
    if logits.dim() == 4 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    if logits.dim() != 3:
        raise ValueError(
            "trajectory instance logits must have shape (K,H,W) or (1,K,H,W), "
            f"got {tuple(instance_logits.shape)}"
        )
    logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)

    num_slots, height, width = logits.shape
    targets = connected_component_instance_targets(
        gt_mask, min_area=min_area, max_instances=num_slots
    ).to(device=logits.device, dtype=logits.dtype)
    zeros = torch.zeros_like(logits)
    if targets.numel() == 0:
        return F.binary_cross_entropy_with_logits(logits, zeros)

    if targets.shape[-2:] != logits.shape[-2:]:
        targets = F.interpolate(
            targets.unsqueeze(0),
            size=logits.shape[-2:],
            mode='nearest',
        ).squeeze(0)

    probs = torch.nan_to_num(torch.sigmoid(logits), nan=0.0, posinf=1.0, neginf=0.0)
    with torch.no_grad():
        probs_flat = probs.detach().flatten(1)
        target_flat = targets.flatten(1)
        intersection = probs_flat @ target_flat.t()
        pred_sum = probs_flat.sum(dim=1, keepdim=True)
        target_sum = target_flat.sum(dim=1).unsqueeze(0)
        dice_cost = 1.0 - (2.0 * intersection + eps) / (
            pred_sum + target_sum + eps
        )
        dice_cost = torch.nan_to_num(
            dice_cost, nan=1.0, posinf=1.0, neginf=0.0
        ).clamp_(0.0, 1.0)
        pred_indices_np, target_indices_np = linear_sum_assignment(
            dice_cost.detach().cpu().numpy()
        )

    pred_indices = torch.as_tensor(pred_indices_np, device=logits.device, dtype=torch.long)
    target_indices = torch.as_tensor(target_indices_np, device=logits.device, dtype=torch.long)
    matched_logits = logits[pred_indices]
    matched_targets = targets[target_indices]

    bce = F.binary_cross_entropy_with_logits(matched_logits, matched_targets)
    matched_probs = torch.nan_to_num(
        torch.sigmoid(matched_logits), nan=0.0, posinf=1.0, neginf=0.0
    )
    reduce_dims = tuple(range(1, matched_probs.dim()))
    intersection = (matched_probs * matched_targets).sum(dim=reduce_dims)
    denominator = matched_probs.sum(dim=reduce_dims) + matched_targets.sum(dim=reduce_dims)
    dice = 1.0 - (2.0 * intersection + eps) / (denominator + eps)
    matched_loss = bce + dice.mean()

    if pred_indices.numel() < num_slots and unmatched_weight > 0:
        unmatched_mask = torch.ones(num_slots, dtype=torch.bool, device=logits.device)
        unmatched_mask[pred_indices] = False
        unmatched_logits = logits[unmatched_mask]
        background_loss = F.binary_cross_entropy_with_logits(
            unmatched_logits,
            torch.zeros_like(unmatched_logits),
        )
        matched_loss = matched_loss + float(unmatched_weight) * background_loss

    return matched_loss


@torch.no_grad()
def stitch_pattern_targets(gt_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Derive count-free component center, direction and spacing targets.

    Targets come only from connected components in the existing binary mask.
    They therefore describe the repeated stitch layout without hard-coding the
    observed six-component pattern or a fixed pixel pitch.
    """
    if gt_mask.dim() == 3 and gt_mask.shape[0] == 1:
        mask_2d = gt_mask.squeeze(0)
    elif gt_mask.dim() == 2:
        mask_2d = gt_mask
    else:
        raise ValueError(
            "Pattern targets expect one mask with shape (1, H, W) or (H, W), "
            f"got {tuple(gt_mask.shape)}"
        )
    mask_u8 = (mask_2d.detach().float().cpu().numpy() > 0.5).astype(np.uint8)
    height, width = mask_u8.shape
    component_count, labels, _, centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
    center = np.zeros((height, width), dtype=np.float32)
    direction = np.zeros((2, height, width), dtype=np.float32)
    spacing = np.zeros((height, width), dtype=np.float32)
    spacing_valid = np.zeros((height, width), dtype=np.float32)
    components = []

    for label in range(1, component_count):
        ys, xs = np.where(labels == label)
        if xs.size < 4:
            continue
        coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=0)
        covariance = np.cov(coords)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        # Direction sign is canonicalized only for the regression target.  The
        # relation sampler itself uses both +/- neighbours and is sign-invariant.
        if axis[0] < 0:
            axis = -axis
        direction[0, labels == label] = axis[0]
        direction[1, labels == label] = axis[1]
        components.append((float(centroids[label, 0]), float(centroids[label, 1])))

    if components:
        # Soft centers remain stable after the trainer resizes predictions to
        # the original image grid.  The radius scales gently with image size.
        radius = max(2, int(round(min(height, width) / 80.0)))
        sigma = max(1.0, radius / 1.5)
        for cx, cy in components:
            cv2.circle(center, (int(round(cx)), int(round(cy))), radius, 1.0, -1)
        center = cv2.GaussianBlur(center, (0, 0), sigmaX=sigma, sigmaY=sigma)
        center /= max(float(center.max()), 1e-6)

        # Sort along the image x-axis rather than assuming a component count.
        # This is appropriate for the observed near-horizontal stitch rows and
        # remains valid under small affine/rotational augmentation.
        ordered = sorted(components, key=lambda point: point[0])
        for index, (cx, cy) in enumerate(ordered):
            distances = []
            if index > 0:
                px, py = ordered[index - 1]
                distances.append(float(np.hypot(cx - px, cy - py)))
            if index + 1 < len(ordered):
                nx, ny = ordered[index + 1]
                distances.append(float(np.hypot(nx - cx, ny - cy)))
            if not distances:
                continue
            spacing_value = float(np.mean(distances) / max(width, 1))
            cv2.circle(spacing, (int(round(cx)), int(round(cy))), radius, spacing_value, -1)
            cv2.circle(spacing_valid, (int(round(cx)), int(round(cy))), radius, 1.0, -1)

    device, dtype = gt_mask.device, gt_mask.dtype
    return {
        'center': torch.from_numpy(center).unsqueeze(0).to(device=device, dtype=dtype),
        'direction': torch.from_numpy(direction).to(device=device, dtype=dtype),
        'spacing': torch.from_numpy(spacing).unsqueeze(0).to(device=device, dtype=dtype),
        'spacing_valid': torch.from_numpy(spacing_valid).unsqueeze(0).to(device=device, dtype=dtype),
        'direction_valid': torch.from_numpy(mask_u8.astype(np.float32)).unsqueeze(0).to(device=device, dtype=dtype),
    }


@torch.no_grad()
def contour_core_target(gt_mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Create three-class labels: background=0, boundary=1, interior=2."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("contour_kernel_size must be odd and >= 3")
    if gt_mask.dim() == 3 and gt_mask.shape[0] == 1:
        mask_2d = gt_mask.squeeze(0)
    elif gt_mask.dim() == 2:
        mask_2d = gt_mask
    else:
        raise ValueError(
            "Contour target expects one mask with shape (1, H, W) or (H, W), "
            f"got {tuple(gt_mask.shape)}"
        )
    mask_u8 = (mask_2d.detach().float().cpu().numpy() > 0.5).astype(np.uint8)
    eroded = cv2.erode(
        mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        iterations=1,
    )
    target = np.zeros(mask_u8.shape, dtype=np.int64)
    target[(mask_u8 > 0) & (eroded == 0)] = 1
    target[eroded > 0] = 2
    return torch.from_numpy(target).to(device=gt_mask.device, dtype=torch.long)


# --- Main C3Net Loss Class ---

class C3NetLoss(nn.Module):
    """
    Combined loss function for C3Net with deep supervision and finalized settings.

    Implements:
    - Deep supervision weights (w_edge, w_loc, w_final).
    - Instance weighting for loc and final losses based on object size.
    - Intermediate edge loss: Spatially weighted Focal + TV regularization.
    - Intermediate localization loss: Instance Weighted (Focal + Tversky (Precision Focus)).
    - Final loss: Instance Weighted (Focal + Tversky (Recall Focus) + Weighted Edge Dice).

    Args:
        w_edge (float): Weight for intermediate edge loss. Default: 1.0.
        w_loc (float): Weight for intermediate localization loss. Default: 1.0.
        w_final (float): Weight for final fusion head loss. Default: 1.0.
        focal_alpha (float): Alpha for Focal Loss component. Default: 0.25.
        focal_gamma (float): Gamma for Focal Loss component. Default: 2.0.
        tversky_alpha_loc (float): Alpha for Tversky in loc loss (Precision). Default: 0.7.
        tversky_beta_loc (float): Beta for Tversky in loc loss (Precision). Default: 0.3.
        tversky_alpha_final (float): Alpha for Tversky in final loss (Recall). Default: 0.3.
        tversky_beta_final (float): Beta for Tversky in final loss (Recall). Default: 0.7.
        final_edge_dice_weight (float): Weight for edge Dice in final loss. Default: 0.1.
        edge_focal_weight (float): Max spatial weight near GT edges for interm. edge loss. Default: 5.0.
        edge_reg_weight (float): Weight for TV regularization in interm. edge loss. Default: 0.1.
        edge_dist_cutoff (int): Iterations for edge weight approx. Default: 3.
        instance_weight_k (float): Factor for instance weighting (0 disables). Default: 2.0.
        instance_weight_low_thresh (float): Lower FG ratio threshold. Default: 0.01.
        instance_weight_high_thresh (float): Upper FG ratio threshold. Default: 0.9.
        eps (float): Epsilon for numerical stability. Default: 1e-7.
    """
    def __init__(
        self,
        w_edge: float = 1.0,
        w_loc: float = 1.0,
        w_final: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        # Separate Tversky parameters based on reasoning
        tversky_alpha_loc: float = 0.6, # Precision focus for Loc Branch loss
        tversky_beta_loc: float = 0.4,
        tversky_alpha_final: float = 0.4, # Recall focus for Final loss
        tversky_beta_final: float = 0.6,
        # Final edge loss weight (decisive low value)
        final_edge_dice_weight: float = 0.1,
        # Intermediate edge loss params
        edge_focal_weight: float = 5.0,
        edge_reg_weight: float = 0.1,
        edge_dist_cutoff: int = 3,
        # Instance weighting params
        instance_weight_k: float = 2.0,
        instance_weight_low_thresh: float = 0.01,
        instance_weight_high_thresh: float = 0.9,        topology_weight: float = 0.15,
        boundary_weight: float = 0.10,
        centerline_weight: float = 0.4,
        centerline_focal_weight: float = 1.0,
        centerline_dilation_kernel_size: int = 3,
        skeleton_iterations: int = 8,
        boundary_kernel_size: int = 3,
        sdf_weight: float = 0.0,
        sdf_truncation: float = 8.0,
        pattern_center_weight: float = 0.0,
        pattern_direction_weight: float = 0.0,
        pattern_spacing_weight: float = 0.0,
        pattern_spacing_min_norm: float = 0.06,
        pattern_spacing_max_norm: float = 0.30,
        contour_refine_weight: float = 0.0,
        contour_kernel_size: int = 3,
        ribbon_mask_weight: float = 0.0,
        ribbon_width_weight: float = 0.0,
        ribbon_width_max: float = 5.5,
        thickness_prior_weight: float = 0.0,
        thickness_prior_max_half_width: float = 5.5,
        instance_aggregate_weight: float = 0.0,
        trajectory_instance_weight: float = 0.0,
        trajectory_unmatched_weight: float = 0.15,
        trajectory_instance_min_area: int = 4,
        prototype_context_weight: float = 0.0,
        prototype_warmup_epochs: int = 5,
        prototype_hard_negative_start: int = 10,
        prototype_hard_negative_weight: float = 2.0,
        prototype_hard_negative_threshold: float = 0.5,
        hard_negative_suppression_weight: float = 0.0,
        hard_negative_safety_weight: float = 0.0,
        hard_negative_warmup_epochs: int = 5,
        hard_negative_mining_threshold: float = 0.5,
        texture_hard_negative_weight: float = 0.0,
        texture_foreground_safety_weight: float = 0.0,
        texture_warmup_epochs: int = 5,
        texture_hard_negative_quantile: float = 0.85,
        afm_coarse_weight: float = 0.0,
        afm_coarse_warmup_epochs: int = 5,
        afm_aux_weight: float = 0.0,
        afm_aux_warmup_epochs: int = 5,
        weak_stitch_weight: float = 0.0,
        weak_stitch_hard_positive_weight: float = 0.25,
        weak_stitch_hard_negative_weight: float = 0.0,
        weak_stitch_hard_negative_threshold: float = 0.60,
        weak_stitch_consistency_weight: float = 0.0,
        weak_stitch_warmup_epochs: int = 5,
        weak_stitch_visibility_threshold: float = 0.5,
        eps: float = 1e-7
    ):
        super().__init__()
        # --- Validation ---
        if not all(w >= 0 for w in [
            w_edge, w_loc, w_final, final_edge_dice_weight, centerline_weight
        ]):
            raise ValueError("Loss weights must be >= 0.")
        if not (0 < instance_weight_low_thresh < instance_weight_high_thresh < 1):
             raise ValueError("Instance weight thresholds: 0 < low < high < 1.")
        if not (tversky_alpha_loc >= 0 and tversky_beta_loc >= 0 and (tversky_alpha_loc + tversky_beta_loc) > eps):
             raise ValueError("Tversky loc alpha/beta must be non-negative and sum > eps.")
        if not (tversky_alpha_final >= 0 and tversky_beta_final >= 0 and (tversky_alpha_final + tversky_beta_final) > eps):
             raise ValueError("Tversky final alpha/beta must be non-negative and sum > eps.")
        if not (0 <= focal_alpha <= 1): raise ValueError("focal_alpha must be in [0, 1]")
        if not (focal_gamma >= 0): raise ValueError("focal_gamma must be >= 0")
        if not (edge_focal_weight >= 1.0): raise ValueError("edge_focal_weight must be >= 1.0")
        if not (edge_reg_weight >= 0): raise ValueError("edge_reg_weight must be >= 0")
        if not (edge_dist_cutoff >= 1): raise ValueError("edge_dist_cutoff must be >= 1")
        if not (instance_weight_k >= 0): raise ValueError("instance_weight_k must be >= 0")
        if not (topology_weight >= 0): raise ValueError("topology_weight must be >= 0")
        if not (boundary_weight >= 0): raise ValueError("boundary_weight must be >= 0")
        if not (centerline_weight >= 0): raise ValueError("centerline_weight must be >= 0")
        if centerline_focal_weight < 0:
            raise ValueError("centerline_focal_weight must be >= 0")
        if not (sdf_weight >= 0): raise ValueError("sdf_weight must be >= 0")
        if not (sdf_truncation > 0): raise ValueError("sdf_truncation must be > 0")
        if not all(weight >= 0 for weight in [
            pattern_center_weight, pattern_direction_weight, pattern_spacing_weight
        ]):
            raise ValueError("Pattern relation loss weights must be >= 0")
        if not (0 < pattern_spacing_min_norm < pattern_spacing_max_norm < 1):
            raise ValueError("Require 0 < pattern_spacing_min_norm < pattern_spacing_max_norm < 1")
        if contour_refine_weight < 0:
            raise ValueError("contour_refine_weight must be >= 0")
        if contour_kernel_size < 3 or contour_kernel_size % 2 == 0:
            raise ValueError("contour_kernel_size must be odd and >= 3")
        if ribbon_mask_weight < 0 or ribbon_width_weight < 0:
            raise ValueError("G1 ribbon loss weights must be >= 0")
        if ribbon_width_max <= 0:
            raise ValueError("ribbon_width_max must be positive")
        if thickness_prior_weight < 0:
            raise ValueError("thickness_prior_weight must be >= 0")
        if thickness_prior_max_half_width <= 0:
            raise ValueError("thickness_prior_max_half_width must be positive")
        if instance_aggregate_weight < 0:
            raise ValueError("instance_aggregate_weight must be >= 0")
        if trajectory_instance_weight < 0:
            raise ValueError("trajectory_instance_weight must be >= 0")
        if trajectory_unmatched_weight < 0:
            raise ValueError("trajectory_unmatched_weight must be >= 0")
        if trajectory_instance_min_area < 1:
            raise ValueError("trajectory_instance_min_area must be >= 1")
        if prototype_context_weight < 0:
            raise ValueError("prototype_context_weight must be >= 0")
        if prototype_warmup_epochs < 0 or prototype_hard_negative_start < 0:
            raise ValueError("prototype schedule epochs must be >= 0")
        if prototype_hard_negative_weight < 1.0:
            raise ValueError("prototype_hard_negative_weight must be >= 1")
        if not 0.0 < prototype_hard_negative_threshold < 1.0:
            raise ValueError("prototype_hard_negative_threshold must be in (0, 1)")
        if hard_negative_suppression_weight < 0 or hard_negative_safety_weight < 0:
            raise ValueError("hard-negative loss weights must be >= 0")
        if hard_negative_warmup_epochs < 0:
            raise ValueError("hard_negative_warmup_epochs must be >= 0")
        if not 0.0 < hard_negative_mining_threshold < 1.0:
            raise ValueError("hard_negative_mining_threshold must be in (0, 1)")
        if texture_hard_negative_weight < 0 or texture_foreground_safety_weight < 0:
            raise ValueError("texture TSCB loss weights must be >= 0")
        if texture_warmup_epochs < 0:
            raise ValueError("texture_warmup_epochs must be >= 0")
        if not 0.0 < texture_hard_negative_quantile < 1.0:
            raise ValueError("texture_hard_negative_quantile must be in (0, 1)")
        if afm_coarse_weight < 0:
            raise ValueError("afm_coarse_weight must be >= 0")
        if afm_coarse_warmup_epochs < 0:
            raise ValueError("afm_coarse_warmup_epochs must be >= 0")
        if afm_aux_weight < 0:
            raise ValueError("afm_aux_weight must be >= 0")
        if afm_aux_warmup_epochs < 0:
            raise ValueError("afm_aux_warmup_epochs must be >= 0")
        if (
            weak_stitch_weight < 0
            or weak_stitch_hard_positive_weight < 0
            or weak_stitch_hard_negative_weight < 0
            or weak_stitch_consistency_weight < 0
        ):
            raise ValueError("weak-stitch loss weights must be >= 0")
        if not 0.0 < weak_stitch_hard_negative_threshold < 1.0:
            raise ValueError("weak_stitch_hard_negative_threshold must be in (0, 1)")
        if weak_stitch_warmup_epochs < 0:
            raise ValueError("weak_stitch_warmup_epochs must be >= 0")
        if not 0.0 < weak_stitch_visibility_threshold < 1.0:
            raise ValueError("weak_stitch_visibility_threshold must be in (0, 1)")
        if not (skeleton_iterations >= 0): raise ValueError("skeleton_iterations must be >= 0")
        if boundary_kernel_size % 2 == 0 or boundary_kernel_size < 3:
            raise ValueError("boundary_kernel_size must be odd and >= 3")
        if centerline_dilation_kernel_size < 1 or centerline_dilation_kernel_size % 2 == 0:
            raise ValueError("centerline_dilation_kernel_size must be odd and >= 1")

        # --- Store Parameters ---
        self.w_edge, self.w_loc, self.w_final = w_edge, w_loc, w_final
        self.w_centerline = centerline_weight
        self.centerline_focal_weight = centerline_focal_weight
        self.focal_alpha, self.focal_gamma = focal_alpha, focal_gamma
        # Store separate Tversky params
        self.tversky_alpha_loc, self.tversky_beta_loc = tversky_alpha_loc, tversky_beta_loc
        self.tversky_alpha_final, self.tversky_beta_final = tversky_alpha_final, tversky_beta_final
        # Store final edge weight
        self.final_edge_dice_weight = final_edge_dice_weight
        # Store other params
        self.edge_focal_weight = edge_focal_weight
        self.edge_reg_weight = edge_reg_weight
        self.edge_dist_cutoff = max(1, edge_dist_cutoff) # Ensure at least 1 iter
        self.instance_weight_k = instance_weight_k
        self.instance_weight_low_thresh = instance_weight_low_thresh
        self.instance_weight_high_thresh = instance_weight_high_thresh
        self.eps = eps
        self.topology_weight = topology_weight
        self.boundary_weight = boundary_weight
        self.centerline_dilation_kernel_size = centerline_dilation_kernel_size
        self.skeleton_iterations = skeleton_iterations
        self.boundary_kernel_size = boundary_kernel_size
        self.sdf_weight = sdf_weight
        self.sdf_truncation = sdf_truncation
        self.pattern_center_weight = pattern_center_weight
        self.pattern_direction_weight = pattern_direction_weight
        self.pattern_spacing_weight = pattern_spacing_weight
        self.pattern_spacing_min_norm = pattern_spacing_min_norm
        self.pattern_spacing_max_norm = pattern_spacing_max_norm
        self.contour_refine_weight = contour_refine_weight
        self.contour_kernel_size = contour_kernel_size
        self.ribbon_mask_weight = ribbon_mask_weight
        self.ribbon_width_weight = ribbon_width_weight
        self.ribbon_width_max = ribbon_width_max
        self.thickness_prior_weight = thickness_prior_weight
        self.thickness_prior_max_half_width = thickness_prior_max_half_width
        self.instance_aggregate_weight = instance_aggregate_weight
        self.trajectory_instance_weight = trajectory_instance_weight
        self.trajectory_unmatched_weight = trajectory_unmatched_weight
        self.trajectory_instance_min_area = trajectory_instance_min_area
        self.prototype_context_weight = prototype_context_weight
        self.prototype_warmup_epochs = int(prototype_warmup_epochs)
        self.prototype_hard_negative_start = int(prototype_hard_negative_start)
        self.prototype_hard_negative_weight = prototype_hard_negative_weight
        self.prototype_hard_negative_threshold = prototype_hard_negative_threshold
        self.current_epoch = 0
        self.hard_negative_suppression_weight = hard_negative_suppression_weight
        self.hard_negative_safety_weight = hard_negative_safety_weight
        self.hard_negative_warmup_epochs = int(hard_negative_warmup_epochs)
        self.hard_negative_mining_threshold = hard_negative_mining_threshold
        self.texture_hard_negative_weight = texture_hard_negative_weight
        self.texture_foreground_safety_weight = texture_foreground_safety_weight
        self.texture_warmup_epochs = int(texture_warmup_epochs)
        self.texture_hard_negative_quantile = texture_hard_negative_quantile
        self.afm_coarse_weight = float(afm_coarse_weight)
        self.afm_coarse_warmup_epochs = int(afm_coarse_warmup_epochs)
        self.afm_aux_weight = float(afm_aux_weight)
        self.afm_aux_warmup_epochs = int(afm_aux_warmup_epochs)
        self.weak_stitch_weight = float(weak_stitch_weight)
        self.weak_stitch_hard_positive_weight = float(weak_stitch_hard_positive_weight)
        self.weak_stitch_hard_negative_weight = float(weak_stitch_hard_negative_weight)
        self.weak_stitch_hard_negative_threshold = float(
            weak_stitch_hard_negative_threshold
        )
        self.weak_stitch_consistency_weight = float(weak_stitch_consistency_weight)
        self.weak_stitch_warmup_epochs = int(weak_stitch_warmup_epochs)
        self.weak_stitch_visibility_threshold = float(weak_stitch_visibility_threshold)

        # Avg Pooling for intermediate edge loss spatial weights
        self.edge_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False)

    # --- Helper Methods ---

    def set_epoch(self, epoch: int) -> None:
        """Set epoch-dependent auxiliary schedules without global state."""
        self.current_epoch = int(epoch)

    def _compute_prototype_context_loss_single(
        self, pred_logits: torch.Tensor, gt_mask: torch.Tensor
    ) -> torch.Tensor:
        """P3 BCE with max-pooled thin-stitch targets and delayed hard negatives."""
        if pred_logits.ndim == 3:
            pred_logits = pred_logits.unsqueeze(0)
        if gt_mask.ndim == 2:
            gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
        elif gt_mask.ndim == 3:
            gt_mask = gt_mask.unsqueeze(0)
        if pred_logits.ndim != 4 or gt_mask.ndim != 4:
            raise ValueError("Prototype context loss expects CxHxW or BxCxHxW tensors")
        # A max-pooled target keeps foreground cells for sub-P3 thin stitches.
        target = F.adaptive_max_pool2d(gt_mask.float(), pred_logits.shape[-2:])
        per_pixel = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        weights = torch.ones_like(per_pixel)
        if self.current_epoch >= self.prototype_hard_negative_start:
            hard_background = (
                (target < 0.5)
                & (torch.sigmoid(pred_logits).detach() >= self.prototype_hard_negative_threshold)
            )
            weights = weights + (self.prototype_hard_negative_weight - 1.0) * hard_background.float()
        return (per_pixel * weights).sum() / (weights.sum() + self.eps)

    def _compute_hard_negative_losses_single(
        self,
        hard_negative_logits: torch.Tensor,
        safety_logits: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Learn a coarse safety map and mine its false-positive backgrounds."""
        def as_bchw(x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(0) if x.ndim == 3 else x
        hard_negative_logits = as_bchw(hard_negative_logits)
        safety_logits = as_bchw(safety_logits)
        if gt_mask.ndim == 2:
            gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
        elif gt_mask.ndim == 3:
            gt_mask = gt_mask.unsqueeze(0)
        target = F.adaptive_max_pool2d(gt_mask.float(), safety_logits.shape[-2:])
        safety_loss = focal_loss_with_logits(
            safety_logits, target, alpha=self.focal_alpha,
            gamma=self.focal_gamma, eps=self.eps,
        )
        hard_target = (
            (target < 0.5)
            & (torch.sigmoid(safety_logits).detach() >= self.hard_negative_mining_threshold)
        ).float()
        per_pixel = F.binary_cross_entropy_with_logits(
            hard_negative_logits, hard_target, reduction='none'
        )
        weights = 1.0 + 2.0 * hard_target
        suppression_loss = (per_pixel * weights).sum() / (weights.sum() + self.eps)
        return suppression_loss, safety_loss

    def _compute_texture_semantic_losses_single(
        self,
        texture_confidence: torch.Tensor,
        edge_logits: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mine only strong edge responses in GT background for TSCB.

        ``texture_confidence`` is a probability produced by the parameter-free
        repeat cue, not a segmentation logit.  The main term asks it to be high
        only on the per-image top-q background edge responses.  The small safety
        term is essential: without it the auxiliary map is unconstrained on
        foreground and could suppress weak stitches.
        """
        def as_bchw(x: torch.Tensor) -> torch.Tensor:
            return x.unsqueeze(0) if x.ndim == 3 else x

        texture_confidence = as_bchw(texture_confidence).clamp(
            min=self.eps, max=1.0 - self.eps
        )
        edge_logits = as_bchw(edge_logits)
        gt_mask = as_bchw(gt_mask).float()
        if texture_confidence.shape[-2:] != gt_mask.shape[-2:]:
            texture_confidence = F.interpolate(
                texture_confidence, size=gt_mask.shape[-2:],
                mode='bilinear', align_corners=False,
            )
        if edge_logits.shape[-2:] != gt_mask.shape[-2:]:
            edge_logits = F.interpolate(
                edge_logits, size=gt_mask.shape[-2:],
                mode='bilinear', align_corners=False,
            )

        foreground = gt_mask >= 0.5
        background = ~foreground
        detached_edge = torch.sigmoid(edge_logits).detach()
        # torch.quantile does not support AMP FP16 tensors.  Mining is a
        # non-differentiable selection operation, so FP32 is both safe and
        # preferable here; the model path and its gradients stay unchanged.
        background_scores = detached_edge[background].float()
        if background_scores.numel() == 0:
            hard_background_loss = texture_confidence.sum() * 0.0
        else:
            threshold = torch.quantile(
                background_scores, self.texture_hard_negative_quantile
            )
            hard_background = background & (detached_edge.float() >= threshold)
            selected = texture_confidence[hard_background]
            hard_background_loss = -torch.log(selected).mean()

        if foreground.any():
            foreground_safety_loss = -torch.log1p(
                -texture_confidence[foreground]
            ).mean()
        else:
            foreground_safety_loss = texture_confidence.sum() * 0.0
        return hard_background_loss, foreground_safety_loss

    def _compute_instance_weight(self, gt_mask: torch.Tensor) -> float:
        """ Calculates sample weight based on foreground ratio. """
        if self.instance_weight_k <= 0: return 1.0
        if gt_mask.dim() == 3 and gt_mask.shape[0] == 1: gt_mask_2d = gt_mask.squeeze(0)
        elif gt_mask.dim() == 2: gt_mask_2d = gt_mask
        else: logger.warning(f"Instance weight GT mask shape unexpected: {gt_mask.shape}. Using weight 1.0."); return 1.0
        num_pixels = float(gt_mask_2d.numel());
        if num_pixels == 0: return 1.0
        foreground_pixels = gt_mask_2d.sum().float()
        fg_ratio = torch.clamp(foreground_pixels / (num_pixels + self.eps), self.eps, 1.0 - self.eps)
        max_weight = 1.0 + self.instance_weight_k; weight = 1.0
        if fg_ratio <= self.instance_weight_low_thresh or fg_ratio >= self.instance_weight_high_thresh: weight = max_weight
        else:
            mid = 0.5; scale = (mid - fg_ratio) / (mid - self.instance_weight_low_thresh + self.eps) if fg_ratio < mid else (fg_ratio - mid) / (self.instance_weight_high_thresh - mid + self.eps)
            weight = 1.0 + self.instance_weight_k * scale
        return min(max(weight.item() if isinstance(weight, torch.Tensor) else weight, 1.0), max_weight)

    @torch.no_grad()
    def _compute_spatial_weights_edge(self, gt_edges: torch.Tensor) -> torch.Tensor:
        """ Approximates distance transform via Average Pooling for edge spatial weights. """
        if gt_edges.dim() == 3 and gt_edges.shape[0] == 1: gt_edges_in = gt_edges # Input is (1, H, W)
        elif gt_edges.dim() == 2: gt_edges_in = gt_edges.unsqueeze(0) # H, W -> 1, H, W
        else: logger.warning(f"Edge spatial weight GT edges shape unexpected: {gt_edges.shape}. Using as is."); gt_edges_in = gt_edges
        blurred_edges = gt_edges_in.unsqueeze(1).float() # Add C dim: (1, 1, H, W)
        for _ in range(self.edge_dist_cutoff): blurred_edges = self.edge_pool(blurred_edges)
        blurred_edges = blurred_edges.squeeze(1).clamp(0.0, 1.0) # Remove C dim -> (1, H, W)
        spatial_weight_map = 1.0 + (self.edge_focal_weight - 1.0) * blurred_edges
        return torch.clamp(spatial_weight_map, min=1.0, max=self.edge_focal_weight) # Shape (1, H, W)

    def _compute_edge_loss_single(self, pred_edge_logits: torch.Tensor, gt_edges: torch.Tensor) -> torch.Tensor:
        """ Computes Weighted Focal + TV loss for a single edge sample (inputs B=1, C=1, H, W). """
        # 1. Compute Spatial Weights
        spatial_weight_map = self._compute_spatial_weights_edge(gt_edges) # (1, H, W)

        # 2. Calculate Weighted Focal Loss (Mean)
        focal_loss_val = focal_loss_with_logits(
            inputs=pred_edge_logits, targets=gt_edges,
            alpha=self.focal_alpha, gamma=self.focal_gamma,
            weight=spatial_weight_map, eps=self.eps # Pass spatial weights
        )

        # 3. Calculate Background TV Regularization (Mean)
        tv_reg_mean = torch.tensor(0.0, device=pred_edge_logits.device, dtype=pred_edge_logits.dtype)
        if self.edge_reg_weight > 0:
            with torch.no_grad():
                # Background mask where spatial weight is close to 1
                background_mask = (spatial_weight_map <= 1.0 + self.eps).float() # (1, H, W)
            # Need to ensure pred_edge_logits is B=1,C=1,H,W for TV helper
            if pred_edge_logits.dim() == 3: pred_edge_logits_bchw = pred_edge_logits.unsqueeze(0)
            else: pred_edge_logits_bchw = pred_edge_logits # Assume already BCHW
            pred_edge_probs = torch.sigmoid(pred_edge_logits_bchw) # (1, 1, H, W)
            # Apply background mask (needs compatible shape)
            tv_input = pred_edge_probs * background_mask.unsqueeze(1) # Add C dim to mask
            tv_reg_mean = total_variation_loss(tv_input, eps=self.eps)

        loss = focal_loss_val + self.edge_reg_weight * tv_reg_mean
        return loss

    # Modified to accept specific alpha/beta for Tversky
    def _compute_segmentation_loss_single(
        self,
        pred_logits: torch.Tensor,
        gt_mask: torch.Tensor,
        tversky_alpha: float, # Pass alpha
        tversky_beta: float   # Pass beta
        ) -> torch.Tensor:
        """ Computes Focal + Tversky loss for a single sample with specific alpha/beta. """
        # Assume inputs are B=1, C=1, H, W
        # 1. Calculate Focal Loss (Mean)
        focal = focal_loss_with_logits(
            inputs=pred_logits, targets=gt_mask,
            alpha=self.focal_alpha, gamma=self.focal_gamma,
            weight=None, eps=self.eps # No spatial weight here
        )
        # 2. Calculate Tversky Loss with passed alpha/beta
        probs = torch.sigmoid(pred_logits)
        tversky = tversky_loss(
            inputs=probs, targets=gt_mask,
            alpha=tversky_alpha, beta=tversky_beta, # Use passed params
            smooth=self.eps
        )
        # 3. Combine Focal and Tversky losses
        loss = focal + tversky
        return loss

    def _compute_afm_coarse_loss_single(
        self, afm_logits: torch.Tensor, gt_mask: torch.Tensor
    ) -> torch.Tensor:
        """Supervise AFM as a coarse *presence* prior at its native P2 grid.

        Max pooling retains a positive target whenever a thin stitch crosses a
        coarse cell.  This is intentionally an auxiliary prior, not a second
        full-resolution final mask, so its loss remains low-weighted.
        """
        if afm_logits.ndim == 3:
            afm_logits = afm_logits.unsqueeze(0)
        if gt_mask.ndim == 2:
            gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
        elif gt_mask.ndim == 3:
            gt_mask = gt_mask.unsqueeze(0)
        if afm_logits.ndim != 4 or gt_mask.ndim != 4:
            raise ValueError("AFM coarse loss expects CxHxW or BxCxHxW tensors")
        target = F.adaptive_max_pool2d(gt_mask.float(), afm_logits.shape[-2:])
        bce = F.binary_cross_entropy_with_logits(afm_logits, target)
        overlap = tversky_loss(
            torch.sigmoid(afm_logits), target,
            alpha=0.5, beta=0.5, smooth=self.eps,
        )
        return bce + overlap

    def forward(
        self,
        predictions: Dict[str, List[torch.Tensor]],  # Dict: stage -> List[Tensor(C=1, H, W)]
        gt_masks: List[torch.Tensor],                # List[Tensor(C=1, H, W)]
        gt_edges: List[torch.Tensor],                # List[Tensor(C=1, H, W)]
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the final combined C3Net loss across a batch using list inputs.
        Includes weighted edge Dice loss for Final loss.
        """
        final_preds_list = predictions['final_mask']
        loc_preds_list = predictions['object_mask']
        edge_preds_list = predictions['edge_mask']
        centerline_preds_list = predictions.get('centerline_mask')
        sdf_preds_list = predictions.get('sdf_mask')
        pattern_center_preds_list = predictions.get('pattern_center')
        pattern_direction_preds_list = predictions.get('pattern_direction')
        pattern_spacing_preds_list = predictions.get('pattern_spacing')
        contour_preds_list = predictions.get('contour_logits')
        ribbon_preds_list = predictions.get('ribbon_mask')
        ribbon_width_preds_list = predictions.get('ribbon_width')
        thickness_prior_preds_list = predictions.get('thickness_width_prior')
        instance_aggregate_preds_list = predictions.get('instance_aggregate_mask')
        instance_logits_preds_list = predictions.get('instance_logits')
        prototype_context_preds_list = predictions.get('prototype_context_mask')
        hard_negative_preds_list = predictions.get('hard_negative_mask')
        hard_negative_safety_preds_list = predictions.get('hard_negative_safety_mask')
        texture_confidence_preds_list = predictions.get('texture_confidence')
        afm_coarse_preds_list = predictions.get('afm_coarse_mask')
        afm_aux_preds_list = predictions.get('afm_aux_mask')
        weak_stitch_preds_list = predictions.get('weak_stitch_mask')
        weak_stitch_consistency_preds_list = predictions.get(
            'weak_stitch_consistency_mask'
        )

        batch_size = len(gt_masks)
        if not (
            len(final_preds_list) == batch_size and
            len(loc_preds_list) == batch_size and
            len(edge_preds_list) == batch_size and
            len(gt_edges) == batch_size
        ):
            raise ValueError(f"Input lists must have same length (batch size). Mismatch found.")
        if self.w_centerline > 0:
            if centerline_preds_list is None:
                raise ValueError(
                    "centerline_weight > 0 but predictions['centerline_mask'] is missing."
                )
            if len(centerline_preds_list) != batch_size:
                raise ValueError("centerline_mask list length must match batch size.")
        if self.sdf_weight > 0:
            if sdf_preds_list is None:
                raise ValueError(
                    "sdf_weight > 0 but predictions['sdf_mask'] is missing. "
                    "Enable decoder.mobile_fpn.sdf_auxiliary."
                )
            if len(sdf_preds_list) != batch_size:
                raise ValueError("sdf_mask list length must match batch size.")
        if self.thickness_prior_weight > 0:
            if thickness_prior_preds_list is None:
                raise ValueError(
                    "thickness_prior_weight > 0 but predictions['thickness_width_prior'] "
                    "is missing. Enable decoder.mobile_fpn.thickness_router."
                )
            if len(thickness_prior_preds_list) != batch_size:
                raise ValueError("thickness_width_prior list length must match batch size.")
        pattern_enabled = any(weight > 0 for weight in [
            self.pattern_center_weight,
            self.pattern_direction_weight,
            self.pattern_spacing_weight,
        ])
        if pattern_enabled:
            if any(predictions_list is None for predictions_list in [
                pattern_center_preds_list,
                pattern_direction_preds_list,
                pattern_spacing_preds_list,
            ]):
                raise ValueError(
                    "Pattern relation loss is enabled but a pattern prediction is missing. "
                    "Enable decoder.mobile_fpn.stitch_pattern_relation."
                )
            if not all(len(predictions_list) == batch_size for predictions_list in [
                pattern_center_preds_list,
                pattern_direction_preds_list,
                pattern_spacing_preds_list,
            ]):
                raise ValueError("Pattern prediction list lengths must match batch size.")
        if self.contour_refine_weight > 0:
            if contour_preds_list is None:
                raise ValueError(
                    "contour_refine_weight > 0 but predictions['contour_logits'] is missing. "
                    "Enable decoder.mobile_fpn.contour_refine."
                )
            if len(contour_preds_list) != batch_size:
                raise ValueError("contour_logits list length must match batch size.")
        if self.ribbon_mask_weight > 0:
            if ribbon_preds_list is None:
                raise ValueError(
                    "ribbon_mask_weight > 0 but predictions['ribbon_mask'] is missing. "
                    "Enable decoder.mobile_fpn.high_res_refine.geometry_ribbon."
                )
            if len(ribbon_preds_list) != batch_size:
                raise ValueError("ribbon_mask list length must match batch size.")
        if self.ribbon_width_weight > 0:
            if ribbon_width_preds_list is None:
                raise ValueError(
                    "ribbon_width_weight > 0 but predictions['ribbon_width'] is missing. "
                    "Enable decoder.mobile_fpn.high_res_refine.geometry_ribbon."
                )
            if len(ribbon_width_preds_list) != batch_size:
                raise ValueError("ribbon_width list length must match batch size.")
        if self.instance_aggregate_weight > 0:
            if instance_aggregate_preds_list is None:
                raise ValueError(
                    "instance_aggregate_weight > 0 but "
                    "predictions['instance_aggregate_mask'] is missing. Enable "
                    "decoder.mobile_fpn.instance_conditioned_highres."
                )
            if len(instance_aggregate_preds_list) != batch_size:
                raise ValueError(
                    "instance_aggregate_mask list length must match batch size."
                )
        if self.trajectory_instance_weight > 0:
            if instance_logits_preds_list is None:
                raise ValueError(
                    "trajectory_instance_weight > 0 but "
                    "predictions['instance_logits'] is missing. Enable "
                    "decoder.mobile_fpn.instance_conditioned_highres."
                )
            if len(instance_logits_preds_list) != batch_size:
                raise ValueError("instance_logits list length must match batch size.")
        if self.prototype_context_weight > 0:
            if prototype_context_preds_list is None:
                raise ValueError(
                    "prototype_context_weight > 0 but predictions['prototype_context_mask'] "
                    "is missing. Enable decoder.mobile_fpn.prototype_context."
                )
            if len(prototype_context_preds_list) != batch_size:
                raise ValueError("prototype_context_mask list length must match batch size.")
        hard_negative_enabled = (
            self.hard_negative_suppression_weight > 0
            or self.hard_negative_safety_weight > 0
        )
        if hard_negative_enabled:
            if hard_negative_preds_list is None or hard_negative_safety_preds_list is None:
                raise ValueError(
                    "Hard-negative losses require hard_negative_mask and "
                    "hard_negative_safety_mask. Enable hard_negative_suppression."
                )
            if (
                len(hard_negative_preds_list) != batch_size
                or len(hard_negative_safety_preds_list) != batch_size
            ):
                raise ValueError("Hard-negative prediction list lengths must match batch size.")
        texture_semantic_enabled = (
            self.texture_hard_negative_weight > 0
            or self.texture_foreground_safety_weight > 0
        )
        if texture_semantic_enabled:
            if texture_confidence_preds_list is None:
                raise ValueError(
                    "Texture losses require predictions['texture_confidence']. "
                    "Enable decoder.mobile_fpn.texture_semantic_contradiction."
                )
            if len(texture_confidence_preds_list) != batch_size:
                raise ValueError("texture_confidence list length must match batch size.")
        if self.afm_coarse_weight > 0:
            if afm_coarse_preds_list is None:
                raise ValueError(
                    "afm_coarse_weight > 0 but predictions['afm_coarse_mask'] is missing. "
                    "Enable decoder.mobile_fpn.afm_coarse_supervision."
                )
            if len(afm_coarse_preds_list) != batch_size:
                raise ValueError("afm_coarse_mask list length must match batch size.")
        if self.afm_aux_weight > 0:
            if afm_aux_preds_list is None:
                raise ValueError(
                    "afm_aux_weight > 0 but predictions['afm_aux_mask'] is missing. "
                    "Enable decoder.mobile_fpn.afm_aux_supervision."
                )
            if len(afm_aux_preds_list) != batch_size:
                raise ValueError("afm_aux_mask list length must match batch size.")
        if self.weak_stitch_weight > 0:
            if weak_stitch_preds_list is None:
                raise ValueError(
                    "weak_stitch_weight > 0 but predictions['weak_stitch_mask'] is "
                    "missing. Enable decoder.mobile_fpn.weak_stitch_recovery."
                )
            if len(weak_stitch_preds_list) != batch_size:
                raise ValueError("weak_stitch_mask list length must match batch size.")
        if self.weak_stitch_consistency_weight > 0:
            if weak_stitch_consistency_preds_list is None:
                raise ValueError(
                    "weak_stitch_consistency_weight > 0 but "
                    "predictions['weak_stitch_consistency_mask'] is missing. "
                    "Enable weak_stitch_recovery.bidirectional_consistency."
                )
            if len(weak_stitch_consistency_preds_list) != batch_size:
                raise ValueError(
                    "weak_stitch_consistency_mask list length must match batch size."
                )

        # --- Loss Accumulation Initialization ---
        output_device = final_preds_list[0].device if batch_size > 0 else torch.device('cpu')
        zero_loss = torch.tensor(0.0, device=output_device, dtype=torch.float32)
        accumulated_losses = {
            'loc': [],
            'final': [],
            'edge': [],
            'centerline': [],
            'topology': [],
            'boundary': [],
            'sdf': [],
            'pattern_center': [],
            'pattern_direction': [],
            'pattern_spacing': [],
            'contour': [],
            'ribbon_mask': [],
            'ribbon_width': [],
            'thickness_prior': [],
            'instance_aggregate': [],
            'trajectory_instance': [],
            'prototype_context': [],
            'hard_negative_suppression': [],
            'hard_negative_safety': [],
            'texture_hard_negative': [],
            'texture_foreground_safety': [],
            'afm_coarse': [],
            'afm_aux': [],
            'weak_stitch': [],
            'weak_stitch_consistency': [],
        }
        instance_weights_list = []

        # --- Per-Sample Loss Calculation Loop ---
        for i in range(batch_size):
            # Get tensors for sample i, move to device, ensure float
            final_pred_i = final_preds_list[i]
            loc_pred_i = loc_preds_list[i]
            edge_pred_i = edge_preds_list[i]
            centerline_pred_i = (
                centerline_preds_list[i] if centerline_preds_list is not None else None
            )
            sdf_pred_i = sdf_preds_list[i] if sdf_preds_list is not None else None
            pattern_center_pred_i = (
                pattern_center_preds_list[i] if pattern_center_preds_list is not None else None
            )
            pattern_direction_pred_i = (
                pattern_direction_preds_list[i] if pattern_direction_preds_list is not None else None
            )
            pattern_spacing_pred_i = (
                pattern_spacing_preds_list[i] if pattern_spacing_preds_list is not None else None
            )
            contour_pred_i = contour_preds_list[i] if contour_preds_list is not None else None
            ribbon_pred_i = ribbon_preds_list[i] if ribbon_preds_list is not None else None
            ribbon_width_pred_i = (
                ribbon_width_preds_list[i] if ribbon_width_preds_list is not None else None
            )
            thickness_prior_pred_i = (
                thickness_prior_preds_list[i]
                if thickness_prior_preds_list is not None else None
            )
            instance_aggregate_pred_i = (
                instance_aggregate_preds_list[i]
                if instance_aggregate_preds_list is not None else None
            )
            instance_logits_pred_i = (
                instance_logits_preds_list[i]
                if instance_logits_preds_list is not None else None
            )
            prototype_context_pred_i = (
                prototype_context_preds_list[i]
                if prototype_context_preds_list is not None else None
            )
            hard_negative_pred_i = (
                hard_negative_preds_list[i] if hard_negative_preds_list is not None else None
            )
            hard_negative_safety_pred_i = (
                hard_negative_safety_preds_list[i]
                if hard_negative_safety_preds_list is not None else None
            )
            texture_confidence_pred_i = (
                texture_confidence_preds_list[i]
                if texture_confidence_preds_list is not None else None
            )
            afm_coarse_pred_i = (
                afm_coarse_preds_list[i]
                if afm_coarse_preds_list is not None else None
            )
            afm_aux_pred_i = (
                afm_aux_preds_list[i]
                if afm_aux_preds_list is not None else None
            )
            weak_stitch_pred_i = (
                weak_stitch_preds_list[i]
                if weak_stitch_preds_list is not None else None
            )
            weak_stitch_consistency_pred_i = (
                weak_stitch_consistency_preds_list[i]
                if weak_stitch_consistency_preds_list is not None else None
            )
            gt_mask_i = gt_masks[i]
            gt_edge_i = gt_edges[i]

            # --- Calculate Instance Weight ---
            instance_weight = self._compute_instance_weight(gt_mask_i)
            instance_weights_list.append(instance_weight)

            # --- Intermediate Edge Loss ---
            loss_edge_i = zero_loss.clone()
            if self.w_edge > 0:
                loss_edge_i = self._compute_edge_loss_single(edge_pred_i, gt_edge_i)
            accumulated_losses['edge'].append(loss_edge_i)

            # --- Auxiliary Centerline Loss ---
            loss_centerline_i = zero_loss.clone()
            if self.w_centerline > 0 and centerline_pred_i is not None:
                loss_centerline_i = centerline_loss_with_logits(
                    centerline_pred_i,
                    gt_mask_i,
                    iterations=self.skeleton_iterations,
                    dilation_kernel_size=self.centerline_dilation_kernel_size,
                    focal_alpha=self.focal_alpha,
                    focal_gamma=self.focal_gamma,
                    focal_weight=self.centerline_focal_weight,
                    smooth=self.eps,
                )
            accumulated_losses['centerline'].append(loss_centerline_i)

            # --- Intermediate Localization Loss (Precision Focus) ---
            loss_loc_i = self._compute_segmentation_loss_single(
                loc_pred_i, gt_mask_i,
                tversky_alpha=self.tversky_alpha_loc,
                tversky_beta=self.tversky_beta_loc
            )
            accumulated_losses['loc'].append(loss_loc_i)

            # --- AFM Coarse-Prior Supervision ---
            # Kept at native P2 resolution with a max-pooled target so it
            # learns a usable stitch-presence prior for HighRes.
            loss_afm_coarse_i = zero_loss.clone()
            if self.afm_coarse_weight > 0 and afm_coarse_pred_i is not None:
                loss_afm_coarse_i = self._compute_afm_coarse_loss_single(
                    afm_coarse_pred_i, gt_mask_i
                )
            accumulated_losses['afm_coarse'].append(loss_afm_coarse_i)

            # --- M0-style AFM Auxiliary Segmentation Loss ---
            # The trainer has already bilinearly resized this P2 prediction to
            # the GT shape.  This keeps AFM spatially selective rather than
            # changing its task into broad coarse-cell presence prediction.
            loss_afm_aux_i = zero_loss.clone()
            if self.afm_aux_weight > 0 and afm_aux_pred_i is not None:
                loss_afm_aux_i = self._compute_segmentation_loss_single(
                    afm_aux_pred_i,
                    gt_mask_i,
                    tversky_alpha=self.tversky_alpha_loc,
                    tversky_beta=self.tversky_beta_loc,
                )
            accumulated_losses['afm_aux'].append(loss_afm_aux_i)

            # --- Weak-stitch evidence supervision ---
            # The candidate head is supervised on the complete GT so it keeps
            # background discrimination.  After warm-up, the selected loss is
            # symmetric: missed GT foreground is recovered while high-score
            # GT background is explicitly rejected.
            loss_weak_stitch_i = zero_loss.clone()
            weak_positive = None
            hard_negative = None
            if self.weak_stitch_weight > 0 and weak_stitch_pred_i is not None:
                full_candidate_loss = self._compute_segmentation_loss_single(
                    weak_stitch_pred_i,
                    gt_mask_i,
                    tversky_alpha=self.tversky_alpha_loc,
                    tversky_beta=self.tversky_beta_loc,
                )
                loc_prob_detached = torch.sigmoid(loc_pred_i.detach())
                weak_positive = (
                    (gt_mask_i > 0.5)
                    & (loc_prob_detached < self.weak_stitch_visibility_threshold)
                ).to(dtype=weak_stitch_pred_i.dtype)
                weak_positive_count = weak_positive.sum()
                if weak_positive_count.item() > 0:
                    hard_positive_loss = -(
                        weak_positive * F.logsigmoid(weak_stitch_pred_i)
                    ).sum() / (weak_positive_count + self.eps)
                else:
                    hard_positive_loss = zero_loss.clone()
                candidate_prob_detached = torch.sigmoid(weak_stitch_pred_i.detach())
                hard_negative = (
                    (gt_mask_i <= 0.5)
                    & (candidate_prob_detached > self.weak_stitch_hard_negative_threshold)
                ).to(dtype=weak_stitch_pred_i.dtype)
                hard_negative_count = hard_negative.sum()
                if hard_negative_count.item() > 0:
                    hard_negative_loss = -(
                        hard_negative * F.logsigmoid(-weak_stitch_pred_i)
                    ).sum() / (hard_negative_count + self.eps)
                else:
                    hard_negative_loss = zero_loss.clone()
                loss_weak_stitch_i = (
                    full_candidate_loss
                    + self.weak_stitch_hard_positive_weight * hard_positive_loss
                    + self.weak_stitch_hard_negative_weight * hard_negative_loss
                )
            accumulated_losses['weak_stitch'].append(loss_weak_stitch_i)

            # --- Bidirectional semantic-detail consistency ---
            # Only weak positives and high-confidence candidate false
            # positives supervise this map.  It therefore validates recovery
            # evidence without turning every ordinary background pixel into a
            # texture-suppression target.
            loss_weak_stitch_consistency_i = zero_loss.clone()
            if (
                self.weak_stitch_consistency_weight > 0
                and weak_stitch_consistency_pred_i is not None
            ):
                if weak_positive is None or hard_negative is None:
                    raise RuntimeError(
                        "Weak-stitch consistency requires the weak-stitch candidate "
                        "loss to be enabled."
                    )
                selection = (weak_positive + hard_negative).clamp(max=1.0)
                selection_count = selection.sum()
                if selection_count.item() > 0:
                    consistency_target = weak_positive.to(
                        dtype=weak_stitch_consistency_pred_i.dtype
                    )
                    consistency_bce = F.binary_cross_entropy_with_logits(
                        weak_stitch_consistency_pred_i,
                        consistency_target,
                        reduction='none',
                    )
                    loss_weak_stitch_consistency_i = (
                        consistency_bce * selection.to(dtype=consistency_bce.dtype)
                    ).sum() / (selection_count + self.eps)
            accumulated_losses['weak_stitch_consistency'].append(
                loss_weak_stitch_consistency_i
            )

            # --- Final Segmentation Loss (Recall Focus + Edge Dice) ---
            # 1. Region Loss Component
            loss_final_region_i = self._compute_segmentation_loss_single(
                final_pred_i, gt_mask_i,
                tversky_alpha=self.tversky_alpha_final,
                tversky_beta=self.tversky_beta_final
            )
            # 2. Edge Dice Loss Component
            loss_final_edge_i = zero_loss.clone()
            if self.final_edge_dice_weight > 0:
                loss_final_edge_i = dice_loss_on_edges(
                    torch.sigmoid(final_pred_i), gt_mask_i, gt_edge_i, smooth=self.eps
                )
            loss_topology_i = zero_loss.clone()
            if self.topology_weight > 0:
                loss_topology_i = soft_cldice_loss(
                    final_pred_i,
                    gt_mask_i,
                    iterations=self.skeleton_iterations,
                    smooth=self.eps,
                )
            loss_boundary_i = zero_loss.clone()
            if self.boundary_weight > 0:
                loss_boundary_i = boundary_dice_loss(
                    final_pred_i,
                    gt_edge_i,
                    kernel_size=self.boundary_kernel_size,
                    smooth=self.eps,
                )
            # 3. Combine final loss components
            loss_final_i = (
                loss_final_region_i +
                self.final_edge_dice_weight * loss_final_edge_i +
                self.topology_weight * loss_topology_i +
                self.boundary_weight * loss_boundary_i
            )
            accumulated_losses['final'].append(loss_final_i)
            accumulated_losses['topology'].append(loss_topology_i)
            accumulated_losses['boundary'].append(loss_boundary_i)

            # Truncated signed-distance supervision is a decoder-only
            # auxiliary.  It uses neither centerlines nor topology/boundary-Dice.
            loss_sdf_i = zero_loss.clone()
            if self.sdf_weight > 0 and sdf_pred_i is not None:
                sdf_target_i = truncated_signed_distance_map(
                    gt_mask_i, truncation=self.sdf_truncation
                )
                loss_sdf_i = F.smooth_l1_loss(
                    torch.tanh(sdf_pred_i), sdf_target_i, reduction='mean'
                )
            accumulated_losses['sdf'].append(loss_sdf_i)

            # G1 supervises the rendered geometric ribbon directly, so the
            # centerline/width representation learns before it is mixed into
            # the base HighRes logits.
            loss_ribbon_mask_i = zero_loss.clone()
            if self.ribbon_mask_weight > 0 and ribbon_pred_i is not None:
                loss_ribbon_mask_i = self._compute_segmentation_loss_single(
                    ribbon_pred_i,
                    gt_mask_i,
                    tversky_alpha=self.tversky_alpha_final,
                    tversky_beta=self.tversky_beta_final,
                )
            accumulated_losses['ribbon_mask'].append(loss_ribbon_mask_i)

            loss_ribbon_width_i = zero_loss.clone()
            if self.ribbon_width_weight > 0 and ribbon_width_pred_i is not None:
                loss_ribbon_width_i = ribbon_width_loss(
                    ribbon_width_pred_i,
                    gt_mask_i,
                    max_half_width=self.ribbon_width_max,
                    skeleton_iterations=self.skeleton_iterations,
                    # Width is defined at the medial axis itself; unlike the
                    # tolerant centerline BCE target it should not be dilated.
                    centerline_dilation_kernel_size=1,
                    eps=self.eps,
                )
            accumulated_losses['ribbon_width'].append(loss_ribbon_width_i)

            loss_thickness_prior_i = zero_loss.clone()
            if self.thickness_prior_weight > 0 and thickness_prior_pred_i is not None:
                loss_thickness_prior_i = sample_thickness_prior_loss(
                    thickness_prior_pred_i,
                    gt_mask_i,
                    max_half_width=self.thickness_prior_max_half_width,
                    skeleton_iterations=self.skeleton_iterations,
                    centerline_dilation_kernel_size=self.centerline_dilation_kernel_size,
                    eps=self.eps,
                )
            accumulated_losses['thickness_prior'].append(loss_thickness_prior_i)

            loss_instance_aggregate_i = zero_loss.clone()
            if (
                self.instance_aggregate_weight > 0
                and instance_aggregate_pred_i is not None
            ):
                loss_instance_aggregate_i = self._compute_segmentation_loss_single(
                    instance_aggregate_pred_i,
                    gt_mask_i,
                    tversky_alpha=self.tversky_alpha_final,
                    tversky_beta=self.tversky_beta_final,
                )
            accumulated_losses['instance_aggregate'].append(
                loss_instance_aggregate_i
            )

            loss_trajectory_instance_i = zero_loss.clone()
            if (
                self.trajectory_instance_weight > 0
                and instance_logits_pred_i is not None
            ):
                loss_trajectory_instance_i = trajectory_instance_hungarian_loss(
                    instance_logits_pred_i,
                    gt_mask_i,
                    unmatched_weight=self.trajectory_unmatched_weight,
                    min_area=self.trajectory_instance_min_area,
                    eps=self.eps,
                )
            accumulated_losses['trajectory_instance'].append(
                loss_trajectory_instance_i
            )

            loss_prototype_context_i = zero_loss.clone()
            if (
                self.prototype_context_weight > 0
                and self.current_epoch >= self.prototype_warmup_epochs
                and prototype_context_pred_i is not None
            ):
                loss_prototype_context_i = self._compute_prototype_context_loss_single(
                    prototype_context_pred_i, gt_mask_i
                )
            accumulated_losses['prototype_context'].append(loss_prototype_context_i)

            loss_hard_negative_i = zero_loss.clone()
            loss_hard_negative_safety_i = zero_loss.clone()
            if hard_negative_enabled and hard_negative_pred_i is not None:
                (
                    loss_hard_negative_i,
                    loss_hard_negative_safety_i,
                ) = self._compute_hard_negative_losses_single(
                    hard_negative_pred_i, hard_negative_safety_pred_i, gt_mask_i
                )
                if self.current_epoch < self.hard_negative_warmup_epochs:
                    loss_hard_negative_i = zero_loss.clone()
            accumulated_losses['hard_negative_suppression'].append(loss_hard_negative_i)
            accumulated_losses['hard_negative_safety'].append(loss_hard_negative_safety_i)

            loss_texture_hard_negative_i = zero_loss.clone()
            loss_texture_foreground_safety_i = zero_loss.clone()
            if texture_semantic_enabled and texture_confidence_pred_i is not None:
                (
                    loss_texture_hard_negative_i,
                    loss_texture_foreground_safety_i,
                ) = self._compute_texture_semantic_losses_single(
                    texture_confidence_pred_i, edge_pred_i, gt_mask_i
                )
                if self.current_epoch < self.texture_warmup_epochs:
                    loss_texture_hard_negative_i = zero_loss.clone()
            accumulated_losses['texture_hard_negative'].append(loss_texture_hard_negative_i)
            accumulated_losses['texture_foreground_safety'].append(
                loss_texture_foreground_safety_i
            )

            loss_pattern_center_i = zero_loss.clone()
            loss_pattern_direction_i = zero_loss.clone()
            loss_pattern_spacing_i = zero_loss.clone()
            if pattern_enabled:
                targets = stitch_pattern_targets(gt_mask_i)
                if self.pattern_center_weight > 0:
                    loss_pattern_center_i = focal_loss_with_logits(
                        pattern_center_pred_i,
                        targets['center'],
                        alpha=0.75,
                        gamma=2.0,
                        eps=self.eps,
                    )
                if self.pattern_direction_weight > 0:
                    valid = targets['direction_valid']
                    loss_pattern_direction_i = F.smooth_l1_loss(
                        torch.tanh(pattern_direction_pred_i) * valid,
                        targets['direction'] * valid,
                        reduction='sum',
                    ) / (2.0 * valid.sum() + self.eps)
                if self.pattern_spacing_weight > 0:
                    valid = targets['spacing_valid']
                    spacing_norm = self.pattern_spacing_min_norm + (
                        self.pattern_spacing_max_norm - self.pattern_spacing_min_norm
                    ) * torch.sigmoid(pattern_spacing_pred_i)
                    loss_pattern_spacing_i = F.smooth_l1_loss(
                        spacing_norm * valid,
                        targets['spacing'] * valid,
                        reduction='sum',
                    ) / (valid.sum() + self.eps)
            accumulated_losses['pattern_center'].append(loss_pattern_center_i)
            accumulated_losses['pattern_direction'].append(loss_pattern_direction_i)
            accumulated_losses['pattern_spacing'].append(loss_pattern_spacing_i)

            loss_contour_i = zero_loss.clone()
            if self.contour_refine_weight > 0 and contour_pred_i is not None:
                contour_target_i = contour_core_target(
                    gt_mask_i, kernel_size=self.contour_kernel_size
                )
                class_weights = torch.tensor(
                    [0.10, 1.50, 1.00], device=contour_pred_i.device,
                    dtype=contour_pred_i.dtype,
                )
                loss_contour_i = F.cross_entropy(
                    contour_pred_i.unsqueeze(0), contour_target_i.unsqueeze(0),
                    weight=class_weights,
                )
            accumulated_losses['contour'].append(loss_contour_i)

        # --- Average Losses Across Batch ---
        instance_weights_tensor = torch.tensor(instance_weights_list, device=output_device).float()
        total_instance_weight_accum = instance_weights_tensor.sum()

        # Calculate weighted average for loc and final loss
        avg_loss_loc = (instance_weights_tensor * torch.stack(accumulated_losses['loc'])).sum() / (total_instance_weight_accum + self.eps)
        avg_loss_final = (instance_weights_tensor * torch.stack(accumulated_losses['final'])).sum() / (total_instance_weight_accum + self.eps)
        # Simple average for edge loss
        avg_loss_edge = torch.stack(accumulated_losses['edge']).mean() if self.w_edge > 0 else zero_loss.clone()
        avg_loss_centerline = torch.stack(accumulated_losses['centerline']).mean() if self.w_centerline > 0 else zero_loss.clone()
        avg_loss_topology = torch.stack(accumulated_losses['topology']).mean() if self.topology_weight > 0 else zero_loss.clone()
        avg_loss_boundary = torch.stack(accumulated_losses['boundary']).mean() if self.boundary_weight > 0 else zero_loss.clone()
        avg_loss_sdf = torch.stack(accumulated_losses['sdf']).mean() if self.sdf_weight > 0 else zero_loss.clone()
        avg_loss_pattern_center = torch.stack(accumulated_losses['pattern_center']).mean() if self.pattern_center_weight > 0 else zero_loss.clone()
        avg_loss_pattern_direction = torch.stack(accumulated_losses['pattern_direction']).mean() if self.pattern_direction_weight > 0 else zero_loss.clone()
        avg_loss_pattern_spacing = torch.stack(accumulated_losses['pattern_spacing']).mean() if self.pattern_spacing_weight > 0 else zero_loss.clone()
        avg_loss_contour = torch.stack(accumulated_losses['contour']).mean() if self.contour_refine_weight > 0 else zero_loss.clone()
        avg_loss_ribbon_mask = torch.stack(accumulated_losses['ribbon_mask']).mean() if self.ribbon_mask_weight > 0 else zero_loss.clone()
        avg_loss_ribbon_width = torch.stack(accumulated_losses['ribbon_width']).mean() if self.ribbon_width_weight > 0 else zero_loss.clone()
        avg_loss_thickness_prior = torch.stack(accumulated_losses['thickness_prior']).mean() if self.thickness_prior_weight > 0 else zero_loss.clone()
        avg_loss_instance_aggregate = torch.stack(accumulated_losses['instance_aggregate']).mean() if self.instance_aggregate_weight > 0 else zero_loss.clone()
        avg_loss_trajectory_instance = torch.stack(accumulated_losses['trajectory_instance']).mean() if self.trajectory_instance_weight > 0 else zero_loss.clone()
        avg_loss_prototype_context = torch.stack(accumulated_losses['prototype_context']).mean() if self.prototype_context_weight > 0 else zero_loss.clone()
        avg_loss_hard_negative = torch.stack(accumulated_losses['hard_negative_suppression']).mean() if hard_negative_enabled else zero_loss.clone()
        avg_loss_hard_negative_safety = torch.stack(accumulated_losses['hard_negative_safety']).mean() if hard_negative_enabled else zero_loss.clone()
        avg_loss_texture_hard_negative = torch.stack(
            accumulated_losses['texture_hard_negative']
        ).mean() if texture_semantic_enabled else zero_loss.clone()
        avg_loss_texture_foreground_safety = torch.stack(
            accumulated_losses['texture_foreground_safety']
        ).mean() if texture_semantic_enabled else zero_loss.clone()
        avg_loss_afm_coarse = torch.stack(
            accumulated_losses['afm_coarse']
        ).mean() if self.afm_coarse_weight > 0 else zero_loss.clone()
        avg_loss_afm_aux = torch.stack(
            accumulated_losses['afm_aux']
        ).mean() if self.afm_aux_weight > 0 else zero_loss.clone()
        avg_loss_weak_stitch = torch.stack(
            accumulated_losses['weak_stitch']
        ).mean() if self.weak_stitch_weight > 0 else zero_loss.clone()
        avg_loss_weak_stitch_consistency = torch.stack(
            accumulated_losses['weak_stitch_consistency']
        ).mean() if self.weak_stitch_consistency_weight > 0 else zero_loss.clone()
        if self.afm_coarse_warmup_epochs > 0:
            afm_coarse_scale = min(
                1.0,
                float(self.current_epoch + 1) / float(self.afm_coarse_warmup_epochs),
            )
        else:
            afm_coarse_scale = 1.0
        effective_afm_coarse_weight = self.afm_coarse_weight * afm_coarse_scale
        if self.afm_aux_warmup_epochs > 0:
            afm_aux_scale = min(
                1.0,
                float(self.current_epoch + 1) / float(self.afm_aux_warmup_epochs),
            )
        else:
            afm_aux_scale = 1.0
        effective_afm_aux_weight = self.afm_aux_weight * afm_aux_scale
        # Unlike ordinary auxiliary losses, weak-pixel mining must not start
        # before the localization head has a meaningful miss map.  Keep it
        # strictly off during the configured warm-up, then enable it fully.
        weak_stitch_scale = (
            0.0
            if self.current_epoch < self.weak_stitch_warmup_epochs
            else 1.0
        )
        effective_weak_stitch_weight = self.weak_stitch_weight * weak_stitch_scale

        # --- Combine Detection Losses with Overall Branch Weights ---
        detection_loss = (self.w_final * avg_loss_final +
                          self.w_loc * avg_loss_loc +
                          self.w_edge * avg_loss_edge +
                          self.w_centerline * avg_loss_centerline)
        # --- Total Loss ---
        total_loss = (
            detection_loss +
            self.sdf_weight * avg_loss_sdf +
            self.pattern_center_weight * avg_loss_pattern_center +
            self.pattern_direction_weight * avg_loss_pattern_direction +
            self.pattern_spacing_weight * avg_loss_pattern_spacing +
            self.contour_refine_weight * avg_loss_contour +
            self.ribbon_mask_weight * avg_loss_ribbon_mask +
            self.ribbon_width_weight * avg_loss_ribbon_width +
            self.thickness_prior_weight * avg_loss_thickness_prior +
            self.instance_aggregate_weight * avg_loss_instance_aggregate +
            self.trajectory_instance_weight * avg_loss_trajectory_instance +
            self.prototype_context_weight * avg_loss_prototype_context +
            self.hard_negative_suppression_weight * avg_loss_hard_negative +
            self.hard_negative_safety_weight * avg_loss_hard_negative_safety +
            self.texture_hard_negative_weight * avg_loss_texture_hard_negative +
            self.texture_foreground_safety_weight * avg_loss_texture_foreground_safety +
            effective_afm_coarse_weight * avg_loss_afm_coarse +
            effective_afm_aux_weight * avg_loss_afm_aux +
            effective_weak_stitch_weight * avg_loss_weak_stitch +
            effective_weak_stitch_weight
            * self.weak_stitch_consistency_weight
            * avg_loss_weak_stitch_consistency
        )

        # --- Return Dictionary of Losses ---
        loss_dict = {
            'loss': total_loss,
            'loss_final': avg_loss_final.detach(),
            'loss_loc': avg_loss_loc.detach(),
            'loss_edge': avg_loss_edge.detach(),
            'loss_centerline': avg_loss_centerline.detach(),
            'loss_topology': avg_loss_topology.detach(),
            'loss_boundary': avg_loss_boundary.detach(),
            'loss_sdf': avg_loss_sdf.detach(),
            'loss_pattern_center': avg_loss_pattern_center.detach(),
            'loss_pattern_direction': avg_loss_pattern_direction.detach(),
            'loss_pattern_spacing': avg_loss_pattern_spacing.detach(),
            'loss_contour': avg_loss_contour.detach(),
            'loss_ribbon_mask': avg_loss_ribbon_mask.detach(),
            'loss_ribbon_width': avg_loss_ribbon_width.detach(),
            'loss_thickness_prior': avg_loss_thickness_prior.detach(),
            'loss_instance_aggregate': avg_loss_instance_aggregate.detach(),
            'loss_trajectory_instance': avg_loss_trajectory_instance.detach(),
            'loss_prototype_context': avg_loss_prototype_context.detach(),
            'loss_hard_negative': avg_loss_hard_negative.detach(),
            'loss_hard_negative_safety': avg_loss_hard_negative_safety.detach(),
            'loss_texture_hard_negative': avg_loss_texture_hard_negative.detach(),
            'loss_texture_foreground_safety': avg_loss_texture_foreground_safety.detach(),
            'loss_afm_coarse': avg_loss_afm_coarse.detach(),
            'afm_coarse_effective_weight': torch.tensor(
                effective_afm_coarse_weight, device=output_device
            ),
            'loss_afm_aux': avg_loss_afm_aux.detach(),
            'afm_aux_effective_weight': torch.tensor(
                effective_afm_aux_weight, device=output_device
            ),
            'loss_weak_stitch': avg_loss_weak_stitch.detach(),
            'weak_stitch_effective_weight': torch.tensor(
                effective_weak_stitch_weight, device=output_device
            ),
            'loss_weak_stitch_consistency': avg_loss_weak_stitch_consistency.detach(),
        }
        return loss_dict


# =============================================================================
# Knowledge Distillation Loss
# =============================================================================
def kd_loss_fn(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """
    KL-divergence distillation loss between teacher and student logits.

    Standard formulation from Hinton et al. (2015):
        KD_loss = KL(softmax(T/蟿) || softmax(S/蟿)) 脳 蟿虏

    The 蟿虏 factor compensates for the gradient scaling introduced by
    temperature softening, keeping the KD gradient magnitude roughly
    independent of 蟿.

    This is a per-pixel loss suitable for segmentation / COD tasks:
    logits are treated as 2-class (fg/bg) and KL is averaged over all
    spatial locations and batch samples.

    Args:
        student_logits: (B, 1, H, W) raw logits from student model.
        teacher_logits: (B, 1, H, W) raw logits from teacher model.
        temperature:    Softening temperature 蟿. Higher = softer labels.

    Returns:
        Scalar KD loss (mean over batch, averaged over spatial dims).
    """
    # Resize teacher logits to match student spatial size if needed
    if teacher_logits.shape[-2:] != student_logits.shape[-2:]:
        teacher_logits = F.interpolate(
            teacher_logits,
            size=student_logits.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

    # Flatten to (N, 2) where N = B*H*W and 2 = [bg_logit, fg_logit]
    B = student_logits.shape[0]
    s_flat = student_logits.view(B, -1)          # (B, H*W)
    t_flat = teacher_logits.view(B, -1)          # (B, H*W)

    # Build 2-class logits: [neg, pos]
    s_2class = torch.stack([-s_flat, s_flat], dim=-1)  # (B, H*W, 2)
    t_2class = torch.stack([-t_flat, t_flat], dim=-1)  # (B, H*W, 2)

    # Soften with temperature
    s_soft = F.log_softmax(s_2class / temperature, dim=-1)
    t_soft = F.softmax(t_2class / temperature, dim=-1)

    # KL divergence: sum over class dim, mean over batch AND spatial dims
    #   sum over classes 鈫?then divide by (B 脳 H脳W) for per-pixel mean
    #   脳 蟿虏 to compensate temperature scaling (Hinton et al. 2015)
    kd = F.kl_div(s_soft, t_soft, reduction='sum', log_target=False)
    H, W = student_logits.shape[-2:]
    kd = kd / (B * H * W) * (temperature ** 2)

    return kd


def compute_distillation_loss(
    student_outputs: Dict[str, torch.Tensor],
    teacher_outputs: Dict[str, torch.Tensor],
    distill_keys: Union[List[str], Dict[str, float]],
    temperature: float = 4.0,
) -> torch.Tensor:
    """
    Aggregate KD loss across multiple output heads.

    Args:
        student_outputs: Dict mapping output name 鈫?student logits tensor
                         (each shape B脳1脳H脳W).
        teacher_outputs: Dict mapping output name 鈫?teacher logits tensor.
        distill_keys:    Output names to distill on. Accepts either a list
                         (equal weight for each head) or a dict of
                         {output_name: weight}.
        temperature:     KL temperature 蟿.

    Returns:
        Scalar KD loss summed over all specified output heads.
    """
    if isinstance(distill_keys, dict):
        distill_items = distill_keys.items()
    else:
        distill_items = [(key, 1.0) for key in distill_keys]

    total_kd = None
    for key, weight in distill_items:
        weight = float(weight)
        if weight <= 0:
            continue
        if key in student_outputs and key in teacher_outputs:
            kd = kd_loss_fn(
                student_outputs[key],
                teacher_outputs[key],
                temperature,
            ) * weight
            total_kd = kd if total_kd is None else total_kd + kd

    if total_kd is None:
        reference = next(
            (value for value in student_outputs.values() if torch.is_tensor(value)),
            None,
        )
        if reference is not None:
            return reference.new_tensor(0.0)
        return torch.tensor(0.0)

    return total_kd


