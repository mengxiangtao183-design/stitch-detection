"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
import csv
import json
import math
from tqdm import tqdm
from pathlib import Path

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data_RGB import get_test_data
from MPRNet import MPRNet
from skimage import img_as_ubyte
from skimage.metrics import structural_similarity
from PIL import Image
from pdb import set_trace as stx


VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def collect_reference_files(reference_dir):
    if not reference_dir.exists():
        return {}
    return {
        path.stem: path
        for path in sorted(reference_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    }


def image_to_float(path, expected_shape=None):
    image = Image.open(path).convert('RGB')
    if expected_shape is not None:
        h, w = expected_shape[:2]
        if image.size != (w, h):
            image = image.resize((w, h), Image.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def compute_image_metrics(pred, target):
    pred = np.clip(pred.astype(np.float32), 0.0, 1.0)
    target = np.clip(target.astype(np.float32), 0.0, 1.0)
    diff = pred - target
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    psnr = float('inf') if mse == 0 else 10.0 * math.log10(1.0 / mse)
    ssim = float(structural_similarity(pred, target, channel_axis=2, data_range=1.0))
    return {
        'psnr': psnr,
        'ssim': ssim,
        'mae': mae,
        'mse': mse,
    }


def compute_center_roi_metrics(pred, target, top_ratio, bottom_ratio):
    """Compute metrics on a fixed vertical region of interest.

    The sewing-stitch trajectory is physically centred in this dataset. The
    default [1/3, 2/3] band is therefore reported as ROI metrics in addition
    to—not instead of—the full-image restoration metrics.
    """
    height = pred.shape[0]
    top = int(round(height * float(top_ratio)))
    bottom = int(round(height * float(bottom_ratio)))
    top = max(0, min(height - 1, top))
    bottom = max(top + 1, min(height, bottom))
    return compute_image_metrics(pred[top:bottom], target[top:bottom]), top, bottom


def get_middle_stripe_bounds(padded_height, image_height, stripe_count, overlap, align=4):
    """Return the visible non-overlapping core of the middle Stage-1 stripe.

    The calculation matches the model's stripe placement. Shared regions with
    the upper and lower stripes are excluded. Inputs are padded before Stage 1,
    so bounds are derived from that height and clipped to the saved image.
    """
    stripe_count = int(stripe_count)
    if stripe_count < 1 or stripe_count % 2 == 0:
        raise ValueError('Middle-stripe metrics require an odd positive stripe count.')
    if not 0.0 <= float(overlap) < 0.5:
        raise ValueError('stage1_overlap must be in [0.0, 0.5).')
    if padded_height % align != 0:
        raise ValueError('Padded height must be divisible by the alignment.')

    effective_stripes = stripe_count - float(overlap) * (stripe_count - 1)
    stripe_height = int(math.ceil((padded_height / effective_stripes) / align) * align)
    stripe_height = min(padded_height, max(align, stripe_height))
    max_start = padded_height - stripe_height
    starts = []
    for index in range(stripe_count):
        start = int(round((max_start * index / (stripe_count - 1)) / align) * align)
        starts.append(min(max_start, max(0, start)))
    starts[0] = 0
    starts[-1] = max_start

    middle_index = stripe_count // 2
    middle_start = starts[middle_index]
    middle_end = middle_start + stripe_height
    top = max(middle_start, starts[middle_index - 1] + stripe_height)
    bottom = min(middle_end, starts[middle_index + 1])
    if bottom <= top:
        raise ValueError('Middle stripe has no non-overlapping core.')

    top = max(0, min(image_height - 1, top))
    bottom = max(top + 1, min(image_height, bottom))
    return top, bottom


def mean_or_none(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(values)) if values else None


def summarize_rows(rows, dataset=None):
    selected = [row for row in rows if dataset is None or row['dataset'] == dataset]
    return {
        'dataset': dataset if dataset is not None else 'overall',
        'samples': len(selected),
        'restored_psnr': mean_or_none([row['restored_psnr'] for row in selected]),
        'restored_ssim': mean_or_none([row['restored_ssim'] for row in selected]),
        'restored_mae': mean_or_none([row['restored_mae'] for row in selected]),
        'restored_mse': mean_or_none([row['restored_mse'] for row in selected]),
        'restored_roi_psnr': mean_or_none([row['restored_roi_psnr'] for row in selected]),
        'restored_roi_ssim': mean_or_none([row['restored_roi_ssim'] for row in selected]),
        'restored_roi_mae': mean_or_none([row['restored_roi_mae'] for row in selected]),
        'restored_roi_mse': mean_or_none([row['restored_roi_mse'] for row in selected]),
        'input_psnr': mean_or_none([row['input_psnr'] for row in selected]),
        'input_ssim': mean_or_none([row['input_ssim'] for row in selected]),
        'input_mae': mean_or_none([row['input_mae'] for row in selected]),
        'input_mse': mean_or_none([row['input_mse'] for row in selected]),
        'input_roi_psnr': mean_or_none([row['input_roi_psnr'] for row in selected]),
        'input_roi_ssim': mean_or_none([row['input_roi_ssim'] for row in selected]),
        'input_roi_mae': mean_or_none([row['input_roi_mae'] for row in selected]),
        'input_roi_mse': mean_or_none([row['input_roi_mse'] for row in selected]),
    }


def summarize_dataset_macro(dataset_summaries):
    """Report an equal-weight mean across evaluated datasets.

    Dataset sizes differ substantially (for example RX has 500 images while
    DB/XJ/SD each have 100), so the headline overall score must not be driven
    by the largest dataset. Per-dataset metrics remain image means; this
    summary is their macro average.
    """
    metric_keys = [
        'restored_psnr', 'restored_ssim', 'restored_mae', 'restored_mse',
        'restored_roi_psnr', 'restored_roi_ssim', 'restored_roi_mae',
        'restored_roi_mse', 'input_psnr', 'input_ssim', 'input_mae',
        'input_mse', 'input_roi_psnr', 'input_roi_ssim', 'input_roi_mae',
        'input_roi_mse',
    ]
    summary = {
        'dataset': 'overall',
        'samples': sum(item['samples'] for item in dataset_summaries),
        'aggregation': 'macro_mean_over_datasets',
        'datasets': [item['dataset'] for item in dataset_summaries],
    }
    for key in metric_keys:
        summary[key] = mean_or_none([item[key] for item in dataset_summaries])
    return summary


def write_metrics(metrics_dir, rows, summaries, weights, stage1_stripes, stage1_overlap):
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / 'per_image_metrics.csv'
    fieldnames = [
        'dataset',
        'filename',
        'restored_psnr',
        'restored_ssim',
        'restored_mae',
        'restored_mse',
        'restored_roi_psnr',
        'restored_roi_ssim',
        'restored_roi_mae',
        'restored_roi_mse',
        'input_psnr',
        'input_ssim',
        'input_mae',
        'input_mse',
        'input_roi_psnr',
        'input_roi_ssim',
        'input_roi_mae',
        'input_roi_mse',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = metrics_dir / 'eval_summary.json'
    summary = {
        'weights': weights,
        'middle_stripe': {
            'stripe_count': stage1_stripes,
            'overlap': stage1_overlap,
            'region': 'non_overlapping_core',
            'description': 'Exact visible middle Stage-1 stripe core; shared stripe overlaps are excluded.',
        },
        'summaries': summaries,
        'per_image_csv': str(csv_path),
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return csv_path, summary_path

parser = argparse.ArgumentParser(description='Image Deblurring using MPRNet')

parser.add_argument('--input_dir', default='H:/DATASET', type=str,
                    help='Dataset root containing DB/XJ/SD/RX folders.')
parser.add_argument('--result_dir',
                    default='H:/C3Netqing/Deblurring/checkpoints/train/Deblurring/results/MPRNet/restored',
                    type=str,
                    help='Directory for restored results.')
parser.add_argument('--weights',
                    default='H:/C3Netqing/Deblurring/checkpoints/train/Deblurring/models/MPRNet/model_latest.pth',
                    type=str,
                    help='Path to weights. Default uses the latest trained checkpoint.')
parser.add_argument('--dataset', default=None, type=str,
                    help='Single test dataset name. Kept for backward compatibility.')
parser.add_argument('--datasets', nargs='+', default=['DB', 'XJ', 'SD', 'RX'],
                    help='Datasets to test when --dataset is not provided.')
parser.add_argument('--split', default='test', type=str,
                    help='Dataset split under each dataset root.')
parser.add_argument('--input_subdir', default='Imgs1', type=str,
                    help='Input image subdirectory, e.g. Imgs1 for degraded images.')
parser.add_argument('--clear_subdir', default='clear', type=str,
                    help='Reference image subdirectory for metric calculation.')
parser.add_argument('--metrics_dir', default=None, type=str,
                    help='Directory for per-image CSV and summary JSON. Default: <result_dir>/metrics.')
parser.add_argument('--no_metrics', action='store_true',
                    help='Only save restored images and skip metric calculation.')
parser.add_argument('--num_workers', default=0, type=int,
                    help='DataLoader workers. Keep 0 on Windows for stability.')
parser.add_argument('--max_images', default=0, type=int,
                    help='Optional limit per dataset for quick smoke tests. 0 means all images.')
parser.add_argument('--roi_top_ratio', default=1.0 / 3.0, type=float,
                    help='Top of the fixed vertical ROI as an image-height ratio.')
parser.add_argument('--roi_bottom_ratio', default=2.0 / 3.0, type=float,
                    help='Bottom of the fixed vertical ROI as an image-height ratio.')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--n_feat', default=96, type=int, help='MPRNet base feature channels.')
parser.add_argument('--scale_unetfeats', default=48, type=int, help='MPRNet U-Net feature growth.')
parser.add_argument('--scale_orsnetfeats', default=32, type=int, help='MPRNet ORSNet feature growth.')
parser.add_argument('--num_cab', default=8, type=int, help='Number of CAB blocks in ORSNet ORB.')
parser.add_argument('--edge_refine', action='store_true', help='Enable stitch edge-guided refinement module.')
parser.add_argument('--edge_refine_mid', default=24, type=int, help='Hidden channels in edge refinement module.')
parser.add_argument('--frequency_refine', action='store_true',
                    help='Enable final RGB frequency residual refinement.')
parser.add_argument('--frequency_refine_hidden', default=16, type=int,
                    help='Hidden channels in the frequency residual refiner.')
parser.add_argument('--frequency_band_gate', action='store_true',
                    help='Enable smooth band-pass correction and per-image frequency strength gate.')
parser.add_argument('--frequency_band_low_start', default=0.05, type=float)
parser.add_argument('--frequency_band_low_end', default=0.10, type=float)
parser.add_argument('--frequency_band_high_start', default=0.30, type=float)
parser.add_argument('--frequency_band_high_end', default=0.40, type=float)
parser.add_argument('--frequency_disagreement_gate', action='store_true',
                    help='Use Stage-3/Stage-2 disagreement to gate band-pass frequency correction.')
parser.add_argument('--frequency_disagreement_threshold', default=0.0120, type=float)
parser.add_argument('--frequency_disagreement_temperature', default=0.0020, type=float)
parser.add_argument('--frequency_disagreement_min_strength', default=0.10, type=float)
parser.add_argument('--light_adapt', action='store_true', help='Enable light adaptive input correction for MPRNet.')
parser.add_argument('--light_target_luma', default=0.50, type=float, help='Target luminance for light adaptation.')
parser.add_argument('--light_min_gain', default=0.75, type=float, help='Minimum gain for light adaptation.')
parser.add_argument('--light_max_gain', default=1.80, type=float, help='Maximum gain for light adaptation.')
parser.add_argument('--light_min_gamma', default=0.65, type=float, help='Minimum gamma for light adaptation.')
parser.add_argument('--light_max_gamma', default=1.35, type=float, help='Maximum gamma for light adaptation.')
parser.add_argument('--light_blend', default=0.75, type=float, help='Blend strength for light adaptation.')
parser.add_argument('--light_dark_threshold', default=0.45, type=float, help='Luminance threshold for dark-image correction.')
parser.add_argument('--light_bright_threshold', default=0.70, type=float, help='Luminance threshold for bright-image correction.')
parser.add_argument('--trajectory_lightweight', action='store_true',
                    help='Enable stripe Stage 1, full-image directional NAF Stage 2, and NAF Stage 3.')
parser.add_argument('--stage1_tristripe', action='store_true',
                    help='Enable three-stripe shared-pair Stage 1 with original Stage 2/3.')
parser.add_argument('--stage1_stripes', default=3, type=int, help='Number of full-width Stage-1 stripes.')
parser.add_argument('--stage1_overlap', default=0.25, type=float, help='Stage-1 stripe overlap ratio.')
parser.add_argument('--stage1_stripe_transformer', action='store_true',
                    help='Enable row-wise Stripe Transformer at the Stage-1 bottleneck.')
parser.add_argument('--stage1_stripe_transformer_heads', default=4, type=int,
                    help='Attention heads for the Stage-1 Stripe Transformer.')
parser.add_argument('--stage2_stripe_transformer', action='store_true',
                    help='Enable row-wise Stripe Transformer at the Stage-2 bottleneck.')
parser.add_argument('--stage2_stripe_transformer_heads', default=4, type=int,
                    help='Attention heads for the Stage-2 Stripe Transformer.')
parser.add_argument('--stage2_overlap', default=0.0, type=float,
                    help='Upper/lower overlap ratio for original Stage 2 in tri-stripe mode.')
parser.add_argument('--stage2_pair_aligned', action='store_true',
                    help='Use [stripe1+stripe2] and [stripe2+stripe3] as Stage-2 branches.')
parser.add_argument('--stage2_tmt', action='store_true',
                    help='Replace Stage-2 Encoder/Decoder/SAM23 with trajectory memory transfer modules.')
parser.add_argument('--stage2_tmt_horizontal_kernel', default=15, type=int,
                    help='Horizontal strip kernel in Stage-2 TMT.')
parser.add_argument('--stage2_tmt_vertical_kernel', default=5, type=int,
                    help='Vertical strip kernel in Stage-2 TMT.')
parser.add_argument('--gated_csff', action='store_true',
                    help='Enable trajectory-calibrated CSFF in the original Stage-2 encoder.')
parser.add_argument('--gated_csff_horizontal_kernel', default=7, type=int,
                    help='Horizontal context kernel in trajectory-calibrated CSFF.')
parser.add_argument('--stitch_trajectory_alignment', action='store_true',
                    help='Align Stage-1 pair features to Stage-2 using learned trajectory fields.')
parser.add_argument('--stam_horizontal_radius', default=6, type=int,
                    help='Maximum horizontal sampling displacement for STAM.')
parser.add_argument('--stam_vertical_radius', default=1, type=int,
                    help='Maximum vertical sampling displacement for STAM.')
parser.add_argument('--stam_hidden_channels', default=32, type=int,
                    help='Hidden channels in each STAM trajectory estimator.')
parser.add_argument('--dual_domain_orsnet', action='store_true',
                    help='Replace Stage-3 ORSNet with selective hybrid dual-domain refinement.')
parser.add_argument('--ddrb_residual_init', default=0.3, type=float,
                    help='Initial learnable residual strength in dual-domain blocks.')
parser.add_argument('--simple_gated_orsnet', action='store_true',
                    help='Replace Stage-3 ORS CAB blocks with attention-free Simple-Gated ORS blocks.')
parser.add_argument('--sgors_blocks', default=8, type=int,
                    help='Simple-Gated restoration blocks in each SG-ORS cell.')
parser.add_argument('--sgors_residual_init', default=0.10, type=float,
                    help='Initial residual scale in each Simple-Gated restoration block.')
parser.add_argument('--hybrid_simple_gated_orsnet', action='store_true',
                    help='Keep pretrained ORS CAB paths and add parallel Simple-Gated residual adapters.')
parser.add_argument('--hsgors_blocks', default=4, type=int,
                    help='Simple-Gated blocks in each hybrid ORS adapter.')
parser.add_argument('--hsgors_residual_init', default=0.05, type=float,
                    help='Initial residual scale in each hybrid SG adapter block.')
parser.add_argument('--hsgors_adapter_scale_init', default=0.02, type=float,
                    help='Initial strength of each hybrid SG adapter correction.')
parser.add_argument('--hsgors_v2', action='store_true',
                    help='Use HSGORS-v2 trajectory-gated ORS adapters.')
parser.add_argument('--hsgors_v2_horizontal_kernel', default=15, type=int,
                    help='Horizontal strip kernel in HSGORS-v2 adapters.')
parser.add_argument('--hsgors_v2_vertical_kernel', default=5, type=int,
                    help='Vertical tolerance kernel in HSGORS-v2 adapters.')
parser.add_argument('--stage3_tepr', action='store_true',
                    help='Use the ORB/CAB-free trajectory evidence pyramid recovery module.')
parser.add_argument('--stage3_tepr_transition', action='store_true',
                    help='Load the training-only legacy recovery path for a transition checkpoint.')
parser.add_argument('--stage3_tepr_freeze_legacy', action='store_true',
                    help='Freeze legacy transition modules. Training-only flag; ignored by pure inference.')
parser.add_argument('--stage3_tepr_blocks', default=3, type=int,
                    help='Trajectory state-propagation blocks in TEPR.')
parser.add_argument('--stage3_tepr_horizontal_kernel', default=15, type=int,
                    help='Horizontal strip kernel in TEPR.')
parser.add_argument('--stage3_tepr_vertical_kernel', default=5, type=int,
                    help='Vertical tolerance kernel in TEPR.')
parser.add_argument('--stage3_trr', action='store_true',
                    help='Replace Stage-3 ORS recovery with trajectory residual restorer adapters.')
parser.add_argument('--stage3_trr_horizontal_kernel', default=15, type=int,
                    help='Horizontal strip kernel in Stage-3 TRR.')
parser.add_argument('--stage3_trr_vertical_kernel', default=5, type=int,
                    help='Vertical strip kernel in Stage-3 TRR.')
parser.add_argument('--stage3_trr_hidden_channels', default=32, type=int,
                    help='Hidden channels in Stage-3 TRR adapters.')
parser.add_argument('--stage2_naf_blocks', default=2, type=int, help='NAF blocks per Stage-2 level.')
parser.add_argument('--stage2_horizontal_kernel', default=15, type=int, help='Horizontal bottleneck kernel.')
parser.add_argument('--stage2_vertical_kernel', default=5, type=int, help='Vertical bottleneck kernel.')
parser.add_argument('--stage3_naf_blocks', default=3, type=int, help='NAF blocks per Stage-3 fusion level.')
parser.add_argument('--tpt_rnet_core', action='store_true',
                    help='Use the rewritten TPT-RNet pairwise pyramid backend.')
parser.add_argument('--tptr_blocks', default=2, type=int,
                    help='Directional topology blocks per TPT-RNet pyramid level.')
parser.add_argument('--tptr_tcfm', action='store_true',
                    help='Inject three-scale Stage-1 Encoder/Decoder states via TCFM.')
parser.add_argument('--tptr_tcfm_horizontal_kernel', default=7, type=int,
                    help='Horizontal trajectory context kernel in TPT-RNet TCFM.')
parser.add_argument('--tristripe_context_pyramid', action='store_true',
                    help='Replace inherited Stage-1 modules with TriStripe Context Pyramid.')
parser.add_argument('--tristripe_context_blocks', default=2, type=int,
                    help='Directional blocks per TriStripe Context Pyramid scale.')
parser.add_argument('--horizontal_trajectory_blocks', action='store_true',
                    help='Use horizontal-prior trajectory blocks in the new TPT-RNet.')
parser.add_argument('--hmt_net_core', action='store_true',
                    help='Use the HMT-Net shared-memory reconstruction core.')
parser.add_argument('--hmt_blocks', default=3, type=int,
                    help='Horizontal trajectory blocks per HMT-Net level.')
parser.add_argument('--evidence_canvas_net', action='store_true',
                    help='Use the independent evidence-canvas reconstruction core.')
parser.add_argument('--evidence_canvas_blocks', default=2, type=int,
                    help='Directional blocks per EvidenceCanvasNet scale.')
parser.add_argument('--tfr_net_core', action='store_true',
                    help='Use the TFR-Net dual-pair frequency residual core.')
parser.add_argument('--tfr_blocks', default=8, type=int,
                    help='Full-resolution restoration blocks in TFR-Net.')
parser.add_argument('--tfr_horizontal_kernel', default=15, type=int,
                    help='Horizontal motion-filter kernel in TFR-Net.')
parser.add_argument('--tfr_vertical_kernel', default=5, type=int,
                    help='Vertical tolerance kernel in TFR-Net.')
parser.add_argument('--tfr_refiner_steps', default=3, type=int,
                    help='Shared terminal residual-refiner steps in TFR-Net.')
parser.add_argument('--stripe_topology_reconstruction', action='store_true',
                    help='Enable STRD multi-scale line-topology reconstruction decoder.')
parser.add_argument('--strd_hidden', default=32, type=int,
                    help='Hidden channels in STRD.')
parser.add_argument('--strd_num_blocks', default=2, type=int,
                    help='Topology continuity blocks at each STRD scale.')
parser.add_argument('--interstage_residual_reassembly', action='store_true',
                    help='Enable the image-space inter-stage residual reassembly (IRR) head.')
parser.add_argument('--irr_hidden', default=24, type=int,
                    help='Hidden channels in the IRR head.')
parser.add_argument('--irr_num_blocks', default=1, type=int,
                    help='Depthwise refinement blocks per IRR scale.')
parser.add_argument('--irr_delta_init', default=1.0e-4, type=float,
                    help='IRR residual-head initialization std; kept for checkpoint architecture parity.')

args = parser.parse_args()

if not 0.0 <= args.roi_top_ratio < args.roi_bottom_ratio <= 1.0:
    raise ValueError('ROI ratios must satisfy 0 <= top < bottom <= 1.')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = MPRNet(
    n_feat=args.n_feat,
    scale_unetfeats=args.scale_unetfeats,
    scale_orsnetfeats=args.scale_orsnetfeats,
    num_cab=args.num_cab,
    edge_refine=args.edge_refine,
    edge_refine_mid=args.edge_refine_mid,
    frequency_refine=args.frequency_refine,
    frequency_refine_hidden=args.frequency_refine_hidden,
    frequency_band_gate=args.frequency_band_gate,
    frequency_band_low_start=args.frequency_band_low_start,
    frequency_band_low_end=args.frequency_band_low_end,
    frequency_band_high_start=args.frequency_band_high_start,
    frequency_band_high_end=args.frequency_band_high_end,
    frequency_disagreement_gate=args.frequency_disagreement_gate,
    frequency_disagreement_threshold=args.frequency_disagreement_threshold,
    frequency_disagreement_temperature=args.frequency_disagreement_temperature,
    frequency_disagreement_min_strength=args.frequency_disagreement_min_strength,
    light_adapt=args.light_adapt,
    light_target_luma=args.light_target_luma,
    light_min_gain=args.light_min_gain,
    light_max_gain=args.light_max_gain,
    light_min_gamma=args.light_min_gamma,
    light_max_gamma=args.light_max_gamma,
    light_blend=args.light_blend,
    light_dark_threshold=args.light_dark_threshold,
    light_bright_threshold=args.light_bright_threshold,
    trajectory_lightweight=args.trajectory_lightweight,
    stage1_tristripe=args.stage1_tristripe,
    stage1_stripes=args.stage1_stripes,
    stage1_overlap=args.stage1_overlap,
    stage1_stripe_transformer=args.stage1_stripe_transformer,
    stage1_stripe_transformer_heads=args.stage1_stripe_transformer_heads,
    stage2_stripe_transformer=args.stage2_stripe_transformer,
    stage2_stripe_transformer_heads=args.stage2_stripe_transformer_heads,
    stage2_overlap=args.stage2_overlap,
    stage2_pair_aligned=args.stage2_pair_aligned,
    stage2_tmt=args.stage2_tmt,
    stage2_tmt_horizontal_kernel=args.stage2_tmt_horizontal_kernel,
    stage2_tmt_vertical_kernel=args.stage2_tmt_vertical_kernel,
    gated_csff=args.gated_csff,
    gated_csff_horizontal_kernel=args.gated_csff_horizontal_kernel,
    stitch_trajectory_alignment=args.stitch_trajectory_alignment,
    stam_horizontal_radius=args.stam_horizontal_radius,
    stam_vertical_radius=args.stam_vertical_radius,
    stam_hidden_channels=args.stam_hidden_channels,
    dual_domain_orsnet=args.dual_domain_orsnet,
    ddrb_residual_init=args.ddrb_residual_init,
    simple_gated_orsnet=args.simple_gated_orsnet,
    sgors_blocks=args.sgors_blocks,
    sgors_residual_init=args.sgors_residual_init,
    hybrid_simple_gated_orsnet=args.hybrid_simple_gated_orsnet,
    hsgors_blocks=args.hsgors_blocks,
    hsgors_residual_init=args.hsgors_residual_init,
    hsgors_adapter_scale_init=args.hsgors_adapter_scale_init,
    hsgors_v2=args.hsgors_v2,
    hsgors_v2_horizontal_kernel=args.hsgors_v2_horizontal_kernel,
    hsgors_v2_vertical_kernel=args.hsgors_v2_vertical_kernel,
    stage3_tepr=args.stage3_tepr,
    stage3_tepr_transition=args.stage3_tepr_transition,
    stage3_tepr_freeze_legacy=args.stage3_tepr_freeze_legacy,
    stage3_tepr_blocks=args.stage3_tepr_blocks,
    stage3_tepr_horizontal_kernel=args.stage3_tepr_horizontal_kernel,
    stage3_tepr_vertical_kernel=args.stage3_tepr_vertical_kernel,
    stage3_trr=args.stage3_trr,
    stage3_trr_horizontal_kernel=args.stage3_trr_horizontal_kernel,
    stage3_trr_vertical_kernel=args.stage3_trr_vertical_kernel,
    stage3_trr_hidden_channels=args.stage3_trr_hidden_channels,
    stage2_naf_blocks=args.stage2_naf_blocks,
    stage2_horizontal_kernel=args.stage2_horizontal_kernel,
    stage2_vertical_kernel=args.stage2_vertical_kernel,
    stage3_naf_blocks=args.stage3_naf_blocks,
    tpt_rnet_core=args.tpt_rnet_core,
    tptr_blocks=args.tptr_blocks,
    tptr_tcfm=args.tptr_tcfm,
    tptr_tcfm_horizontal_kernel=args.tptr_tcfm_horizontal_kernel,
    tristripe_context_pyramid=args.tristripe_context_pyramid,
    tristripe_context_blocks=args.tristripe_context_blocks,
    horizontal_trajectory_blocks=args.horizontal_trajectory_blocks,
    hmt_net_core=args.hmt_net_core,
    hmt_blocks=args.hmt_blocks,
    evidence_canvas_net=args.evidence_canvas_net,
    evidence_canvas_blocks=args.evidence_canvas_blocks,
    tfr_net_core=args.tfr_net_core,
    tfr_blocks=args.tfr_blocks,
    tfr_horizontal_kernel=args.tfr_horizontal_kernel,
    tfr_vertical_kernel=args.tfr_vertical_kernel,
    tfr_refiner_steps=args.tfr_refiner_steps,
    stripe_topology_reconstruction=args.stripe_topology_reconstruction,
    strd_hidden=args.strd_hidden,
    strd_num_blocks=args.strd_num_blocks,
    interstage_residual_reassembly=args.interstage_residual_reassembly,
    irr_hidden=args.irr_hidden,
    irr_num_blocks=args.irr_num_blocks,
    irr_delta_init=args.irr_delta_init,
)

if not os.path.exists(args.weights):
    raise FileNotFoundError(f"Weights not found: {args.weights}")

utils.load_checkpoint(model_restoration, args.weights)
print("===> Testing using weights:", args.weights)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_restoration = model_restoration.to(device)
# DataParallel adds no value on one GPU and can intermittently reject a
# perfectly valid cuda:0 model on Windows during long evaluation runs.
if torch.cuda.device_count() > 1:
    model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

datasets = [args.dataset] if args.dataset else args.datasets
metrics_rows = []
summary_rows = []

with torch.no_grad():
    for dataset in datasets:
        rgb_dir_test = Path(args.input_dir) / dataset / args.split / args.input_subdir
        if not rgb_dir_test.exists():
            print(f"===> Skipping {dataset}: input directory not found: {rgb_dir_test}")
            continue
        clear_dir_test = Path(args.input_dir) / dataset / args.split / args.clear_subdir
        reference_files = {} if args.no_metrics else collect_reference_files(clear_dir_test)
        if not args.no_metrics and not reference_files:
            print(f"===> Metrics disabled for {dataset}: reference directory not found or empty: {clear_dir_test}")

        test_dataset = get_test_data(str(rgb_dir_test), img_options={})
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=True,
        )

        result_dir = Path(args.result_dir) / dataset
        utils.mkdir(str(result_dir))
        dataset_rows = []

        print(f"===> Testing {dataset}: {rgb_dir_test} -> {result_dir}")
        for ii, data_test in enumerate(tqdm(test_loader, desc=dataset), 0):
            if args.max_images > 0 and ii >= args.max_images:
                break
            input_ = data_test[0].to(device, non_blocking=True)
            input_for_metrics = data_test[0].permute(0, 2, 3, 1).cpu().numpy()
            filenames = data_test[1]

            # Padding in case images are not multiples of 8.
            factor = 8
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor - 1) // factor) * factor, ((w + factor - 1) // factor) * factor
            padh = H - h
            padw = W - w
            if padh > 0 or padw > 0:
                input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            restored = model_restoration(input_)
            restored = torch.clamp(restored[0], 0, 1)

            # Unpad images to original dimensions.
            if padh > 0 or padw > 0:
                restored = restored[:, :, :h, :w]

            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                utils.save_img(str(result_dir / f"{filenames[batch]}.png"), restored_img)

                if args.no_metrics:
                    continue

                filename = filenames[batch]
                reference_path = reference_files.get(filename)
                if reference_path is None:
                    continue

                target = image_to_float(reference_path, expected_shape=restored[batch].shape)
                restored_metrics = compute_image_metrics(restored[batch], target)
                input_metrics = compute_image_metrics(input_for_metrics[batch], target)
                middle_stripe_align = 8 if args.evidence_canvas_net else 4
                middle_top, middle_bottom = get_middle_stripe_bounds(
                    H,
                    h,
                    args.stage1_stripes,
                    args.stage1_overlap,
                    align=middle_stripe_align,
                )
                restored_roi_metrics = compute_image_metrics(
                    restored[batch][middle_top:middle_bottom],
                    target[middle_top:middle_bottom],
                )
                input_roi_metrics = compute_image_metrics(
                    input_for_metrics[batch][middle_top:middle_bottom],
                    target[middle_top:middle_bottom],
                )
                row = {
                    'dataset': dataset,
                    'filename': filename,
                    'restored_psnr': restored_metrics['psnr'],
                    'restored_ssim': restored_metrics['ssim'],
                    'restored_mae': restored_metrics['mae'],
                    'restored_mse': restored_metrics['mse'],
                    'restored_roi_psnr': restored_roi_metrics['psnr'],
                    'restored_roi_ssim': restored_roi_metrics['ssim'],
                    'restored_roi_mae': restored_roi_metrics['mae'],
                    'restored_roi_mse': restored_roi_metrics['mse'],
                    'input_psnr': input_metrics['psnr'],
                    'input_ssim': input_metrics['ssim'],
                    'input_mae': input_metrics['mae'],
                    'input_mse': input_metrics['mse'],
                    'input_roi_psnr': input_roi_metrics['psnr'],
                    'input_roi_ssim': input_roi_metrics['ssim'],
                    'input_roi_mae': input_roi_metrics['mae'],
                    'input_roi_mse': input_roi_metrics['mse'],
                }
                dataset_rows.append(row)
                metrics_rows.append(row)

        if dataset_rows:
            summary = summarize_rows(dataset_rows, dataset=dataset)
            summary_rows.append(summary)
            print(
                f"===> Metrics {dataset}: "
                f"PSNR={summary['restored_psnr']:.4f}, "
                f"SSIM={summary['restored_ssim']:.5f}, "
                f"MiddleStripe_PSNR={summary['restored_roi_psnr']:.4f}, "
                f"MiddleStripe_SSIM={summary['restored_roi_ssim']:.5f}, "
                f"MAE={summary['restored_mae']:.5f}, "
                f"Input_PSNR={summary['input_psnr']:.4f}"
            )

if metrics_rows:
    summary_rows.append(summarize_dataset_macro(summary_rows))
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else Path(args.result_dir) / 'metrics'
    csv_path, summary_path = write_metrics(
        metrics_dir,
        metrics_rows,
        summary_rows,
        args.weights,
        args.stage1_stripes,
        args.stage1_overlap,
    )
    print(f"===> Saved per-image metrics: {csv_path}")
    print(f"===> Saved metric summary: {summary_path}")
