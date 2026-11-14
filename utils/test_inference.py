"""
Comprehensive Inference & Speed Test Tool for C3Net (COD).

This script provides:
1. Pure model inference speed benchmarking (excluding pre/post-processing)
2. End-to-end inference speed testing (full pipeline)
3. Multi-batch throughput measurement
4. Detection result visualization (original, restored, mask, overlay)
5. CSV summary export

Fully compatible with C3Net (MobileNetV4 backbone, CoordAtt).
Handles DataParallel checkpoints and both CPU / GPU (FP16) inference.

Usage:
    python utils/test_inference.py \
        --config configs/default.yaml \
        --checkpoint checkpoints/model_best.pth \
        --input path/to/image_or_dir \
        --output ./inference_results \
        --mode both \
        --num_test 100 \
        --batch_sizes 1,2,4,8 \
        --device cuda \
        --fp16

Author : C3Net Team
Date   : 08-06-2026
"""

import sys
import os

# Ensure project root is on sys.path so we can import project modules
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import csv
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

# Project imports
from com_models.registry import build_model as build_configured_model
from utils.image_processor import CODImageProcessor

# Optional: matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend by default
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Optional: cv2 for overlay blending
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('test_inference')

SEP = "=" * 70
SEP2 = "-" * 70


# ===========================================================================
# Utility helpers
# ===========================================================================

def _load_config(path: str) -> Dict:
    """Load YAML configuration file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def _load_checkpoint(path: str, device: torch.device) -> Dict:
    """Load checkpoint, handling DataParallel 'module.' prefix."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(p, map_location=device)
    if 'model_state_dict' not in ckpt:
        # Try loading as raw state_dict
        ckpt = {'model_state_dict': ckpt}
    # Strip 'module.' prefix
    sd = ckpt['model_state_dict']
    new_sd = OrderedDict()
    for k, v in sd.items():
        new_sd[k[7:] if k.startswith('module.') else k] = v
    ckpt['model_state_dict'] = new_sd
    return ckpt


def _gather_images(input_path: str) -> List[Path]:
    """Collect image paths from a file or directory."""
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if p.is_file():
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
            return [p]
        else:
            raise ValueError(f"Unsupported file extension: {p.suffix}")
    # Directory: gather all image files
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    files = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
    if not files:
        raise RuntimeError(f"No image files found in directory: {input_path}")
    return files


# ===========================================================================
# Model builder
# ===========================================================================

def build_model(config: Dict, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Instantiate C3Net from config and load trained weights.

    Args:
        config: Full project config dictionary (must contain 'model' key).
        checkpoint_path: Path to .pth checkpoint.
        device: torch device.

    Returns:
        C3Net model in eval mode on the target device.
    """
    logger.info("Building configured segmentation model ...")
    model_cfg = config['model']
    model = build_configured_model(model_cfg)
    ckpt = _load_checkpoint(checkpoint_path, device)
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    model.to(device)
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded.  Total params: {total:,}  Trainable: {trainable:,}")
    return model


# ===========================================================================
# Image pre-processing / post-processing
# ===========================================================================

class InferenceProcessor:
    """
    Thin wrapper around CODImageProcessor + output post-processing.

    Handles:
    - Loading & normalising images to model input size.
    - Converting model logit output to binary mask at original resolution.
    """

    def __init__(self, config: Dict):
        ip_cfg = config['model']['image_processing']
        self.target_size = ip_cfg.get('target_size', 320)
        self.processor = CODImageProcessor(
            target_size=self.target_size,
            normalize_mean=tuple(ip_cfg.get('normalize_mean', [0.485, 0.456, 0.406])),
            normalize_std=tuple(ip_cfg.get('normalize_std', [0.229, 0.224, 0.225])),
        )

    def preprocess(self, image_path: Union[str, Path]) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess a single image.

        Returns:
            model_input:  (1, 3, H, W) normalised tensor.
            original_bgr: Original image as (H, W, 3) uint8 BGR (for OpenCV) or RGB.
        """
        original = self._load_original(image_path)
        tensor_input = self.processor.process_image(image_path)  # (3, H, W)
        return tensor_input.unsqueeze(0), original  # add batch dim

    @staticmethod
    def _load_original(path: Union[str, Path]) -> np.ndarray:
        """Load raw image as (H, W, 3) RGB uint8 numpy array."""
        from PIL import Image
        img = Image.open(str(path)).convert('RGB')
        return np.array(img)

    @staticmethod
    def postprocess(
        logits: torch.Tensor,
        original_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Convert final_mask logits to a binary mask at original image resolution.

        Args:
            logits: (1, 1, H_pred, W_pred) raw logits from model.
            original_shape: (H_orig, W_orig) target size.

        Returns:
            Binary mask (H_orig, W_orig) uint8 [0, 255].
        """
        # Sigmoid 鈫?probabilities
        prob = torch.sigmoid(logits)
        # Resize to original resolution
        prob = F.interpolate(
            prob, size=original_shape, mode='bilinear', align_corners=False
        )
        mask = (prob.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
        return mask


# ===========================================================================
# Speed test
# ===========================================================================

def _cuda_sync(device: torch.device):
    """Synchronise CUDA if applicable."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class SpeedBenchmark:
    """
    Measure inference speed: pure model, end-to-end, and batch throughput.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor: InferenceProcessor,
        device: torch.device,
        use_fp16: bool = True,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.use_fp16 = use_fp16 and (device.type == 'cuda')

    # ------------------------------------------------------------------
    # Pure model inference
    # ------------------------------------------------------------------
    def _run_pure_batch(self, tensor: torch.Tensor) -> float:
        """Single pure-model forward pass (no pre/post).  Returns seconds."""
        x = tensor.to(self.device)
        _cuda_sync(self.device)
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            with torch.no_grad():
                _ = self.model(x)
        _cuda_sync(self.device)
        return time.perf_counter() - t0

    def benchmark_pure_model(
        self, image_paths: List[Path], num_iter: int = 100
    ) -> Dict:
        """
        Pure model speed (excludes image loading, pre-processing, post-processing).

        Uses the first valid image to build the input tensor.
        """
        logger.info(SEP2)
        logger.info("Pure Model Inference Benchmark")
        logger.info(SEP2)

        # Build fixed input tensor from first image
        img_path = image_paths[0]
        tensor, _ = self.processor.preprocess(img_path)  # (1, 3, H, W)
        logger.info(f"Input tensor shape: {tuple(tensor.shape)}")

        # Warm-up
        logger.info(f"Warming up ({min(20, num_iter // 5)} iterations) ...")
        for _ in range(min(20, num_iter // 5)):
            self._run_pure_batch(tensor)

        # Timed iterations
        logger.info(f"Running {num_iter} timed iterations ...")
        times = []
        for _ in tqdm(range(num_iter), desc='Pure model'):
            t = self._run_pure_batch(tensor)
            times.append(t)

        stats = _compute_stats(times, 'pure_model')
        _print_stats_table(stats, 'Pure Model Inference')
        return stats

    # ------------------------------------------------------------------
    # End-to-end inference
    # ------------------------------------------------------------------
    def benchmark_e2e(
        self, image_paths: List[Path], num_iter: int = 100
    ) -> Dict:
        """
        Full pipeline: load 鈫?preprocess 鈫?infer 鈫?postprocess.
        Cycles through available images if more than one.
        """
        logger.info(SEP2)
        logger.info("End-to-End Inference Benchmark")
        logger.info(SEP2)

        n_images = len(image_paths)
        logger.info(f"Images available: {n_images}")

        # Warm-up
        logger.info(f"Warming up ({min(10, num_iter // 10)} iterations) ...")
        for i in range(min(10, num_iter // 10)):
            self._run_e2e_single(image_paths[i % n_images])

        # Timed
        logger.info(f"Running {num_iter} timed iterations ...")
        times = []
        for i in tqdm(range(num_iter), desc='End-to-end'):
            t = self._run_e2e_single(image_paths[i % n_images])
            times.append(t)

        stats = _compute_stats(times, 'e2e')
        _print_stats_table(stats, 'End-to-End Inference')
        return stats

    def _run_e2e_single(self, img_path: Path) -> float:
        """Single end-to-end inference.  Returns seconds."""
        img_path_str = str(img_path)
        _cuda_sync(self.device)
        t0 = time.perf_counter()

        # Preprocess
        tensor, original = self.processor.preprocess(img_path_str)

        # Forward
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            with torch.no_grad():
                outputs = self.model(tensor.to(self.device))

        # Postprocess
        _ = self.processor.postprocess(
            outputs['final_mask'],
            original.shape[:2],
        )
        _cuda_sync(self.device)
        return time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Batch throughput
    # ------------------------------------------------------------------
    def benchmark_throughput(
        self,
        image_paths: List[Path],
        batch_sizes: List[int] = (1, 2, 4, 8, 16),
        num_iter: int = 50,
    ) -> List[Dict]:
        """Measure throughput (images/sec) for different batch sizes."""
        logger.info(SEP2)
        logger.info("Batch Throughput Benchmark")
        logger.info(SEP2)

        # Build reference tensor
        tensor_single, _ = self.processor.preprocess(image_paths[0])
        _, C, H, W = tensor_single.shape

        results = []
        for bs in batch_sizes:
            batch_tensor = tensor_single.repeat(bs, 1, 1, 1).to(self.device)
            # Warm-up
            for _ in range(min(10, num_iter // 5)):
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    with torch.no_grad():
                        _ = self.model(batch_tensor)
                _cuda_sync(self.device)

            # Timed
            times = []
            for _ in range(num_iter):
                _cuda_sync(self.device)
                t0 = time.perf_counter()
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    with torch.no_grad():
                        _ = self.model(batch_tensor)
                _cuda_sync(self.device)
                times.append(time.perf_counter() - t0)

            stats = {
                'batch_size': bs,
                'avg_time_s': float(np.mean(times)),
                'median_time_s': float(np.median(times)),
                'throughput_imgs': bs / float(np.mean(times)),
                'throughput_fps': bs / float(np.mean(times)),
            }
            results.append(stats)

        # Print table
        logger.info(f"  {'Batch':>6s}  {'Avg(ms)':>10s}  {'Median(ms)':>12s}  {'Imgs/s':>10s}")
        logger.info(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*10}")
        for r in results:
            logger.info(
                f"  {r['batch_size']:6d}  {r['avg_time_s']*1000:10.2f}  "
                f"{r['median_time_s']*1000:12.2f}  {r['throughput_imgs']:10.2f}"
            )
        return results


# ===========================================================================
# Statistics helpers
# ===========================================================================

def _compute_stats(times: List[float], label: str) -> Dict:
    arr = np.array(times)
    return {
        'test': label,
        'iterations': len(arr),
        'mean_s': float(np.mean(arr)),
        'median_s': float(np.median(arr)),
        'std_s': float(np.std(arr)),
        'min_s': float(np.min(arr)),
        'max_s': float(np.max(arr)),
        'fps': 1.0 / float(np.mean(arr)) if float(np.mean(arr)) > 0 else 0.0,
    }


def _print_stats_table(stats: Dict, title: str):
    logger.info(f"\n  {title}")
    logger.info(f"  {'-'*50}")
    logger.info(f"  Iterations  : {stats['iterations']}")
    logger.info(f"  Mean        : {stats['mean_s']*1000:.3f} ms  ({stats['fps']:.2f} FPS)")
    logger.info(f"  Median      : {stats['median_s']*1000:.3f} ms")
    logger.info(f"  Std         : {stats['std_s']*1000:.3f} ms")
    logger.info(f"  Min         : {stats['min_s']*1000:.3f} ms")
    logger.info(f"  Max         : {stats['max_s']*1000:.3f} ms")


# ===========================================================================
# Visualization
# ===========================================================================

class Visualizer:
    """Generate and save detection result figures."""

    def __init__(self, output_dir: Path, show: bool = False):
        if not HAS_MPL:
            raise ImportError(
                "matplotlib is required for visualisation.  Install with: pip install matplotlib"
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.show = show
        if show:
            matplotlib.use('TkAgg')  # switch to interactive

    def visualise_single(
        self,
        original: np.ndarray,           # (H, W, 3) RGB
        mask: np.ndarray,               # (H, W) uint8 [0, 255]
        filename: str,
        inference_time_s: float,
        restored: Optional[np.ndarray] = None,  # (H, W, 3) RGB restored image
    ) -> Path:
        """
        Create a multi-panel figure and save to disk.

        Layout (2脳2):
            [Original]          [Input Preview]
            [Predicted Mask]    [Overlay]
        """
        fps = 1.0 / inference_time_s if inference_time_s > 0 else 0.0

        # Determine number of panels
        has_restored = restored is not None
        ncols = 2
        nrows = 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10))
        axes = axes.flatten()

        # Panel 0: Original
        axes[0].imshow(original)
        axes[0].set_title(f'Original\n{original.shape[1]}脳{original.shape[0]}', fontsize=10)
        axes[0].axis('off')

        # Panel 1: Input preview
        if has_restored:
            axes[1].imshow(restored)
            axes[1].set_title('Input Preview', fontsize=10)
        else:
            axes[1].text(0.5, 0.5, 'No restored output',
                         ha='center', va='center', fontsize=14,
                         transform=axes[1].transAxes)
            axes[1].set_title('Input Preview', fontsize=10)
        axes[1].axis('off')

        # Panel 2: Predicted mask
        axes[2].imshow(mask, cmap='gray', vmin=0, vmax=255)
        axes[2].set_title('Predicted Mask', fontsize=10)
        axes[2].axis('off')

        # Panel 3: Overlay
        overlay = self._build_overlay(original, mask)
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay', fontsize=10)
        axes[3].axis('off')

        # Global title with timing info
        fig.suptitle(
            f'{filename}  |  {inference_time_s*1000:.1f} ms  |  {fps:.1f} FPS',
            fontsize=12, fontweight='bold',
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save
        save_name = f'{Path(filename).stem}_infer{inference_time_s*1000:.0f}ms.png'
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if self.show:
            logger.info(f"Saved: {save_path}")
        return save_path

    @staticmethod
    def _build_overlay(original_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Overlay mask (red) on original image.
        Falls back to pure-numpy blending if cv2 is unavailable.
        """
        mask_bin = (mask > 127).astype(np.uint8)
        if HAS_CV2:
            # cv2.addWeighted with red colourisation
            coloured = np.zeros_like(original_rgb)
            coloured[:, :, 2] = mask_bin * 255  # red channel
            overlay = cv2.addWeighted(original_rgb, 0.6, coloured, 0.4, 0)
            return overlay
        else:
            # NumPy fallback
            alpha = 0.4
            overlay = original_rgb.copy().astype(np.float32)
            overlay[mask_bin > 0, 0] = overlay[mask_bin > 0, 0] * (1 - alpha) + 255 * alpha  # R
            overlay[mask_bin > 0, 1] = overlay[mask_bin > 0, 1] * (1 - alpha)  # G
            overlay[mask_bin > 0, 2] = overlay[mask_bin > 0, 2] * (1 - alpha)  # B
            return np.clip(overlay, 0, 255).astype(np.uint8)


# ===========================================================================
# Main inference / test driver
# ===========================================================================

def run_inference_and_tests(args):
    """Orchestrate config loading, model setup, speed tests, and visualisation."""

    # --- 1. Load config & checkpoint ---------------------------------------
    logger.info(SEP)
    logger.info("C3Net Inference & Speed Test Tool")
    logger.info(SEP)

    config = _load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and device.type == 'cpu':
        logger.warning("CUDA requested but not available 鈥?falling back to CPU.")
    logger.info(f"Device: {device}  |  FP16: {args.fp16 and device.type == 'cuda'}")

    # --- 2. Build model ----------------------------------------------------
    model = build_model(config, args.checkpoint, device)

    # --- 3. Prepare processor & gather images ------------------------------
    processor = InferenceProcessor(config)
    image_paths = _gather_images(args.input)
    logger.info(f"Found {len(image_paths)} image(s) to process.")

    # --- 4. Speed benchmarks -----------------------------------------------
    speed_results = {}
    if args.mode in ('speed', 'both'):
        bench = SpeedBenchmark(model, processor, device, use_fp16=args.fp16)

        speed_results['pure_model'] = bench.benchmark_pure_model(
            image_paths, num_iter=args.num_test
        )
        speed_results['e2e'] = bench.benchmark_e2e(
            image_paths, num_iter=args.num_test
        )
        speed_results['throughput'] = bench.benchmark_throughput(
            image_paths,
            batch_sizes=[int(x.strip()) for x in args.batch_sizes.split(',')],
            num_iter=max(1, args.num_test // 2),
        )

    # --- 5. Visualisation --------------------------------------------------
    csv_rows = []
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ('visualize', 'both'):
        logger.info(SEP2)
        logger.info("Visualisation")
        logger.info(SEP2)
        vis = Visualizer(output_dir, show=args.show)

        for img_path in tqdm(image_paths, desc='Visualising'):
            try:
                # Preprocess
                tensor, original = processor.preprocess(img_path)

                # Forward
                _cuda_sync(device)
                t0 = time.perf_counter()
                with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == 'cuda'):
                    with torch.no_grad():
                        outputs = model(tensor.to(device))
                _cuda_sync(device)
                infer_time = time.perf_counter() - t0

                # Postprocess mask
                mask = processor.postprocess(
                    outputs['final_mask'], original.shape[:2]
                )

                # Save visualization
                out_path = vis.visualise_single(
                    original=original,
                    mask=mask,
                    filename=img_path.name,
                    inference_time_s=infer_time,
                )
                csv_rows.append({
                    'filename': img_path.name,
                    'inference_ms': infer_time * 1000.0,
                    'output': str(out_path),
                })
            except Exception as exc:
                logger.exception("Failed to visualise %s: %s", img_path, exc)

    if csv_rows:
        csv_path = output_dir / 'summary.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info("CSV summary saved: %s", csv_path)


def parse_args():
    parser = argparse.ArgumentParser(description='C3Net inference utility')
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--input', required=True, help='Input image file or directory')
    parser.add_argument('--output', default='runs/inference', help='Output directory')
    parser.add_argument('--mode', choices=['visualize', 'speed', 'both'], default='visualize')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision on CUDA')
    parser.add_argument('--show', action='store_true', help='Enable interactive matplotlib backend')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--runs', type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_inference_and_tests(args)


if __name__ == '__main__':
    main()
