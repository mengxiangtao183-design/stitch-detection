"""
Evaluator module for Camouflaged Object Detection.

This module provides:
1. Model evaluation on test datasets
2. Comprehensive metric computation and analysis
3. Quality-based result categorization and storage
4. Visual result organization and validation
5. Performance analysis and timing statistics

Processing Flow:
1. Model Loading and Warmup
   - Checkpoint loading
   - Device placement
   - Inference warmup runs

2. Batch Processing
   - Efficient data handling
   - Mixed precision inference
   - Multi-scale prediction resizing

3. Result Analysis
   - Metric computation
   - Quality categorization
   - Statistical aggregation

4. Result Organization
   - Category-based storage
   - Visual result generation
   - Performance summaries

Input Requirements:
    DataLoader providing batches with:
    - images: torch.Tensor[B, C, H, W]  # Normalized images
    - masks: List[torch.Tensor[1, H, W]] # Original size masks
    - names: List[str]                   # Image filenames
"""

import logging
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import cv2
import torch
import torch.nn.functional as F
import numpy
import torch.serialization

# 1. 娣诲姞瀹夊叏鍏ㄥ眬瀵硅薄锛堢敤numpy._core鏇挎崲搴熷純鐨刵umpy.core锛?#torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
import numpy as np
from tqdm import tqdm

from com_models.registry import build_model
from utils.checkpoint_utils import load_model_checkpoint
# from com_models.OPNet_model import OPNet_Wrapper as OPNet
# from com_models.fselnet_model import fselnet_Wrapper as fselnet
# from com_models.sam2_model import SAM2_Wrapper as SAM2
from utils.metrics import MetricsProcessor
from utils.visualization import ResultVisualizer
from utils.run_manager import DirectoryManager
from utils.checkpoint_utils import normalize_state_dict_keys

@dataclass
class EvaluationResult:
    """
    Container for single sample evaluation results.

    Attributes:
        filename (str): Source image filename
        seg_pred (torch.Tensor): Final segmentation prediction
        seg_stages (List[torch.Tensor]): Multi-scale predictions
        metrics (Dict[str, float]): Performance metrics
        original_image (np.ndarray): Input image for visualization
    """
    filename: str
    preds: Dict[str, np.ndarray]  # Resized final prediction
    metrics: Dict[str, float]
    original_image: np.ndarray

class ResultManager:
    """
    Manages evaluation result storage and organization.
    
    Features:
    - Quality-based categorization (good/medium/bad)
    - Result visualization management
    - Metric logging and organization
    - Directory structure maintenance
    """
    
    def __init__(self, dir_manager: DirectoryManager):
        """
        Initialize result management system.
        
        Args:
            dir_manager: Provides directory structure for results
                    
        Directory Structure:
            visualizations/
            鈹溾攢鈹€ good/
            鈹?  鈹溾攢鈹€ segmentation/
            鈹?  鈹斺攢鈹€ edges/
            鈹溾攢鈹€ medium/
            鈹?  鈹溾攢鈹€ segmentation/
            鈹?  鈹斺攢鈹€ edges/
            鈹斺攢鈹€ bad/
                鈹溾攢鈹€ segmentation/
                鈹斺攢鈹€ edges/
        """
        # Store directory manager reference
        self.run_dirs = dir_manager.run_dirs
        self.dataset_dirs = {}  # Store dataset-specific directories
        
        # Log initialization
        logging.info("Initialized result directory structure")
    
    def setup_dataset_directories(self, dataset_name: str):
        """Create directory structure for a specific dataset."""
        dataset_root = self.run_dirs.root / dataset_name
        
        # Create visualization directories
        viz_dir = dataset_root / 'visualizations'
        # Create metrics directory
        metrics_dir = dataset_root / 'metrics'

        for category in ['good', 'medium', 'bad']:
            (viz_dir / category / 'segmentation').mkdir(parents=True, exist_ok=True)
            (viz_dir / category / 'edges').mkdir(parents=True, exist_ok=True)
            (metrics_dir / category).mkdir(parents=True, exist_ok=True)
            
        # Store paths for this dataset
        self.dataset_dirs[dataset_name] = {
            'root': dataset_root,
            'visualizations': viz_dir,
            'metrics': metrics_dir
        }

    def determine_quality_category(self, metrics: Dict[str, float]) -> str:
        """
        Determine result quality.

        For stitch trajectory detection, tolerance-aware trajectory F1 is the
        primary quality signal. Fall back to the classic COD S/F measures when
        older metric dictionaries do not include trajectory metrics.
        
        Args:
            metrics: Dictionary containing evaluation metrics
            
        Returns:
            Category string ('good', 'medium', or 'bad')
        """
        trajectory_f1 = metrics.get('trajectory_f1')
        if trajectory_f1 is not None:
            if trajectory_f1 >= 0.90:
                return 'good'
            if trajectory_f1 >= 0.75:
                return 'medium'
            return 'bad'

        s_alpha = metrics['s_alpha']
        weighted_f = metrics['weighted_f']
        
        # Both measures must meet threshold for category
        if s_alpha >= 0.8 and weighted_f >= 0.8:
            return 'good'
        elif s_alpha >= 0.6 and weighted_f >= 0.6:
            return 'medium'
        return 'bad'
    
    def save_prediction(self, result: EvaluationResult, dataset_name: str) -> None:
        """
        Save prediction results in organized structure.
        
        Organization:
        1. Quality-based categorization
        2. Separate storage for segmentation and edges
        3. Multi-scale visualization generation
        4. Metric data storage
        
        Args:
            result: Evaluation data container
            dataset_name: Current dataset identifier
            
        Directory Structure:
            dataset/
            鈹溾攢鈹€ visualizations/
            鈹?  鈹溾攢鈹€ good/
            鈹?  鈹溾攢鈹€ medium/
            鈹?  鈹斺攢鈹€ bad/
            鈹斺攢鈹€ metrics/
        """

        # Ensure dataset directories exist
        if dataset_name not in self.dataset_dirs:
            self.setup_dataset_directories(dataset_name)
            
        # Determine quality category
        category = self.determine_quality_category(result.metrics)        
        # Get paths for this dataset
        viz_dir = self.dataset_dirs[dataset_name]['visualizations']
        # Save predictions in appropriate category
        category_dir = viz_dir / category
        
        try:
            # Save segmentation visualizations
            seg_vis = ResultVisualizer(category_dir / 'segmentation')
            seg_preds = {}
            edge_pred = {}
            for key, pred in result.preds.items():
                if key != 'edge_mask':
                    seg_preds[key] = pred
                else:
                    edge_pred[key] = pred
            # Save all visualizations including stages
            seg_vis.save_all_visualizations(
                result.filename,
                seg_preds,
                result.original_image,
            )
        
            
            # Save edge visualizations
            edge_vis = ResultVisualizer(category_dir / 'edges')
            edge_vis.save_all_visualizations(
                result.filename,
                edge_pred,
                result.original_image
            )
            
            # Save metrics
            metrics_file = self.dataset_dirs[dataset_name]['metrics'] / category / f'{result.filename}_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(result.metrics, f, indent=4)
                
        except Exception as e:
            logging.error(f"Failed to save prediction {result.filename}: {str(e)}")
            
            
            
    def get_category_summary(self, dataset_name: str) -> Dict[str, Dict[str, int]]:
        """
        Generate summary of prediction distribution across categories for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset to summarize
            
        Returns:
            Dictionary containing:
                counts: Number of predictions in each category
                total: Total number of predictions
                
        Example:
            {
                'counts': {'good': 150, 'medium': 80, 'bad': 20},
                'total': 250
            }
        """
        if dataset_name not in self.dataset_dirs:
            raise ValueError(f"Dataset {dataset_name} not found in results")
            
        summary = {'counts': {}, 'total': 0}
        metrics_dir = self.dataset_dirs[dataset_name]['metrics']
        
        # Count predictions in each category
        for category in ['good', 'medium', 'bad']:
            # Metrics files are one-per-sample and avoid counting auxiliary
            # masks such as object_mask as separate samples.
            metric_files = list((metrics_dir / category).glob('*_metrics.json'))
            count = len(metric_files)
                
            summary['counts'][category] = count
            summary['total'] += count
            
        return summary


class Evaluator:
    """
    C3Net Camouflaged Object Detection evaluator.
    
    This class handles the complete evaluation pipeline:
    1. Model loading and initialization
    2. Batch processing and inference
    3. Metric computation and tracking
    4. Result organization and storage
    
    Features:
    - Efficient batch processing
    - Comprehensive timing analysis
    - Quality-based result categorization
    - Memory-optimized prediction handling
    
    The evaluation process tracks multiple metrics:
    - Structure measure (S伪)
    - Weighted F-measure (F尾w)
    - Mean Absolute Error (M)
    - Enhanced-alignment measure (E蠒)
    - Mean F-measure (F尾m)
    """
    
    def __init__(
        self,
        model_path: str,
        dir_manager: DirectoryManager,
        model_config: Dict,
        device: torch.device,
        batch_size: int,
        branch_diagnostics: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize evaluator components.
        
        Args:
            model_path: Path to model checkpoint (.pth file)
            dir_manager: Directory manager for result organization
            model_config: Dictionary containing model configuration
            
        Components Initialized:
        1. Model: C3Net loaded from checkpoint
        2. Metrics Calculator: For performance evaluation
        3. Result Manager: For output organization
        4. Timing Tracker: For performance analysis
        """
        # Setup device
        self.device = device
        self.model_config = model_config
        self.branch_diagnostics_cfg = branch_diagnostics or {}
        self.branch_diagnostics_enabled = bool(
            self.branch_diagnostics_cfg.get('enable', False)
        )
        self.branch_diagnostics_threshold = float(
            self.branch_diagnostics_cfg.get('threshold', 0.5)
        )
        self.branch_diagnostics_max_samples = max(
            0,
            int(self.branch_diagnostics_cfg.get('max_visualizations_per_dataset', 24)),
        )
        self._branch_diagnostics: Dict[str, Dict[str, Any]] = {}
        logging.info(f"Using device: {self.device}")
        if self.branch_diagnostics_enabled:
            logging.info(
                "Branch diagnostics enabled (threshold=%.3f, max_samples=%d)",
                self.branch_diagnostics_threshold,
                self.branch_diagnostics_max_samples,
            )
        
        try:
            # Initialize model
            self.model = self._load_model(model_path, model_config, batch_size)
            logging.info(f"Model loaded from: {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise
            
        # Initialize components
        self.result_manager = ResultManager(dir_manager)
        self.metrics_processor = MetricsProcessor()
        
        # Initialize timing statistics
        self.timing_stats = {
            'inference_times': [],    # Individual inference times
            'processing_times': [],   # Total batch processing times
            'total_time': 0          # Complete evaluation time
        }
        
    def _load_model(self, model_path: str, model_config: Dict, batch_size: int) -> torch.nn.Module:
       """
       Load C3Net model from checkpoint and prepare for evaluation.
       
       Args:
           model_path: Path to model checkpoint (.pth file)
           
       Returns:
           Loaded model in evaluation mode
           
       Raises:
           RuntimeError: If model loading fails
           FileNotFoundError: If checkpoint file not found
       """
       try:
            # Initialize model architecture
            model = build_model(model_config)

            # Load checkpoint
            checkpoint = load_model_checkpoint(model_path, map_location=self.device)
            state_dict = normalize_state_dict_keys(checkpoint['model_state_dict'])
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logging.warning(f"Missing keys when loading evaluation checkpoint: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys when loading evaluation checkpoint: {unexpected_keys}")

            # Prepare for evaluation
            model = model.to(self.device)
            model.eval()

            # Proper warm-up with multiple runs and synchronization
            target_size = model_config['image_processing']['target_size']
            dummy_input = torch.randn(batch_size, 3, target_size, target_size, device=self.device)

            # Multiple warm-up runs
            with torch.inference_mode():
                for _ in range(3):  # Usually 2-3 runs is enough
                    _ = model(dummy_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()  # Ensure GPU completion
                
            return model
           
       except FileNotFoundError:
           logging.error(f"Model checkpoint not found: {model_path}")
           raise
       except Exception as e:
           logging.error(f"Error loading model: {str(e)}")
           raise RuntimeError(f"Failed to load model: {str(e)}")
       
    def _denormalize_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Denormalize image tensor back to 0-255 range.
        
        Args:
            image_tensor: Input image tensor [C, H, W] normalized with ImageNet stats
            
        Returns:
            numpy array [H, W, C] with values in [0, 255]
        """
        mean = torch.tensor(self.model_config['image_processing']['normalize_mean']).view(3, 1, 1).to(image_tensor.device)
        std = torch.tensor(self.model_config['image_processing']['normalize_std']).view(3, 1, 1).to(image_tensor.device)
        
        # Denormalize
        denorm_image = image_tensor * std + mean
        
        # Convert to numpy and correct format
        image_np = (denorm_image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        return image_np

    @staticmethod
    def _empty_binary_stats() -> Dict[str, int]:
        """Create pixel counters used by the optional branch diagnostic."""
        return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

    @staticmethod
    def _update_binary_stats(
        stats: Dict[str, int], pred: np.ndarray, target: np.ndarray
    ) -> None:
        stats['tp'] += int(np.logical_and(pred, target).sum())
        stats['fp'] += int(np.logical_and(pred, ~target).sum())
        stats['fn'] += int(np.logical_and(~pred, target).sum())
        stats['tn'] += int(np.logical_and(~pred, ~target).sum())

    @staticmethod
    def _summarize_binary_stats(stats: Dict[str, int]) -> Dict[str, float]:
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        precision = 1.0 if tp + fp == 0 and tp + fn == 0 else tp / max(tp + fp, 1)
        recall = 1.0 if tp + fn == 0 else tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        iou = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'iou': float(iou),
            'tp_pixels': int(tp),
            'fp_pixels': int(fp),
            'fn_pixels': int(fn),
            'tn_pixels': int(stats['tn']),
        }

    @staticmethod
    def _extract_boundary(mask: np.ndarray) -> np.ndarray:
        """Match the project's 1-pixel morphological boundary convention."""
        mask_u8 = mask.astype(np.uint8)
        if int(mask_u8.sum()) == 0:
            return np.zeros_like(mask_u8, dtype=bool)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return (mask_u8 - cv2.erode(mask_u8, kernel, iterations=1)) > 0

    def _init_branch_diagnostics(self, dataset_name: str) -> None:
        if not self.branch_diagnostics_enabled:
            return
        self._branch_diagnostics[dataset_name] = {
            'branches': {
                'loc': self._empty_binary_stats(),
                'edge_boundary': self._empty_binary_stats(),
                'afm': self._empty_binary_stats(),
                'highres_final': self._empty_binary_stats(),
            },
            'error_support': {
                'final_fp_pixels': 0,
                'final_fn_pixels': 0,
                'fp_highres_added_pixels': 0,
                'fp_afm_carried_pixels': 0,
                'fp_loc_supported_pixels': 0,
                'fp_edge_supported_pixels': 0,
                'fp_no_branch_support_pixels': 0,
                'fn_loc_missed_pixels': 0,
                'fn_edge_missed_pixels': 0,
                'fn_afm_missed_pixels': 0,
                'fn_lost_after_afm_pixels': 0,
            },
            'samples': [],
            'num_images': 0,
        }

    @staticmethod
    def _uint8_probability(probability: np.ndarray) -> np.ndarray:
        return np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _record_branch_diagnostic(
        self,
        dataset_name: str,
        filename: str,
        original_image: np.ndarray,
        gt: np.ndarray,
        probabilities: Dict[str, np.ndarray],
    ) -> None:
        """Accumulate read-only branch evidence and retain the worst samples."""
        state = self._branch_diagnostics[dataset_name]
        threshold = self.branch_diagnostics_threshold
        gt_bin = gt.astype(bool)
        if original_image.shape[:2] != gt_bin.shape:
            original_image = cv2.resize(
                original_image,
                (gt_bin.shape[1], gt_bin.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        branch_bin = {
            key: value >= threshold for key, value in probabilities.items()
        }

        self._update_binary_stats(state['branches']['loc'], branch_bin['loc'], gt_bin)
        self._update_binary_stats(
            state['branches']['edge_boundary'],
            branch_bin['edge'],
            self._extract_boundary(gt_bin),
        )
        self._update_binary_stats(state['branches']['afm'], branch_bin['afm'], gt_bin)
        self._update_binary_stats(
            state['branches']['highres_final'], branch_bin['final'], gt_bin
        )

        final_fp = branch_bin['final'] & ~gt_bin
        final_fn = ~branch_bin['final'] & gt_bin
        support = state['error_support']
        support['final_fp_pixels'] += int(final_fp.sum())
        support['final_fn_pixels'] += int(final_fn.sum())
        # These are evidence/support categories, not causal claims.  A pixel
        # can be supported by more than one earlier branch.
        support['fp_highres_added_pixels'] += int((final_fp & ~branch_bin['afm']).sum())
        support['fp_afm_carried_pixels'] += int((final_fp & branch_bin['afm']).sum())
        support['fp_loc_supported_pixels'] += int((final_fp & branch_bin['loc']).sum())
        support['fp_edge_supported_pixels'] += int((final_fp & branch_bin['edge']).sum())
        support['fp_no_branch_support_pixels'] += int(
            (final_fp & ~branch_bin['afm'] & ~branch_bin['loc'] & ~branch_bin['edge']).sum()
        )
        support['fn_loc_missed_pixels'] += int((final_fn & ~branch_bin['loc']).sum())
        support['fn_edge_missed_pixels'] += int((final_fn & ~branch_bin['edge']).sum())
        support['fn_afm_missed_pixels'] += int((final_fn & ~branch_bin['afm']).sum())
        support['fn_lost_after_afm_pixels'] += int((gt_bin & branch_bin['afm'] & ~branch_bin['final']).sum())
        state['num_images'] += 1

        if self.branch_diagnostics_max_samples <= 0:
            return
        final_intersection = int((branch_bin['final'] & gt_bin).sum())
        final_union = int((branch_bin['final'] | gt_bin).sum())
        sample_iou = 1.0 if final_union == 0 else final_intersection / final_union
        candidate = {
            'filename': filename,
            'iou': float(sample_iou),
            'error_pixels': int(final_fp.sum() + final_fn.sum()),
            'original_image': original_image.copy(),
            'gt': gt_bin.astype(np.uint8),
            'maps': {
                key: self._uint8_probability(value) for key, value in probabilities.items()
            },
        }
        state['samples'].append(candidate)
        state['samples'].sort(key=lambda item: (item['iou'], -item['error_pixels'], item['filename']))
        del state['samples'][self.branch_diagnostics_max_samples:]

    @staticmethod
    def _diagnostic_panel(image: np.ndarray, title: str) -> np.ndarray:
        panel = image.copy()
        cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        return panel

    def _save_branch_diagnostic_samples(self, root: Path, samples: List[Dict[str, Any]]) -> None:
        samples_dir = root / 'samples'
        samples_dir.mkdir(parents=True, exist_ok=True)
        for rank, sample in enumerate(samples, start=1):
            rgb = sample['original_image']
            base_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            gt = sample['gt'].astype(bool)
            maps = sample['maps']
            final = maps['final'] >= int(round(self.branch_diagnostics_threshold * 255.0))
            loc = maps['loc'] >= int(round(self.branch_diagnostics_threshold * 255.0))
            edge = maps['edge'] >= int(round(self.branch_diagnostics_threshold * 255.0))
            afm = maps['afm'] >= int(round(self.branch_diagnostics_threshold * 255.0))

            gt_panel = cv2.cvtColor((gt.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
            error_panel = base_bgr.copy()
            tp = final & gt
            fp = final & ~gt
            fn = ~final & gt
            error_panel[tp] = (0, 180, 0)       # green: correct foreground
            error_panel[fp] = (0, 0, 255)       # red: false positive
            error_panel[fn] = (255, 0, 0)       # blue: false negative

            source_panel = np.zeros_like(base_bgr)
            source_panel[final & ~gt & ~afm] = (0, 0, 255)          # HighRes-added FP
            source_panel[final & ~gt & afm & ~loc] = (0, 165, 255)  # AFM carried, no loc support
            source_panel[final & ~gt & afm & loc] = (255, 0, 255)   # loc-supported FP
            source_panel[final & ~gt & afm & edge] = (0, 255, 255)  # edge-supported FP
            source_panel[~final & gt & ~loc] = (255, 0, 0)          # loc-missed FN
            source_panel[~final & gt & loc] = (180, 0, 0)           # later-stage FN

            panels = [
                self._diagnostic_panel(base_bgr, 'RGB'),
                self._diagnostic_panel(gt_panel, 'GT'),
                self._diagnostic_panel(error_panel, 'Final: TP green / FP red / FN blue'),
                self._diagnostic_panel(source_panel, 'Error evidence'),
            ]
            for key, title in [('loc', 'Loc probability'), ('edge', 'Edge probability'), ('afm', 'AFM probability'), ('final', 'HighRes final probability')]:
                heat = cv2.applyColorMap(maps[key], cv2.COLORMAP_TURBO)
                panels.append(self._diagnostic_panel(heat, title))
            mosaic = np.vstack([
                np.hstack(panels[:4]),
                np.hstack(panels[4:]),
            ])
            stem = Path(sample['filename']).stem
            safe_stem = ''.join(char if char.isalnum() or char in '._-' else '_' for char in stem)
            output_path = samples_dir / f'{rank:02d}_iou{sample["iou"]:.3f}_{safe_stem}.png'
            cv2.imwrite(str(output_path), mosaic)

    def _finalize_branch_diagnostics(self, dataset_name: str) -> None:
        if not self.branch_diagnostics_enabled:
            return
        state = self._branch_diagnostics[dataset_name]
        root = self.result_manager.run_dirs.root / 'branch_diagnostics' / dataset_name
        root.mkdir(parents=True, exist_ok=True)
        support = state['error_support'].copy()
        total_fp = max(support['final_fp_pixels'], 1)
        total_fn = max(support['final_fn_pixels'], 1)
        support.update({
            'fp_highres_added_ratio_of_final_fp': support['fp_highres_added_pixels'] / total_fp,
            'fp_afm_carried_ratio_of_final_fp': support['fp_afm_carried_pixels'] / total_fp,
            'fp_loc_supported_ratio_of_final_fp': support['fp_loc_supported_pixels'] / total_fp,
            'fp_edge_supported_ratio_of_final_fp': support['fp_edge_supported_pixels'] / total_fp,
            'fp_no_branch_support_ratio_of_final_fp': support['fp_no_branch_support_pixels'] / total_fp,
            'fn_loc_missed_ratio_of_final_fn': support['fn_loc_missed_pixels'] / total_fn,
            'fn_edge_missed_ratio_of_final_fn': support['fn_edge_missed_pixels'] / total_fn,
            'fn_afm_missed_ratio_of_final_fn': support['fn_afm_missed_pixels'] / total_fn,
            'fn_lost_after_afm_ratio_of_final_fn': support['fn_lost_after_afm_pixels'] / total_fn,
        })
        summary = {
            'description': (
                'Read-only branch evidence diagnostic. Support ratios show whether a final error '
                'was also active in an earlier branch; they must not be interpreted as causal attribution.'
            ),
            'threshold': self.branch_diagnostics_threshold,
            'num_images': state['num_images'],
            'branch_binary_metrics': {
                name: self._summarize_binary_stats(stats)
                for name, stats in state['branches'].items()
            },
            'final_error_support': support,
            'saved_worst_samples': [
                {
                    'filename': item['filename'],
                    'binary_iou': item['iou'],
                    'final_error_pixels': item['error_pixels'],
                }
                for item in state['samples']
            ],
        }
        with open(root / 'summary.json', 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        self._save_branch_diagnostic_samples(root, state['samples'])
        logging.info("Branch diagnostic saved to: %s", root)
    
    def evaluate(self, test_loader: torch.utils.data.DataLoader, dataset_name: str) -> Dict[str, float]:
        """
        Evaluate model on test dataset.
        
        Workflow:
        1. Process batches with timing analysis
        2. Compute metrics for each prediction
        3. Save results organized by quality
        4. Track timing statistics
        5. Generate evaluation summary
        
        Args:
            dataset_name: Name of the test dataset
            
        Returns:
            Dictionary of averaged evaluation metrics:
            - s_alpha: Structure measure
            - weighted_f: Weighted F-measure
            - mae: Mean Absolute Error
            - e_phi: Enhanced-alignment measure
            - mean_f: Mean F-measure
            
        Performance Tracking:
        - Per-batch inference time
        - Total processing time
        - Memory usage monitoring
        """
        # Initialize metric tracking. Metrics are accumulated dynamically so
        # trajectory-specific fields can be added without special plumbing.
        total_metrics = {}
        
        # Setup directories for this dataset
        self.result_manager.setup_dataset_directories(dataset_name)
        self._init_branch_diagnostics(dataset_name)

        # Start evaluation timing
        eval_start = time.time()
        logging.info("Starting evaluation...")
        
        total_samples_processed = 0

        try:
            # Process each batch
            progress = tqdm(test_loader, desc=f'Evaluating {dataset_name}')
            for batch_idx, batch in enumerate(progress):
                batch_metrics, batch_size = self._process_batch(dataset_name, batch)
                
                # Update total metrics
                for k, v in batch_metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v
                
                # Update total samples processed
                total_samples_processed += batch_size
                    
                # Update progress bar with running averages
                self._update_progress(progress, total_metrics, total_samples_processed)
                    
            # Compute final averaged metrics
            avg_metrics = {k: v/total_samples_processed for k, v in total_metrics.items()}
            
            # Save timing statistics
            self.timing_stats['total_time'] = time.time() - eval_start
            self.timing_stats['total_samples'] = total_samples_processed
            self._save_evaluation_summary(dataset_name, avg_metrics)
            self._finalize_branch_diagnostics(dataset_name)
            
            return avg_metrics
            
        except Exception as e:
            logging.error(f"Evaluation failed: {str(e)}")
            raise

    def _process_batch(self, dataset_name: str, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], int]:
        """
        Process single batch of test samples.
        
        Processing Steps:
        1. Data Preparation
        - Device transfer
        - Input normalization
        
        2. Model Inference
        - Mixed precision prediction
        - Timing measurement
        
        3. Result Processing
        - Prediction resizing
        - Metric computation
        - Result storage
        
        Args:
            dataset_name: Name of current dataset
            batch: Data dictionary containing:
                - images: Input tensor [B, C, H, W]
                - masks: Ground truth list
                - names: Filename list
                
        Returns:
            tuple: (batch_metrics, batch_size)
                - batch_metrics: Accumulated metrics
                - batch_size: Number of processed samples
                
        Note:
            Processes samples individually to optimize memory usage
        """
        # Start batch timing
        batch_start = time.time()
        
        # Prepare inputs
        images = batch['images'].to(self.device)
        masks = [m.to(self.device) for m in batch['masks']]
        filenames = batch['names']
        batch_size = len(filenames)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()  # Ensure data transfer is complete
        
        batch_metrics = {}
        
        # Model inference with per-image timing. CUDA operations are
        # asynchronous, so a synchronization is required after the forward
        # pass; otherwise the recorded value only measures CPU launch time.
        with torch.inference_mode():
            inference_start = time.perf_counter()
            # External comparison models implement the compact ``forward(x)``
            # contract.  Only C3Net branch diagnostics require the optional
            # intermediate-output argument.
            if self.branch_diagnostics_enabled:
                outputs = self.model(images, return_intermediates=True)
            else:
                outputs = self.model(images)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            self.timing_stats['inference_times'].append(
                (time.perf_counter() - inference_start) / max(batch_size, 1)
            )

            processed_output = {}
            # Process all mask outputs
            for key, preds in outputs.items():
                # Structural auxiliary heads are training supervision only and
                # may have two direction channels, so they are intentionally
                # excluded from image visualizations and test metrics.
                if key in {
                    'pattern_center', 'pattern_direction', 'pattern_spacing',
                    'sdf_mask', 'contour_logits', 'instance_logits',
                    'afm_coarse_mask', 'afm_aux_mask', 'weak_stitch_mask',
                    'weak_stitch_consistency_mask',
                }:
                    continue
                if preds.dim() >= 4 and preds.shape[1] != 1:
                    continue
                processed_preds = [F.interpolate(
                    preds[i:i+1], size=masks[i].shape[-2:], mode='bilinear', align_corners=False
                    ).squeeze(0) for i, _ in enumerate(preds)]
                processed_output[key] = processed_preds
            

            for idx, filename in enumerate(filenames):
                # Get original image for visualization
                original_image = self._denormalize_image(images[idx])
                

                file_preds = {k: v[idx] for k, v in processed_output.items()}

                if self.branch_diagnostics_enabled:
                    required_keys = {'final_mask', 'object_mask', 'edge_mask', 'afm_mask'}
                    missing_keys = required_keys.difference(file_preds)
                    if missing_keys:
                        raise RuntimeError(
                            "Branch diagnostics requested but the model did not return: "
                            f"{sorted(missing_keys)}"
                        )
                    probabilities = {
                        'final': file_preds['final_mask'].sigmoid().detach().cpu().numpy().squeeze(0),
                        'loc': file_preds['object_mask'].sigmoid().detach().cpu().numpy().squeeze(0),
                        'edge': file_preds['edge_mask'].sigmoid().detach().cpu().numpy().squeeze(0),
                        'afm': file_preds['afm_mask'].sigmoid().detach().cpu().numpy().squeeze(0),
                    }
                    gt_binary = (masks[idx].detach().cpu().numpy().squeeze(0) >= 0.5)
                    self._record_branch_diagnostic(
                        dataset_name=dataset_name,
                        filename=filename,
                        original_image=original_image,
                        gt=gt_binary,
                        probabilities=probabilities,
                    )

                # Compute metrics (only segmentation metrics, no edge metrics)
                sample_metrics = self.metrics_processor.compute_metrics(
                    seg_pred=[file_preds['final_mask']],
                    seg_gt=[masks[idx]],
                    include_trajectory_metrics=True
                )

                viz_preds = {
                    key: value.sigmoid().detach().cpu().numpy().squeeze(0)
                    if torch.is_tensor(value) else np.asarray(value).squeeze(0)
                    for key, value in file_preds.items()
                }

                # Update batch metrics
                for k, v in sample_metrics.items():
                    batch_metrics[k] = batch_metrics.get(k, 0.0) + v

                # Save results
                result = EvaluationResult(
                    filename=filename,
                    preds=viz_preds,
                    metrics=sample_metrics,
                    original_image=original_image
                )
                self.result_manager.save_prediction(result, dataset_name)
                
        # Track batch processing time
        self.timing_stats['processing_times'].append(
            time.time() - batch_start
        )
        
        return batch_metrics, batch_size

    def _update_progress(self, progress_bar: tqdm, total_metrics: Dict[str, float], total_samples: int) -> None:
        """Update progress bar with running averages."""
        avg_inference = np.mean(self.timing_stats['inference_times'][-10:]) * 1000
        
        # Compute running averages
        running_metrics = {k: v/total_samples for k, v in total_metrics.items()}
        
        progress_bar.set_postfix({
            'S-measure': f"{running_metrics['s_alpha']:.3f}",
            'F-measure': f"{running_metrics['weighted_f']:.3f}",
            'Traj-F': f"{running_metrics.get('trajectory_f1', 0.0):.3f}",
            'Inf(ms)': f"{avg_inference:.1f}"
        })

    def _save_evaluation_summary(self, dataset_name: str, metrics: Dict[str, float]):
        """
        Save comprehensive evaluation summary.
        
        Args:
            metrics: Dictionary of evaluation metrics
            
        Saves:
        1. Final averaged metrics
        2. Timing statistics
        3. Category distribution
        4. Performance analysis
        """
        # Prepare timing summary
        timing_summary = {
            'total_time': self.timing_stats['total_time'],
            'avg_inference_time': np.mean(self.timing_stats['inference_times']),
            'avg_processing_time': np.mean(self.timing_stats['processing_times']),
            'total_samples': self.timing_stats['total_samples']
        }
        
        # Get category distribution
        category_summary = self.result_manager.get_category_summary(dataset_name)
        
        # Save comprehensive summary
        summary = {
            'metrics': metrics,
            'timing': timing_summary,
            'categories': category_summary
        }
        
        # Save in dataset directory
        summary_file = self.result_manager.dataset_dirs[dataset_name]['root'] / 'evaluation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)

        # Log final results
        self._log_final_results(dataset_name, summary)

    def _log_final_results(self, dataset_name: str, summary: Dict) -> None:
        """
        Log final evaluation results.
        
        Args:
            summary: Dictionary containing evaluation summary
        """
        metrics = summary['metrics']
        timing = summary['timing']
        categories = summary['categories']
        
        logging.info(f"\nEvaluation Results for {dataset_name}:")
        logging.info("-------------------")
        logging.info(f"Total samples: {timing['total_samples']}")
        logging.info(f"Total time: {timing['total_time']:.2f}s")
        logging.info(f"Average inference time: {timing['avg_inference_time']*1000:.2f}ms")
        
        logging.info("\nMetrics:")
        logging.info(f"S-measure (S伪): {metrics['s_alpha']:.4f}")
        logging.info(f"Weighted F-measure (F尾w): {metrics['weighted_f']:.4f}")
        logging.info(f"Mean Absolute Error (M): {metrics['mae']:.4f}")
        logging.info(f"Enhanced-alignment (E_phi): {metrics['e_phi']:.4f}")
        logging.info(f"Mean F-measure (F尾m): {metrics['mean_f']:.4f}")
        
        if 'trajectory_f1' in metrics:
            logging.info(f"Trajectory F1 @2px: {metrics['trajectory_f1']:.4f}")
            logging.info(f"Trajectory F1 @1px: {metrics['trajectory_f1_1px']:.4f}")
            logging.info(f"Trajectory F1 @3px: {metrics['trajectory_f1_3px']:.4f}")
            logging.info(f"Binary IoU @0.5: {metrics['binary_iou']:.4f}")
            logging.info(f"Binary F1 @0.5: {metrics['binary_f1']:.4f}")

        logging.info("\nQuality Distribution:")
        total = categories['total']
        for category, count in categories['counts'].items():
            percentage = (count / total) * 100 if total > 0 else 0.0
            logging.info(f"{category.capitalize()}: {count} ({percentage:.1f}%)")
