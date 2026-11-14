"""
Performance Metrics Processor for C3Net Camouflaged Object Detection.

This module implements the standard evaluation metrics used in the C3Net paper (Section 4)
for both training monitoring and benchmark evaluation. The implementation uses the PySODMetrics 
library (https://github.com/lartpang/PySODMetrics) which provides optimized implementations
of standard COD evaluation metrics used by the research community.

PySODMetrics Integration:
   We use this library because it provides:
   1. Standard implementations used in COD research for fair comparisons
   2. Consistent results with published benchmarks
   
   The library provides five core metrics:
   - Structure measure (Sα): Via Smeasure class
   - Enhanced-alignment (Eφ): Via Emeasure class 
   - Weighted F-measure (Fβw): Via WeightedFmeasure class
   - Mean Absolute Error (M): Via MAE class
   - Mean F-measure (Fβm): Via Fmeasure class

Standard Metrics (as per COD benchmarks):
1. Structure measure (Sα): Evaluates structural similarity considering both region
  and object-aware structural similarity. Critical for COD as it captures how well
  the prediction preserves object structure.

2. Enhanced-alignment measure (Eφ): More sensitive to region-level matching quality.
  Important for camouflaged objects where boundaries may be ambiguous.

3. Weighted F-measure (Fβw): Applies adaptive weighting to precision and recall.
  Better handles the foreground-background imbalance common in COD.

4. Mean Absolute Error (M): Average pixel-wise prediction error.
  Provides a straightforward measure of prediction accuracy.

5. Mean F-measure (Fβm): Traditional F-measure averaged across thresholds.
  Ensures robust evaluation across different operating points.

Performance Optimizations:
- Parallel metric computation using CPU process pool
- Efficient GPU-CPU data transfer with async operations
- Batched tensor operations on GPU side
- Memory-efficient tensor handling
- Smart data movement to minimize copying

Usage Contexts:
1. Training Monitoring:
  - Real-time performance tracking
  - Lightweight computation for quick feedback
  - Focus on segmentation metrics

2. Benchmark Evaluation:
  - Comprehensive metric computation
  - Both segmentation and edge detection evaluation
  - Full statistical analysis

Example Usage:
   # Initialize processor
   metrics = MetricsProcessor()
   
   # Training monitoring
   metrics_dict = metrics.compute_metrics(
       seg_pred=model_output['predictions'][-1],  # Final scale prediction
       seg_gt=batch['masks']
   )
   
   # Full evaluation with edge detection
   metrics_dict = metrics.compute_metrics(
       seg_pred=model_output['predictions'][-1],
       seg_gt=batch['masks'],
       edge_pred=model_output['edge'],
       edge_gt=batch['edges']
   )

Hardware Notes:
   - Optimized for H100 GPU (93.6GB VRAM)
   - Efficient utilization of 64 CPU cores
   - Process pool handles batch size 42 optimally
   - Memory management tuned for 193GB system RAM
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Optional, Union, Tuple
from py_sod_metrics import (
    Smeasure, Emeasure, WeightedFmeasure, MAE, Fmeasure
)
from multiprocessing import Pool, cpu_count
import torch.cuda.amp as amp

class MetricsProcessor:
    """
    Unified metrics computation for C3Net training and evaluation.
    
    This processor handles performance measurement in both training and evaluation
    contexts, computing the five standard COD metrics efficiently. It optimizes
    computation by:
    
    1. GPU-side Operations:
       - Sigmoid activation
       - Tensor quantization
       - Batch processing
       
    2. CPU Parallel Processing:
       - Multi-process metric computation
       - Efficient numpy operations
       - Memory-optimized data handling
       
    3. Data Movement:
       - Asynchronous GPU-CPU transfers
       - Batched memory operations
       - Minimal data copying
       
    Hardware Utilization:
       - Optimized for H100 GPU (93.6GB VRAM)
       - Efficient use of 64 CPU cores
       - Handles batch size 42 optimally
    """
    
    def __init__(self, num_processes: Optional[int] = None):
        """
        Initialize metrics processor.
        
        Args:
            num_processes: Number of processes for parallel computation.
                         Defaults to min(batch_size, cpu_count()-1)
        """
        self.num_processes = num_processes or min(42, cpu_count() - 1)
        # Pre-initialize process pool
        self.pool = Pool(processes=self.num_processes)
        
    @staticmethod
    def _initialize_metrics():
        """Initialize metric calculators for a process."""
        return {
            'sm': Smeasure(),
            'em': Emeasure(),
            'wfm': WeightedFmeasure(),
            'mae': MAE(),
            'fm': Fmeasure()
        }

    @staticmethod
    def _f1_score(precision: float, recall: float) -> float:
        """Compute F1 with stable handling for empty predictions/masks."""
        denom = precision + recall
        if denom <= 0:
            return 0.0
        return 2.0 * precision * recall / denom

    @staticmethod
    def _binary_prf(pred_bin: np.ndarray, gt_bin: np.ndarray) -> Tuple[float, float, float]:
        """Compute exact binary precision/recall/F1 at the selected threshold."""
        pred_sum = int(pred_bin.sum())
        gt_sum = int(gt_bin.sum())

        if pred_sum == 0 and gt_sum == 0:
            return 1.0, 1.0, 1.0
        if pred_sum == 0 or gt_sum == 0:
            return 0.0, 0.0, 0.0

        tp = np.logical_and(pred_bin, gt_bin).sum()
        precision = float(tp / pred_sum)
        recall = float(tp / gt_sum)
        return precision, recall, MetricsProcessor._f1_score(precision, recall)

    @staticmethod
    def _tolerance_prf(
        pred_bin: np.ndarray,
        gt_bin: np.ndarray,
        radius: int
    ) -> Tuple[float, float, float]:
        """
        Compute trajectory precision/recall/F1 with spatial tolerance.

        Precision asks whether predicted trajectory pixels land near the GT
        trajectory. Recall asks whether GT trajectory pixels are covered by a
        nearby prediction. This is better suited to thin stitch tracks than
        exact pixel overlap.
        """
        pred_sum = int(pred_bin.sum())
        gt_sum = int(gt_bin.sum())

        if pred_sum == 0 and gt_sum == 0:
            return 1.0, 1.0, 1.0
        if pred_sum == 0 or gt_sum == 0:
            return 0.0, 0.0, 0.0

        if radius <= 0:
            return MetricsProcessor._binary_prf(pred_bin, gt_bin)

        kernel_size = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        pred_u8 = pred_bin.astype(np.uint8)
        gt_u8 = gt_bin.astype(np.uint8)

        gt_near = cv2.dilate(gt_u8, kernel) > 0
        pred_near = cv2.dilate(pred_u8, kernel) > 0

        precision = float(np.logical_and(pred_bin, gt_near).sum() / pred_sum)
        recall = float(np.logical_and(gt_bin, pred_near).sum() / gt_sum)
        return precision, recall, MetricsProcessor._f1_score(precision, recall)

    @staticmethod
    def _mask_boundary(mask_bin: np.ndarray) -> np.ndarray:
        """Extract a one-pixel binary boundary from a foreground mask."""
        mask_u8 = mask_bin.astype(np.uint8)
        if int(mask_u8.sum()) == 0:
            return mask_u8.astype(bool)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(mask_u8, kernel, iterations=1)
        return (mask_u8 - eroded) > 0

    @staticmethod
    def _compute_trajectory_metrics(
        pred: np.ndarray,
        gt: np.ndarray,
        radii: Tuple[int, ...],
        primary_radius: int,
        threshold: float,
    ) -> Dict[str, float]:
        """Compute binary and tolerance-aware trajectory metrics."""
        if np.issubdtype(pred.dtype, np.floating):
            pred_bin = pred >= threshold
        else:
            pred_bin = pred >= int(round(threshold * 255.0))
        gt_bin = gt >= 128

        exact_p, exact_r, exact_f = MetricsProcessor._binary_prf(pred_bin, gt_bin)
        intersection = int(np.logical_and(pred_bin, gt_bin).sum())
        union = int(np.logical_or(pred_bin, gt_bin).sum())
        # Use the same 0.5-thresholded masks as binary precision/recall/F1.
        # The empty/empty convention is IoU=1, consistent with _binary_prf.
        binary_iou = 1.0 if union == 0 else float(intersection / union)
        results = {
            "binary_precision": exact_p,
            "binary_recall": exact_r,
            "binary_f1": exact_f,
            "binary_iou": binary_iou,
        }

        pred_boundary = MetricsProcessor._mask_boundary(pred_bin)
        gt_boundary = MetricsProcessor._mask_boundary(gt_bin)
        boundary_p, boundary_r, boundary_f1 = MetricsProcessor._tolerance_prf(
            pred_boundary, gt_boundary, radius=1
        )
        results.update({
            "boundary_precision_1px": boundary_p,
            "boundary_recall_1px": boundary_r,
            "boundary_f1_1px": boundary_f1,
        })

        for radius in radii:
            precision, recall, f1 = MetricsProcessor._tolerance_prf(pred_bin, gt_bin, radius)
            results[f"trajectory_precision_{radius}px"] = precision
            results[f"trajectory_recall_{radius}px"] = recall
            results[f"trajectory_f1_{radius}px"] = f1

            if radius == primary_radius:
                results["trajectory_precision"] = precision
                results["trajectory_recall"] = recall
                results["trajectory_f1"] = f1

        if "trajectory_f1" not in results:
            precision, recall, f1 = MetricsProcessor._tolerance_prf(
                pred_bin, gt_bin, primary_radius
            )
            results["trajectory_precision"] = precision
            results["trajectory_recall"] = recall
            results["trajectory_f1"] = f1

        return results
        
    @staticmethod
    def _process_single_sample(data: Tuple[np.ndarray, np.ndarray, bool, Tuple[int, ...], int, float, Optional[np.ndarray]]) -> Dict[str, float]:
        """
        Process metrics for a single sample in a separate process.
        
        Args:
            data: Tuple of (prediction, ground_truth) as numpy arrays
            
        Returns:
            Dictionary of computed metrics for the sample
        """
        pred, gt, include_trajectory_metrics, radii, primary_radius, threshold, trajectory_pred = data
        metrics = MetricsProcessor._initialize_metrics()
        
        # Update all metrics for this sample
        for name, metric in metrics.items():
            metric.step(pred=pred, gt=gt)
            
        # Get results
        results = {
            "sm": metrics['sm'].get_results()["sm"],
            "wfm": metrics['wfm'].get_results()["wfm"],
            "mae": metrics['mae'].get_results()["mae"],
            "em": metrics['em'].get_results()["em"]["adp"],
            "fm": metrics['fm'].get_results()["fm"]["curve"].mean()
        }

        if include_trajectory_metrics:
            results.update(
                MetricsProcessor._compute_trajectory_metrics(
                    pred=trajectory_pred if trajectory_pred is not None else pred,
                    gt=gt,
                    radii=radii,
                    primary_radius=primary_radius,
                    threshold=threshold,
                )
            )

        return results

    @torch.no_grad()
    def compute_metrics(
        self,
        seg_pred: List[torch.Tensor],
        seg_gt: List[torch.Tensor],
        edge_pred: Optional[List[torch.Tensor]] = None,
        edge_gt: Optional[List[torch.Tensor]] = None,
        include_trajectory_metrics: bool = False,
        trajectory_radii: Tuple[int, ...] = (1, 2, 3, 5),
        trajectory_primary_radius: int = 2,
        trajectory_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute metrics efficiently using multiprocessing.
        
        Workflow:
        1. Batch process predictions on GPU (sigmoid + uint8 conversion)
        2. Async transfer to CPU
        3. Parallel process metrics across CPU cores
        4. Aggregate results
        
        Args:
            seg_pred: Segmentation predictions [0,1] after sigmoid
                     Either List[Tensor[1,H,W]] or Tensor[B,1,H,W]
            seg_gt: Ground truth masks {0,1}
                   List[Tensor[H,W]]
            edge_pred: Optional edge predictions [0,1]
                      List[Tensor[1,H,W]]
            edge_gt: Optional edge ground truth {0,1}
                    List[Tensor[H,W]]
            include_trajectory_metrics: If True, also compute thresholded
                    exact and tolerance-aware trajectory metrics.
            trajectory_radii: Pixel tolerances for trajectory F1.
            trajectory_primary_radius: Radius exposed as trajectory_f1.
            trajectory_threshold: Probability threshold for binary trajectory.
                    
        Returns:
            Dictionary of averaged metrics
        """
        try:
            # Prepare prepare predictions (convert to uint8)
            seg_prob_processed = [p.sigmoid() for p in seg_pred]
            seg_pred_processed = [(p * 255).byte() for p in seg_prob_processed]

            # Prepare ground truth (convert to uint8)
            seg_gt_processed = [(g * 255).byte() for g in seg_gt]
            
            # Async transfer to CPU and convert to numpy
            # Using non_blocking=True for async transfer
            seg_pred_np = [p.cpu().numpy().squeeze() for p in seg_pred_processed]
            seg_prob_np = [
                p.cpu().numpy().squeeze()
                for p in seg_prob_processed
            ] if include_trajectory_metrics else [None] * len(seg_pred_processed)
            seg_gt_np = [g.cpu().numpy().squeeze() for g in seg_gt_processed]
            
            # Create processing pairs
            seg_pairs = [
                (
                    pred,
                    gt,
                    include_trajectory_metrics,
                    trajectory_radii,
                    trajectory_primary_radius,
                    trajectory_threshold,
                    traj_pred,
                )
                for pred, gt, traj_pred in zip(seg_pred_np, seg_gt_np, seg_prob_np)
            ]
            
            # Process segmentation metrics in parallel
            seg_results = self.pool.map(self._process_single_sample, seg_pairs)
            
            # Process edge metrics if present
            if edge_pred is not None and edge_gt is not None:
                edge_pred_processed = [(p.sigmoid() * 255).byte() for p in edge_pred]
                edge_gt_processed = [(g * 255).byte() for g in edge_gt]
                edge_pred_np = [p.cpu().numpy().squeeze() for p in edge_pred_processed]
                edge_gt_np = [g.cpu().numpy().squeeze() for g in edge_gt_processed]
                edge_pairs = [
                    (pred, gt, False, trajectory_radii, trajectory_primary_radius, trajectory_threshold, None)
                    for pred, gt in zip(edge_pred_np, edge_gt_np)
                ]
                edge_results = self.pool.map(self._process_single_sample, edge_pairs)
            else:
                edge_results = None
            
            # Aggregate results
            final_metrics = self._aggregate_results(seg_results, edge_results)
            
            return final_metrics
            
        except Exception as e:
            print(f"Error in metric computation: {str(e)}")
            raise
            
    def _aggregate_results(
        self, 
        seg_results: List[Dict[str, float]], 
        edge_results: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, float]:
        """
        Aggregate metrics from multiple samples.
        
        Args:
            seg_results: List of segmentation metrics per sample
            edge_results: Optional list of edge metrics per sample
            
        Returns:
            Dictionary of averaged metrics
        """
        # Average segmentation metrics
        n_samples = len(seg_results)
        metrics = {
            "s_alpha": sum(r["sm"] for r in seg_results) / n_samples,
            "weighted_f": sum(r["wfm"] for r in seg_results) / n_samples,
            "mae": sum(r["mae"] for r in seg_results) / n_samples,
            "e_phi": sum(r["em"] for r in seg_results) / n_samples,
            "mean_f": sum(r["fm"] for r in seg_results) / n_samples
        }

        metric_keys = {"sm", "wfm", "mae", "em", "fm"}
        extra_keys = [k for k in seg_results[0].keys() if k not in metric_keys]
        for key in extra_keys:
            metrics[key] = sum(r[key] for r in seg_results) / n_samples
        
        # Add edge metrics if present
        if edge_results:
            metrics.update({
                "edge_mae": sum(r["mae"] for r in edge_results) / n_samples,
                "edge_f": sum(r["fm"] for r in edge_results) / n_samples
            })
            
        return metrics
        
    def __del__(self):
        """Cleanup process pool on deletion."""
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()
