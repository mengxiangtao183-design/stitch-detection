"""
Training engine for Camouflaged Object Detection models.

This module provides a comprehensive training implementation with:
1. Automatic mixed precision training
2. Multi-scale segmentation and edge supervision
3. Resource-optimized batch processing
4. Metric tracking and model checkpointing
5. Metric tracking and model checkpointing

Core Components:
- TrainingMonitor: Handles metric tracking and checkpoint management
- Trainer: Manages training loops and optimization

Processing Pipeline:
1. Data Loading: Non-blocking tensor transfers
2. Forward Pass: Mixed precision inference
3. Multi-scale Loss: Segmentation and edge losses
4. Optimization: Gradient scaling and clipping
5. Metric Tracking: Comprehensive performance monitoring
"""
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
import logging
import json
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import MetricsProcessor
from utils.loss_functions import C3NetLoss, compute_distillation_loss
from utils.run_manager import DirectoryManager
from com_models.registry import build_model
from models.c3net import C3Net
from utils.checkpoint_utils import (
    load_model_checkpoint,
    normalize_state_dict_keys,
    save_model_checkpoint,
)
# from com_models.OPNet_model import OPNet_Wrapper as OPNet
# from com_models.fselnet_model import fselnet_Wrapper as fselnet
# from com_models.sam2_model import SAM2_Wrapper as SAM2
from utils.data_loader import get_training_loaders
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts


class TrainingMonitor:
    """
    Tracks and manages training metrics and model checkpoints.

    This class handles:
    1. Per-batch statistic aggregation
    2. Best model selection via weighted F-measure
    3. Metric history management
    4. Atomic checkpoint saving

    Attributes:
        metrics_file (Path): JSON file for metric storage
        checkpoint_dir (Path): Directory for model checkpoints
        batch_stats (defaultdict): Running statistics tracker
        epoch_start (float): Epoch timing reference
        history (dict): Complete training history
            epochs: List of epoch statistics
            best_metrics: Best performance records

    """

    def __init__(self, dir_manager: DirectoryManager, monitor_metric: str = 'weighted_f'):
        """Initialize training monitor with directory structure."""
        # Output paths
        self.metrics_file = dir_manager.run_dirs.metrics_file
        self.checkpoint_dir = dir_manager.run_dirs.checkpoints
        self.monitor_metric = monitor_metric

        # Statistics tracking
        self.batch_stats = defaultdict(lambda: {'sum': 0.0, 'count': 0})

        # Epoch tracking
        self.epoch_start = None
        # Initialize or load history
        self.history = {
            "epochs": [],
            "best_metrics": {
                "weighted_f": 0.0,
                self.monitor_metric: 0.0,
                "s_alpha": 0.0,
                "mae": float("inf"),
            }
        }

        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                loaded = json.load(f)
                self.history = loaded
    def start_epoch(self):
        """Reset statistics and start timing for new epoch."""
        self.batch_stats.clear()
        self.epoch_start = time.time()

    def update_batch(self, metrics: Dict[str, float], timing: Dict[str, float], batch_size: int):
        """
        Updates running statistics with batch results.

        Accumulates weighted averages of metrics and timing statistics,
        accounting for variable batch sizes.

        Args:
            metrics: Current batch metric values
            timing: Timing measurements (data_time, forward_time, etc.)
            batch_size: Number of samples in batch

        Updates:
            - Running sums and counts for each statistic
            - Enables proper weighted averaging
        """
        # Update running statistics
        for key, value in {**metrics, **timing}.items():
            value = float(value) if torch.is_tensor(value) else value
            self.batch_stats[key]['sum'] += value * batch_size
            self.batch_stats[key]['count'] += batch_size

    def get_current_stats(self) -> Dict[str, float]:
        """Get current averaged statistics."""
        return {
            key: stats['sum'] / stats['count']
            for key, stats in self.batch_stats.items()
            if stats['count'] > 0
        }

    def check_best_model(self, current_metrics: Dict[str, float]) -> bool:
        """Return True when the configured validation metric improves."""
        current_value = current_metrics.get(self.monitor_metric)
        if current_value is None:
            raise KeyError(
                f"Configured monitor metric '{self.monitor_metric}' is absent from validation metrics."
            )
        is_best = current_value > self.history['best_metrics'].get(self.monitor_metric, float('-inf'))
        if is_best:
            logging.info(f"New best model -> {self.monitor_metric}: {current_value:.4f}")
            self.history['best_metrics'].update({
                self.monitor_metric: current_value,
                'weighted_f': current_metrics.get('weighted_f', 0.0),
                's_alpha': current_metrics.get('s_alpha', 0.0),
                'mae': current_metrics.get('mae', float('inf')),
            })
        return is_best

    def save_history(self) -> None:
        """Persist the metric history atomically on Windows and POSIX systems."""
        tmp_path = self.metrics_file.with_suffix('.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        tmp_path.replace(self.metrics_file)

    def save_epoch(self, epoch: int, phase: str):
        """
        Save epoch results to metrics file.

        Args:
            epoch: Current epoch number
            phase: Training phase ('train' or 'val')
        """
        stats = self.get_current_stats()
        epoch_time = time.time() - self.epoch_start

        # Split into metrics and timing
        metrics = {k: v for k, v in stats.items() if not k.endswith('_time')}
        timing = {k: v for k, v in stats.items() if k.endswith('_time')}
        timing['epoch_time'] = epoch_time

        # Ensure epoch entry exists
        while len(self.history['epochs']) <= epoch:
            self.history['epochs'].append({"epoch": len(self.history['epochs'])})

        # Update epoch data
        self.history['epochs'][epoch][phase] = {
            "metrics": metrics,
            "timing": timing
        }

        # Save complete history structure
        self.save_history()
        if phase == 'val':
            # Build log message with available metrics
            log_parts = [
                f"Epoch {epoch} ({phase})",
            ]
            if 'weighted_f' in stats:
                log_parts.append(f"F-measure: {stats['weighted_f']:.4f}")
            if 's_alpha' in stats:
                log_parts.append(f"S-alpha: {stats['s_alpha']:.4f}")
            if 'mae' in stats:
                log_parts.append(f"MAE: {stats['mae']:.4f}")
            log_parts.append(f"Loss: {stats['loss']:.4f}")
            log_parts.append(f"Time: {epoch_time:.2f}s")
            logging.info(" - ".join(log_parts))
        else:
            log_parts = [
                f"Epoch {epoch} ({phase})",
                f"Loss: {stats['loss']:.4f}",
            ]
            if 'loss_centerline' in stats:
                log_parts.append(f"Center: {stats['loss_centerline']:.4f}")
            if 'loss_sdf' in stats:
                log_parts.append(f"SDF: {stats['loss_sdf']:.4f}")
            if 'loss_prototype_context' in stats:
                log_parts.append(f"Prototype: {stats['loss_prototype_context']:.4f}")
            if 'loss_hard_negative' in stats:
                log_parts.append(f"HardNeg: {stats['loss_hard_negative']:.4f}")
            if 'loss_texture_hard_negative' in stats:
                log_parts.append(f"TextureHN: {stats['loss_texture_hard_negative']:.4f}")
            if 'loss_texture_foreground_safety' in stats:
                log_parts.append(f"TextureSafe: {stats['loss_texture_foreground_safety']:.4f}")
            if 'loss_pattern_center' in stats:
                log_parts.append(f"PatternCenter: {stats['loss_pattern_center']:.4f}")
            if 'loss_contour' in stats:
                log_parts.append(f"Contour: {stats['loss_contour']:.4f}")
            log_parts.append(f"Time: {epoch_time:.2f}s")
            logging.info(" - ".join(log_parts))


class Trainer:
    """
    Manages complete training pipeline for Camouflaged Object Detection.

    Implements:
    1. Multi-scale training with automatic mixed precision
    2. Joint segmentation and edge supervision
    3. Layer-wise learning rate optimization
    4. Comprehensive validation and checkpointing
    5. Checkpointing and early stopping
    

    Processing Flow:
    1. Batch Loading: Images, masks, and edge maps
    2. Forward Pass: Generate multi-scale predictions
    3. Loss Computation: Combined segmentation and edge losses
    4. Optimization: Scaled gradients with clipping
    5. Validation: Multi-metric performance tracking

    Args:
        config (Dict): Model and training configuration
        dir_manager (DirectoryManager): Output directory handler
        device (torch.device): Computing device
    """

    def __init__(
        self,
        config: Dict,
        dir_manager: DirectoryManager,
        device: torch.device,
    ):
        # Basic setup
        self.config = config['training']
        self.model_config = config['model']
        self.device = device
        self.mode_config = self.config.get('mode', {'type': 'train'})
        # Initialize model and training components
        self.model = self._initialize_model()

        # =====================================================================
        # Knowledge Distillation: Teacher Model
        # =====================================================================
        self.distill_cfg = self.model_config.get('distillation', {})
        self.distill_enabled = self.distill_cfg.get('enabled', False)
        self.teacher = None
        if self.distill_enabled:
            self._init_teacher()

        # Training parameters
        self.batch_size = self.config['batch_size']
        self.num_epochs = self.config['num_epochs']
        self.grad_clip = self.config.get('gradient_clip', 1.0)
        self.early_stop_patience = self.config.get('early_stop_patience', 15)
        self.save_freq = self.config.get('save_freq', 1)
        self.save_best = self.config.get('save_best', True)
        self.monitor_metric = self.config.get('monitor_metric', 'weighted_f')
        self.include_trajectory_metrics = self.config.get(
            'include_trajectory_metrics',
            self.monitor_metric in {
                'binary_iou',
                'binary_f1',
                'boundary_f1_1px',
                'trajectory_f1',
            },
        )

        # Initialize components
        self._setup_optimization()
        self.criterion = C3NetLoss(**self.config['loss']).to(device)
        self.monitor = TrainingMonitor(dir_manager, self.monitor_metric)
        self.metrics_processor = MetricsProcessor()

        # Mixed precision setup
        self.use_amp = self.config.get('use_amp', True)
        self.scaler = amp.GradScaler(enabled=self.use_amp)
        # Set starting epoch
        self.start_epoch = 0

        # Load checkpoint if resuming or fine-tuning
        if self.mode_config.get('checkpoint_path'):
            self._load_checkpoint()

    def _initialize_model(self) -> torch.nn.Module:
        """Initialize model and wrap with DataParallel if multiple GPUs are available."""
        model = build_model(self.model_config)

        # Check for multiple GPUs and wrap with DataParallel
        if torch.cuda.device_count() > 1:
            gpu_count = torch.cuda.device_count()
            logging.info(f"Using {gpu_count} GPUs with torch.nn.DataParallel.")
            model = torch.nn.DataParallel(model)  # Wrap the model
        else:
            logging.info("Using single GPU or CPU.")

        model = model.to(self.device)  # Move wrapped or unwrapped model to target device
        base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        self._configure_trainable_parameters(base_model)

        return model

    def _init_teacher(self) -> None:
        """
        Build and load the frozen DINOv2 teacher for knowledge distillation.

        The teacher is constructed via C3Net.create_teacher() which:
        1. Instantiates C3Net with backbone_type='dinov2' (no CoordAtt / stem_projections).
        2. Loads the checkpoint with strict=False.
        3. Freezes all parameters and sets eval() mode.

        The teacher stays on the same device as the student and is only used
        during training (not validation) to generate soft labels.
        """
        checkpoint = self.distill_cfg.get('teacher_checkpoint')
        if not checkpoint:
            raise ValueError(
                "distillation.enabled=True but teacher_checkpoint is not set.")

        logging.info(f"=== Loading Teacher Model for Distillation ===")
        logging.info(f"  Checkpoint: {checkpoint}")
        logging.info(f"  Temperature: {self.distill_cfg.get('temperature', 4.0)}")
        logging.info(f"  KD weight:   {self.distill_cfg.get('kd_weight', 0.5)}")
        logging.info(f"  Distill on:  {self.distill_cfg.get('distill_outputs', [])}")

        self.teacher = C3Net.create_teacher(
            config=self.model_config,
            checkpoint_path=checkpoint,
            device=self.device,
        )

        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        logging.info(f"Teacher loaded: {teacher_params:,} parameters (frozen).")

    def _configure_trainable_parameters(self, model: torch.nn.Module) -> None:
        """
        Configure model trainability for fine-tuning.

        Applies optional fine-tuning freezes.




        Args:
            model: C3Net model to be configured
        """
        # --- Traditional fine-tuning controls (preserved from original) ---
        # Freeze encoder if specified
        if self.mode_config.get('freeze_encoder', False):
            if hasattr(model, 'backbone'):
                for param in model.backbone.parameters():
                    param.requires_grad = False
                logging.info("Backbone layers frozen for fine-tuning")
            else:
                logging.warning("freeze_encoder requested but model has no backbone attribute")

        # Freeze specified layers
        for layer_name in self.mode_config.get('frozen_layers', []):
            if hasattr(model, layer_name):
                for param in getattr(model, layer_name).parameters():
                    param.requires_grad = False
                logging.info(f"Layer {layer_name} frozen for fine-tuning")

        # G2-style safe calibration can train only a newly introduced head on
        # top of an immutable checkpoint.  Prefixes use model.named_parameters
        # paths, e.g. ``mobile_decoder.geometry_ribbon_head``.
        trainable_prefixes = self.mode_config.get('trainable_parameter_prefixes', [])
        if trainable_prefixes:
            normalized_prefixes = tuple(str(prefix).rstrip('.') for prefix in trainable_prefixes)
            trainable_count = 0
            for name, param in model.named_parameters():
                param.requires_grad = any(
                    name == prefix or name.startswith(prefix + '.')
                    for prefix in normalized_prefixes
                )
                trainable_count += int(param.requires_grad)
            logging.info(
                "Fine-tuning only parameter prefixes %s (%d tensors trainable)",
                normalized_prefixes, trainable_count,
            )

    def _set_frozen_modules_eval(self) -> None:
        """Prevent frozen BatchNorm statistics from drifting during fine-tuning."""
        base_model = (
            self.model.module if isinstance(self.model, torch.nn.DataParallel)
            else self.model
        )
        for name, module in base_model.named_modules():
            if name and not any(param.requires_grad for param in module.parameters()):
                module.eval()
    def _load_checkpoint(self) -> None:
        """
        Load and validate checkpoint, handling DataParallel state dict keys.
        """
        checkpoint_path = Path(self.mode_config['checkpoint_path'])
        logging.info(f"Loading checkpoint from: {checkpoint_path}")

        try:
            checkpoint = load_model_checkpoint(checkpoint_path, map_location=self.device)

            if 'model_state_dict' not in checkpoint:
                raise ValueError("Invalid checkpoint: missing model state")

            saved_state_dict = normalize_state_dict_keys(checkpoint['model_state_dict'])

            # Load into the base model (access via .module if DP is currently active)
            base_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            missing_keys, unexpected_keys = base_model.load_state_dict(
                saved_state_dict, strict=False
            )

            if missing_keys:
                logging.warning(f"Missing keys when loading state_dict: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys when loading state_dict: {unexpected_keys}")

            # --- Load other states (optimizer, scheduler, etc.) ---
            if self.mode_config['type'] == 'resume':
                self.start_epoch = checkpoint.get('epoch', -1) + 1
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    logging.warning("No optimizer state found, using fresh optimizer.")
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                else:
                    logging.warning("No scheduler state found, using fresh scheduler.")
                if 'scaler' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler'])
                else:
                    logging.warning("No AMP scaler state found, using fresh scaler.")
                if 'history' in checkpoint:
                    self.monitor.history = checkpoint['history']
                # ... (rest of resume logic) ...
                logging.info(f"Resumed training from epoch {self.start_epoch}")
            else:  # finetune
                lr_scale = self.mode_config.get('finetune_lr_scale', 0.1)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * lr_scale
                logging.info(f"Fine-tuning with learning rate scale: {lr_scale}")
                self.monitor.history = {  # Reset history for finetune
                    'epochs': [],
                    'best_metrics': {
                        'weighted_f': 0.0, 's_alpha': 0.0, 'mae': float('inf'),
                    }
                }

            # Validate configuration compatibility
            if 'config' in checkpoint:
                self._validate_config_compatibility(checkpoint['config'])

        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}", exc_info=True)  # Added exc_info
            raise RuntimeError(f"Failed to load checkpoint: {checkpoint_path}")

    def _validate_config_compatibility(self, checkpoint_config: Dict) -> None:
        """
        Validate and log configuration differences between checkpoint and current setup.

        Args:
            checkpoint_config: Configuration from checkpoint

        Notes:
            Logs warnings for significant differences but doesn't block execution
        """
        try:
            # Check model configuration
            if 'model' in checkpoint_config:
                current_model = self.model_config
                saved_model = checkpoint_config['model']

                # Check critical parameters
                if saved_model.get('name') != current_model.get('name'):
                    logging.warning(f"Model architecture mismatch: saved={saved_model.get('name')}, current={current_model.get('name')}")

                if saved_model.get('encoder', {}).get('variant') != current_model.get('encoder', {}).get('variant'):
                    logging.warning("Encoder variant mismatch between checkpoint and current configuration")

            # Check training configuration
            if 'training' in checkpoint_config:
                # Log significant training parameter changes
                current_train = self.config
                saved_train = checkpoint_config['training']

                # Check batch size compatibility
                if saved_train.get('batch_size') != current_train.get('batch_size'):
                    logging.info(f"Batch size changed: saved={saved_train.get('batch_size')}, current={current_train.get('batch_size')}")

                # Check learning rate differences
                if saved_train.get('optimizer', {}).get('learning_rate') != current_train.get('optimizer', {}).get('learning_rate'):
                    logging.info("Learning rate configuration has changed from checkpoint")

        except Exception as e:
            logging.warning(f"Configuration validation error: {e}")
            logging.warning("Continuing with current configuration")

    def _setup_optimization(self):
        """Configure optimization with layerwise learning rates."""
        # ================== Read configuration correctly ==================
        train_cfg = self.config                    # self.config is already config['training']
        opt_cfg = train_cfg.get('optimizer', {})
        sched_cfg = train_cfg.get('scheduler', {})

        logging.info("Setting up optimizer and scheduler...")

        # Get parameter groups
        param_groups = self._get_param_groups()

        # Create Optimizer
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=opt_cfg.get('weight_decay', 0.01)
        )

        # Create Scheduler (Cosine Annealing)
        base_lr = float(opt_cfg.get('learning_rate', 0.0001))
        total_epochs = int(train_cfg.get('num_epochs', 200))
        eta_min = float(sched_cfg.get('eta_min', 1e-6))

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs,
            eta_min=eta_min
        )

        logging.info(f"Optimizer setup complete. Base LR: {base_lr}, Weight decay: {opt_cfg.get('weight_decay', 0.01)}")

    def _get_param_groups(self) -> List[Dict]:
        """
        Create parameter groups for different learning rates.

        Supports:
        - encoder: reduced LR (scale * base_lr), no weight decay
        - encoder_norm: reduced LR (scale * base_lr), no weight decay reduced LR (0.1 * base_lr), no weight decay  [NEW]
        - decoder: full base_lr, with weight decay
        - decoder_norm: full base_lr, no weight decay

        Returns:
            List of parameter group dictionaries for AdamW.
        """
        train_cfg = self.config
        opt_cfg = train_cfg.get('optimizer', {})

        base_lr = float(opt_cfg.get('learning_rate', 0.0001))
        encoder_lr_scale = float(opt_cfg.get('encoder_lr_ratio', 0.2))

        # Parameter groups
        encoder_params = {'params': [], 'lr': base_lr * encoder_lr_scale, 'weight_decay': 0.0}
        encoder_norm_params = {'params': [], 'lr': base_lr * encoder_lr_scale, 'weight_decay': 0.0}
        decoder_params = {'params': [], 'lr': base_lr,
                          'weight_decay': float(opt_cfg.get('weight_decay', 0.01))}
        decoder_norm_params = {'params': [], 'lr': base_lr, 'weight_decay': 0.0}
        # Group parameters by name
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            is_norm = ('norm' in name.lower() or 'bn' in name.lower() or 'bias' in name.lower())
            # --- Encoder (backbone) parameters ---
            if 'encoder' in name.lower() or 'backbone' in name.lower() or 'stem_projections' in name.lower():
                if is_norm:
                    encoder_norm_params['params'].append(p)
                else:
                    encoder_params['params'].append(p)
            # --- Decoder and all other parameters ---
            else:
                if is_norm:
                    decoder_norm_params['params'].append(p)
                else:
                    decoder_params['params'].append(p)

        # Build final list, skipping empty groups
        all_groups = [
            encoder_params, encoder_norm_params,
            decoder_params, decoder_norm_params,
        ]
        param_groups = [g for g in all_groups if len(g['params']) > 0]

        # Log group sizes
        for g in param_groups:
            logging.info(f"  Param group (lr={g['lr']:.2e}, wd={g['weight_decay']}): "
                         f"{len(g['params'])} params")

        return param_groups

    def _process_batch(self, batch: Dict[str, torch.Tensor],
                       is_train: bool) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Processes single batch through complete training pipeline.

        Steps:
        1. Data Transfer: Non-blocking device placement
        2. Forward Pass: Mixed precision inference
        3. Prediction Resizing: Multi-scale alignment
        4. Loss Computation: Segmentation and edge losses
        5. Optimization: Gradient scaling and updates (training only)
        6. Metric Computation: Performance measurements

        Args:
            batch: Input data containing:
                - images: Input images
                - masks: Ground truth segmentations
                - edges: Ground truth edge maps
            is_train: Whether in training mode

        Returns:
            metrics: Performance measurements
            timing: Processing time breakdown
        """

        timing = {}
        batch_start = time.time()

        # Data loading
        data_start = time.time()
        images = batch['images'].to(self.device, non_blocking=True)
        masks = [m.to(self.device, non_blocking=True) for m in batch['masks']]
        edges = [e.to(self.device, non_blocking=True) for e in batch['edges']]
        # Legacy edge maps in DB/XJ/SD are stored at 392x392 while the clear
        # images and segmentation GT are 320x320.  Edge supervision must use
        # the same target grid as the segmentation losses for every ablation.
        edges = [
            edge if edge.shape[-2:] == mask.shape[-2:] else F.interpolate(
                edge.unsqueeze(0), size=mask.shape[-2:], mode='nearest'
            ).squeeze(0)
            for edge, mask in zip(edges, masks)
        ]
        timing['data_time'] = time.time() - data_start

        # Forward pass (student)
        forward_start = time.time()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                outputs = self.model(images)
        timing['forward_time'] = time.time() - forward_start

        # =====================================================================
        # Knowledge Distillation: Teacher forward (training only, no_grad)
        # Run under autocast for consistency with student forward.
        # =====================================================================
        teacher_outputs = None
        if self.distill_enabled and is_train and self.teacher is not None:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                with torch.no_grad():
                    teacher_outputs = self.teacher(images)
        # Prediction resizing
        resize_start = time.time()
        # Resize predictions to match target sizes
        resized_dict = {}
        for key, value in outputs.items():
            # STPC is supervised on its native P3 grid.  Keeping this output at
            # 40x40 lets the loss generate a max-pooled thin-stitch target
            # instead of blurring the target through bilinear resizing.
            if key in {
                'prototype_context_mask',
                'hard_negative_mask',
                'hard_negative_safety_mask',
                'afm_coarse_mask',
            }:
                resized_dict[key] = [value[i] for i in range(value.shape[0])]
                continue
            resized_preds = [F.interpolate(
                value[i:i+1], size=masks[i].shape[-2:], mode='bilinear', align_corners=False
                ).squeeze(0) for i, _ in enumerate(value)]
            resized_dict[key] = resized_preds
        timing['resize_time'] = time.time() - resize_start

        # Loss computation (GT loss + optional KD loss, both under autocast)
        loss_start = time.time()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss_dict = self.criterion(
                predictions=resized_dict,
                gt_masks=masks,
                gt_edges=edges,
            )

            # =================================================================
            # Knowledge Distillation: add KD loss to total.
            #   total = (1 - 伪) 脳 GT_loss  +  伪 脳 KD_loss
            # Computed on raw logits (kd_loss_fn handles spatial alignment).
            # =================================================================
            if teacher_outputs is not None:
                kd_loss_val = compute_distillation_loss(
                    student_outputs=outputs,
                    teacher_outputs=teacher_outputs,
                    distill_keys=self.distill_cfg.get('distill_outputs',
                        ['final_mask', 'object_mask', 'edge_mask']),
                    temperature=self.distill_cfg.get('temperature', 4.0),
                )
                kd_weight = self.distill_cfg.get('kd_weight', 0.5)
                gt_loss = loss_dict['loss']
                loss_dict['loss'] = (1.0 - kd_weight) * gt_loss + kd_weight * kd_loss_val
                loss_dict['kd_loss'] = kd_loss_val.detach()

        timing['loss_time'] = time.time() - loss_start
        # Optimization
        if is_train:
            backward_start = time.time()
            self.optimizer.zero_grad(set_to_none=True)
            total_loss = loss_dict['loss']
            if not torch.isfinite(total_loss).all():
                # A single corrupt batch must not permanently poison model
                # parameters.  The warning makes the cause auditable while
                # leaving the last finite model state untouched.
                logging.warning("Skipped non-finite training loss for current batch")
                loss_dict['nonfinite_batch_skipped'] = total_loss.detach().new_tensor(1.0)
            else:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)

                if self.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                else:
                    grad_norm = torch.zeros((), device=self.device)
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            grad_norm = grad_norm + parameter.grad.detach().float().pow(2).sum()
                    grad_norm = grad_norm.sqrt()

                if torch.isfinite(grad_norm).all():
                    self.scaler.step(self.optimizer)
                else:
                    # With AMP, GradScaler.step also records the overflow and
                    # safely skips the optimizer update.  In FP32, simply
                    # clear the gradients and retain the last finite weights.
                    logging.warning("Skipped optimizer step with non-finite gradient norm")
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                    self.optimizer.zero_grad(set_to_none=True)
                    loss_dict['nonfinite_batch_skipped'] = total_loss.detach().new_tensor(1.0)
                self.scaler.update()
            timing['backward_time'] = time.time() - backward_start
            metrics = loss_dict
        else:
            # Metrics computation
            metric_start = time.time()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    metrics = self.metrics_processor.compute_metrics(
                        seg_pred=resized_dict['final_mask'],
                        seg_gt=masks,
                        edge_pred=resized_dict['edge_mask'],
                        edge_gt=edges,
                        include_trajectory_metrics=self.include_trajectory_metrics,
                    )
            metrics.update(loss_dict)
            timing['metric_time'] = time.time() - metric_start

        timing['batch_time'] = time.time() - batch_start

        return metrics, timing

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Executes single training epoch.

        Processing:
        1. Sets model to training mode
        2. Iterates through batches with progress tracking
        3. Processes each batch with optimization
        4. Updates monitoring statistics

        Args:
            loader: Training data loader
            epoch: Current epoch number

        Returns:
            dict: Averaged epoch metrics
        """

        self.model.train()
        self.criterion.set_epoch(epoch)
        self._set_frozen_modules_eval()
        self.monitor.start_epoch()

        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
        for batch_idx, batch in enumerate(pbar):
            # Process batch
            metrics, timing = self._process_batch(batch, is_train=True)

            # Update monitoring
            self.monitor.update_batch(
                metrics=metrics,
                timing=timing,
                batch_size=len(batch['images'])
            )

            # Update progress bar (handle missing keys gracefully for different train modes)
            postfix_dict = {
                'loss': f"{metrics.get('loss', 0):.4f}",
                'time': f"{timing['batch_time']:.3f}s",
            }
            if 'loss_loc' in metrics:
                postfix_dict['object_loss'] = f"{metrics['loss_loc']:.4f}"
            if 'loss_edge' in metrics:
                postfix_dict['edge_loss'] = f"{metrics['loss_edge']:.4f}"
            if 'loss_centerline' in metrics:
                postfix_dict['center_loss'] = f"{metrics['loss_centerline']:.4f}"
            if 'loss_sdf' in metrics:
                postfix_dict['sdf_loss'] = f"{metrics['loss_sdf']:.4f}"
            if 'loss_prototype_context' in metrics:
                postfix_dict['proto_loss'] = f"{metrics['loss_prototype_context']:.4f}"
            if 'loss_hard_negative' in metrics:
                postfix_dict['hardneg_loss'] = f"{metrics['loss_hard_negative']:.4f}"
            if 'loss_texture_hard_negative' in metrics:
                postfix_dict['texture_hn'] = f"{metrics['loss_texture_hard_negative']:.4f}"
            if 'loss_texture_foreground_safety' in metrics:
                postfix_dict['texture_safe'] = f"{metrics['loss_texture_foreground_safety']:.4f}"
            if 'loss_afm_coarse' in metrics:
                postfix_dict['afm_coarse'] = f"{metrics['loss_afm_coarse']:.4f}"
            if 'loss_afm_aux' in metrics:
                postfix_dict['afm_aux'] = f"{metrics['loss_afm_aux']:.4f}"
            if 'loss_weak_stitch' in metrics:
                postfix_dict['weak_stitch'] = f"{metrics['loss_weak_stitch']:.4f}"
            if 'loss_weak_stitch_consistency' in metrics:
                postfix_dict['weak_cons'] = f"{metrics['loss_weak_stitch_consistency']:.4f}"
            if 'loss_pattern_center' in metrics:
                postfix_dict['pattern_center'] = f"{metrics['loss_pattern_center']:.4f}"
            if 'loss_contour' in metrics:
                postfix_dict['contour_loss'] = f"{metrics['loss_contour']:.4f}"
            if 'loss_final' in metrics:
                postfix_dict['final_loss'] = f"{metrics['loss_final']:.4f}"
            pbar.set_postfix(postfix_dict)

        return self.monitor.get_current_stats()

    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Performs validation pass.

        Processing:
        1. Sets model to evaluation mode
        2. Processes batches without gradients
        3. Computes comprehensive metrics (including PSNR/SSIM if applicable)
        4. Tracks performance statistics

        Args:
            loader: Validation data loader
            epoch: Current epoch number

        Returns:
            dict: Complete validation metrics
        """
        self.model.eval()
        self.criterion.set_epoch(epoch)
        self.monitor.start_epoch()
        pbar = tqdm(loader, desc='Validation')
        for batch_idx, batch in enumerate(pbar):
            metrics, timing = self._process_batch(batch, is_train=False)

            self.monitor.update_batch(
                metrics=metrics,
                timing=timing,
                batch_size=len(batch['images'])
            )

            # Update progress bar (handle missing keys gracefully for different train modes)
            postfix_dict = {
                'loss': f"{metrics.get('loss', 0):.4f}",
                'time': f"{timing['batch_time']:.3f}s",
            }
            if 'weighted_f' in metrics:
                postfix_dict['f_measure'] = f"{metrics['weighted_f']:.4f}"
            if 'binary_iou' in metrics:
                postfix_dict['binary_iou'] = f"{metrics['binary_iou']:.4f}"
            if 'binary_f1' in metrics:
                postfix_dict['binary_f1'] = f"{metrics['binary_f1']:.4f}"
            if 'boundary_f1_1px' in metrics:
                postfix_dict['boundary_f1'] = f"{metrics['boundary_f1_1px']:.4f}"
            if 'trajectory_f1' in metrics:
                postfix_dict['traj_f1'] = f"{metrics['trajectory_f1']:.4f}"
            if 's_alpha' in metrics:
                postfix_dict['s_alpha'] = f"{metrics['s_alpha']:.4f}"
            if 'mae' in metrics:
                postfix_dict['mae'] = f"{metrics['mae']:.4f}"
            pbar.set_postfix(postfix_dict)

        return self.monitor.get_current_stats()

    def train(self, dataset_dirs: List[str]):
        """Main training loop."""
        try:
            # Setup data loaders
            train_loader, val_loader = get_training_loaders(
                dataset_dirs=dataset_dirs,
                model_config=self.model_config,
                batch_size=self.batch_size,
                num_workers=self.config['num_workers'],
                val_ratio=self.config['val_ratio'],
                max_samples_per_dataset=self.config.get('max_samples_per_dataset'),
                stratified_validation=self.config.get('stratified_validation', False),
                split_seed=self.config.get('seed', 42),
            )

            # Log setup
            logging.info(f"Training samples: {len(train_loader.dataset)}")
            if val_loader:
                logging.info(f"Validation samples: {len(val_loader.dataset)}")

            # Initialize tracking variables
            best_monitor_metric = self.monitor.history['best_metrics'].get(
                self.monitor_metric, float('-inf')
            )
            early_stop_counter = 0
            min_delta = self.config.get('min_delta', 1e-4)

            # Training loop
            for epoch in range(self.start_epoch, self.num_epochs):
                # Training phase
                train_metrics = self.train_epoch(train_loader, epoch)
                self.monitor.save_epoch(epoch, 'train')

                # Validation phase
                if val_loader:
                    val_metrics = self.validate(val_loader, epoch)
                    self.monitor.save_epoch(epoch, 'val')

                    # Learning rate adjustment
                    self.scheduler.step()
                    current_monitor_metric = val_metrics.get(self.monitor_metric)
                    if current_monitor_metric is None:
                        raise KeyError(
                            f"Configured monitor metric '{self.monitor_metric}' is absent from validation metrics."
                        )
                    improvement = current_monitor_metric - best_monitor_metric

                    if improvement > min_delta:
                        best_monitor_metric = current_monitor_metric
                        early_stop_counter = 0

                        # Save model if best (respect save_best config)
                        if self.save_best and self.monitor.check_best_model(val_metrics):
                            self._save_checkpoint(epoch, val_metrics, is_best=True)
                    else:
                        early_stop_counter += 1

                    if early_stop_counter >= self.early_stop_patience:
                        logging.info("Early stopping triggered")
                        break

                # Regular checkpointing
                if (epoch + 1) % self.save_freq == 0:
                    self._save_checkpoint(
                        epoch,
                        val_metrics if val_loader else train_metrics,
                        is_best=False
                    )

                # Memory cleanup
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Training error: {str(e)}", exc_info=True)
            raise

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save training checkpoint, handling DataParallel."""
        # If using DataParallel, save the underlying model's state_dict
        if isinstance(self.model, torch.nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        checkpoint_metadata = {
            'epoch': epoch,
            'metrics': metrics,
            'config': {
                'training': self.config,
                'model': self.model_config
            },
            'history': self.monitor.history,
            'mode': self.mode_config,
        }

        filename = 'model_best.pth' if is_best else f'checkpoint_{epoch:03d}.pth'
        save_path = self.monitor.checkpoint_dir / filename
        # Do not use torch.save here: its pickle writers intermittently crash
        # this Windows/PyTorch setup during repeated best-checkpoint overwrites.
        # Optimizer state is intentionally omitted; these files are stable model
        # snapshots for evaluation or model-only resume.
        save_model_checkpoint(save_path, model_state_dict, checkpoint_metadata)

        logging.info(f"Saved checkpoint: {save_path}")


