"""
Entry point for C3Net Camouflaged Object Detection.

This module provides a command-line interface for:
1. Training the model with optional validation
2. Evaluating on test datasets
3. Making predictions on new images

Directory Structure:
    ./
    ├── configs/
    │   └── default.    # Default configuration
    ├── checkpoints/
    │   └── model_best.pth  # Default model checkpoint
    └── results/            # Operation outputs

Example Usage:
    Training:
        python main.py train --config configs/default.yaml 
    
    Evaluation:
        python main.py evaluate --model checkpoints/best.pth 
        
    Prediction:
        python main.py predict --model checkpoints/best.pth --input path/to/image
"""

import argparse
import copy
import logging
import os
import random
import sys

# Must be set before NumPy/PyTorch import their OpenMP runtimes.  The Windows
# c3net environment otherwise intermittently crashes in python310.dll after
# several training/checkpoint cycles.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import yaml
from pathlib import Path
from typing import Dict, Optional
import json
import warnings
import numpy
import torch.serialization


warnings.filterwarnings('ignore', category=UserWarning, module='py_sod_metrics.sod_metrics')

from ptflops import get_model_complexity_info

import torch

from engine.trainer import Trainer
from engine.evaluator import Evaluator
from engine.predictor import Predictor
from utils.data_loader import get_test_loaders
from utils.run_manager import DirectoryManager
from com_models.registry import build_model
from utils.checkpoint_utils import load_model_checkpoint


'''
import compare models
'''
# from com_models.OPNet_model import OPNet_Wrapper as OPNet   
# from com_models.fselnet_model import fselnet_Wrapper as fselnet
# from com_models.PVTNet_model import Model as PVTNet   
# from com_models.sam2_model import SAM2_Wrapper as SAM2



# Constants for default paths
DEFAULT_CONFIG_PATH = Path('./configs/default.yaml')
DEFAULT_MODEL_PATH = Path('./checkpoints/model_best.pth')


def set_seed(seed: int) -> None:
    """Make controlled ablation runs reproducible."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ConfigurationManager:
    """
    Manages configuration loading and validation.
    
    Features:
    - Hierarchical config loading
    - Path validation
    - Config merging
    - Default handling
    """
    
    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Recursively merge a small experiment override onto a base config."""
        result = copy.deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = ConfigurationManager._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _load_config_file(config_path: Path, seen: Optional[set] = None) -> Dict:
        """Load a YAML config with optional relative ``base_config`` inheritance."""
        resolved = config_path.resolve()
        seen = set() if seen is None else seen
        if resolved in seen:
            raise RuntimeError(f"Circular base_config reference: {resolved}")
        seen.add(resolved)
        try:
            with resolved.open('r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise RuntimeError(f"Invalid configuration file: {resolved}") from e

        base_name = config.pop('base_config', None)
        if not base_name:
            return config
        base_path = (resolved.parent / base_name).resolve()
        if not base_path.exists():
            raise RuntimeError(f"base_config not found: {base_path}")
        return ConfigurationManager._deep_merge(
            ConfigurationManager._load_config_file(base_path, seen), config
        )

    @staticmethod
    def load_config(config_path: Optional[Path] = None) -> Dict:
        """
        Load configuration with fallback to defaults.
        
        Args:
            config_path: Optional path to config file
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            RuntimeError: If no valid configuration found
        """
        # Try user config first
        if config_path and config_path.exists():
            try:
                return ConfigurationManager._load_config_file(config_path)
            except (yaml.Error, RuntimeError) as e:
                logging.error(f"Error in config file {config_path}: {e}")
                raise
        
        # Try default config
        if DEFAULT_CONFIG_PATH.exists():
            try:
                return ConfigurationManager._load_config_file(DEFAULT_CONFIG_PATH)
            except (yaml.YAMLError, RuntimeError) as e:
                logging.error(f"Error in default config: {e}")
                raise RuntimeError("Invalid default configuration file") from e
                
        raise RuntimeError(
            "No valid configuration found. Ensure either:\n"
            f"1. User config exists at specified path, or\n"
            f"2. Default config exists at {DEFAULT_CONFIG_PATH}"
        )
    
    @staticmethod
    def load_model_config(model_path: Optional[Path] = None) -> Dict:
        """
        Load model configuration from checkpoint.
        
        Args:
            model_path: Optional path to model checkpoint
            
        Returns:
            Model configuration dictionary
            
        Raises:
            RuntimeError: If no valid model checkpoint found
        """
        # Determine model path
        checkpoint_path = model_path or DEFAULT_MODEL_PATH
        
        if not checkpoint_path.exists():
            raise RuntimeError(
                "No valid model checkpoint found. Ensure either:\n"
                f"1. Model exists at specified path, or\n"
                f"2. Default model exists at {DEFAULT_MODEL_PATH}"
            )
            
        try:
            checkpoint = load_model_checkpoint(checkpoint_path, map_location='cpu')
            if 'config' not in checkpoint:
                raise ValueError("Checkpoint does not contain configuration")
            return checkpoint['config']
            
        except Exception as e:
            logging.error(f"Error loading model checkpoint: {e}")
            raise RuntimeError(f"Invalid model checkpoint: {checkpoint_path}")
def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed command line arguments
        
    Raises:
        ArgumentError: If arguments are invalid for specified mode
        
    Notes:
        - Config defaults to './configs/default.yaml'
        - Model defaults to './checkpoints/model_best.pth'
        - Input path required only for predict mode
    """
    parser = argparse.ArgumentParser(
        description='C3Net: Context-Contrast Network for Camouflaged Object Detection'
    )
    
    parser.add_argument('mode', 
                        nargs='?',
                       choices=['train', 'evaluate', 'predict'],
                       help='Operation mode',
                       default='train')
    
    parser.add_argument('--config',
                       type=Path,
                       help=f'Path to config file (default: {DEFAULT_CONFIG_PATH})')
    
    parser.add_argument('--model',
                       type=Path,
                       help=f'Path to model checkpoint (default: {DEFAULT_MODEL_PATH})')
    
    parser.add_argument('--input',
                       type=Path,
                       help='Input image or directory for prediction')
    
    args = parser.parse_args()
    
    # Validate predict mode requirements
    if args.mode == 'predict' and not args.input:
        parser.error("predict mode requires --input argument")
        
    return args

def setup_logging(dir_manager: DirectoryManager) -> None:
    """
    Configure logging system.
    
    Features:
    - Timestamp-based logging
    - Both file and console output
    - Proper formatting
    
    Args:
        dir_manager: Manages output directories
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(dir_manager.run_dirs.log_file),
            logging.StreamHandler()
        ]
    )


def train(config: Dict, model_path: Optional[Path], dir_manager: DirectoryManager) -> None:
    """
    Execute model training with support for resuming and fine-tuning.
    
    Args:
        config: Training configuration dictionary
        model_path: Optional path to model checkpoint from command line
        dir_manager: Output directory manager
        
    Notes:
        Priority for checkpoint loading:
        1. Command line --model argument (overrides config for resume/finetune)
        2. Config-specified checkpoint_path
        3. Default 'train' mode if neither is specified
    """
    logging.info("Initializing training...")

    # Determine training mode and checkpoint
    training_mode = config['training'].get('mode', {'type': 'train'})
    mode_type = training_mode.get('type', 'train')
    
    if mode_type not in ['train', 'resume', 'finetune']:
        raise ValueError(f"Invalid training mode: {mode_type}")
    
    # Handle checkpoint path priority
    if not model_path:
        model_path = Path(training_mode.get('checkpoint_path', ''))
    if mode_type in ['resume', 'finetune']:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for {mode_type} mode: {model_path}"
            )
        training_mode['checkpoint_path'] = model_path
        
            
    # Load model configuration from checkpoint if resuming
    if mode_type == 'resume' and training_mode['checkpoint_path']:
        checkpoint = load_model_checkpoint(training_mode['checkpoint_path'], map_location='cpu')
        if 'config' in checkpoint:
            logging.info("Updating configuration from checkpoint for resume mode")
            # Update only model and training configuration
            config['model'].update(checkpoint['config']['model'])
            config['training'].update(checkpoint['config']['training'])
            # Preserve the current mode configuration
            config['training']['mode'] = training_mode
    
    logging.info(f"Training Mode: {mode_type.upper()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Training on device: {device}")
    
    # Setup dataset paths
    dataset_paths = config['training']['datasets']
    if not dataset_paths:
        raise ValueError("No dataset paths provided in configuration")
    
    logging.info(f"Training on datasets: {dataset_paths}")
    
    # Initialize trainer
    trainer = Trainer(
        config=config,
        dir_manager=dir_manager,
        device=device
    )
    
    # Start training
    trainer.train(dataset_paths)

def evaluate(config: Dict, model_path: Path, dir_manager: DirectoryManager) -> None:
    """
    Execute model evaluation.
    
    Features:
    - Multi-dataset evaluation
    - Comprehensive metrics
    - Result visualization
    
    Args:
        config: Evaluation configuration
        model_path: Path to model checkpoint
        dir_manager: Output directory manager
        
    Notes:
        Uses settings from config for:
        - Batch processing
        - Dataset paths
        - Evaluation metrics
    """
    logging.info("Starting evaluation...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Evaluating on device: {device}")


    dataset_paths = config['evaluation']['datasets']
    if not dataset_paths:
        raise ValueError("No dataset paths provided for evaluation")

    test_loaders = get_test_loaders(
        dataset_dirs=dataset_paths,
        model_config=config['model'],
        batch_size=config['evaluation']['batch_size'],
        num_workers=config['evaluation']['num_workers']
    )

    evaluator = Evaluator(
        model_path=model_path,
        model_config=config['model'],
        dir_manager=dir_manager,
        device=device,
        batch_size=config['evaluation']['batch_size'],
        branch_diagnostics=config['evaluation'].get('branch_diagnostics', {}),
    )
    
    all_metrics = {}
    
    # Evaluate each test set
    for dataset_name, loader in test_loaders.items():
        logging.info(f"\nEvaluating on {dataset_name}:")
        metrics = evaluator.evaluate(loader, dataset_name)
        all_metrics[dataset_name] = metrics
        
        # Log metrics
        logging.info("\nEvaluation Results:")
        logging.info(f"Structure measure (Sα): {metrics['s_alpha']:.4f}")
        logging.info(f"Weighted F-measure (Fβw): {metrics['weighted_f']:.4f}")
        logging.info(f"Mean Absolute Error (M): {metrics['mae']:.4f}")
        logging.info(f"Enhanced-alignment (E_phi): {metrics['e_phi']:.4f}")
        logging.info(f"Mean F-measure (Fβm): {metrics['mean_f']:.4f}")

        if 'trajectory_f1' in metrics:
            logging.info(f"Trajectory F1 @2px: {metrics['trajectory_f1']:.4f}")
            logging.info(f"Trajectory F1 @1px: {metrics['trajectory_f1_1px']:.4f}")
            logging.info(f"Binary F1 @0.5: {metrics['binary_f1']:.4f}")

    # Save overall metrics summary
    metrics_path = dir_manager.run_dirs.root / 'metrics_summary.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    logging.info(f"Metrics saved to {metrics_path}")

def predict(config: Dict, model_path: Path, input_path: Path, 
           dir_manager: DirectoryManager) -> None:
    """
    Execute model prediction.
    
    Features:
    - Single image processing
    - Batch directory processing
    - Result visualization
    
    Args:
        config: Prediction configuration
        model_path: Path to model checkpoint
        input_path: Path to input image/directory
        dir_manager: Output directory manager
        
    Notes:
        Uses settings from config for:
        - Image processing
        - Output formatting
        - Batch handling
    """

    logging.info("Starting prediction...")
    
    predictor = Predictor(
        model_path=model_path,
        model_config=config['model'],
        dir_manager=dir_manager,
        batch_size=config['prediction'].get('batch_size')
    )

    # Handle directory or single file input
    if input_path.is_dir():
        logging.info(f"Processing directory: {input_path}")
        results = predictor.predict_directory(
            input_dir=str(input_path),
            output_size=config['prediction'].get('output_size')
        )
        logging.info(f"Processed {len(results)} images")
    else:
        logging.info(f"Processing single image: {input_path}")
        seg_pred, edge_pred, original_image = predictor.predict_single(
            image_path=str(input_path),
            output_size=config['prediction'].get('output_size')
        )
        # Save results
        predictor.result_manager.save_prediction(
            input_path.name,
            seg_pred,
            edge_pred,
            original_image
        )
        logging.info(f"Processing complete, results saved")

def print_model_info(config: Dict) -> None:
    """
    Print model architecture and complexity information.
    Ensures cleanup of model from memory after printing.
    
    Args:
        config: Configuration dictionary containing model settings
        
    Note:
        This function temporarily creates a model instance for information
        gathering, then ensures proper cleanup to free memory.
    """
    logging.info("Analyzing model architecture and complexity...")
    
    try:
        # Initialize on appropriate device11111111111111111111111111111111111111111111111111111111111111
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        model = build_model(config['model']).to(device)
        model.eval()



        
        # Print model architecture
        logging.info("\nModel Architecture:")
        logging.info("=" * 50)
        logging.info(model)
        logging.info("=" * 50)
        
        # Calculate complexity
        input_size = (3, config['model']['image_processing']['target_size'],
                     config['model']['image_processing']['target_size'])
        
        with torch.cuda.device(device):
            flops, params = get_model_complexity_info(
                model,
                input_size,
                as_strings=True,
                print_per_layer_stat=False,
                verbose=False
            )
            
        logging.info("\nModel Complexity Analysis:")
        logging.info("-" * 30)
        logging.info(f"Computational Cost: {flops}")
        logging.info(f"Number of Parameters: {params}")
        logging.info("-" * 30 + "\n")
        
    except Exception as e:
        logging.warning(f"Could not complete model analysis: {str(e)}")
        logging.warning("Continuing with execution...")
        
    finally:
        # Explicit cleanup
        try:
            del model
            torch.cuda.empty_cache()
            logging.debug("Model cleanup completed")
        except NameError:
            # Model wasn't successfully created
            pass

def main() -> None:
    """
    Main execution pipeline.
    
    Flow:
    1. Argument Parsing:
       - Mode selection
       - Optional paths
    
    2. Configuration Loading:
       - User config or default配置文件
       - Model config from checkpoint
       - Setting validation验证
    
    3. Directory Setup:
       - Output organization
       - Logging configuration
    
    4. Mode Execution:
       - train: Full training pipeline
       - evaluate: Dataset evaluation
       - predict: Image processing
    """
    try:
        # Parse arguments解析参数
        args = parse_args()

        # Load appropriate configurations
        config = ConfigurationManager.load_config(args.config)#指定的配置文件（或默认路径）加载项目的核心配置，比如模型参数、数据路径、训练超参数等

        if args.mode in ['evaluate', 'predict']:#从已训练的模型断点（args.model）中加载模型配置
            model_config = ConfigurationManager.load_model_config(args.model)
            # Update model settings from checkpoint
            config['model'].update(model_config['model'])

        # Keep every log, metric file, and checkpoint in a directory named
        # after the experiment structure rather than an opaque timestamp.
        dir_manager = DirectoryManager(args.mode, config.get('model', {}).get('name'))
        setup_logging(dir_manager)

        seed = config.get('training', {}).get('seed')
        if seed is not None:
            set_seed(int(seed))
            logging.info(f"Random seed fixed to {seed}")

        # Log configuration记录当前运行模式和完整配置
        logging.info(f"Running in {args.mode} mode")
        logging.info("Configuration:")
        logging.info(yaml.dump(config, default_flow_style=False))

        # Print model information
        print_model_info(config)

        # Execute requested mode
        if args.mode == 'train':
            train(config, args.model, dir_manager)
        elif args.mode == 'evaluate':
            evaluate(config, args.model or DEFAULT_MODEL_PATH, dir_manager)
        else:  # predict
            predict(config, args.model or DEFAULT_MODEL_PATH, 
                   args.input, dir_manager)
        
        logging.info("Process completed successfully")
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
