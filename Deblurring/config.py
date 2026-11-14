#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:35:48 2019

@author: aditya
"""

r"""This module provides package-wide configuration management."""
from typing import Any, List
import yaml

try:
    from yacs.config import CfgNode as CN
except ModuleNotFoundError:
    class CN(dict):
        """Minimal yacs-like config node fallback with attribute access."""

        def __getattr__(self, attr):
            if attr in self:
                value = self[attr]
                if isinstance(value, dict) and not isinstance(value, CN):
                    value = CN(value)
                    self[attr] = value
                return value
            raise AttributeError(attr)

        def __setattr__(self, key, value):
            self[key] = value

        def merge_from_file(self, config_yaml):
            with open(config_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self._merge_dict(data)

        def merge_from_list(self, config_override):
            for idx in range(0, len(config_override), 2):
                key = config_override[idx]
                value = config_override[idx + 1]
                self._set_by_dotted_key(key, value)

        def freeze(self):
            return None

        def dump(self, stream=None):
            data = self.to_plain_dict()
            if stream is None:
                return yaml.safe_dump(data, sort_keys=False)
            yaml.safe_dump(data, stream, sort_keys=False)

        def to_plain_dict(self):
            plain = {}
            for key, value in self.items():
                if isinstance(value, CN):
                    plain[key] = value.to_plain_dict()
                else:
                    plain[key] = value
            return plain

        def _merge_dict(self, other):
            for key, value in other.items():
                if isinstance(value, dict):
                    node = self.get(key, CN())
                    if not isinstance(node, CN):
                        node = CN(node)
                    node._merge_dict(value)
                    self[key] = node
                else:
                    self[key] = value

        def _set_by_dotted_key(self, dotted_key, value):
            keys = dotted_key.split(".")
            node = self
            for key in keys[:-1]:
                if key not in node or not isinstance(node[key], CN):
                    node[key] = CN()
                node = node[key]
            node[keys[-1]] = value


class Config(object):
    r"""
    A collection of all the required configuration parameters. This class is a nested dict-like
    structure, with nested keys accessible as attributes. It contains sensible default values for
    all the parameters, which may be overriden by (first) through a YAML file and (second) through
    a list of attributes and values.

    Extended Summary
    ----------------
    This class definition contains default values corresponding to ``joint_training`` phase, as it
    is the final training phase and uses almost all the configuration parameters. Modification of
    any parameter after instantiating this class is not possible, so you must override required
    parameter values in either through ``config_yaml`` file or ``config_override`` list.

    Parameters
    ----------
    config_yaml: str
        Path to a YAML file containing configuration parameters to override.
    config_override: List[Any], optional (default= [])
        A list of sequential attributes and values of parameters to override. This happens after
        overriding from YAML file.

    Examples
    --------
    Let a YAML file named "config.yaml" specify these parameters to override::

        ALPHA: 1000.0
        BETA: 0.5

    >>> _C = Config("config.yaml", ["OPTIM.BATCH_SIZE", 2048, "BETA", 0.7])
    >>> _C.ALPHA  # default: 100.0
    1000.0
    >>> _C.BATCH_SIZE  # default: 256
    2048
    >>> _C.BETA  # default: 0.1
    0.7

    Attributes
    ----------
    """

    def __init__(self, config_yaml: str, config_override: List[Any] = []):

        self._C = CN()
        self._C.GPU = [0]
        self._C.VERBOSE = False

        self._C.MODEL = CN()
        self._C.MODEL.MODE = 'global'
        self._C.MODEL.SESSION = 'ps128_bs1'
        self._C.MODEL.N_FEAT = 96
        self._C.MODEL.SCALE_UNETFEATS = 48
        self._C.MODEL.SCALE_ORSNETFEATS = 32
        self._C.MODEL.NUM_CAB = 8
        self._C.MODEL.EDGE_REFINE = False
        self._C.MODEL.EDGE_REFINE_MID = 24
        self._C.MODEL.FREQUENCY_REFINE = False
        self._C.MODEL.FREQUENCY_REFINE_HIDDEN = 16
        self._C.MODEL.FREQUENCY_BAND_GATE = False
        self._C.MODEL.FREQUENCY_BAND_LOW_START = 0.05
        self._C.MODEL.FREQUENCY_BAND_LOW_END = 0.10
        self._C.MODEL.FREQUENCY_BAND_HIGH_START = 0.30
        self._C.MODEL.FREQUENCY_BAND_HIGH_END = 0.40
        self._C.MODEL.FREQUENCY_DISAGREEMENT_GATE = False
        self._C.MODEL.FREQUENCY_DISAGREEMENT_THRESHOLD = 0.0120
        self._C.MODEL.FREQUENCY_DISAGREEMENT_TEMPERATURE = 0.0020
        self._C.MODEL.FREQUENCY_DISAGREEMENT_MIN_STRENGTH = 0.10
        self._C.MODEL.LIGHT_ADAPT = False
        self._C.MODEL.LIGHT_TARGET_LUMA = 0.50
        self._C.MODEL.LIGHT_MIN_GAIN = 0.75
        self._C.MODEL.LIGHT_MAX_GAIN = 1.80
        self._C.MODEL.LIGHT_MIN_GAMMA = 0.65
        self._C.MODEL.LIGHT_MAX_GAMMA = 1.35
        self._C.MODEL.LIGHT_BLEND = 0.75
        self._C.MODEL.LIGHT_DARK_THRESHOLD = 0.45
        self._C.MODEL.LIGHT_BRIGHT_THRESHOLD = 0.70
        self._C.MODEL.TRAJECTORY_LIGHTWEIGHT = False
        self._C.MODEL.STAGE1_TRISTRIPE = False
        self._C.MODEL.STAGE1_STRIPES = 3
        self._C.MODEL.STAGE1_OVERLAP = 0.25
        self._C.MODEL.STAGE1_STRIPE_TRANSFORMER = False
        self._C.MODEL.STAGE1_STRIPE_TRANSFORMER_HEADS = 4
        self._C.MODEL.STAGE2_STRIPE_TRANSFORMER = False
        self._C.MODEL.STAGE2_STRIPE_TRANSFORMER_HEADS = 4
        self._C.MODEL.STAGE2_OVERLAP = 0.0
        self._C.MODEL.STAGE2_PAIR_ALIGNED = False
        self._C.MODEL.STAGE2_TMT = False
        self._C.MODEL.STAGE2_TMT_HORIZONTAL_KERNEL = 15
        self._C.MODEL.STAGE2_TMT_VERTICAL_KERNEL = 5
        self._C.MODEL.GATED_CSFF = False
        self._C.MODEL.GATED_CSFF_HORIZONTAL_KERNEL = 7
        self._C.MODEL.STITCH_TRAJECTORY_ALIGNMENT = False
        self._C.MODEL.STAM_HORIZONTAL_RADIUS = 6
        self._C.MODEL.STAM_VERTICAL_RADIUS = 1
        self._C.MODEL.STAM_HIDDEN_CHANNELS = 32
        self._C.MODEL.DUAL_DOMAIN_ORSNET = False
        self._C.MODEL.DDRB_RESIDUAL_INIT = 0.3
        self._C.MODEL.SIMPLE_GATED_ORSNET = False
        self._C.MODEL.SGORS_BLOCKS = 8
        self._C.MODEL.SGORS_RESIDUAL_INIT = 0.10
        self._C.MODEL.HYBRID_SIMPLE_GATED_ORSNET = False
        self._C.MODEL.HSGORS_BLOCKS = 4
        self._C.MODEL.HSGORS_RESIDUAL_INIT = 0.05
        self._C.MODEL.HSGORS_ADAPTER_SCALE_INIT = 0.02
        self._C.MODEL.HSGORS_V2 = False
        self._C.MODEL.HSGORS_V2_HORIZONTAL_KERNEL = 15
        self._C.MODEL.HSGORS_V2_VERTICAL_KERNEL = 5
        self._C.MODEL.STAGE3_TEPR = False
        self._C.MODEL.STAGE3_TEPR_TRANSITION = False
        self._C.MODEL.STAGE3_TEPR_FREEZE_LEGACY = True
        self._C.MODEL.STAGE3_TEPR_BLOCKS = 3
        self._C.MODEL.STAGE3_TEPR_HORIZONTAL_KERNEL = 15
        self._C.MODEL.STAGE3_TEPR_VERTICAL_KERNEL = 5
        self._C.MODEL.STAGE3_TRR = False
        self._C.MODEL.STAGE3_TRR_HORIZONTAL_KERNEL = 15
        self._C.MODEL.STAGE3_TRR_VERTICAL_KERNEL = 5
        self._C.MODEL.STAGE3_TRR_HIDDEN_CHANNELS = 32
        self._C.MODEL.STAGE2_NAF_BLOCKS = 2
        self._C.MODEL.STAGE2_HORIZONTAL_KERNEL = 15
        self._C.MODEL.STAGE2_VERTICAL_KERNEL = 5
        self._C.MODEL.STAGE3_NAF_BLOCKS = 3
        self._C.MODEL.TPTRNET_CORE = False
        self._C.MODEL.TPTR_BLOCKS = 2
        self._C.MODEL.TPTR_TCFM = False
        self._C.MODEL.TPTR_TCFM_HORIZONTAL_KERNEL = 7
        self._C.MODEL.TRISTRIP_CONTEXT_PYRAMID = False
        self._C.MODEL.TRISTRIP_CONTEXT_BLOCKS = 2
        self._C.MODEL.TRISTRIPE_CONTEXT_PYRAMID = False
        self._C.MODEL.TRISTRIPE_CONTEXT_BLOCKS = 2
        self._C.MODEL.HORIZONTAL_TRAJECTORY_BLOCKS = False
        self._C.MODEL.HMT_NET_CORE = False
        self._C.MODEL.HMT_BLOCKS = 3
        self._C.MODEL.EVIDENCE_CANVAS_NET = False
        self._C.MODEL.EVIDENCE_CANVAS_BLOCKS = 2
        self._C.MODEL.TFR_NET_CORE = False
        self._C.MODEL.TFR_BLOCKS = 8
        self._C.MODEL.TFR_HORIZONTAL_KERNEL = 15
        self._C.MODEL.TFR_VERTICAL_KERNEL = 5
        self._C.MODEL.TFR_REFINER_STEPS = 3
        self._C.MODEL.STRIPE_TOPOLOGY_RECONSTRUCTION = False
        self._C.MODEL.STRD_HIDDEN = 32
        self._C.MODEL.STRD_NUM_BLOCKS = 2
        self._C.MODEL.INTERSTAGE_RESIDUAL_REASSEMBLY = False
        self._C.MODEL.IRR_HIDDEN = 24
        self._C.MODEL.IRR_NUM_BLOCKS = 1
        self._C.MODEL.IRR_DELTA_INIT = 1.0e-4

        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 1
        self._C.OPTIM.NUM_EPOCHS = 100
        self._C.OPTIM.NEPOCH_DECAY = [100]
        self._C.OPTIM.LR_INITIAL = 0.0002
        self._C.OPTIM.LR_MIN = 0.0002
        self._C.OPTIM.BETA1 = 0.5
        self._C.OPTIM.ACCUM_ITERS = 1
        self._C.OPTIM.FREQUENCY_LR_MULTIPLIER = 1.0
        self._C.OPTIM.STAGE2_TMT_LR_MULTIPLIER = 1.0
        self._C.OPTIM.STAGE3_TRR_LR_MULTIPLIER = 1.0
        self._C.OPTIM.STAGE3_TEPR_LR_MULTIPLIER = 1.0
        self._C.OPTIM.STAM_LR_MULTIPLIER = 1.0
        self._C.OPTIM.TPTR_LR_MULTIPLIER = 1.0
        self._C.OPTIM.HMT_LR_MULTIPLIER = 1.0
        self._C.OPTIM.STRD_LR_MULTIPLIER = 1.0
        self._C.OPTIM.IRR_LR_MULTIPLIER = 1.0
        self._C.OPTIM.SGORS_LR_MULTIPLIER = 1.0

        self._C.TRAINING = CN()
        self._C.TRAINING.VAL_AFTER_EVERY = 3
        self._C.TRAINING.DO_VALIDATE = True
        self._C.TRAINING.RESUME = False
        self._C.TRAINING.SAVE_IMAGES = False
        self._C.TRAINING.TRAIN_DIR = 'images_dir/train'
        self._C.TRAINING.VAL_DIR = 'images_dir/val'
        self._C.TRAINING.TRAIN_DIRS = []
        self._C.TRAINING.VAL_DIRS = []
        # Optional explicit paired manifests.  When both are supplied, they
        # override directory scanning and random validation splitting.  This
        # is used only for paired cross-model comparison protocols.
        self._C.TRAINING.TRAIN_MANIFEST = ''
        self._C.TRAINING.VAL_MANIFEST = ''
        self._C.TRAINING.MANIFEST_DATA_ROOT = ''
        self._C.TRAINING.SAVE_DIR = 'checkpoints'
        self._C.TRAINING.TRAIN_PS = 320
        self._C.TRAINING.VAL_PS = 320
        self._C.TRAINING.INPUT_SUBDIR = 'input'
        self._C.TRAINING.TARGET_SUBDIR = 'target'
        self._C.TRAINING.EDGE_SUBDIR = ''
        # 0 keeps every matched image; a positive value takes the first N
        # sorted paired images from each TRAIN_DIR independently.
        self._C.TRAINING.MAX_IMAGES_PER_TRAIN_DIR = 0
        self._C.TRAINING.VAL_SPLIT = 0.1
        self._C.TRAINING.SPLIT_SEED = 42
        self._C.TRAINING.STRATIFIED_VAL_SPLIT = False
        self._C.TRAINING.RESEED_BEFORE_DATA = False
        self._C.TRAINING.EPOCH_DETERMINISTIC_DATA = False
        self._C.TRAINING.NUM_WORKERS = 8
        self._C.TRAINING.SEED = 1234
        self._C.TRAINING.FINETUNE = False
        self._C.TRAINING.PRETRAINED_PATH = ''
        self._C.TRAINING.USE_AMP = True
        self._C.TRAINING.CUDNN_BENCHMARK = False
        self._C.TRAINING.CUDNN_DETERMINISTIC = True
        self._C.TRAINING.CHECKPOINT_EVERY = 3
        self._C.TRAINING.SAVE_BEST = True
        self._C.TRAINING.SAVE_BEST_EVERY = 1
        self._C.TRAINING.SAVE_OPTIMIZER = True
        self._C.TRAINING.STAGE3_TEPR_TRANSITION_EPOCHS = 0
        self._C.TRAINING.SAFE_CHECKPOINTS = False

        self._C.AUGMENT = CN()
        self._C.AUGMENT.ENABLE = True
        self._C.AUGMENT.COLOR_PROB = 0.7
        self._C.AUGMENT.BRIGHTNESS = 0.25
        self._C.AUGMENT.CONTRAST = 0.25
        self._C.AUGMENT.SATURATION = 0.30
        self._C.AUGMENT.HUE = 0.03
        self._C.AUGMENT.GAMMA_PROB = 0.5
        self._C.AUGMENT.GAMMA_MIN = 0.70
        self._C.AUGMENT.GAMMA_MAX = 1.50
        self._C.AUGMENT.INPUT_NOISE_PROB = 0.25
        self._C.AUGMENT.INPUT_NOISE_STD = 0.015
        self._C.AUGMENT.LOW_LIGHT_PROB = 0.0
        self._C.AUGMENT.LOW_LIGHT_FACTOR_MIN = 0.35
        self._C.AUGMENT.LOW_LIGHT_FACTOR_MAX = 0.85
        self._C.AUGMENT.LOW_LIGHT_GAMMA_MIN = 1.10
        self._C.AUGMENT.LOW_LIGHT_GAMMA_MAX = 1.80
        self._C.AUGMENT.PRESERVE_HORIZONTAL_TRAJECTORY = False

        self._C.LOSS = CN()
        self._C.LOSS.STAGE_WEIGHTS = [1.0, 0.5, 0.25]
        self._C.LOSS.EDGE_WEIGHT = 0.05
        self._C.LOSS.DISTILL_WEIGHT = 0.10
        self._C.LOSS.STRIPE_CONSISTENCY_WEIGHT = 0.00
        self._C.LOSS.FREQUENCY_CONSISTENCY_WEIGHT = 0.00
        self._C.LOSS.FREQUENCY_CONSISTENCY_LOW_START = 0.05
        self._C.LOSS.FREQUENCY_CONSISTENCY_LOW_END = 0.10
        self._C.LOSS.FREQUENCY_CONSISTENCY_HIGH_START = 0.30
        self._C.LOSS.FREQUENCY_CONSISTENCY_HIGH_END = 0.40
        self._C.LOSS.FREQUENCY_CONSISTENCY_LOG_MAG_WEIGHT = 0.25
        self._C.LOSS.FREQUENCY_CONSISTENCY_STAGE_WEIGHTS = [1.0, 0.5, 0.25]
        self._C.LOSS.STRUCTURE_TEXTURE_WEIGHT = 0.00
        self._C.LOSS.STRUCTURE_TEXTURE_SIGMA = 1.50

        self._C.DISTILL = CN()
        self._C.DISTILL.ENABLE = False
        self._C.DISTILL.TEACHER_CONFIG = ''
        self._C.DISTILL.TEACHER_WEIGHTS = ''
        self._C.DISTILL.TEACHER_N_FEAT = 96
        self._C.DISTILL.TEACHER_SCALE_UNETFEATS = 48
        self._C.DISTILL.TEACHER_SCALE_ORSNETFEATS = 32
        self._C.DISTILL.TEACHER_NUM_CAB = 8
        self._C.DISTILL.TEACHER_EDGE_REFINE = False
        self._C.DISTILL.TEACHER_EDGE_REFINE_MID = 24
        self._C.DISTILL.TEACHER_AMP = False

        # Override parameter values from YAML file first, then from override list.
        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

        # Make an instantiated object of this class immutable.
        self._C.freeze()

    def dump(self, file_path: str):
        r"""Save config at the specified file path.

        Parameters
        ----------
        file_path: str
            (YAML) path to save config at.
        """
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __repr__(self):
        return self._C.__repr__()
