import argparse
import faulthandler
import os
import random
import shutil
import sys
import time
import traceback
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Sampler, Subset
from tqdm import tqdm

import losses
import utils
from config import Config
from data_RGB import get_training_data, get_validation_data
from MPRNet import MPRNet, build_mprnet_from_config
from warmup_scheduler import GradualWarmupScheduler


CONFIG_PATH = Path(__file__).with_name('training.yml')


def parse_args():
    parser = argparse.ArgumentParser(description='Train Deblurring/MPRNet variants.')
    parser.add_argument(
        '--config',
        type=str,
        default=str(CONFIG_PATH),
        help='Path to a Deblurring training YAML file.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Optional TRAINING.SEED override for paired reproducibility runs.',
    )
    parser.add_argument(
        '--session',
        type=str,
        default=None,
        help='Optional MODEL.SESSION override; use a unique value per seed.',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Optional OPTIM.NUM_EPOCHS override.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Optional OPTIM.BATCH_SIZE override.',
    )
    parser.add_argument(
        '--no-auto-resume',
        action='store_true',
        help='Force a fresh run instead of using an existing session checkpoint.',
    )
    return parser.parse_args()


def _safetensors_path(manifest_path: str) -> str:
    return str(Path(manifest_path).with_suffix('.safetensors'))


def atomic_torch_save(payload, save_path: str, use_safetensors=False) -> None:
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if use_safetensors:
        if 'optimizer' in payload:
            raise ValueError('Safe checkpoints do not serialize optimizer state.')
        from safetensors.torch import save_file

        weights_path = _safetensors_path(save_path)
        weights_tmp = f"{weights_path}.{os.getpid()}.tmp"
        # Cloning breaks shared storages, which safetensors intentionally
        # rejects, while also ensuring no CUDA storage reaches serialization.
        cpu_state = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in payload['state_dict'].items()
        }
        save_file(cpu_state, weights_tmp)
        os.replace(weights_tmp, weights_path)

        manifest = {
            key: checkpoint_to_cpu(value)
            for key, value in payload.items()
            if key != 'state_dict'
        }
        manifest.update({
            'checkpoint_format': 'safetensors_manifest_v1',
            'state_dict_file': os.path.basename(weights_path),
        })
        manifest_tmp = f"{save_path}.{os.getpid()}.tmp"
        torch.save(manifest, manifest_tmp)
        os.replace(manifest_tmp, save_path)
        return

    tmp_path = f"{save_path}.{os.getpid()}.tmp"
    # Saving CUDA storages can enter the CUDA serialization path while a
    # training stream is still active.  Checkpoints are only restored into the
    # target device later, so serialize a CPU-only snapshot instead.
    torch.save(checkpoint_to_cpu(payload), tmp_path)
    os.replace(tmp_path, save_path)


def atomic_checkpoint_alias(source_path: str, target_path: str) -> None:
    """Create an immutable checkpoint snapshot without serializing tensors again."""
    target_dir = os.path.dirname(target_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    source_checkpoint = torch.load(source_path, map_location='cpu')
    if source_checkpoint.get('checkpoint_format') == 'safetensors_manifest_v1':
        source_weights = os.path.join(
            os.path.dirname(source_path), source_checkpoint['state_dict_file']
        )
        target_weights = _safetensors_path(target_path)
        weights_tmp = f"{target_weights}.{os.getpid()}.tmp"
        try:
            os.link(source_weights, weights_tmp)
        except OSError:
            shutil.copyfile(source_weights, weights_tmp)
        os.replace(weights_tmp, target_weights)

        target_manifest = dict(source_checkpoint)
        target_manifest['state_dict_file'] = os.path.basename(target_weights)
        manifest_tmp = f"{target_path}.{os.getpid()}.tmp"
        torch.save(target_manifest, manifest_tmp)
        os.replace(manifest_tmp, target_path)
        return

    tmp_path = f"{target_path}.{os.getpid()}.tmp"
    try:
        # All checkpoints of a run are on the same volume.  A hard link avoids
        # another 246 MB write and, unlike a copy, adds no serialization load.
        os.link(source_path, tmp_path)
    except OSError:
        # Some filesystems do not support hard links; retain the same atomic
        # replacement semantics with a byte-for-byte copy in that case.
        shutil.copyfile(source_path, tmp_path)
    os.replace(tmp_path, target_path)


def checkpoint_to_cpu(value):
    """Recursively copy checkpoint tensors to CPU for safe serialization."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: checkpoint_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [checkpoint_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(checkpoint_to_cpu(item) for item in value)
    return value


def build_checkpoint_payload(epoch, model, optimizer, save_optimizer: bool):
    """Build the smallest checkpoint appropriate for the configured run."""
    payload = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer_lrs': [group['lr'] for group in optimizer.param_groups],
    }
    if save_optimizer:
        payload['optimizer'] = optimizer.state_dict()
    return payload


def enable_crash_logging(result_dir: str):
    """Persist Python and native-fault diagnostics that would otherwise vanish."""
    log_path = os.path.join(result_dir, 'fatal_errors.log')
    error_log = open(log_path, 'a', encoding='utf-8', buffering=1)
    error_log.write(f"\n{'=' * 80}\nTraining process started: {time.ctime()}\n")
    faulthandler.enable(file=error_log, all_threads=True)
    print(f"==> Crash diagnostics: {log_path}")
    default_excepthook = sys.excepthook

    def log_uncaught_exception(exc_type, exc_value, exc_traceback):
        error_log.write(f"\nUnhandled exception at {time.ctime()}:\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=error_log)
        error_log.flush()
        default_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = log_uncaught_exception
    return error_log


def resolve_resume_checkpoint(model_dir: str):
    preferred = Path(model_dir) / "model_latest.pth"
    if preferred.exists() and preferred.stat().st_size > 0:
        return str(preferred)

    candidates = sorted(
        (path for path in Path(model_dir).glob("*.pth") if path.stat().st_size > 0),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])
    return None

# 随机种子
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EpochRandomSampler(Sampler):
    """Deterministic per-epoch sampler that remains stable after a resume."""

    def __init__(self, data_source, seed):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 1

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(
            torch.randperm(len(self.data_source), generator=generator).tolist()
        )

    def __len__(self):
        return len(self.data_source)


def build_dataloaders(opt, device: torch.device, do_validate: bool = True):
    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR
    train_dirs = list(opt.TRAINING.TRAIN_DIRS)
    val_dirs = list(opt.TRAINING.VAL_DIRS)
    input_subdir = opt.TRAINING.INPUT_SUBDIR
    target_subdir = opt.TRAINING.TARGET_SUBDIR
    edge_subdir = opt.TRAINING.EDGE_SUBDIR if hasattr(opt.TRAINING, 'EDGE_SUBDIR') else ''
    num_workers = int(opt.TRAINING.NUM_WORKERS)
    val_split = float(opt.TRAINING.VAL_SPLIT)
    max_images_per_train_dir = int(
        getattr(opt.TRAINING, 'MAX_IMAGES_PER_TRAIN_DIR', 0)
    )
    train_manifest = str(getattr(opt.TRAINING, 'TRAIN_MANIFEST', '') or '')
    val_manifest = str(getattr(opt.TRAINING, 'VAL_MANIFEST', '') or '')
    manifest_data_root = str(
        getattr(opt.TRAINING, 'MANIFEST_DATA_ROOT', '') or ''
    )
    manifest_protocol = bool(train_manifest or val_manifest)
    if manifest_protocol:
        if not train_manifest or not val_manifest:
            raise ValueError(
                'Manifest protocol requires both TRAINING.TRAIN_MANIFEST '
                'and TRAINING.VAL_MANIFEST.'
            )
        if not manifest_data_root:
            raise ValueError(
                'Manifest protocol requires TRAINING.MANIFEST_DATA_ROOT.'
            )
        if not Path(train_manifest).is_file() or not Path(val_manifest).is_file():
            raise FileNotFoundError(
                f'Manifest files not found: {train_manifest} / {val_manifest}'
            )
        if not Path(manifest_data_root).is_dir():
            raise FileNotFoundError(
                f'Manifest data root not found: {manifest_data_root}'
            )

    resolved_train_dirs = train_dirs if train_dirs else [train_dir]
    resolved_val_dirs = val_dirs if val_dirs else ([val_dir] if val_dir else [])

    full_train_dataset = get_training_data(
        resolved_train_dirs,
        {'patch_size': opt.TRAINING.TRAIN_PS, 'augment': opt.AUGMENT},
        input_subdir=input_subdir,
        target_subdir=target_subdir,
        edge_subdir=edge_subdir,
        max_pairs_per_root=max_images_per_train_dir,
        manifest_path=train_manifest or None,
        manifest_data_root=manifest_data_root or None,
    )

    if manifest_protocol and do_validate:
        train_dataset = full_train_dataset
        val_dataset = get_validation_data(
            resolved_train_dirs,
            {'patch_size': opt.TRAINING.VAL_PS},
            input_subdir=input_subdir,
            target_subdir=target_subdir,
            max_pairs_per_root=max_images_per_train_dir,
            manifest_path=val_manifest,
            manifest_data_root=manifest_data_root,
        )
        print(
            f'==> Explicit manifest protocol: train={len(train_dataset)}, '
            f'val={len(val_dataset)}'
        )
    elif not do_validate:
        train_dataset = full_train_dataset
        val_dataset = None
    elif resolved_val_dirs:
        train_dataset = full_train_dataset
        val_dataset = get_validation_data(
            resolved_val_dirs,
            {'patch_size': opt.TRAINING.VAL_PS},
            input_subdir=input_subdir,
            target_subdir=target_subdir,
            max_pairs_per_root=max_images_per_train_dir,
        )
    else:
        full_val_dataset = get_validation_data(
            resolved_train_dirs,
            {'patch_size': opt.TRAINING.VAL_PS},
            input_subdir=input_subdir,
            target_subdir=target_subdir,
            max_pairs_per_root=max_images_per_train_dir,
        )
        total_size = len(full_train_dataset)
        split_seed = int(
            getattr(opt.TRAINING, 'SPLIT_SEED', opt.TRAINING.SEED)
        )
        stratified_split = bool(
            getattr(opt.TRAINING, 'STRATIFIED_VAL_SPLIT', False)
        )
        if stratified_split:
            groups = {}
            for index, filename in enumerate(full_train_dataset.inp_filenames):
                root = str(Path(filename).parent.parent)
                groups.setdefault(root, []).append(index)

            train_indices = []
            val_indices = []
            for group_index, root in enumerate(sorted(groups)):
                group_indices = groups[root]
                generator = torch.Generator().manual_seed(split_seed + group_index)
                order = torch.randperm(
                    len(group_indices), generator=generator
                ).tolist()
                group_val_size = max(1, int(len(group_indices) * val_split))
                group_val_size = min(group_val_size, len(group_indices) - 1)
                shuffled = [group_indices[position] for position in order]
                train_indices.extend(shuffled[:-group_val_size])
                val_indices.extend(shuffled[-group_val_size:])
            print(
                f"==> Stratified validation split: "
                f"groups={len(groups)}, train={len(train_indices)}, "
                f"val={len(val_indices)}, split_seed={split_seed}"
            )
        else:
            val_size = max(1, int(total_size * val_split))
            train_size = max(1, total_size - val_size)
            split_generator = torch.Generator().manual_seed(split_seed)
            shuffled_indices = torch.randperm(
                total_size, generator=split_generator
            ).tolist()
            train_indices = shuffled_indices[:train_size]
            val_indices = shuffled_indices[train_size:]
        if not val_indices:
            val_indices = train_indices[-1:]
            train_indices = train_indices[:-1]

        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_val_dataset, val_indices)

    loader_kwargs = {
        'num_workers': num_workers,
        'drop_last': False,
        'pin_memory': device.type == 'cuda',
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True

    epoch_deterministic = bool(
        getattr(opt.TRAINING, 'EPOCH_DETERMINISTIC_DATA', False)
    )
    if epoch_deterministic:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=opt.OPTIM.BATCH_SIZE,
            sampler=EpochRandomSampler(train_dataset, int(opt.TRAINING.SEED)),
            shuffle=False,
            **loader_kwargs,
        )
    else:
        data_order_generator = torch.Generator().manual_seed(
            int(opt.TRAINING.SEED)
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=opt.OPTIM.BATCH_SIZE,
            shuffle=True,
            generator=data_order_generator,
            **loader_kwargs,
        )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=min(16, opt.OPTIM.BATCH_SIZE),
            shuffle=False,
            **loader_kwargs,
        )
    return train_loader, val_loader


def normalize_stage_weights(weights, num_outputs):
    weights = list(weights) if weights else [1.0] * num_outputs
    if len(weights) < num_outputs:
        weights.extend([weights[-1]] * (num_outputs - len(weights)))
    return [float(weight) for weight in weights[:num_outputs]]


def build_teacher(opt, device):
    if not bool(opt.DISTILL.ENABLE):
        return None
    teacher_weights = opt.DISTILL.TEACHER_WEIGHTS
    if not teacher_weights:
        return None
    if not os.path.exists(teacher_weights):
        raise FileNotFoundError(f"Teacher weights not found: {teacher_weights}")

    teacher_config = str(getattr(opt.DISTILL, 'TEACHER_CONFIG', '') or '')
    if teacher_config:
        teacher_cfg = Config(teacher_config)
        teacher = build_mprnet_from_config(teacher_cfg.MODEL).to(device)
    else:
        teacher = MPRNet(
            n_feat=int(opt.DISTILL.TEACHER_N_FEAT),
            scale_unetfeats=int(opt.DISTILL.TEACHER_SCALE_UNETFEATS),
            scale_orsnetfeats=int(opt.DISTILL.TEACHER_SCALE_ORSNETFEATS),
            num_cab=int(opt.DISTILL.TEACHER_NUM_CAB),
            edge_refine=bool(getattr(opt.DISTILL, 'TEACHER_EDGE_REFINE', False)),
            edge_refine_mid=int(getattr(opt.DISTILL, 'TEACHER_EDGE_REFINE_MID', 24)),
        ).to(device)
    utils.load_checkpoint(teacher, teacher_weights)
    utils.freeze(teacher)
    teacher.eval()
    print(f"==> Distillation teacher loaded: {teacher_weights}")
    return teacher


def restore_scheduler_after_resume(scheduler, optimizer, completed_epochs):
    """Restore warmup/cosine scheduler state without stepping before an update."""
    completed_epochs = max(0, int(completed_epochs))
    scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]
    if completed_epochs <= scheduler.total_epoch:
        scheduler.last_epoch = completed_epochs
        return

    # GradualWarmupScheduler switches to its after-scheduler immediately after
    # the completed warmup epochs. Its own last_epoch then stays at total+1.
    scheduler.last_epoch = scheduler.total_epoch + 1
    scheduler.finished = True
    if scheduler.after_scheduler is not None:
        scheduler.after_scheduler.last_epoch = completed_epochs - (scheduler.total_epoch + 1)
        scheduler.after_scheduler._last_lr = list(scheduler._last_lr)


def main(config_path: str = str(CONFIG_PATH), config_overrides=None) -> None:
    opt = Config(str(config_path), config_overrides or [])

    gpus = ','.join([str(i) for i in opt.GPU])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    cudnn_benchmark = bool(getattr(opt.TRAINING, 'CUDNN_BENCHMARK', False))
    cudnn_deterministic = bool(getattr(opt.TRAINING, 'CUDNN_DETERMINISTIC', True))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    set_seeds(int(opt.TRAINING.SEED))

    start_epoch = 1
    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION

    result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
    model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)

    utils.mkdir(result_dir)
    utils.mkdir(model_dir)
    crash_log = enable_crash_logging(result_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = bool(opt.TRAINING.USE_AMP) and device.type == 'cuda'
    accum_iters = max(1, int(opt.OPTIM.ACCUM_ITERS))
    do_validate = bool(opt.TRAINING.DO_VALIDATE)
    progress_update_every = 10
    scaler = GradScaler(enabled=use_amp)

    model_restoration = build_mprnet_from_config(opt.MODEL).to(device)
    student_params = sum(p.numel() for p in model_restoration.parameters())
    print(f"==> Student model parameters: {student_params / 1e6:.3f}M")
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    new_lr = opt.OPTIM.LR_INITIAL
    frequency_lr_multiplier = float(
        getattr(opt.OPTIM, 'FREQUENCY_LR_MULTIPLIER', 1.0)
    )
    if frequency_lr_multiplier <= 0:
        raise ValueError('OPTIM.FREQUENCY_LR_MULTIPLIER must be positive.')

    stage2_tmt_lr_multiplier = float(
        getattr(opt.OPTIM, 'STAGE2_TMT_LR_MULTIPLIER', 1.0)
    )
    if stage2_tmt_lr_multiplier <= 0:
        raise ValueError('OPTIM.STAGE2_TMT_LR_MULTIPLIER must be positive.')

    stage3_trr_lr_multiplier = float(
        getattr(opt.OPTIM, 'STAGE3_TRR_LR_MULTIPLIER', 1.0)
    )
    if stage3_trr_lr_multiplier <= 0:
        raise ValueError('OPTIM.STAGE3_TRR_LR_MULTIPLIER must be positive.')

    stage3_tepr_lr_multiplier = float(
        getattr(opt.OPTIM, 'STAGE3_TEPR_LR_MULTIPLIER', 1.0)
    )
    if stage3_tepr_lr_multiplier <= 0:
        raise ValueError('OPTIM.STAGE3_TEPR_LR_MULTIPLIER must be positive.')

    stam_lr_multiplier = float(
        getattr(opt.OPTIM, 'STAM_LR_MULTIPLIER', 1.0)
    )
    if stam_lr_multiplier <= 0:
        raise ValueError('OPTIM.STAM_LR_MULTIPLIER must be positive.')

    irr_lr_multiplier = float(
        getattr(opt.OPTIM, 'IRR_LR_MULTIPLIER', 1.0)
    )
    if irr_lr_multiplier <= 0:
        raise ValueError('OPTIM.IRR_LR_MULTIPLIER must be positive.')

    strd_lr_multiplier = float(
        getattr(opt.OPTIM, 'STRD_LR_MULTIPLIER', 1.0)
    )
    if strd_lr_multiplier <= 0:
        raise ValueError('OPTIM.STRD_LR_MULTIPLIER must be positive.')

    tptr_lr_multiplier = float(
        getattr(opt.OPTIM, 'TPTR_LR_MULTIPLIER', 1.0)
    )
    if tptr_lr_multiplier <= 0:
        raise ValueError('OPTIM.TPTR_LR_MULTIPLIER must be positive.')

    hmt_lr_multiplier = float(
        getattr(opt.OPTIM, 'HMT_LR_MULTIPLIER', 1.0)
    )
    if hmt_lr_multiplier <= 0:
        raise ValueError('OPTIM.HMT_LR_MULTIPLIER must be positive.')

    sgors_lr_multiplier = float(
        getattr(opt.OPTIM, 'SGORS_LR_MULTIPLIER', 1.0)
    )
    if sgors_lr_multiplier <= 0:
        raise ValueError('OPTIM.SGORS_LR_MULTIPLIER must be positive.')

    special_param_groups = []
    special_param_ids = set()
    if model_restoration.frequency_refine_enabled and frequency_lr_multiplier != 1.0:
        frequency_params = list(model_restoration.frequency_refine.parameters())
        special_param_ids.update(id(param) for param in frequency_params)
        special_param_groups.append(
            {
                'params': frequency_params,
                'lr': new_lr * frequency_lr_multiplier,
                'name': 'frequency_refine',
            }
        )
        print(
            f"==> Frequency refiner learning rate: "
            f"{new_lr * frequency_lr_multiplier:.6g} "
            f"({frequency_lr_multiplier:g}x backbone)"
        )

    if (
        getattr(model_restoration, 'stage2_tmt_enabled', False)
        and stage2_tmt_lr_multiplier != 1.0
    ):
        stage2_tmt_tokens = (
            'tmt_units',
            'trajectory_refiners',
            'feature_refiner',
            'image_residual',
            'image_scale',
        )
        stage2_tmt_params = [
            param for name, param in model_restoration.named_parameters()
            if any(token in name for token in stage2_tmt_tokens)
        ]
        special_param_ids.update(id(param) for param in stage2_tmt_params)
        special_param_groups.append(
            {
                'params': stage2_tmt_params,
                'lr': new_lr * stage2_tmt_lr_multiplier,
                'name': 'stage2_tmt',
            }
        )
        print(
            f"==> Stage-2 TMT learning rate: "
            f"{new_lr * stage2_tmt_lr_multiplier:.6g} "
            f"({stage2_tmt_lr_multiplier:g}x backbone)"
        )

    if (
        getattr(model_restoration, 'stage3_trr_enabled', False)
        and stage3_trr_lr_multiplier != 1.0
    ):
        stage3_trr_tokens = (
            'trr1',
            'trr2',
            'trr3',
            'stage3_synthesis',
        )
        stage3_trr_params = [
            param for name, param in model_restoration.named_parameters()
            if any(token in name for token in stage3_trr_tokens)
        ]
        special_param_ids.update(id(param) for param in stage3_trr_params)
        special_param_groups.append(
            {
                'params': stage3_trr_params,
                'lr': new_lr * stage3_trr_lr_multiplier,
                'name': 'stage3_trr',
            }
        )
        print(
            f"==> Stage-3 TRR learning rate: "
            f"{new_lr * stage3_trr_lr_multiplier:.6g} "
            f"({stage3_trr_lr_multiplier:g}x backbone)"
        )

    if (
        getattr(model_restoration, 'stage3_tepr_enabled', False)
        and stage3_tepr_lr_multiplier != 1.0
    ):
        tepr_params = list(model_restoration.stage3_tepr.parameters())
        tepr_params += list(model_restoration.tepr_head.parameters())
        special_param_ids.update(id(param) for param in tepr_params)
        special_param_groups.append(
            {
                'params': tepr_params,
                'lr': new_lr * stage3_tepr_lr_multiplier,
                'name': 'stage3_tepr',
            }
        )
        print(
            f"==> Stage-3 TEPR learning rate: "
            f"{new_lr * stage3_tepr_lr_multiplier:.6g} "
            f"({stage3_tepr_lr_multiplier:g}x backbone)"
        )

    if (
        model_restoration.stitch_trajectory_alignment is not None
        and stam_lr_multiplier != 1.0
    ):
        stam_params = list(model_restoration.stitch_trajectory_alignment.parameters())
        special_param_ids.update(id(param) for param in stam_params)
        special_param_groups.append(
            {
                'params': stam_params,
                'lr': new_lr * stam_lr_multiplier,
                'name': 'stam',
            }
        )
        print(
            f"==> STAM learning rate: {new_lr * stam_lr_multiplier:.6g} "
            f"({stam_lr_multiplier:g}x backbone)"
        )

    if (
        model_restoration.interstage_residual_reassembly_enabled
        and irr_lr_multiplier != 1.0
    ):
        irr_params = list(
            model_restoration.interstage_residual_reassembly.parameters()
        )
        special_param_ids.update(id(param) for param in irr_params)
        special_param_groups.append(
            {
                'params': irr_params,
                'lr': new_lr * irr_lr_multiplier,
                'name': 'irr',
            }
        )
        print(
            f"==> IRR learning rate: {new_lr * irr_lr_multiplier:.6g} "
            f"({irr_lr_multiplier:g}x backbone)"
        )

    if (
        model_restoration.stripe_topology_reconstruction is not None
        and strd_lr_multiplier != 1.0
    ):
        strd_params = list(
            model_restoration.stripe_topology_reconstruction.parameters()
        )
        special_param_ids.update(id(param) for param in strd_params)
        special_param_groups.append(
            {
                'params': strd_params,
                'lr': new_lr * strd_lr_multiplier,
                'name': 'strd',
            }
        )
        print(
            f"==> STRD learning rate: {new_lr * strd_lr_multiplier:.6g} "
            f"({strd_lr_multiplier:g}x backbone)"
        )

    if (
        model_restoration.tpt_rnet_core is not None
        and tptr_lr_multiplier != 1.0
    ):
        tptr_params = list(model_restoration.tpt_rnet_core.parameters())
        if model_restoration.stage1_bridge is not None:
            tptr_params += list(model_restoration.stage1_bridge.parameters())
        special_param_ids.update(id(param) for param in tptr_params)
        special_param_groups.append(
            {
                'params': tptr_params,
                'lr': new_lr * tptr_lr_multiplier,
                'name': 'tpt_rnet_core',
            }
        )
        print(
            f"==> TPT-RNet core learning rate: "
            f"{new_lr * tptr_lr_multiplier:.6g} "
            f"({tptr_lr_multiplier:g}x backbone)"
        )

    if (
        model_restoration.hmt_net_core is not None
        and hmt_lr_multiplier != 1.0
    ):
        hmt_params = list(model_restoration.hmt_net_core.parameters())
        special_param_ids.update(id(param) for param in hmt_params)
        special_param_groups.append(
            {
                'params': hmt_params,
                'lr': new_lr * hmt_lr_multiplier,
                'name': 'hmt_main',
            }
        )
        print(
            f"==> HMT main learning rate: "
            f"{new_lr * hmt_lr_multiplier:.6g} "
            f"({hmt_lr_multiplier:g}x backbone)"
        )

    if (
        (model_restoration.simple_gated_orsnet_enabled
         or model_restoration.hybrid_simple_gated_orsnet_enabled
         or getattr(model_restoration, 'hsgors_v2_enabled', False))
        and sgors_lr_multiplier != 1.0
    ):
        if (
            model_restoration.hybrid_simple_gated_orsnet_enabled
            or getattr(model_restoration, 'hsgors_v2_enabled', False)
        ):
            sgors_params = list(model_restoration.stage3_orsnet.adapter1.parameters())
            sgors_params += list(model_restoration.stage3_orsnet.adapter2.parameters())
            sgors_params += list(model_restoration.stage3_orsnet.adapter3.parameters())
            sgors_group_name = (
                'hsgors_v2_adapters'
                if getattr(model_restoration, 'hsgors_v2_enabled', False)
                else 'hybrid_sgors_adapters'
            )
        else:
            sgors_params = list(model_restoration.stage3_orsnet.orb1.parameters())
            sgors_params += list(model_restoration.stage3_orsnet.orb2.parameters())
            sgors_params += list(model_restoration.stage3_orsnet.orb3.parameters())
            sgors_group_name = 'sgors'
        special_param_ids.update(id(param) for param in sgors_params)
        special_param_groups.append(
            {
                'params': sgors_params,
                'lr': new_lr * sgors_lr_multiplier,
                'name': sgors_group_name,
            }
        )
        print(
            f"==> SG-ORS learning rate: {new_lr * sgors_lr_multiplier:.6g} "
            f"({sgors_lr_multiplier:g}x backbone)"
        )

    if special_param_groups:
        backbone_params = [
            param for param in model_restoration.parameters()
            if param.requires_grad and id(param) not in special_param_ids
        ]
        optimizer_params = [
            {'params': backbone_params, 'lr': new_lr, 'name': 'backbone'},
            *special_param_groups,
        ]
    else:
        optimizer_params = [
            param for param in model_restoration.parameters()
            if param.requires_grad
        ]

    optimizer = optim.Adam(
        optimizer_params,
        lr=new_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        opt.OPTIM.NUM_EPOCHS - warmup_epochs,
        eta_min=opt.OPTIM.LR_MIN,
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_epoch=warmup_epochs,
        after_scheduler=scheduler_cosine,
    )

    best_psnr = 0
    best_epoch = 0
    save_best = bool(opt.TRAINING.SAVE_BEST)
    save_optimizer = bool(getattr(opt.TRAINING, 'SAVE_OPTIMIZER', True))
    safe_checkpoints = bool(
        getattr(opt.TRAINING, 'SAFE_CHECKPOINTS', False)
    )
    if safe_checkpoints and save_optimizer:
        raise ValueError(
            'TRAINING.SAFE_CHECKPOINTS requires SAVE_OPTIMIZER=False.'
        )
    save_best_every = max(1, int(opt.TRAINING.SAVE_BEST_EVERY))
    checkpoint_every = max(
        1,
        int(getattr(opt.TRAINING, 'CHECKPOINT_EVERY', opt.TRAINING.VAL_AFTER_EVERY)),
    )

    path_chk_rest = resolve_resume_checkpoint(model_dir) if opt.TRAINING.RESUME else None
    if path_chk_rest is not None:
        utils.load_checkpoint(model_restoration, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        resume_checkpoint = torch.load(path_chk_rest, map_location='cpu')
        if 'optimizer' in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint['optimizer'])
        else:
            saved_lrs = resume_checkpoint.get('optimizer_lrs', [])
            for group, saved_lr in zip(optimizer.param_groups, saved_lrs):
                group['lr'] = float(saved_lr)
            print(
                '==> Resume checkpoint has no optimizer state; '
                'restored model, epoch, and saved learning rates.'
            )
        restore_scheduler_after_resume(scheduler, optimizer, start_epoch - 1)
        new_lr = optimizer.param_groups[0]['lr']
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')
    elif opt.TRAINING.FINETUNE and opt.TRAINING.PRETRAINED_PATH:
        pretrained_path = opt.TRAINING.PRETRAINED_PATH
        print('------------------------------------------------------------------------------')
        print(f"==> Fine-tuning from pretrained weights: {pretrained_path}")
        print('------------------------------------------------------------------------------')
        incompatible, skipped = utils.load_checkpoint_partial(model_restoration, pretrained_path)
        print(
            f"==> Partially loaded pretrained weights: "
            f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}, "
            f"skipped_shape={len(skipped)}"
        )
        if getattr(model_restoration, 'stage3_tepr_transition_enabled', False):
            model_restoration.initialize_stage3_tepr_from_legacy()
            print('==> TEPR transition initialized its synthesis projection from the legacy RGB head.')
    elif opt.TRAINING.RESUME:
        print('------------------------------------------------------------------------------')
        print(f"==> No checkpoint found in {model_dir}; starting a new training run")
        print('------------------------------------------------------------------------------')

    if not do_validate:
        print("==> Validation is disabled (TRAINING.DO_VALIDATE=False)")
    if not save_best:
        print("==> Best checkpoint saving is disabled (TRAINING.SAVE_BEST=False)")
    elif save_best_every > 1:
        print(f"==> Best checkpoint save frequency: every {save_best_every} validation round(s)")
    if checkpoint_every != int(opt.TRAINING.VAL_AFTER_EVERY):
        print(f"==> Epoch checkpoint save frequency: every {checkpoint_every} epoch(s)")
    if not save_optimizer:
        print("==> Checkpoints save model weights only (optimizer state omitted)")

    if len(device_ids) > 1:
        model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

    criterion_char = losses.CharbonnierLoss()
    criterion_edge = losses.EdgeLoss()
    criterion_edge_map = losses.EdgeMapLoss()
    stage_weights = normalize_stage_weights(opt.LOSS.STAGE_WEIGHTS, 3)
    edge_supervision_weight = float(opt.LOSS.EDGE_WEIGHT)
    distill_weight = float(opt.LOSS.DISTILL_WEIGHT)
    stripe_consistency_weight = float(
        getattr(opt.LOSS, 'STRIPE_CONSISTENCY_WEIGHT', 0.0)
    )
    frequency_consistency_weight = float(
        getattr(opt.LOSS, 'FREQUENCY_CONSISTENCY_WEIGHT', 0.0)
    )
    structure_texture_weight = float(
        getattr(opt.LOSS, 'STRUCTURE_TEXTURE_WEIGHT', 0.0)
    )
    structure_texture_sigma = float(
        getattr(opt.LOSS, 'STRUCTURE_TEXTURE_SIGMA', 1.50)
    )
    if structure_texture_weight < 0.0:
        raise ValueError('LOSS.STRUCTURE_TEXTURE_WEIGHT must be non-negative.')
    if structure_texture_sigma <= 0.0:
        raise ValueError('LOSS.STRUCTURE_TEXTURE_SIGMA must be positive.')
    criterion_frequency = None
    frequency_stage_weights = []
    if frequency_consistency_weight > 0.0:
        criterion_frequency = losses.BandLimitedFrequencyLoss(
            low_start=float(getattr(opt.LOSS, 'FREQUENCY_CONSISTENCY_LOW_START', 0.05)),
            low_end=float(getattr(opt.LOSS, 'FREQUENCY_CONSISTENCY_LOW_END', 0.10)),
            high_start=float(getattr(opt.LOSS, 'FREQUENCY_CONSISTENCY_HIGH_START', 0.30)),
            high_end=float(getattr(opt.LOSS, 'FREQUENCY_CONSISTENCY_HIGH_END', 0.40)),
            log_magnitude_weight=float(
                getattr(opt.LOSS, 'FREQUENCY_CONSISTENCY_LOG_MAG_WEIGHT', 0.25)
            ),
        )
        frequency_stage_weights = list(
            getattr(opt.LOSS, 'FREQUENCY_CONSISTENCY_STAGE_WEIGHTS', [1.0])
        )
        print(
            "==> Frequency consistency loss enabled: "
            f"weight={frequency_consistency_weight:g}, "
            f"stage_weights={frequency_stage_weights}"
        )
    use_edge_supervision = bool(opt.MODEL.EDGE_REFINE) and edge_supervision_weight > 0
    teacher_model = build_teacher(opt, device)
    teacher_use_amp = bool(getattr(opt.DISTILL, 'TEACHER_AMP', False)) and use_amp
    print(
        f"==> cuDNN: benchmark={cudnn_benchmark}, deterministic={cudnn_deterministic}; "
        f"teacher_amp={teacher_use_amp}"
    )

    if bool(getattr(opt.TRAINING, 'RESEED_BEFORE_DATA', False)):
        # Model construction differs between M2 and M5-v3. Re-seeding here
        # prevents those extra initializations from changing paired data order.
        set_seeds(int(opt.TRAINING.SEED))
    train_loader, val_loader = build_dataloaders(opt, device, do_validate=do_validate)

    tepr_transition_epochs = int(
        getattr(opt.TRAINING, 'STAGE3_TEPR_TRANSITION_EPOCHS', 0)
    )
    tepr_transition_enabled = getattr(
        model_restoration, 'stage3_tepr_transition_enabled', False
    )
    if tepr_transition_enabled and tepr_transition_epochs < 2:
        raise ValueError(
            'TRAINING.STAGE3_TEPR_TRANSITION_EPOCHS must be at least 2 '
            'when Stage-3 TEPR transition is enabled.'
        )
    if tepr_transition_enabled:
        print(
            '==> Stage-3 TEPR transition: legacy path is frozen; '
            f'alpha ramps from 0 to 1 over {tepr_transition_epochs} epoch(s).'
        )

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
    print('===> Loading datasets')

    total_batches = len(train_loader) * (opt.OPTIM.NUM_EPOCHS - start_epoch + 1)
    progress = tqdm(total=total_batches, desc='Training', dynamic_ncols=True)
    try:
        for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
            if tepr_transition_enabled:
                tepr_alpha = min(
                    1.0,
                    float(epoch - 1) / float(max(1, tepr_transition_epochs - 1)),
                )
                transition_model = (
                    model_restoration.module
                    if isinstance(model_restoration, nn.DataParallel)
                    else model_restoration
                )
                transition_model.set_stage3_tepr_transition_alpha(tepr_alpha)
            else:
                tepr_alpha = 1.0
            if bool(getattr(opt.TRAINING, 'EPOCH_DETERMINISTIC_DATA', False)):
                set_seeds(int(opt.TRAINING.SEED) + epoch)
                if hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
            sampled_epoch_loss = 0.0
            sampled_loss_count = 0
            save_best_checkpoint = False

            model_restoration.train()
            optimizer.zero_grad(set_to_none=True)
            for batch_idx, data in enumerate(train_loader, 0):

                target = data[0].to(device)
                input_ = data[1].to(device)
                edge_target = data[3].to(device) if use_edge_supervision and len(data) > 3 else None

                with autocast(enabled=use_amp):
                    return_aux = (
                        use_edge_supervision
                        or stripe_consistency_weight > 0
                        or structure_texture_weight > 0
                    )
                    model_output = model_restoration(input_, return_edges=return_aux)
                    if isinstance(model_output, tuple):
                        restored, extras = model_output
                    else:
                        restored, extras = model_output, {}

                    active_stage_weights = normalize_stage_weights(stage_weights, len(restored))
                    loss_char = sum(
                        weight * criterion_char(stage_output, target)
                        for weight, stage_output in zip(active_stage_weights, restored)
                    )
                    loss_edge = sum(
                        weight * criterion_edge(stage_output, target)
                        for weight, stage_output in zip(active_stage_weights, restored)
                    )
                    loss = loss_char + (0.05 * loss_edge)

                    loss_edge_map = target.new_tensor(0.0)
                    if edge_target is not None:
                        loss_edge_map = criterion_edge_map(extras.get('edge'), edge_target)
                        loss = loss + edge_supervision_weight * loss_edge_map

                    loss_stripe_consistency = target.new_tensor(0.0)
                    if stripe_consistency_weight > 0:
                        loss_stripe_consistency = extras.get(
                            'stripe_consistency', loss_stripe_consistency
                        )
                        loss = loss + stripe_consistency_weight * loss_stripe_consistency

                    loss_structure_texture = target.new_tensor(0.0)
                    if structure_texture_weight > 0:
                        kernel_size = max(3, int(round(structure_texture_sigma * 4)) * 2 + 1)
                        pad = kernel_size // 2
                        target_low = F.avg_pool2d(
                            F.pad(target, (pad, pad, pad, pad), mode='reflect'),
                            kernel_size=kernel_size,
                            stride=1,
                        )
                        input_low = F.avg_pool2d(
                            F.pad(input_, (pad, pad, pad, pad), mode='reflect'),
                            kernel_size=kernel_size,
                            stride=1,
                        )
                        target_structure_delta = target_low - input_low
                        target_texture_delta = (
                            (target - target_low) - (input_ - input_low)
                        )
                        predicted_structure = extras.get('structure')
                        predicted_texture = extras.get('texture')
                        if predicted_structure is None or predicted_texture is None:
                            raise ValueError(
                                'Structure-texture loss requires the HMT-Net synthesis outputs.'
                            )
                        loss_structure_texture = (
                            criterion_char(
                                predicted_structure, target_structure_delta
                            )
                            + criterion_char(
                                predicted_texture, target_texture_delta
                            )
                        )
                        loss = loss + structure_texture_weight * loss_structure_texture

                    irr_delta_abs = extras.get(
                        'irr_delta_abs', target.new_tensor(0.0)
                    )

                    loss_frequency_consistency = target.new_tensor(0.0)
                    if criterion_frequency is not None:
                        active_frequency_weights = normalize_stage_weights(
                            frequency_stage_weights, len(restored)
                        )
                        loss_frequency_consistency = sum(
                            weight * criterion_frequency(stage_output, target)
                            for weight, stage_output in zip(active_frequency_weights, restored)
                        )
                        loss = loss + (
                            frequency_consistency_weight
                            * loss_frequency_consistency
                        )

                    loss_distill = target.new_tensor(0.0)
                    if teacher_model is not None and distill_weight > 0:
                        with torch.no_grad():
                            # The frozen teacher is intentionally kept in FP32.
                            # This avoids the unstable cuDNN inference + AMP path
                            # while the student still benefits from AMP training.
                            with autocast(enabled=teacher_use_amp):
                                teacher_restored = teacher_model(
                                    input_, return_intermediate=False
                                )[0]
                        loss_distill = criterion_char(restored[0], teacher_restored)
                        loss = loss + distill_weight * loss_distill

                if not torch.isfinite(loss).all():
                    raise FloatingPointError(
                        f"Non-finite loss at epoch {epoch}, batch {batch_idx + 1}: "
                        f"loss={loss.detach().float().item()}"
                    )

                scaled_loss = loss / accum_iters
                scaler.scale(scaled_loss).backward()

                should_step = ((batch_idx + 1) % accum_iters == 0) or ((batch_idx + 1) == len(train_loader))
                if should_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                should_update_progress = (
                    ((batch_idx + 1) % progress_update_every == 0)
                    or ((batch_idx + 1) == len(train_loader))
                )
                if should_update_progress:
                    loss_value = loss.detach().item()
                    sampled_epoch_loss += loss_value
                    sampled_loss_count += 1
                    avg_loss = sampled_epoch_loss / sampled_loss_count
                    progress.set_postfix(
                        epoch=f'{epoch}/{opt.OPTIM.NUM_EPOCHS}',
                        batch=f'{batch_idx + 1}/{len(train_loader)}',
                        loss=f'{loss_value:.4f}',
                        edge_map=f'{loss_edge_map.detach().item():.4f}',
                        stripe=f'{loss_stripe_consistency.detach().item():.4f}',
                        irr=f'{irr_delta_abs.detach().item():.5f}',
                        freq=f'{loss_frequency_consistency.detach().item():.4f}',
                        distill=f'{loss_distill.detach().item():.4f}',
                        tepr_alpha=f'{tepr_alpha:.2f}',
                        avg_sampled=f'{avg_loss:.4f}',
                        lr=f'{scheduler.get_lr()[0]:.6f}',
                        amp=use_amp,
                        accum=accum_iters,
                    )
                progress.update(1)

            if do_validate and val_loader is not None and epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
                model_restoration.eval()
                psnr_val_rgb = []
                for data_val in val_loader:
                    target = data_val[0].to(device)
                    input_ = data_val[1].to(device)

                    with torch.no_grad():
                        with autocast(enabled=use_amp):
                            restored = model_restoration(input_)
                    restored = restored[0]

                    for res, tar in zip(restored, target):
                        psnr_val_rgb.append(utils.torchPSNR(res, tar))

                psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

                # A transition checkpoint with alpha < 1 still contains the
                # training-only legacy recovery path. It must never be marked
                # as the final best model.
                final_architecture_active = (
                    not tepr_transition_enabled or tepr_alpha >= 1.0 - 1e-8
                )
                if final_architecture_active and psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    if save_best and ((epoch // opt.TRAINING.VAL_AFTER_EVERY) % save_best_every == 0):
                        save_best_checkpoint = True

                progress.set_postfix(
                    epoch=f'{epoch}/{opt.OPTIM.NUM_EPOCHS}',
                    val_psnr=f'{psnr_val_rgb:.4f}',
                    best=f'{best_psnr:.4f}@{best_epoch}',
                    lr=f'{scheduler.get_lr()[0]:.6f}',
                    amp=use_amp,
                    accum=accum_iters,
                )

            scheduler.step()

            latest_checkpoint = os.path.join(model_dir, "model_latest.pth")
            atomic_torch_save(
                build_checkpoint_payload(
                    epoch,
                    model_restoration,
                    optimizer,
                    save_optimizer,
                ),
                latest_checkpoint,
                use_safetensors=safe_checkpoints,
            )
            if epoch % checkpoint_every == 0:
                atomic_checkpoint_alias(
                    latest_checkpoint,
                    os.path.join(model_dir, f"model_epoch_{epoch}.pth"),
                )
            if save_best_checkpoint:
                atomic_checkpoint_alias(
                    latest_checkpoint,
                    os.path.join(model_dir, "model_best.pth"),
                )
    finally:
        progress.close()


if __name__ == '__main__':
    freeze_support()
    args = parse_args()
    overrides = []
    if args.seed is not None:
        overrides.extend(['TRAINING.SEED', args.seed])
    if args.session:
        overrides.extend(['MODEL.SESSION', args.session])
    if args.epochs is not None:
        overrides.extend(['OPTIM.NUM_EPOCHS', args.epochs])
    if args.batch_size is not None:
        overrides.extend(['OPTIM.BATCH_SIZE', args.batch_size])
    if args.no_auto_resume:
        overrides.extend(['TRAINING.RESUME', False])
    main(args.config, overrides)
