import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random
from pathlib import Path

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


def _normalize_roots(rgb_dir):
    if isinstance(rgb_dir, (list, tuple)):
        return [Path(path) for path in rgb_dir]
    return [Path(rgb_dir)]


def _collect_pairs(rgb_dir, input_subdir, target_subdir, max_pairs_per_root=0):
    pairs = []
    for root in _normalize_roots(rgb_dir):
        input_dir = root / input_subdir
        target_dir = root / target_subdir
        if not input_dir.is_dir() or not target_dir.is_dir():
            raise FileNotFoundError(
                f"Required directories not found under {root}: "
                f"{input_dir} / {target_dir}"
            )

        input_files = {
            Path(name).stem: input_dir / name
            for name in sorted(os.listdir(input_dir))
            if is_image_file(name)
        }
        target_files = {
            Path(name).stem: target_dir / name
            for name in sorted(os.listdir(target_dir))
            if is_image_file(name)
        }

        common_stems = sorted(set(input_files) & set(target_files))
        if not common_stems:
            raise ValueError(f"No matching input/target pairs found in {root}")

        max_pairs_per_root = int(max_pairs_per_root or 0)
        if max_pairs_per_root > 0:
            common_stems = common_stems[:max_pairs_per_root]
        pairs.extend((input_files[stem], target_files[stem]) for stem in common_stems)

    return pairs


def _collect_manifest_pairs(manifest_path, data_root):
    """Read explicit ``clear<TAB>blur`` pairs used by the shared V2 protocol.

    The controlled SynBlur filenames include a degradation suffix, so pairing by
    filename stem is not valid for this protocol.  Keeping the manifest as the
    sole source of truth also prevents a MPRNet run from silently using a
    different train/validation split than NAFNet or MBDNet.
    """

    manifest = Path(manifest_path)
    root = Path(data_root)
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if not root.is_dir():
        raise FileNotFoundError(f"Manifest data root not found: {root}")

    pairs = []
    for line_number, line in enumerate(manifest.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        columns = line.split('\t')
        if len(columns) != 2:
            raise ValueError(f"Expected clear<TAB>blur at {manifest}:{line_number}")
        target_rel, input_rel = columns
        target = root / target_rel
        input_ = root / input_rel
        if not target.is_file() or not input_.is_file():
            raise FileNotFoundError(
                f"Manifest pair missing at {manifest}:{line_number}: {target} / {input_}"
            )
        pairs.append((input_, target))

    if not pairs:
        raise ValueError(f"No pairs found in manifest: {manifest}")
    return pairs


def _cfg_get(cfg, name, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _random_positive_factor(strength):
    strength = max(0.0, float(strength))
    return random.uniform(max(0.0, 1.0 - strength), 1.0 + strength)


def _apply_paired_color_augment(inp_img, tar_img, aug_options):
    if not bool(_cfg_get(aug_options, 'ENABLE', False)):
        return inp_img, tar_img

    color_prob = float(_cfg_get(aug_options, 'COLOR_PROB', 0.0))
    if random.random() < color_prob:
        brightness = _random_positive_factor(_cfg_get(aug_options, 'BRIGHTNESS', 0.0))
        contrast = _random_positive_factor(_cfg_get(aug_options, 'CONTRAST', 0.0))
        saturation = _random_positive_factor(_cfg_get(aug_options, 'SATURATION', 0.0))
        hue_strength = min(0.5, max(0.0, float(_cfg_get(aug_options, 'HUE', 0.0))))
        hue = random.uniform(-hue_strength, hue_strength)

        for adjust_fn, value in (
            (TF.adjust_brightness, brightness),
            (TF.adjust_contrast, contrast),
            (TF.adjust_saturation, saturation),
        ):
            inp_img = adjust_fn(inp_img, value)
            tar_img = adjust_fn(tar_img, value)
        if hue_strength > 0:
            inp_img = TF.adjust_hue(inp_img, hue)
            tar_img = TF.adjust_hue(tar_img, hue)

    gamma_prob = float(_cfg_get(aug_options, 'GAMMA_PROB', 0.0))
    if random.random() < gamma_prob:
        gamma_min = float(_cfg_get(aug_options, 'GAMMA_MIN', 1.0))
        gamma_max = float(_cfg_get(aug_options, 'GAMMA_MAX', 1.0))
        if gamma_min > gamma_max:
            gamma_min, gamma_max = gamma_max, gamma_min
        gamma = random.uniform(max(0.1, gamma_min), max(0.1, gamma_max))
        inp_img = TF.adjust_gamma(inp_img, gamma)
        tar_img = TF.adjust_gamma(tar_img, gamma)

    return inp_img, tar_img


def _apply_input_low_light_augment(inp_img, aug_options):
    if not bool(_cfg_get(aug_options, 'ENABLE', False)):
        return inp_img

    prob = float(_cfg_get(aug_options, 'LOW_LIGHT_PROB', 0.0))
    if random.random() >= prob:
        return inp_img

    factor_min = float(_cfg_get(aug_options, 'LOW_LIGHT_FACTOR_MIN', 0.35))
    factor_max = float(_cfg_get(aug_options, 'LOW_LIGHT_FACTOR_MAX', 0.85))
    gamma_min = float(_cfg_get(aug_options, 'LOW_LIGHT_GAMMA_MIN', 1.10))
    gamma_max = float(_cfg_get(aug_options, 'LOW_LIGHT_GAMMA_MAX', 1.80))
    if factor_min > factor_max:
        factor_min, factor_max = factor_max, factor_min
    if gamma_min > gamma_max:
        gamma_min, gamma_max = gamma_max, gamma_min

    brightness = random.uniform(max(0.05, factor_min), max(0.05, factor_max))
    gamma = random.uniform(max(0.1, gamma_min), max(0.1, gamma_max))
    inp_img = TF.adjust_brightness(inp_img, brightness)
    inp_img = TF.adjust_gamma(inp_img, gamma)
    return inp_img


def _apply_input_noise(inp_img, aug_options):
    if not bool(_cfg_get(aug_options, 'ENABLE', False)):
        return inp_img
    noise_prob = float(_cfg_get(aug_options, 'INPUT_NOISE_PROB', 0.0))
    noise_std = float(_cfg_get(aug_options, 'INPUT_NOISE_STD', 0.0))
    if noise_std > 0 and random.random() < noise_prob:
        std = random.uniform(0.0, noise_std)
        inp_img = torch.clamp(inp_img + torch.randn_like(inp_img) * std, 0.0, 1.0)
    return inp_img

class DataLoaderTrain(Dataset):
    def __init__(
        self,
        rgb_dir,
        img_options=None,
        input_subdir='input',
        target_subdir='target',
        edge_subdir=None,
        max_pairs_per_root=0,
        manifest_path=None,
        manifest_data_root=None,
    ):
        super(DataLoaderTrain, self).__init__()

        pairs = (
            _collect_manifest_pairs(manifest_path, manifest_data_root)
            if manifest_path
            else _collect_pairs(rgb_dir, input_subdir, target_subdir, max_pairs_per_root)
        )
        self.inp_filenames = [str(inp) for inp, _ in pairs]
        self.tar_filenames = [str(tar) for _, tar in pairs]

        self.img_options = img_options
        self.augment_options = self.img_options.get('augment') if isinstance(self.img_options, dict) else None
        self.preserve_horizontal_trajectory = bool(getattr(
            self.augment_options, 'PRESERVE_HORIZONTAL_TRAJECTORY', False
        ))
        self.edge_subdir = edge_subdir
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def _find_edge_path(self, tar_path):
        if not self.edge_subdir:
            return None
        target_path = Path(tar_path)
        edge_dir = target_path.parent.parent / self.edge_subdir
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            candidate = edge_dir / f"{target_path.stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]
        edge_path = self._find_edge_path(tar_path)

        inp_img = Image.open(inp_path).convert("RGB")
        tar_img = Image.open(tar_path).convert("RGB")
        edge_img = Image.open(edge_path).convert('L') if edge_path is not None else None

        w,h = tar_img.size
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw!=0 or padh!=0:
            inp_img = TF.pad(inp_img, (0,0,padw,padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')
            if edge_img is not None:
                edge_img = TF.pad(edge_img, (0,0,padw,padh), padding_mode='constant', fill=0)

        inp_img = _apply_input_low_light_augment(inp_img, self.augment_options)
        inp_img, tar_img = _apply_paired_color_augment(inp_img, tar_img, self.augment_options)

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        edge_tensor = TF.to_tensor(edge_img) if edge_img is not None else None

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr     = random.randint(0, hh-ps)
        cc     = random.randint(0, ww-ps)
        # A row-aware trajectory model must not be trained on 90-degree
        # rotations when all target-domain traces run horizontally.  Flips and
        # a 180-degree rotation retain the horizontal trajectory distribution.
        aug = (
            random.choice((0, 1, 2, 4, 8))
            if self.preserve_horizontal_trajectory
            else random.randint(0, 8)
        )

        # Crop patch
        inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]
        tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]
        if edge_tensor is not None:
            edge_tensor = edge_tensor[:, rr:rr+ps, cc:cc+ps]

        # Data Augmentations
        if aug==1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
            if edge_tensor is not None:
                edge_tensor = edge_tensor.flip(1)
        elif aug==2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
            if edge_tensor is not None:
                edge_tensor = edge_tensor.flip(2)
        elif aug==3:
            inp_img = torch.rot90(inp_img,dims=(1,2))
            tar_img = torch.rot90(tar_img,dims=(1,2))
            if edge_tensor is not None:
                edge_tensor = torch.rot90(edge_tensor,dims=(1,2))
        elif aug==4:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=2)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=2)
            if edge_tensor is not None:
                edge_tensor = torch.rot90(edge_tensor,dims=(1,2), k=2)
        elif aug==5:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=3)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=3)
            if edge_tensor is not None:
                edge_tensor = torch.rot90(edge_tensor,dims=(1,2), k=3)
        elif aug==6:
            inp_img = torch.rot90(inp_img.flip(1),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(1),dims=(1,2))
            if edge_tensor is not None:
                edge_tensor = torch.rot90(edge_tensor.flip(1),dims=(1,2))
        elif aug==7:
            inp_img = torch.rot90(inp_img.flip(2),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(2),dims=(1,2))
            if edge_tensor is not None:
                edge_tensor = torch.rot90(edge_tensor.flip(2),dims=(1,2))

        inp_img = _apply_input_noise(inp_img, self.augment_options)
        
        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        if edge_tensor is not None:
            return tar_img, inp_img, filename, edge_tensor
        return tar_img, inp_img, filename

class DataLoaderVal(Dataset):
    def __init__(
        self,
        rgb_dir,
        img_options=None,
        rgb_dir2=None,
        input_subdir='input',
        target_subdir='target',
        max_pairs_per_root=0,
        manifest_path=None,
        manifest_data_root=None,
    ):
        super(DataLoaderVal, self).__init__()

        pairs = (
            _collect_manifest_pairs(manifest_path, manifest_data_root)
            if manifest_path
            else _collect_pairs(rgb_dir, input_subdir, target_subdir, max_pairs_per_root)
        )
        self.inp_filenames = [str(inp) for inp, _ in pairs]
        self.tar_filenames = [str(tar) for _, tar in pairs]

        self.img_options = img_options
        self.sizex       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert("RGB")
        tar_img = Image.open(tar_path).convert("RGB")

        # Validate on center crop
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps,ps))
            tar_img = TF.center_crop(tar_img, (ps,ps))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp).convert("RGB")

        inp = TF.to_tensor(inp)
        return inp, filename
