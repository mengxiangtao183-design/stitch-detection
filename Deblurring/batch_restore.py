"""
Batch restoration script for applying the Deblurring MPRNet model to custom datasets.

Default layout expected per dataset root:

    <dataset_root>/
        train/
            Imgs1/   # degraded inputs
            clear/   # optional clear references for PSNR reporting

Example:
    python Deblurring/batch_restore.py ^
        --weights H:\\path\\to\\model_deblurring.pth
"""

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from MPRNet import MPRNet


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTENSIONS


def load_weights(model: nn.Module, weights_path: str, device: torch.device) -> None:
    """Load MPRNet weights from a few common checkpoint formats."""
    checkpoint = utils.load_checkpoint_file(weights_path)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    normalized_state_dict = OrderedDict()
    for key, value in state_dict.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        normalized_state_dict[normalized_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(normalized_state_dict, strict=False)
    if missing_keys:
        print(f"[WARN] Missing keys when loading weights: {missing_keys}")
    if unexpected_keys:
        print(f"[WARN] Unexpected keys when loading weights: {unexpected_keys}")


class CustomDeblurDataset(Dataset):
    """Reads degraded images from one folder with optional clear references."""

    def __init__(self, input_dir: Path, clear_dir: Optional[Path] = None):
        self.input_dir = input_dir
        self.clear_dir = clear_dir if clear_dir and clear_dir.is_dir() else None
        self.files = sorted([path for path in input_dir.iterdir() if path.is_file() and is_image_file(path)])

        if not self.files:
            raise ValueError(f"No image files found in: {input_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, object]:
        input_path = self.files[index]
        image = Image.open(input_path).convert("RGB")
        image_tensor = to_tensor(image)

        sample = {
            "image": image_tensor,
            "name": input_path.stem,
            "suffix": input_path.suffix,
            "size": image.size,
        }

        if self.clear_dir is not None:
            clear_path = self._find_clear_path(input_path.stem)
            if clear_path is not None:
                clear_image = Image.open(clear_path).convert("RGB")
                sample["clear"] = to_tensor(clear_image)
                sample["clear_path"] = str(clear_path)

        return sample

    def _find_clear_path(self, stem: str) -> Optional[Path]:
        if self.clear_dir is None:
            return None
        for ext in VALID_EXTENSIONS:
            candidate = self.clear_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None


def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    if len(batch) != 1:
        raise ValueError("This script expects batch_size=1 for arbitrary image sizes.")
    return batch[0]


def pad_to_multiple(image: torch.Tensor, factor: int = 8) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad BCHW tensor so height/width are multiples of factor."""
    _, _, h, w = image.shape
    padded_h = ((h + factor - 1) // factor) * factor
    padded_w = ((w + factor - 1) // factor) * factor
    pad_h = padded_h - h
    pad_w = padded_w - w

    if pad_h == 0 and pad_w == 0:
        return image, (0, 0)

    padded = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, (pad_h, pad_w)


def unpad_image(image: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pad_hw
    if pad_h > 0:
        image = image[:, :, :-pad_h, :]
    if pad_w > 0:
        image = image[:, :, :, :-pad_w]
    return image


def compute_psnr(restored: torch.Tensor, clear: torch.Tensor) -> float:
    restored = torch.clamp(restored, 0.0, 1.0)
    clear = torch.clamp(clear, 0.0, 1.0)
    mse = torch.mean((restored - clear) ** 2).item()
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(1.0 / np.sqrt(mse))


def tensor_to_uint8_image(restored: torch.Tensor) -> np.ndarray:
    """Convert BCHW/BHWC float image in [0, 1] to uint8 HWC."""
    restored = torch.clamp(restored, 0.0, 1.0)
    restored_np = restored.permute(0, 2, 3, 1).cpu().numpy()[0]
    return np.clip(np.round(restored_np * 255.0), 0, 255).astype(np.uint8)


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image).save(path)


def resolve_dataset_paths(dataset_roots: List[str], split: str, input_subdir: str, clear_subdir: str) -> List[Dict[str, Path]]:
    resolved = []
    for root in dataset_roots:
        root_path = Path(root)
        split_dir = root_path / split
        input_dir = split_dir / input_subdir
        clear_dir = split_dir / clear_subdir

        if not input_dir.is_dir():
            print(f"[WARN] Skip {root_path}: input dir not found -> {input_dir}")
            continue

        resolved.append(
            {
                "root": root_path,
                "split_dir": split_dir,
                "input_dir": input_dir,
                "clear_dir": clear_dir,
            }
        )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Deblurring/MPRNet to custom degraded datasets.")
    parser.add_argument(
        "--weights",
        type=str,
        default=r"H:\C3Netqing\Deblurring\checkpoints\train\Deblurring\models\MPRNet\model_latest.pth",
        help="Path to MPRNet deblurring weights. Default uses the latest local training checkpoint.",
    )
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        default=[
            r"H:\DATASET\DB",
            r"H:\DATASET\XJ",
            r"H:\DATASET\SD",
        ],
        help="Dataset root directories.",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to process.")
    parser.add_argument("--input-subdir", type=str, default="Imgs1", help="Degraded image subdirectory name.")
    parser.add_argument("--clear-subdir", type=str, default="clear", help="Clear reference subdirectory name.")
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="Imgs1_MPRNet",
        help="Output subdirectory created under each split directory.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size. Use 1 for arbitrary sizes.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--save-json", action="store_true", help="Save per-dataset summary JSON.")
    parser.add_argument("--n-feat", type=int, default=96, help="MPRNet base feature channels.")
    parser.add_argument("--scale-unetfeats", type=int, default=48, help="MPRNet U-Net feature growth.")
    parser.add_argument("--scale-orsnetfeats", type=int, default=32, help="MPRNet ORSNet feature growth.")
    parser.add_argument("--num-cab", type=int, default=8, help="Number of CAB blocks in ORSNet ORB.")
    parser.add_argument("--edge-refine", action="store_true", help="Enable stitch edge-guided refinement module.")
    parser.add_argument("--edge-refine-mid", type=int, default=24, help="Hidden channels in edge refinement module.")
    parser.add_argument("--frequency-refine", action="store_true",
                        help="Enable final RGB frequency residual refinement.")
    parser.add_argument("--frequency-refine-hidden", type=int, default=16,
                        help="Hidden channels in the frequency residual refiner.")
    parser.add_argument("--frequency-band-gate", action="store_true",
                        help="Enable smooth band-pass correction and per-image frequency strength gate.")
    parser.add_argument("--frequency-band-low-start", type=float, default=0.05)
    parser.add_argument("--frequency-band-low-end", type=float, default=0.10)
    parser.add_argument("--frequency-band-high-start", type=float, default=0.30)
    parser.add_argument("--frequency-band-high-end", type=float, default=0.40)
    parser.add_argument("--frequency-disagreement-gate", action="store_true",
                        help="Gate band-pass frequency correction using Stage-3/Stage-2 disagreement.")
    parser.add_argument("--frequency-disagreement-threshold", type=float, default=0.0120)
    parser.add_argument("--frequency-disagreement-temperature", type=float, default=0.0020)
    parser.add_argument("--frequency-disagreement-min-strength", type=float, default=0.10)
    parser.add_argument("--light-adapt", action="store_true", help="Enable light adaptive input correction for MPRNet.")
    parser.add_argument("--light-target-luma", type=float, default=0.50, help="Target luminance for light adaptation.")
    parser.add_argument("--light-min-gain", type=float, default=0.75, help="Minimum gain for light adaptation.")
    parser.add_argument("--light-max-gain", type=float, default=1.80, help="Maximum gain for light adaptation.")
    parser.add_argument("--light-min-gamma", type=float, default=0.65, help="Minimum gamma for light adaptation.")
    parser.add_argument("--light-max-gamma", type=float, default=1.35, help="Maximum gamma for light adaptation.")
    parser.add_argument("--light-blend", type=float, default=0.75, help="Blend strength for light adaptation.")
    parser.add_argument("--light-dark-threshold", type=float, default=0.45, help="Luminance threshold for dark-image correction.")
    parser.add_argument("--light-bright-threshold", type=float, default=0.70, help="Luminance threshold for bright-image correction.")
    parser.add_argument("--trajectory-lightweight", action="store_true", help="Enable the trajectory-aware lightweight MPRNet path.")
    parser.add_argument("--stage1-tristripe", action="store_true",
                        help="Enable three-stripe shared-pair Stage 1 with original Stage 2/3.")
    parser.add_argument("--stage1-stripes", type=int, default=3, help="Number of full-width Stage-1 stripes.")
    parser.add_argument("--stage1-overlap", type=float, default=0.25, help="Stage-1 stripe overlap ratio.")
    parser.add_argument("--stage1-stripe-transformer", action="store_true",
                        help="Enable row-wise Stripe Transformer at the Stage-1 bottleneck.")
    parser.add_argument("--stage1-stripe-transformer-heads", type=int, default=4,
                        help="Attention heads for the Stage-1 Stripe Transformer.")
    parser.add_argument("--stage2-stripe-transformer", action="store_true",
                        help="Enable row-wise Stripe Transformer at the Stage-2 bottleneck.")
    parser.add_argument("--stage2-stripe-transformer-heads", type=int, default=4,
                        help="Attention heads for the Stage-2 Stripe Transformer.")
    parser.add_argument("--stage2-overlap", type=float, default=0.0,
                        help="Upper/lower overlap ratio for original Stage 2 in tri-stripe mode.")
    parser.add_argument("--stage2-pair-aligned", action="store_true",
                        help="Use [stripe1+stripe2] and [stripe2+stripe3] as Stage-2 branches.")
    parser.add_argument("--stage2-naf-blocks", type=int, default=2, help="NAF blocks per Stage-2 level.")
    parser.add_argument("--stage2-horizontal-kernel", type=int, default=15, help="Horizontal bottleneck kernel.")
    parser.add_argument("--stage2-vertical-kernel", type=int, default=5, help="Vertical bottleneck kernel.")
    parser.add_argument("--stage3-naf-blocks", type=int, default=3, help="NAF blocks per Stage-3 fusion level.")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"MPRNet weights not found: {weights_path}\n"
            "Please place the Deblurring checkpoint there or pass --weights explicitly."
        )

    dataset_specs = resolve_dataset_paths(
        dataset_roots=args.dataset_roots,
        split=args.split,
        input_subdir=args.input_subdir,
        clear_subdir=args.clear_subdir,
    )
    if not dataset_specs:
        raise RuntimeError("No valid dataset directories found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading weights from: {weights_path}")

    model = MPRNet(
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
        stage2_naf_blocks=args.stage2_naf_blocks,
        stage2_horizontal_kernel=args.stage2_horizontal_kernel,
        stage2_vertical_kernel=args.stage2_vertical_kernel,
        stage3_naf_blocks=args.stage3_naf_blocks,
    ).to(device)
    load_weights(model, str(weights_path), device)
    if device.type == "cuda":
        model = nn.DataParallel(model)
    model.eval()

    for spec in dataset_specs:
        dataset_name = spec["root"].name
        output_dir = spec["split_dir"] / args.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        clear_dir = spec["clear_dir"] if spec["clear_dir"].is_dir() else None
        dataset = CustomDeblurDataset(spec["input_dir"], clear_dir=clear_dir)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_fn,
        )

        summary = {
            "dataset": dataset_name,
            "input_dir": str(spec["input_dir"]),
            "output_dir": str(output_dir),
            "processed": 0,
            "psnr_samples": 0,
            "avg_psnr": None,
        }
        psnr_values: List[float] = []

        print(f"[INFO] Restoring dataset: {dataset_name}")
        with torch.no_grad():
            for sample in tqdm(loader, desc=f"Restore {dataset_name}"):
                image = sample["image"].unsqueeze(0).to(device)
                padded_image, pad_hw = pad_to_multiple(image, factor=8)

                restored = model(padded_image)
                restored = torch.clamp(restored[0], 0, 1)
                restored = unpad_image(restored, pad_hw)

                restored_img = tensor_to_uint8_image(restored)
                output_path = output_dir / f"{sample['name']}.png"
                save_rgb_image(output_path, restored_img)

                summary["processed"] += 1

                clear_tensor = sample.get("clear")
                if clear_tensor is not None:
                    clear_tensor = clear_tensor.unsqueeze(0).to(device)
                    psnr_values.append(compute_psnr(restored, clear_tensor))

                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if psnr_values:
            summary["psnr_samples"] = len(psnr_values)
            summary["avg_psnr"] = float(np.mean(psnr_values))

        print(
            f"[DONE] {dataset_name}: processed={summary['processed']}, "
            f"avg_psnr={summary['avg_psnr']}"
        )

        if args.save_json:
            summary_path = output_dir / "restore_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
