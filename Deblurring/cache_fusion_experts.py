import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from quality_fusion import build_frozen_expert, forward_padded, load_yaml, resolve_path


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CONFIG_PATH = Path(__file__).resolve().parent.parent / "_archive" / "deblurring_nonpaper_yaml_2026-11-14" / "training_quality_fusion_fast100.yml"


def parse_args():
    parser = argparse.ArgumentParser(description="Cache frozen expert outputs for fusion-gate training.")
    parser.add_argument("--config", default=str(CONFIG_PATH), type=str)
    parser.add_argument("--force", action="store_true", help="Overwrite existing cached PNG files.")
    parser.add_argument("--max_samples", default=0, type=int, help="Optional global sample limit for smoke tests.")
    return parser.parse_args()


def tensor_from_image(path):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1))


def save_tensor_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().clamp(0, 1).cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    array = np.transpose(array, (1, 2, 0))
    array = (array * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(array).save(path)


def collect_pairs(train_dirs, input_subdir, target_subdir, max_per_root):
    pairs = []
    for root in train_dirs:
        root = Path(root)
        input_dir = root / input_subdir
        target_dir = root / target_subdir
        if not input_dir.is_dir() or not target_dir.is_dir():
            raise FileNotFoundError(f"Missing input/target directory under {root}")

        input_files = {
            path.stem: path
            for path in sorted(input_dir.iterdir())
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        }
        target_files = {
            path.stem: path
            for path in sorted(target_dir.iterdir())
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        }
        stems = sorted(set(input_files) & set(target_files))
        if max_per_root > 0:
            stems = stems[:max_per_root]
        domain = root.parent.name if root.name.lower() in {"train", "val", "test"} else root.name
        for stem in stems:
            pairs.append(
                {
                    "domain": domain,
                    "stem": stem,
                    "input": input_files[stem],
                    "clear": target_files[stem],
                }
            )
    return pairs


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    base_dir = cfg_path.parent

    gpus = ",".join(str(gpu) for gpu in cfg.get("GPU", [0]))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    data_cfg = cfg["DATA"]
    cache_cfg = cfg["CACHE"]
    expert_a_cfg = cfg["EXPERTS"]["A"]
    expert_b_cfg = cfg["EXPERTS"]["B"]

    cache_root = Path(resolve_path(cache_cfg["ROOT"], base_dir))
    expert_a_subdir = expert_a_cfg.get("CACHE_SUBDIR", "expert_a")
    expert_b_subdir = expert_b_cfg.get("CACHE_SUBDIR", "expert_b")

    train_dirs = data_cfg.get("TRAIN_DIRS", [])
    max_per_root = int(data_cfg.get("MAX_IMAGES_PER_TRAIN_DIR", 0))
    pairs = collect_pairs(
        train_dirs,
        data_cfg.get("INPUT_SUBDIR", "Imgs1"),
        data_cfg.get("TARGET_SUBDIR", "clear"),
        max_per_root,
    )
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise RuntimeError("No training pairs found for fusion cache.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cache_cfg.get("USE_AMP", True)) and device.type == "cuda"

    expert_a = build_frozen_expert(
        resolve_path(expert_a_cfg["CONFIG"], base_dir),
        resolve_path(expert_a_cfg["WEIGHTS"], base_dir),
        device,
    )
    expert_b = build_frozen_expert(
        resolve_path(expert_b_cfg["CONFIG"], base_dir),
        resolve_path(expert_b_cfg["WEIGHTS"], base_dir),
        device,
    )

    print(f"==> Caching {len(pairs)} samples to {cache_root}")
    print(f"==> Expert A: {expert_a_cfg.get('NAME', expert_a_subdir)}")
    print(f"==> Expert B: {expert_b_cfg.get('NAME', expert_b_subdir)}")

    with torch.no_grad():
        for item in tqdm(pairs, desc="Caching fusion experts", dynamic_ncols=True):
            domain = item["domain"]
            stem = item["stem"]
            out_paths = {
                "input": cache_root / domain / "input" / f"{stem}.png",
                "clear": cache_root / domain / "clear" / f"{stem}.png",
                expert_a_subdir: cache_root / domain / expert_a_subdir / f"{stem}.png",
                expert_b_subdir: cache_root / domain / expert_b_subdir / f"{stem}.png",
            }
            if not args.force and all(path.exists() for path in out_paths.values()):
                continue

            input_tensor = tensor_from_image(item["input"])
            clear_tensor = tensor_from_image(item["clear"])
            batch = input_tensor.unsqueeze(0).to(device, non_blocking=True)

            out_a = forward_padded(expert_a, batch, use_amp=use_amp)
            out_b = forward_padded(expert_b, batch, use_amp=use_amp)

            save_tensor_image(input_tensor, out_paths["input"])
            save_tensor_image(clear_tensor, out_paths["clear"])
            save_tensor_image(out_a, out_paths[expert_a_subdir])
            save_tensor_image(out_b, out_paths[expert_b_subdir])

    print("==> Fusion expert cache complete.")


if __name__ == "__main__":
    main()
