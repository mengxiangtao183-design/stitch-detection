import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import losses
from quality_fusion import QualityFusionGate, load_yaml, resolve_path


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CONFIG_PATH = Path(__file__).with_name("training_quality_fusion_fast100.yml")


def parse_args():
    parser = argparse.ArgumentParser(description="Train quality-aware fusion gate.")
    parser.add_argument("--config", default=str(CONFIG_PATH), type=str)
    return parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_from_image(path):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1))


def collect_cached_samples(cache_root, domains, expert_a_subdir, expert_b_subdir):
    samples = []
    for domain in domains:
        domain_root = Path(cache_root) / domain
        input_dir = domain_root / "input"
        clear_dir = domain_root / "clear"
        expert_a_dir = domain_root / expert_a_subdir
        expert_b_dir = domain_root / expert_b_subdir
        for required in (input_dir, clear_dir, expert_a_dir, expert_b_dir):
            if not required.is_dir():
                raise FileNotFoundError(f"Fusion cache directory not found: {required}")

        input_files = {
            path.stem: path
            for path in sorted(input_dir.iterdir())
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        }
        clear_files = {
            path.stem: path
            for path in sorted(clear_dir.iterdir())
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        }
        expert_a_files = {
            path.stem: path
            for path in sorted(expert_a_dir.iterdir())
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        }
        expert_b_files = {
            path.stem: path
            for path in sorted(expert_b_dir.iterdir())
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        }
        stems = sorted(set(input_files) & set(clear_files) & set(expert_a_files) & set(expert_b_files))
        for stem in stems:
            samples.append(
                {
                    "domain": domain,
                    "stem": stem,
                    "input": input_files[stem],
                    "clear": clear_files[stem],
                    "expert_a": expert_a_files[stem],
                    "expert_b": expert_b_files[stem],
                }
            )
    return samples


class FusionCacheDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = list(samples)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        degraded = tensor_from_image(item["input"])
        clear = tensor_from_image(item["clear"])
        expert_a = tensor_from_image(item["expert_a"])
        expert_b = tensor_from_image(item["expert_b"])

        if self.augment and random.random() < 0.5:
            degraded = degraded.flip(2)
            clear = clear.flip(2)
            expert_a = expert_a.flip(2)
            expert_b = expert_b.flip(2)

        return degraded, clear, expert_a, expert_b, item["domain"], item["stem"]


def split_indices(total, val_split, seed):
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    val_count = max(1, int(round(total * float(val_split))))
    val_count = min(val_count, total - 1)
    return indices[val_count:], indices[:val_count]


def psnr_tensor(pred, target):
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def roi_slice(tensor, top_ratio, bottom_ratio):
    height = tensor.shape[2]
    top = int(round(height * float(top_ratio)))
    bottom = int(round(height * float(bottom_ratio)))
    top = max(0, min(height - 1, top))
    bottom = max(top + 1, min(height, bottom))
    return tensor[:, :, top:bottom, :]


def evaluate(gate, loader, device, top_ratio, bottom_ratio, temperature=1.0):
    gate.eval()
    fused_psnr = []
    fused_roi_psnr = []
    expert_a_psnr = []
    expert_b_psnr = []
    alpha_values = []
    domain_alpha = {}
    with torch.no_grad():
        for degraded, clear, expert_a, expert_b, domains, _ in loader:
            degraded = degraded.to(device, non_blocking=True)
            clear = clear.to(device, non_blocking=True)
            expert_a = expert_a.to(device, non_blocking=True)
            expert_b = expert_b.to(device, non_blocking=True)
            fused, alpha = gate(
                degraded,
                expert_a,
                expert_b,
                temperature=temperature,
            )
            fused = torch.clamp(fused, 0.0, 1.0)

            fused_psnr.extend(psnr_tensor(fused, clear).detach().cpu().tolist())
            fused_roi_psnr.extend(
                psnr_tensor(
                    roi_slice(fused, top_ratio, bottom_ratio),
                    roi_slice(clear, top_ratio, bottom_ratio),
                ).detach().cpu().tolist()
            )
            expert_a_psnr.extend(psnr_tensor(expert_a, clear).detach().cpu().tolist())
            expert_b_psnr.extend(psnr_tensor(expert_b, clear).detach().cpu().tolist())
            alpha_values.extend(alpha.view(-1).detach().cpu().tolist())
            for domain, value in zip(domains, alpha.view(-1).detach().cpu().tolist()):
                domain_alpha.setdefault(domain, []).append(value)

    metrics = {
        "fused_psnr": float(np.mean(fused_psnr)),
        "fused_roi_psnr": float(np.mean(fused_roi_psnr)),
        "expert_a_psnr": float(np.mean(expert_a_psnr)),
        "expert_b_psnr": float(np.mean(expert_b_psnr)),
        "alpha_mean": float(np.mean(alpha_values)),
        "alpha_std": float(np.std(alpha_values)),
    }
    metrics["domain_alpha"] = {
        domain: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "samples": len(values),
        }
        for domain, values in sorted(domain_alpha.items())
    }
    return metrics


def save_checkpoint(path, gate, optimizer, scheduler, epoch, val_metrics, best_psnr, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": gate.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_metrics": val_metrics,
            "best_psnr": best_psnr,
            "config": cfg,
        },
        path,
    )


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    base_dir = cfg_path.parent

    gpus = ",".join(str(gpu) for gpu in cfg.get("GPU", [0]))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    seed = int(cfg.get("SEED", 42))
    set_seeds(seed)

    cache_cfg = cfg["CACHE"]
    fusion_cfg = cfg["FUSION"]
    optim_cfg = cfg["OPTIM"]
    data_cfg = cfg["DATA"]
    expert_a_subdir = cfg["EXPERTS"]["A"].get("CACHE_SUBDIR", "expert_a")
    expert_b_subdir = cfg["EXPERTS"]["B"].get("CACHE_SUBDIR", "expert_b")

    cache_root = resolve_path(cache_cfg["ROOT"], base_dir)
    source_domains = data_cfg.get("SOURCE_DOMAINS", ["DB", "XJ", "SD"])
    samples = collect_cached_samples(cache_root, source_domains, expert_a_subdir, expert_b_subdir)
    if len(samples) < 2:
        raise RuntimeError("Fusion cache must contain at least two samples.")

    train_indices, val_indices = split_indices(
        len(samples),
        data_cfg.get("VAL_SPLIT", 0.1),
        seed,
    )
    train_dataset = Subset(FusionCacheDataset(samples, augment=bool(fusion_cfg.get("AUGMENT_HFLIP", True))), train_indices)
    val_dataset = Subset(FusionCacheDataset(samples, augment=False), val_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = {
        "num_workers": int(optim_cfg.get("NUM_WORKERS", 0)),
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(optim_cfg.get("BATCH_SIZE", 16)),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(optim_cfg.get("VAL_BATCH_SIZE", optim_cfg.get("BATCH_SIZE", 16))),
        shuffle=False,
        **loader_kwargs,
    )

    gate = QualityFusionGate(
        in_channels=int(fusion_cfg.get("IN_CHANNELS", 12)),
        hidden=tuple(fusion_cfg.get("HIDDEN", [32, 48, 64])),
        mlp_hidden=int(fusion_cfg.get("MLP_HIDDEN", 16)),
        use_descriptors=bool(fusion_cfg.get("USE_DESCRIPTORS", False)),
        descriptor_hidden=int(fusion_cfg.get("DESCRIPTOR_HIDDEN", 16)),
        roi_top_ratio=float(fusion_cfg.get("ROI_TOP_RATIO", 1.0 / 3.0)),
        roi_bottom_ratio=float(fusion_cfg.get("ROI_BOTTOM_RATIO", 2.0 / 3.0)),
    ).to(device)
    gate_params = sum(param.numel() for param in gate.parameters())
    print(f"==> Fusion gate parameters: {gate_params / 1e6:.4f}M")
    print(f"==> Train/val cached samples: {len(train_dataset)}/{len(val_dataset)}")

    optimizer = optim.Adam(
        gate.parameters(),
        lr=float(optim_cfg.get("LR_INITIAL", 1.0e-4)),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(optim_cfg.get("NUM_EPOCHS", 25)),
        eta_min=float(optim_cfg.get("LR_MIN", 1.0e-6)),
    )
    use_amp = bool(optim_cfg.get("USE_AMP", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    criterion_char = losses.CharbonnierLoss()

    model_dir = Path(resolve_path(fusion_cfg["MODEL_DIR"], base_dir))
    model_dir.mkdir(parents=True, exist_ok=True)
    latest_path = model_dir / "model_latest.pth"
    best_path = model_dir / "model_best.pth"

    start_epoch = 1
    best_psnr = -1.0
    if bool(optim_cfg.get("RESUME", False)) and latest_path.exists() and latest_path.stat().st_size > 0:
        checkpoint = torch.load(latest_path, map_location="cpu")
        gate.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_psnr = float(
            checkpoint.get(
                "best_psnr",
                checkpoint.get("val_metrics", {}).get("fused_psnr", best_psnr),
            )
        )
        print(f"==> Resuming fusion gate from epoch {start_epoch}")

    route_weight = float(
        fusion_cfg.get("ROUTING_LOSS_WEIGHT", fusion_cfg.get("ALPHA_LOSS_WEIGHT", 0.1))
    )
    routing_mode = str(fusion_cfg.get("ROUTING_MODE", "soft")).lower()
    if routing_mode not in {"soft", "hard_margin"}:
        raise ValueError(f"Unsupported FUSION.ROUTING_MODE: {routing_mode}")
    routing_margin = float(fusion_cfg.get("ROUTING_MARGIN", 0.0))
    soft_target_temperature = float(fusion_cfg.get("ALPHA_TARGET_TEMPERATURE", 0.01))
    roi_target_weight = float(fusion_cfg.get("ROI_TARGET_WEIGHT", 0.0))
    roi_loss_weight = float(fusion_cfg.get("ROI_LOSS_WEIGHT", 0.0))
    oracle_loss_weight = float(fusion_cfg.get("ORACLE_LOSS_WEIGHT", 0.0))
    train_temperature = float(fusion_cfg.get("TRAIN_ALPHA_TEMPERATURE", 1.0))
    val_temperature = float(fusion_cfg.get("INFERENCE_ALPHA_TEMPERATURE", 1.0))
    roi_top_ratio = float(fusion_cfg.get("ROI_TOP_RATIO", 1.0 / 3.0))
    roi_bottom_ratio = float(fusion_cfg.get("ROI_BOTTOM_RATIO", 2.0 / 3.0))

    for epoch in range(start_epoch, int(optim_cfg.get("NUM_EPOCHS", 25)) + 1):
        gate.train()
        running_loss = 0.0
        running_batches = 0
        running_alpha = []
        confident_samples = 0
        route_b_samples = 0
        total_samples = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
        for degraded, clear, expert_a, expert_b, _, _ in progress:
            degraded = degraded.to(device, non_blocking=True)
            clear = clear.to(device, non_blocking=True)
            expert_a = expert_a.to(device, non_blocking=True)
            expert_b = expert_b.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                fused, alpha, logits = gate(
                    degraded,
                    expert_a,
                    expert_b,
                    temperature=train_temperature,
                    return_logits=True,
                )
                e_a = torch.abs(expert_a - clear).mean(dim=(1, 2, 3), keepdim=True)
                e_b = torch.abs(expert_b - clear).mean(dim=(1, 2, 3), keepdim=True)
                roi_clear = roi_slice(clear, roi_top_ratio, roi_bottom_ratio)
                roi_a = roi_slice(expert_a, roi_top_ratio, roi_bottom_ratio)
                roi_b = roi_slice(expert_b, roi_top_ratio, roi_bottom_ratio)
                e_a_roi = torch.abs(roi_a - roi_clear).mean(dim=(1, 2, 3), keepdim=True)
                e_b_roi = torch.abs(roi_b - roi_clear).mean(dim=(1, 2, 3), keepdim=True)
                route_score = (e_a - e_b) + roi_target_weight * (e_a_roi - e_b_roi)
                hard_route_target = (route_score > 0.0).to(logits.dtype)
                loss_char = criterion_char(fused, clear)
                loss_roi = criterion_char(
                    roi_slice(fused, roi_top_ratio, roi_bottom_ratio),
                    roi_clear,
                )
                if routing_mode == "hard_margin":
                    route_target = hard_route_target
                    confident = torch.abs(route_score) >= routing_margin
                    route_elements = F.binary_cross_entropy_with_logits(
                        logits,
                        route_target.detach(),
                        reduction="none",
                    )
                    if confident.any():
                        loss_route = route_elements[confident].mean()
                    else:
                        loss_route = logits.sum() * 0.0
                else:
                    route_target = torch.sigmoid(
                        route_score / max(soft_target_temperature, 1.0e-8)
                    )
                    confident = torch.ones_like(route_score, dtype=torch.bool)
                    loss_route = F.mse_loss(alpha, route_target.detach())
                oracle = torch.where(hard_route_target.bool(), expert_b, expert_a)
                loss_oracle = criterion_char(fused, oracle.detach())
                loss = (
                    loss_char
                    + roi_loss_weight * loss_roi
                    + route_weight * loss_route
                    + oracle_loss_weight * loss_oracle
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu())
            running_batches += 1
            running_alpha.extend(alpha.view(-1).detach().cpu().tolist())
            confident_samples += int(confident.sum().detach().cpu())
            route_b_samples += int((hard_route_target.bool() & confident).sum().detach().cpu())
            total_samples += int(clear.shape[0])
            progress.set_postfix(
                loss=f"{running_loss / max(1, running_batches):.5f}",
                alpha=f"{np.mean(running_alpha):.3f}",
                confident=f"{confident_samples}/{total_samples}",
            )

        scheduler.step()
        val_metrics = evaluate(
            gate,
            val_loader,
            device,
            roi_top_ratio,
            roi_bottom_ratio,
            temperature=val_temperature,
        )
        epoch_best_psnr = max(best_psnr, val_metrics["fused_psnr"])
        save_checkpoint(
            latest_path,
            gate,
            optimizer,
            scheduler,
            epoch,
            val_metrics,
            epoch_best_psnr,
            cfg,
        )
        if val_metrics["fused_psnr"] > best_psnr:
            best_psnr = val_metrics["fused_psnr"]
            save_checkpoint(
                best_path,
                gate,
                optimizer,
                scheduler,
                epoch,
                val_metrics,
                best_psnr,
                cfg,
            )

        print(
            f"==> Epoch {epoch}: "
            f"val_psnr={val_metrics['fused_psnr']:.4f}, "
            f"val_roi_psnr={val_metrics['fused_roi_psnr']:.4f}, "
            f"A={val_metrics['expert_a_psnr']:.4f}, "
            f"B={val_metrics['expert_b_psnr']:.4f}, "
            f"alpha={val_metrics['alpha_mean']:.3f}+/-{val_metrics['alpha_std']:.3f}, "
            f"confident_B={route_b_samples}/{max(1, confident_samples)}, "
            f"best={best_psnr:.4f}"
        )
        print(
            "==> Validation alpha by domain: "
            + ", ".join(
                f"{domain}={stats['mean']:.3f}+/-{stats['std']:.3f}"
                for domain, stats in val_metrics["domain_alpha"].items()
            )
        )

    print(f"==> Fusion gate training complete. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
