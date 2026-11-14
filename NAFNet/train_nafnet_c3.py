"""Train official NAFNet on the DB/XJ/SD/RX stitch deblurring manifests."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from basicsr.models.archs.NAFNet_arch import NAFNet
from basicsr.models.archs.MBDNetP_arch import MBDNetP, MBDNetPPlus, MBDNetPPlusR
from c3net_checkpoint import load_checkpoint_any, save_checkpoint_npz
from c3net_data import (
    DEFAULT_DOMAINS,
    DynamicLineTraceV3Dataset,
    PairedManifestDataset,
    psnr,
    ssim,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NAFNet on industrial stitch images")
    parser.add_argument("--data-root", default="H:/DATASET")
    parser.add_argument("--manifest-dir", default="H:/C3Netqing/MISCFilter/manifests/c3net_all")
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--output-dir", default="checkpoints/C3NetAll_NAFNet_width32")
    parser.add_argument("--patch-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--schedule-epochs",
        type=int,
        default=0,
        help="Cosine schedule horizon; use 30 for a short screening run intended for later resume.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA FP16 autocast with GradScaler during training. Validation and inference remain FP32.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Number of micro-batches accumulated before one optimizer update.",
    )
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--enc-blks", nargs="+", type=int, default=[1, 1, 1, 28])
    parser.add_argument("--middle-blks", type=int, default=1)
    parser.add_argument("--dec-blks", nargs="+", type=int, default=[1, 1, 1, 1])
    parser.add_argument(
        "--arch",
        choices=("nafnet", "mbdnet-p", "mbdnet-p-plus", "mbdnet-p-plus-r"),
        default="nafnet",
    )
    parser.add_argument(
        "--reconstruction-loss",
        choices=("psnr", "charbonnier"),
        default="psnr",
        help="Primary supervised objective; the default preserves existing runs.",
    )
    parser.add_argument("--charbonnier-eps", type=float, default=1e-3)
    parser.add_argument(
        "--ssim-finetune-start",
        type=int,
        default=0,
        help="First epoch that adds SSIM loss; 0 disables it.",
    )
    parser.add_argument("--ssim-weight", type=float, default=0.05)
    parser.add_argument(
        "--horizontal-gradient-weight",
        type=float,
        default=0.0,
        help=(
            "Training-only Charbonnier consistency on horizontal image derivatives. "
            "It does not add a module or any computation at inference."
        ),
    )
    parser.add_argument(
        "--gradient-eps",
        type=float,
        default=1e-3,
        help="Charbonnier epsilon for the optional directional-gradient loss.",
    )
    parser.add_argument("--directional-stages", nargs="+", type=int, default=[3])
    parser.add_argument("--no-directional-middle", action="store_true")
    parser.add_argument("--direction-kernel", type=int, default=7)
    parser.add_argument("--source-style-prob", type=float, default=0.0)
    parser.add_argument("--source-gain", type=float, default=0.0)
    parser.add_argument("--source-bias", type=float, default=0.0)
    parser.add_argument("--source-gamma", type=float, default=0.0)
    parser.add_argument("--source-noise-std", type=float, default=0.0)
    parser.add_argument("--source-band-gain", type=float, default=0.0)
    parser.add_argument(
        "--dynamic-linetrace-v3-train",
        action="store_true",
        help="Generate a fresh, strictly paired line-trace blur from DB/XJ/SD clear targets on every training sample.",
    )
    parser.add_argument("--dynamic-v3-repeats", type=int, default=3)
    parser.add_argument("--dynamic-v3-profile", choices=("mild", "mixed"), default="mixed")
    parser.add_argument("--sffr-decoder-index", type=int, default=1)
    parser.add_argument("--sffr-channels", type=int, default=32)
    parser.add_argument("--detail-width", type=int, default=32)
    parser.add_argument("--detail-blocks", type=int, default=3)
    parser.add_argument("--detail-limit", type=float, default=0.14)
    parser.add_argument("--detail-loss-weight", type=float, default=0.10)
    parser.add_argument("--stripe-token-count", type=int, default=4)
    return parser.parse_args()


def psnr_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    return (10.0 / math.log(10.0)) * torch.log(mse + 1e-8).mean()


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Stable pixel loss with an explicit epsilon recorded in run metadata."""

    return torch.sqrt((prediction - target).square() + epsilon * epsilon).mean()


def horizontal_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Robust consistency of x-direction image derivatives.

    The V2 training inputs are generated with predominantly near-horizontal
    motion blur.  This training-only term therefore restores the frequency
    content most attenuated by the degradation, while the pixel PSNR objective
    remains responsible for colour and all non-directional texture fidelity.
    It deliberately has no mask, condition input, or inference-time branch.
    """

    prediction_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    return torch.sqrt((prediction_dx - target_dx).square() + epsilon * epsilon).mean()


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.arch == "mbdnet-p":
        return MBDNetP(img_channel=3, width=args.width)
    if args.arch == "mbdnet-p-plus":
        return MBDNetPPlus(img_channel=3, width=args.width)
    if args.arch == "mbdnet-p-plus-r":
        return MBDNetPPlusR(img_channel=3, width=args.width)
    return NAFNet(
        img_channel=3,
        width=args.width,
        enc_blk_nums=args.enc_blks,
        middle_blk_num=args.middle_blks,
        dec_blk_nums=args.dec_blks,
    )


def load_checkpoint(
    model: torch.nn.Module,
    scheduler: CosineAnnealingLR,
    optimizer: AdamW,
    path: str,
) -> tuple[int, int]:
    checkpoint = load_checkpoint_any(path)
    state = {key.removeprefix("module."): value for key, value in checkpoint["state_dict"].items()}
    model.load_state_dict(state)
    if checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
        for group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
            group["lr"] = lr
    completed_epoch = int(checkpoint["epoch"])
    if checkpoint.get("epoch_complete", True):
        return completed_epoch + 1, int(checkpoint.get("global_step", 0))
    return completed_epoch, int(checkpoint.get("global_step", 0))


def load_initial_weights(model: torch.nn.Module, path: str) -> None:
    checkpoint = load_checkpoint_any(path)
    state = {key.removeprefix("module."): value for key, value in checkpoint["state_dict"].items()}
    model.load_state_dict(state)
    print(f"Loaded model weights from {path}; optimizer and scheduler were reset", flush=True)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    scheduler: CosineAnnealingLR,
    global_step: int,
    epoch_complete: bool,
) -> None:
    torch.cuda.synchronize()
    save_checkpoint_npz(
        path,
        epoch=epoch,
        state_dict=model.state_dict(),
        scheduler_state=scheduler.state_dict(),
        global_step=global_step,
        epoch_complete=epoch_complete,
    )


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> float:
    model.eval()
    scores: list[torch.Tensor] = []
    for batch_index, (target, source, _) in enumerate(loader):
        target = target.to(device, non_blocking=True)
        source = source.to(device, non_blocking=True)
        prediction = model(source).clamp(0, 1)
        scores.append(psnr(prediction, target))
        if max_batches and batch_index + 1 >= max_batches:
            break
    return torch.cat(scores).mean().item()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("NAFNet C3 training requires a CUDA GPU")
    if args.detail_loss_weight < 0.0:
        raise ValueError("detail-loss-weight must be non-negative")
    if args.schedule_epochs < 0:
        raise ValueError("schedule-epochs must be non-negative")
    if args.charbonnier_eps <= 0.0:
        raise ValueError("charbonnier-eps must be positive")
    if args.ssim_finetune_start < 0:
        raise ValueError("ssim-finetune-start must be non-negative")
    if args.ssim_weight < 0.0:
        raise ValueError("ssim-weight must be non-negative")
    if args.ssim_finetune_start and args.ssim_weight == 0.0:
        raise ValueError("ssim-weight must be positive when ssim-finetune-start is enabled")
    if args.horizontal_gradient_weight < 0.0:
        raise ValueError("horizontal-gradient-weight must be non-negative")
    if args.gradient_eps <= 0.0:
        raise ValueError("gradient-eps must be positive")
    if args.grad_accum_steps <= 0:
        raise ValueError("grad-accum-steps must be a positive integer")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    # RTX 4090 has Tensor Cores.  TF32 applies only to FP32 matrix products;
    # AMP's sensitive reductions still use GradScaler-protected FP32 weights.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    manifest_dir = Path(args.manifest_dir)
    train_manifest = manifest_dir / "train.tsv"
    val_manifest = manifest_dir / "val.tsv"
    if not train_manifest.is_file() or not val_manifest.is_file():
        raise RuntimeError(f"Missing manifests under {manifest_dir}. Run MISCFilter manifest generation first.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.auto_resume and not args.resume:
        latest_checkpoint = output_dir / "model_latest.pth"
        if latest_checkpoint.is_file() and latest_checkpoint.stat().st_size > 1024:
            args.resume = str(latest_checkpoint)
            print(f"Auto-resuming from {args.resume}", flush=True)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    if args.dynamic_linetrace_v3_train:
        train_set = DynamicLineTraceV3Dataset(
            args.data_root,
            train_manifest,
            args.patch_size,
            repeats=args.dynamic_v3_repeats,
            profile=args.dynamic_v3_profile,
        )
        print(
            f"Dynamic V3 paired training enabled: {len(train_set)} samples/epoch, profile={args.dynamic_v3_profile}",
            flush=True,
        )
    else:
        train_set = PairedManifestDataset(
            args.data_root,
            train_manifest,
            args.patch_size,
            True,
            source_style_prob=args.source_style_prob,
            source_gain=args.source_gain,
            source_bias=args.source_bias,
            source_gamma=args.source_gamma,
            source_noise_std=args.source_noise_std,
            source_band_gain=args.source_band_gain,
        )
    val_set = PairedManifestDataset(args.data_root, val_manifest, args.patch_size, False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda")
    model = build_model(args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.9))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.schedule_epochs if args.schedule_epochs else args.epochs,
        eta_min=args.min_lr,
    )

    start_epoch = 1
    step = 0
    if args.resume:
        start_epoch, step = load_checkpoint(model, scheduler, optimizer, args.resume)
        print(f"Resumed from {args.resume}: start_epoch={start_epoch}, global_step={step}", flush=True)
    elif args.init_checkpoint:
        if not Path(args.init_checkpoint).is_file():
            raise FileNotFoundError(f"Initial checkpoint not found: {args.init_checkpoint}")
        load_initial_weights(model, args.init_checkpoint)

    best_psnr = float("-inf")
    metrics_path = output_dir / "metrics.jsonl"
    if args.resume and metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                best_psnr = max(best_psnr, json.loads(line).get("val_psnr", float("-inf")))

    current_epoch = start_epoch
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            print(f"Starting epoch {epoch}/{args.epochs} at global_step={step}", flush=True)
            model.train()
            running_loss = 0.0
            running_ssim_loss = 0.0
            running_horizontal_gradient_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for batch_index, batch in enumerate(train_loader, start=1):
                target, source, _names = batch
                target = target.to(device, non_blocking=True)
                source = source.to(device, non_blocking=True)
                ssim_loss = None
                gradient_loss = None
                with torch.cuda.amp.autocast(enabled=args.amp):
                    prediction = model(source)
                    if args.reconstruction_loss == "charbonnier":
                        loss = charbonnier_loss(prediction, target, args.charbonnier_eps)
                    else:
                        loss = psnr_loss(prediction, target)
                    if args.horizontal_gradient_weight > 0.0:
                        gradient_loss = horizontal_gradient_loss(
                            prediction, target, args.gradient_eps
                        )
                        loss = loss + args.horizontal_gradient_weight * gradient_loss
                        running_horizontal_gradient_loss += gradient_loss.item()
                    if args.ssim_finetune_start and epoch >= args.ssim_finetune_start:
                        # Clamp only the bounded SSIM term.  The reconstruction
                        # objective remains unconstrained and supplies stable
                        # gradients outside the valid RGB range.
                        ssim_loss = 1.0 - ssim(prediction.clamp(0.0, 1.0), target).mean()
                        loss = loss + args.ssim_weight * ssim_loss
                        running_ssim_loss += ssim_loss.item()

                # Accumulated gradients are averaged, so a micro-batch pair has
                # the same gradient scale as one physical batch of four images.
                scaler.scale(loss / args.grad_accum_steps).backward()
                update_due = batch_index % args.grad_accum_steps == 0 or batch_index == len(train_loader)
                if update_due:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1

                running_loss += loss.item()
                if args.log_every_steps and batch_index % args.log_every_steps == 0:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "micro_batch": batch_index,
                                "micro_batches": len(train_loader),
                                "effective_batch_size": args.batch_size * args.grad_accum_steps,
                                "global_step": step,
                                "loss": loss.item(),
                                **(
                                    {"ssim_loss": ssim_loss.item()}
                                    if ssim_loss is not None
                                    else {}
                                ),
                                **(
                                    {"horizontal_gradient_loss": gradient_loss.item()}
                                    if gradient_loss is not None
                                    else {}
                                ),
                            }
                        ),
                        flush=True,
                    )
                if args.max_steps and update_due and step >= args.max_steps:
                    break

            record = {
                "epoch": epoch,
                "steps": step,
                "train_loss": running_loss / max(1, batch_index),
                "lr": optimizer.param_groups[0]["lr"],
            }
            if args.ssim_finetune_start and epoch >= args.ssim_finetune_start:
                record["train_ssim_loss"] = running_ssim_loss / max(1, batch_index)
            if args.horizontal_gradient_weight > 0.0:
                record["train_horizontal_gradient_loss"] = (
                    running_horizontal_gradient_loss / max(1, batch_index)
                )
            if epoch % args.val_every == 0 or (args.max_steps and step >= args.max_steps):
                record["val_psnr"] = validate(model, val_loader, device, args.max_val_batches)
                if record["val_psnr"] > best_psnr:
                    best_psnr = record["val_psnr"]
                    save_checkpoint(output_dir / "model_best.pth", epoch, model, scheduler, step, True)

            save_checkpoint(output_dir / "model_latest.pth", epoch, model, scheduler, step, True)
            if args.checkpoint_every and epoch % args.checkpoint_every == 0:
                save_checkpoint(output_dir / f"model_epoch_{epoch}.pth", epoch, model, scheduler, step, True)

            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

            if args.max_steps and step >= args.max_steps:
                break
            scheduler.step()
    except KeyboardInterrupt:
        save_checkpoint(output_dir / "model_latest.pth", current_epoch, model, scheduler, step, False)
        print(f"Interrupted safely at epoch {current_epoch}, global step {step}.", flush=True)


if __name__ == "__main__":
    main()
