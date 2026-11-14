"""Train MISCFilter on the DB/XJ/SD/RX paired industrial stitch datasets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import kornia
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from c3net_checkpoint import load_checkpoint_any, save_checkpoint_npz
from c3net_data import DEFAULT_DOMAINS, PairedManifestDataset, prepare_manifests, psnr
from loss import losses
from models.MISCFilterNet import MISCKernelNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MISCFilter on industrial stitch images")
    parser.add_argument("--data-root", default="H:/DATASET")
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--manifest-dir", default="manifests/c3net_all")
    parser.add_argument("--output-dir", default="checkpoints/C3NetAll_MISCFilters")
    parser.add_argument("--patch-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--max-val-batches", type=int, default=0, help="Nonzero value limits validation batches")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0, help="Nonzero value limits a smoke run")
    parser.add_argument("--resume", default="")
    parser.add_argument("--auto-resume", action="store_true", help="Resume model_latest.pth when available")
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Archive a model_epoch checkpoint every N epochs")
    parser.add_argument("--log-every-steps", type=int, default=25)
    return parser.parse_args()


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    path: str,
) -> int:
    checkpoint = load_checkpoint_any(path)
    state = checkpoint["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        print("Checkpoint has no optimizer state; Adam momentum is reset.", flush=True)
    completed_epoch = int(checkpoint["epoch"])
    if checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        # Checkpoints created before scheduler persistence use the closed-form
        # state at the completed epoch.
        scheduler.last_epoch = completed_epoch
        restored_lrs = scheduler._get_closed_form_lr()
        for parameter_group, lr in zip(optimizer.param_groups, restored_lrs):
            parameter_group["lr"] = lr
        scheduler._last_lr = restored_lrs
    if checkpoint.get("epoch_complete", True):
        return completed_epoch + 1, int(checkpoint.get("global_step", 0))
    # An interrupted epoch is replayed from its beginning with the latest
    # weights. This is safer than silently skipping its remaining samples.
    return completed_epoch, int(checkpoint.get("global_step", 0))


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    global_step: int,
    epoch_complete: bool,
) -> None:
    """Synchronize custom CUDA kernels and atomically publish a lightweight checkpoint."""
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
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    scores: list[torch.Tensor] = []
    for batch_index, (target, source, _) in enumerate(loader):
        target, source = target.to(device), source.to(device)
        restored, _ = model(source)
        scores.append(psnr(torch.clamp(restored[0], 0, 1), target))
        if max_batches and batch_index + 1 >= max_batches:
            break
    return torch.cat(scores).mean().item()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MISCFilter requires a CUDA GPU")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    manifests = prepare_manifests(
        args.data_root, args.manifest_dir, args.domains, args.val_ratio, args.seed
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.auto_resume and not args.resume:
        latest_checkpoint = output_dir / "model_latest.pth"
        if latest_checkpoint.is_file() and latest_checkpoint.stat().st_size > 1024:
            args.resume = str(latest_checkpoint)
            print(f"Auto-resuming from {args.resume}")
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_set = PairedManifestDataset(args.data_root, manifests["train"], args.patch_size, True)
    val_set = PairedManifestDataset(args.data_root, manifests["val"], args.patch_size, False)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda")
    model = MISCKernelNet().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    start_epoch = 1
    step = 0
    if args.resume:
        start_epoch, step = load_checkpoint(model, optimizer, scheduler, args.resume)
        print(f"Resumed from {args.resume}: start_epoch={start_epoch}, global_step={step}", flush=True)

    char_loss = losses.CharbonnierLoss()
    edge_loss = losses.EdgeLoss()
    fft_loss = losses.fftLoss()
    best_psnr = float("-inf")
    metrics_path = output_dir / "metrics.jsonl"
    if args.resume and metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            best_psnr = max(best_psnr, record.get("val_psnr", float("-inf")))

    current_epoch = start_epoch
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            print(f"Starting epoch {epoch}/{args.epochs} at global_step={step}", flush=True)
            model.train()
            running_loss = 0.0
            for batch_index, (target, source, _) in enumerate(train_loader, start=1):
                target, source = target.to(device, non_blocking=True), source.to(device, non_blocking=True)
                targets = kornia.geometry.transform.build_pyramid(target, 3)
                optimizer.zero_grad(set_to_none=True)
                restored, restored_inter = model(source)
                loss_char = sum(char_loss(pred, ref) for pred, ref in zip(restored, targets))
                loss_inter = sum(char_loss(pred, ref) for pred, ref in zip(restored_inter, targets))
                loss_fft = sum(fft_loss(pred, ref) for pred, ref in zip(restored, targets))
                loss_edge = sum(edge_loss(pred, ref) for pred, ref in zip(restored, targets))
                loss = loss_char + loss_inter + 0.01 * loss_fft + 0.05 * loss_edge
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                step += 1
                if args.log_every_steps and batch_index % args.log_every_steps == 0:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "batch": batch_index,
                                "batches": len(train_loader),
                                "global_step": step,
                                "loss": loss.item(),
                            }
                        ),
                        flush=True,
                    )
                if args.save_every_steps and step % args.save_every_steps == 0:
                    save_checkpoint(output_dir / "model_latest.pth", epoch, model, optimizer, scheduler, step, False)
                if args.max_steps and step >= args.max_steps:
                    break

            record = {
                "epoch": epoch,
                "steps": step,
                "train_loss": running_loss / max(1, len(train_loader)),
                "lr": optimizer.param_groups[0]["lr"],
            }
            if epoch % args.val_every == 0 or (args.max_steps and step >= args.max_steps):
                record["val_psnr"] = validate(model, val_loader, device, args.max_val_batches)
                if record["val_psnr"] > best_psnr:
                    best_psnr = record["val_psnr"]
                    save_checkpoint(output_dir / "model_best.pth", epoch, model, optimizer, scheduler, step, True)
            save_checkpoint(output_dir / "model_latest.pth", epoch, model, optimizer, scheduler, step, True)
            if args.checkpoint_every and epoch % args.checkpoint_every == 0:
                save_checkpoint(
                    output_dir / f"model_epoch_{epoch}.pth",
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    step,
                    True,
                )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record))
            if args.max_steps and step >= args.max_steps:
                break
            scheduler.step()
    except KeyboardInterrupt:
        save_checkpoint(output_dir / "model_latest.pth", current_epoch, model, optimizer, scheduler, step, False)
        print(f"Interrupted safely at epoch {current_epoch}, global step {step}.")


if __name__ == "__main__":
    main()
