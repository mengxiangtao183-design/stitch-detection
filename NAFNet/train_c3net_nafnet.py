"""Train NAFNet on the same DB/XJ/SD/RX split used by MISCFilter."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
MISC_ROOT = ROOT.parent / "MISCFilter"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MISC_ROOT))

from basicsr.models.archs.NAFNet_arch import NAFNet
from c3net_data import DEFAULT_DOMAINS, PairedManifestDataset, prepare_manifests, psnr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NAFNet on industrial stitch deblurring")
    parser.add_argument("--data-root", default="H:/DATASET")
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_DOMAINS))
    parser.add_argument("--manifest-dir", default="H:/C3Netqing/MISCFilter/manifests/c3net_all")
    parser.add_argument("--output-dir", default="checkpoints/C3NetAll_NAFNet_width32")
    parser.add_argument("--patch-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--width", type=int, default=32)
    return parser.parse_args()


def build_model(width: int) -> NAFNet:
    return NAFNet(
        img_channel=3,
        width=width,
        enc_blk_nums=[1, 1, 1, 28],
        middle_blk_num=1,
        dec_blk_nums=[1, 1, 1, 1],
    )


def to_cpu_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    scheduler: CosineAnnealingLR,
    global_step: int,
    epoch_complete: bool,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "state_dict": to_cpu_state(model.state_dict()),
            "scheduler": scheduler.state_dict(),
            "global_step": int(global_step),
            "epoch_complete": bool(epoch_complete),
        },
        temporary_path,
    )
    torch.load(temporary_path, map_location="cpu", weights_only=False)
    temporary_path.replace(path)


def load_checkpoint(
    model: torch.nn.Module,
    scheduler: CosineAnnealingLR,
    path: str,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = {key.removeprefix("module."): value for key, value in checkpoint["state_dict"].items()}
    model.load_state_dict(state)
    if checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = int(checkpoint["epoch"])
    step = int(checkpoint.get("global_step", 0))
    if checkpoint.get("epoch_complete", True):
        return epoch + 1, step
    return epoch, step


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> float:
    model.eval()
    scores: list[torch.Tensor] = []
    for batch_index, (target, source, _) in enumerate(loader):
        target = target.to(device, non_blocking=True)
        source = source.to(device, non_blocking=True)
        restored = model(source).clamp(0, 1)
        scores.append(psnr(restored, target))
        if max_batches and batch_index + 1 >= max_batches:
            break
    return torch.cat(scores).mean().item()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("NAFNet training requires a CUDA GPU")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    manifest_dir = Path(args.manifest_dir)
    required = [manifest_dir / "train.tsv", manifest_dir / "val.tsv"]
    if not all(path.is_file() for path in required):
        prepare_manifests(args.data_root, manifest_dir, args.domains, args.val_ratio, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.auto_resume and not args.resume:
        latest = output_dir / "model_latest.pth"
        if latest.is_file() and latest.stat().st_size > 1024:
            args.resume = str(latest)
            print(f"Auto-resuming from {args.resume}", flush=True)

    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_set = PairedManifestDataset(args.data_root, manifest_dir / "train.tsv", args.patch_size, True)
    val_set = PairedManifestDataset(args.data_root, manifest_dir / "val.tsv", args.patch_size, False)
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
    model = build_model(args.width).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.9), weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    start_epoch = 1
    step = 0
    if args.resume:
        start_epoch, step = load_checkpoint(model, scheduler, args.resume)
        print(f"Resumed from {args.resume}: start_epoch={start_epoch}, global_step={step}", flush=True)

    best_psnr = float("-inf")
    metrics_path = output_dir / "metrics.jsonl"
    if args.resume and metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            best_psnr = max(best_psnr, record.get("val_psnr", float("-inf")))

    criterion = torch.nn.L1Loss()
    current_epoch = start_epoch
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            print(f"Starting epoch {epoch}/{args.epochs} at global_step={step}", flush=True)
            model.train()
            running_loss = 0.0
            for batch_index, (target, source, _) in enumerate(train_loader, start=1):
                target = target.to(device, non_blocking=True)
                source = source.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                restored = model(source)
                loss = criterion(restored, target)
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
