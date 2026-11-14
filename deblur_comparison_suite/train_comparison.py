from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data_protocol import PairedManifestDataset
from metric_protocol import psnr, psnr_loss
from model_factory import available_models, build_model


ROOT = Path(__file__).resolve().parent
REGISTRY = json.loads((ROOT / "model_registry.json").read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair MultiSource-v2 deblurring comparison training")
    parser.add_argument("--model", required=True, choices=available_models())
    parser.add_argument("--data-root", default=REGISTRY["protocol"]["data_root"])
    parser.add_argument("--manifest-dir", default=REGISTRY["protocol"]["manifest_dir"])
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--schedule-epochs", type=int, default=30)
    parser.add_argument("--patch-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--resume", default="")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(model, optimizer, scheduler, scaler, epoch, step, best_psnr, args) -> dict:
    return {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": step,
        "best_psnr": best_psnr,
        "model_name": args.model,
        "args": vars(args),
    }


@torch.no_grad()
def validate(model, loader, device, amp: bool, max_batches: int) -> float:
    model.eval()
    scores: list[torch.Tensor] = []
    for batch_index, (target, source, _) in enumerate(loader, 1):
        target = target.to(device, non_blocking=True)
        source = source.to(device, non_blocking=True)
        # Keep validation in FP32, matching the MBDNet-P+ full-training run.
        prediction = model(source)
        scores.append(psnr(prediction.float().clamp(0, 1), target))
        if max_batches and batch_index >= max_batches:
            break
    return torch.cat(scores).mean().item()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    entry = REGISTRY["models"][args.model]
    protocol = REGISTRY["protocol"]
    args.lr = float(entry.get("learning_rate", protocol["learning_rate"]) if args.lr is None else args.lr)
    args.min_lr = float(entry.get("minimum_learning_rate", protocol["minimum_learning_rate"]) if args.min_lr is None else args.min_lr)
    args.weight_decay = float(entry.get("weight_decay", protocol["weight_decay"]) if args.weight_decay is None else args.weight_decay)
    optimizer_name = entry.get("optimizer", protocol["optimizer"])
    args.betas = tuple(float(value) for value in entry.get("betas", (0.9, 0.9)))
    if args.batch_size <= 0:
        args.batch_size = int(entry["batch_size"])
    if args.grad_accum_steps <= 0:
        args.grad_accum_steps = int(entry["grad_accum_steps"])
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        run_tag = entry.get("run_tag", "multisource_v2_s30_s42")
        output_dir = ROOT / "checkpoints" / f"{args.model}_{run_tag}"
    args.output_dir = str(output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    manifest_dir = Path(args.manifest_dir)
    train_manifest, val_manifest = manifest_dir / "train.tsv", manifest_dir / "val.tsv"
    for path in (train_manifest, val_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps({**vars(args), "registry": entry}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    train_set = PairedManifestDataset(args.data_root, train_manifest, args.patch_size, augment=True)
    val_set = PairedManifestDataset(args.data_root, val_manifest, args.patch_size, augment=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda")
    model = build_model(args.model).to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps({"model": args.model, "parameters": total, "trainable": trainable}), flush=True)
    optimizer_cls = Adam if optimizer_name.lower() == "adam" else AdamW
    optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.betas)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.schedule_epochs), eta_min=args.min_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch, global_step, best_psnr = 1, 0, float("-inf")
    latest = output_dir / "model_latest.pth"
    if args.auto_resume and not args.resume and latest.is_file():
        args.resume = str(latest)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if checkpoint.get("model_name") not in (None, args.model):
            raise ValueError(f"Checkpoint model mismatch: {checkpoint.get('model_name')} != {args.model}")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_psnr = float(checkpoint.get("best_psnr", best_psnr))
        print(f"Auto-resuming from {args.resume} at epoch {start_epoch}", flush=True)

    history_path = output_dir / "history.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        batches = 0
        print(f"Starting epoch {epoch}/{args.epochs} at global_step={global_step}", flush=True)
        for batch_index, (target, source, _) in enumerate(train_loader, 1):
            target = target.to(device, non_blocking=True)
            source = source.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                predictions = model.forward_all(source)
                losses = [psnr_loss(prediction, target) for prediction in predictions]
                loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss for {args.model} at epoch {epoch}, batch {batch_index}. "
                    "The run was stopped before writing an invalid checkpoint."
                )
            scaler.scale(loss / args.grad_accum_steps).backward()
            should_step = batch_index % args.grad_accum_steps == 0 or batch_index == len(train_loader)
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            running_loss += loss.item()
            batches += 1
            if batch_index % args.log_every_steps == 0:
                print(json.dumps({
                    "epoch": epoch,
                    "batch": batch_index,
                    "batches": len(train_loader),
                    "global_step": global_step,
                    "loss": loss.item(),
                }), flush=True)
            if args.max_steps and batch_index >= args.max_steps:
                break

        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": running_loss / max(1, batches),
            "lr": optimizer.param_groups[0]["lr"],
        }
        if epoch % args.val_every == 0:
            record["val_psnr"] = validate(model, val_loader, device, args.amp, args.max_val_batches)
            if record["val_psnr"] > best_psnr:
                best_psnr = record["val_psnr"]
                record["best"] = True
                atomic_torch_save(
                    checkpoint_payload(model, optimizer, scheduler, scaler, epoch, global_step, best_psnr, args),
                    output_dir / "model_best.pth",
                )
        scheduler.step()
        record["best_psnr"] = best_psnr
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        atomic_torch_save(
            checkpoint_payload(model, optimizer, scheduler, scaler, epoch, global_step, best_psnr, args),
            latest,
        )
        if epoch % args.checkpoint_every == 0:
            atomic_torch_save(
                checkpoint_payload(model, optimizer, scheduler, scaler, epoch, global_step, best_psnr, args),
                output_dir / f"model_epoch_{epoch:03d}.pth",
            )
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
