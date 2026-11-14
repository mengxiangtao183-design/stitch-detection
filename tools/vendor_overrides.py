"""Capture or restore local changes in the nested third-party Git repositories.

Usage:
    python tools/vendor_overrides.py capture
    python tools/vendor_overrides.py restore
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "vendor_overrides"
REPOS = (
    "C3Netqing-baseline-dino",
    "MISCFilter",
    "com_models/third_party/PIDNet",
    "deblur_comparison_suite/third_party/AdaRevD",
    "deblur_comparison_suite/third_party/BANet",
    "deblur_comparison_suite/third_party/CascadedGaze",
    "deblur_comparison_suite/third_party/DarkIR",
    "deblur_comparison_suite/third_party/FFTformer",
    "deblur_comparison_suite/third_party/HINet_official",
    "deblur_comparison_suite/third_party/LoFormer",
    "deblur_comparison_suite/third_party/LoFormer_index_only",
    "deblur_comparison_suite/third_party/MIMO-UNet",
    "deblur_comparison_suite/third_party/Restormer",
    "deblur_comparison_suite/third_party/SFNet_official",
    "deblur_comparison_suite/third_party/Stripformer",
    "deblur_comparison_suite/third_party/UFPDeblur",
)
SKIP_PARTS = {
    ".git", ".vscode", "__pycache__", "checkpoints", "datasets", "dataset",
    "results", "outputs", "runs", "logs", "pretrained", "pretrained_models",
}
SKIP_SUFFIXES = {
    ".pyc", ".pth", ".pt", ".ckpt", ".pkl", ".safetensors", ".zip",
    ".npy", ".npz", ".log", ".joblib", ".onnx",
}


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def capture() -> None:
    for name in REPOS:
        repo = ROOT / name
        dest = OUT / name
        dest.mkdir(parents=True, exist_ok=True)
        patch = git(repo, "diff", "--binary", "HEAD", "--", ".", ":!*.pyc", ":!**/__pycache__/**")
        patch_path = dest / "tracked.patch"
        if patch:
            patch_path.write_bytes(patch)
        elif patch_path.exists():
            patch_path.unlink()

        copied = 0
        raw = git(repo, "ls-files", "--others", "--exclude-standard", "-z")
        for item in filter(None, raw.decode("utf-8", "surrogateescape").split("\0")):
            relative = Path(item)
            source = repo / relative
            if not source.is_file() or any(part in SKIP_PARTS for part in relative.parts):
                continue
            if source.suffix.lower() in SKIP_SUFFIXES or source.stat().st_size > 50 * 1024 * 1024:
                continue
            target = dest / "files" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
        print(f"{name}: patch={len(patch)} bytes, new files={copied}")


def restore() -> None:
    for name in REPOS:
        repo = ROOT / name
        dest = OUT / name
        patch_path = dest / "tracked.patch"
        if patch_path.exists():
            subprocess.run(["git", "-C", str(repo), "apply", str(patch_path)], check=True)
        files = dest / "files"
        if files.exists():
            for source in files.rglob("*"):
                if source.is_file():
                    target = repo / source.relative_to(files)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
        print(f"restored {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "restore"))
    args = parser.parse_args()
    capture() if args.action == "capture" else restore()
