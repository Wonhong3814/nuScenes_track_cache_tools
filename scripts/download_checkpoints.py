#!/usr/bin/env python3
"""Download checkpoints needed by the sweep LargeKernel3D pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve


LARGEKERNEL3D_CHECKPOINTS = {
    "largekernel3d_tiny_val.pth": "1qDCareDEyzElFMH0iPuMYkMVozI8qSGQ",
    "largekernel3d_multimodal_test.pth": "1Cipmcq5PFyxObWkJPG9LPUNVnYsrlYBH",
    "largekernel3d_f_multimodal_test.pth": "1MDSOGEtV0BZ_GCWDiedyLe9h1pi-lnnV",
}

DEEPLAB_URL = "https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--download_all", action="store_true", help="Download all LargeKernel3D checkpoints. Default: tiny val only.")
    parser.add_argument("--download_deeplab", action="store_true", help="Also download the multimodal DeeplabV3 ResNet50 pretrain.")
    return parser.parse_args()


def run_gdown(file_id: str, output_path: Path) -> None:
    cmd = [sys.executable, "-m", "gdown", "--id", file_id, "-O", str(output_path)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("gdown is not installed. Install requirements first: python3 -m pip install -r requirements.txt") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"gdown failed for {output_path}. Command: {' '.join(cmd)}") from exc


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    names = list(LARGEKERNEL3D_CHECKPOINTS) if args.download_all else ["largekernel3d_tiny_val.pth"]
    for name in names:
        output_path = checkpoint_dir / name
        if output_path.exists():
            print(f"[skip] {output_path}")
            continue
        print(f"[download] {name}")
        run_gdown(LARGEKERNEL3D_CHECKPOINTS[name], output_path)

    if args.download_deeplab:
        output_path = checkpoint_dir / "deeplabv3_resnet50_coco-cd0a2569.pth"
        if output_path.exists():
            print(f"[skip] {output_path}")
        else:
            print(f"[download] {output_path.name}")
            urlretrieve(DEEPLAB_URL, output_path)

    print(f"[done] checkpoints in {checkpoint_dir}")


if __name__ == "__main__":
    main()

