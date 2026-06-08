#!/usr/bin/env python3
"""Prepare third-party repos after cloning this repository with submodules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", default=None, help="Pipeline repo root. Default: parent of this script directory.")
    parser.add_argument("--nuscenes_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--split", default="sweep_trainval")
    parser.add_argument("--detector", default="largekernel")
    parser.add_argument("--frame_rate", type=int, default=10)
    return parser.parse_args()


def repo_root_from_args(value: str | None) -> Path:
    if value:
        return Path(value).resolve()
    return Path(__file__).resolve().parents[1]


def require_dir(path: Path, hint: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{hint} not found: {path}")


def patch_centerpoint(centerpoint_root: Path) -> list[str]:
    changes: list[str] = []
    backbones = centerpoint_root / "det3d" / "models" / "backbones"
    require_dir(backbones, "CenterPoint backbones directory")

    alias_path = backbones / "scn_largekernel_multimodal.py"
    alias_text = "from .scn_largekernel import SpMiddleResNetFHDLargeKernel\n"
    if not alias_path.exists() or alias_path.read_text() != alias_text:
        alias_path.write_text(alias_text)
        changes.append(str(alias_path))

    models_init = centerpoint_root / "det3d" / "models" / "__init__.py"
    text = models_init.read_text()
    if "ROI heads disabled:" not in text:
        text = text.replace("from .second_stage import * \nfrom .roi_heads import * \n", "from .second_stage import *\ntry:\n    from .roi_heads import *\nexcept ImportError as exc:\n    print(f\"ROI heads disabled: {exc}\")\n")
        text = text.replace("from .second_stage import *\nfrom .roi_heads import *\n", "from .second_stage import *\ntry:\n    from .roi_heads import *\nexcept ImportError as exc:\n    print(f\"ROI heads disabled: {exc}\")\n")
        models_init.write_text(text)
        changes.append(str(models_init))

    return changes


def configure_mctrack(mctrack_root: Path, nuscenes_root: Path, output_root: Path, split: str, detector: str, frame_rate: int) -> list[str]:
    cfg_path = mctrack_root / "config" / "nuscenes.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"MCTrack nuScenes config not found: {cfg_path}")
    with cfg_path.open("r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    cfg["SPLIT"] = split
    cfg["DETECTOR"] = detector
    cfg["DATASET_ROOT"] = str(nuscenes_root)
    cfg["DETECTIONS_ROOT"] = "data/base_version/nuscenes/"
    cfg["SAVE_PATH"] = str(output_root / "mctrack_results")
    cfg["FRAME_RATE"] = int(frame_rate)

    with cfg_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return [str(cfg_path)]


def main() -> None:
    args = parse_args()
    repo_root = repo_root_from_args(args.repo_root)
    nuscenes_root = Path(args.nuscenes_root).resolve()
    output_root = Path(args.output_root).resolve()

    largekernel_root = repo_root / "third_party" / "LargeKernel3D"
    focalsconv_root = repo_root / "third_party" / "FocalsConv"
    centerpoint_root = focalsconv_root / "CenterPoint"
    mctrack_root = repo_root / "third_party" / "MCTrack"

    require_dir(largekernel_root, "LargeKernel3D submodule")
    require_dir(centerpoint_root, "FocalsConv/CenterPoint submodule")
    require_dir(mctrack_root, "MCTrack submodule")
    require_dir(nuscenes_root, "nuScenes root")
    output_root.mkdir(parents=True, exist_ok=True)

    changes = {
        "centerpoint": patch_centerpoint(centerpoint_root),
        "mctrack": configure_mctrack(
            mctrack_root=mctrack_root,
            nuscenes_root=nuscenes_root,
            output_root=output_root,
            split=args.split,
            detector=args.detector,
            frame_rate=args.frame_rate,
        ),
    }

    manifest = {
        "repo_root": str(repo_root),
        "nuscenes_root": str(nuscenes_root),
        "output_root": str(output_root),
        "largekernel3d_root": str(largekernel_root),
        "centerpoint_root": str(centerpoint_root),
        "mctrack_root": str(mctrack_root),
        "split": args.split,
        "detector": args.detector,
        "frame_rate": args.frame_rate,
        "changes": changes,
    }
    manifest_path = output_root / "bootstrap_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[ok] LargeKernel3D: {largekernel_root}")
    print(f"[ok] CenterPoint: {centerpoint_root}")
    print(f"[ok] MCTrack: {mctrack_root}")
    print(f"[done] wrote {manifest_path}")


if __name__ == "__main__":
    main()

