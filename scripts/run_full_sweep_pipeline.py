#!/usr/bin/env python3
"""Run the full sweep-anchor LargeKernel3D -> MCTrack pipeline from this clone."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuscenes_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", default="9")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--split", default="sweep_trainval")
    parser.add_argument("--mctrack_processes", type=int, default=8)
    return parser.parse_args()


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("\n[run] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def latest_results_json(results_root: Path) -> Path:
    candidates = sorted(results_root.glob("nuscenes/*/results.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no MCTrack results.json found under {results_root}")
    return candidates[0]


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    nuscenes_root = Path(args.nuscenes_root).resolve()
    output_root = Path(args.output_root).resolve()
    checkpoint_arg = Path(args.checkpoint)
    checkpoint = checkpoint_arg.resolve() if checkpoint_arg.is_absolute() else (repo_root / checkpoint_arg).resolve()

    centerpoint_root = repo_root / "third_party" / "FocalsConv" / "CenterPoint"
    largekernel_config = repo_root / "third_party" / "LargeKernel3D" / "object-detection" / "configs" / "nusc" / "voxelnet" / "nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_largekernel3d_tiny.py"
    mctrack_root = repo_root / "third_party" / "MCTrack"
    sweep_info_pkl = output_root / "sweep_infos" / "infos_sweep_trainval_1sweep.pkl"
    detector_work_dir = output_root / "largekernel3d_sweep_trainval_1sweep"

    run([
        sys.executable,
        str(repo_root / "scripts" / "bootstrap_third_party.py"),
        "--repo_root",
        str(repo_root),
        "--nuscenes_root",
        str(nuscenes_root),
        "--output_root",
        str(output_root),
        "--split",
        args.split,
        "--frame_rate",
        "10",
    ])

    run([
        sys.executable,
        str(repo_root / "create_nuscenes_sweep_infos.py"),
        "--nuscenes_root",
        str(nuscenes_root),
        "--version",
        args.version,
        "--output",
        str(sweep_info_pkl),
    ])

    detector_env = os.environ.copy()
    detector_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run([
        sys.executable,
        str(repo_root / "run_largekernel3d_sweep_infer.py"),
        "--centerpoint_root",
        str(centerpoint_root),
        "--config",
        str(largekernel_config),
        "--checkpoint",
        str(checkpoint),
        "--nuscenes_root",
        str(nuscenes_root),
        "--version",
        args.version,
        "--sweep_info_pkl",
        str(sweep_info_pkl),
        "--work_dir",
        str(detector_work_dir),
    ], env=detector_env)

    run([
        sys.executable,
        str(repo_root / "convert_sweep_detections_to_mctrack_base.py"),
        "--nuscenes_root",
        str(nuscenes_root),
        "--version",
        args.version,
        "--det_json",
        str(detector_work_dir / "sweep_detections.json"),
        "--save_path",
        str(mctrack_root / "data" / "base_version" / "nuscenes"),
        "--detector",
        "largekernel",
        "--split",
        args.split,
    ])

    run([
        sys.executable,
        "main.py",
        "--dataset",
        "nuscenes",
        "-p",
        str(args.mctrack_processes),
    ], cwd=mctrack_root)

    results_json = latest_results_json(output_root / "mctrack_results")
    run([
        sys.executable,
        str(repo_root / "convert_official_results.py"),
        "--nuscenes_root",
        str(nuscenes_root),
        "--version",
        args.version,
        "--mctrack_result_json",
        str(results_json),
        "--output_dir",
        str(output_root),
    ])

    print(f"[done] final output root: {output_root}")


if __name__ == "__main__":
    main()
