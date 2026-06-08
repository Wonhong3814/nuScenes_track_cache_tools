#!/usr/bin/env python3
"""Run MCTrack on one scene from a large base-version nuScenes JSON file."""

from __future__ import annotations

import argparse
import json
import mmap
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mctrack_root", required=True)
    parser.add_argument("--base_json", required=True)
    parser.add_argument("--scene_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Run only the first N frames of the selected scene. Default: all frames.",
    )
    return parser.parse_args()


def extract_scene_array(base_json: Path, scene_id: str) -> list[dict]:
    key = json.dumps(scene_id).encode("utf-8")
    with base_json.open("rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            key_pos = mm.find(key)
            if key_pos < 0:
                raise KeyError(f"scene_id not found in {base_json}: {scene_id}")
            colon_pos = mm.find(b":", key_pos + len(key))
            if colon_pos < 0:
                raise RuntimeError(f"could not find ':' after scene key {scene_id}")
            start = mm.find(b"[", colon_pos + 1)
            if start < 0:
                raise RuntimeError(f"could not find scene array after scene key {scene_id}")

            depth = 0
            in_string = False
            escape = False
            end = None
            for pos in range(start, len(mm)):
                byte = mm[pos]
                if in_string:
                    if escape:
                        escape = False
                    elif byte == 0x5C:  # backslash
                        escape = True
                    elif byte == 0x22:  # quote
                        in_string = False
                    continue
                if byte == 0x22:
                    in_string = True
                elif byte == 0x5B:  # [
                    depth += 1
                elif byte == 0x5D:  # ]
                    depth -= 1
                    if depth == 0:
                        end = pos + 1
                        break
            if end is None:
                raise RuntimeError(f"could not find end of scene array for {scene_id}")
            return json.loads(mm[start:end])


def main() -> None:
    args = parse_args()
    mctrack_root = Path(args.mctrack_root).resolve()
    base_json = Path(args.base_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(mctrack_root))
    os.chdir(mctrack_root)

    from main import run
    from utils.nusc_utils import save_results_nuscenes, save_results_nuscenes_for_motion

    cfg_path = mctrack_root / "config" / "nuscenes.yaml"
    with cfg_path.open("r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    cfg["SAVE_PATH"] = str(output_dir)

    scene_data_full = extract_scene_array(base_json, args.scene_id)
    scene_data = scene_data_full
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError(f"--max_frames must be positive, got {args.max_frames}")
        scene_data = scene_data_full[: args.max_frames]

    scenes_data = {args.scene_id: scene_data}
    tracking_results: dict = {}
    run(args.scene_id, scenes_data, cfg, SimpleNamespace(), tracking_results)

    save_results_nuscenes(tracking_results, str(output_dir))
    save_results_nuscenes_for_motion(tracking_results, str(output_dir))

    meta = {
        "scene_id": args.scene_id,
        "num_frames": len(scene_data),
        "num_frames_full_scene": len(scene_data_full),
        "max_frames": args.max_frames,
        "base_json": str(base_json),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": "MCTrack tracking is CPU/numpy based; CUDA_VISIBLE_DEVICES does not move this tracker to GPU.",
    }
    with (output_dir / "single_scene_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {output_dir / 'results.json'}")
    print(f"[done] {output_dir / 'results_for_motion.json'}")
    print(f"[done] {output_dir / 'single_scene_meta.json'}")


if __name__ == "__main__":
    main()
