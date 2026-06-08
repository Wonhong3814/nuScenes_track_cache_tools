#!/usr/bin/env python3
"""Convert official MCTrack nuScenes results to flat sweep/GT pickle caches.

This script is intentionally small: it does not run LargeKernel3D, does not run
MCTrack, and does not build mini sequences. It only converts already generated
official tracking JSON results plus nuScenes keyframe GT into the requested
flat dict schemas.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

try:
    from pyquaternion import Quaternion
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pyquaternion is required for nuScenes box yaw conversion") from exc


TRACK_PKL_NAME = "mctrack_largekernel3d_sweep_tracks.pkl"
GT_PKL_NAME = "nuscenes_keyframe_gt.pkl"
LIDAR_SWEEP_PREFIX = "sweeps/LIDAR_TOP/"
KEYFRAME_LIDAR_PREFIX = "samples/LIDAR_TOP/"


class ConvertError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert official MCTrack nuScenes tracking JSON to requested flat caches."
    )
    parser.add_argument("--nuscenes_root", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--mctrack_result_json", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise ConvertError(f"missing file: {path}")
    with path.open("r") as f:
        return json.load(f)


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(rotation: list[float]) -> float:
    return wrap_angle(float(Quaternion(rotation).yaw_pitch_roll[0]))


def box_from_nuscenes_fields(translation: list[float], size_wlh: list[float], rotation: list[float]) -> list[float]:
    if len(translation) != 3 or len(size_wlh) != 3:
        raise ConvertError(f"bad box fields: translation={translation}, size={size_wlh}")
    x, y, z = [float(v) for v in translation]
    w, l, h = [float(v) for v in size_wlh]
    return [x, y, z, l, w, h, yaw_from_quat(rotation)]


def build_nuscenes_maps(nuscenes_root: Path, version: str) -> dict[str, Any]:
    table_root = nuscenes_root / version
    sample_data = load_json(table_root / "sample_data.json")
    samples = load_json(table_root / "sample.json")
    scenes = load_json(table_root / "scene.json")

    scene_name_by_token = {scene["token"]: scene["name"] for scene in scenes}
    sample_to_scene = {
        sample["token"]: scene_name_by_token[sample["scene_token"]]
        for sample in samples
    }

    sweep_token_to_scene: dict[str, str] = {}
    key_lidar_token_to_scene: dict[str, str] = {}
    sample_to_key_lidar: dict[str, str] = {}

    for sd in sample_data:
        filename = sd.get("filename", "")
        if filename.startswith(LIDAR_SWEEP_PREFIX):
            sweep_token_to_scene[sd["token"]] = sample_to_scene[sd["sample_token"]]
        elif filename.startswith(KEYFRAME_LIDAR_PREFIX) and sd.get("is_key_frame", False):
            scene_id = sample_to_scene[sd["sample_token"]]
            key_lidar_token_to_scene[sd["token"]] = scene_id
            sample_to_key_lidar[sd["sample_token"]] = sd["token"]

    if not sweep_token_to_scene:
        raise ConvertError(f"no sweep LIDAR_TOP sample_data found under {LIDAR_SWEEP_PREFIX}")
    if not key_lidar_token_to_scene:
        raise ConvertError(f"no keyframe LIDAR_TOP sample_data found under {KEYFRAME_LIDAR_PREFIX}")

    return {
        "table_root": table_root,
        "sweep_token_to_scene": sweep_token_to_scene,
        "key_lidar_token_to_scene": key_lidar_token_to_scene,
        "sample_to_key_lidar": sample_to_key_lidar,
    }


def convert_tracks(result_json: dict[str, Any], maps: dict[str, Any]) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    results = result_json.get("results")
    if not isinstance(results, dict):
        raise ConvertError("MCTrack result JSON must have a dict field named 'results'")

    sweep_token_to_scene = maps["sweep_token_to_scene"]
    track_data: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    bad_frame_ids: list[str] = []

    for frame_id, entries in results.items():
        scene_id = sweep_token_to_scene.get(frame_id)
        if scene_id is None:
            bad_frame_ids.append(frame_id)
            continue
        if not isinstance(entries, list):
            raise ConvertError(f"results[{frame_id}] must be a list")
        frame_tracks: dict[str, dict[str, Any]] = {}
        for entry in entries:
            track_id = entry.get("tracking_id")
            if track_id is None:
                raise ConvertError(f"missing tracking_id in frame {frame_id}: {entry}")
            class_name = entry.get("tracking_name")
            if class_name is None:
                raise ConvertError(f"missing tracking_name in frame {frame_id}, track {track_id}")
            score = entry.get("tracking_score")
            if score is None:
                raise ConvertError(f"missing tracking_score in frame {frame_id}, track {track_id}")
            bbox = box_from_nuscenes_fields(entry["translation"], entry["size"], entry["rotation"])
            frame_tracks[str(track_id)] = {
                "bbox": bbox,
                "class_name": str(class_name),
                "score": float(score),
            }
        track_data.setdefault(scene_id, {})[frame_id] = frame_tracks

    if bad_frame_ids:
        preview = ", ".join(bad_frame_ids[:5])
        raise ConvertError(
            "tracking result frame ids are not sweep LIDAR_TOP sample_data_tokens. "
            f"First bad ids: {preview}. Official MCTrack default nuScenes output is usually keyed by keyframe sample_token; "
            "that does not satisfy the requested sweep-frame schema."
        )
    return track_data


def convert_keyframe_gt(nuscenes_root: Path, version: str, maps: dict[str, Any]) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    table_root = nuscenes_root / version
    anns = load_json(table_root / "sample_annotation.json")
    instances = load_json(table_root / "instance.json")
    categories = load_json(table_root / "category.json")

    instance_to_category = {item["token"]: item["category_token"] for item in instances}
    category_name = {item["token"]: item["name"] for item in categories}
    sample_to_key_lidar = maps["sample_to_key_lidar"]
    key_lidar_token_to_scene = maps["key_lidar_token_to_scene"]

    gt_data: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for ann in anns:
        frame_id = sample_to_key_lidar.get(ann["sample_token"])
        if frame_id is None:
            continue
        scene_id = key_lidar_token_to_scene[frame_id]
        cat_token = instance_to_category.get(ann["instance_token"])
        class_name = category_name.get(cat_token, "unknown")
        bbox = box_from_nuscenes_fields(ann["translation"], ann["size"], ann["rotation"])
        gt_data.setdefault(scene_id, {}).setdefault(frame_id, {})[ann["instance_token"]] = {
            "bbox": bbox,
            "class_name": str(class_name),
        }
    return gt_data


def validate_flat_outputs(track_data: dict[str, Any], gt_data: dict[str, Any], maps: dict[str, Any]) -> None:
    if "tracks" in track_data:
        raise ConvertError("track_data must not contain a top-level 'tracks' key")
    if "gt" in gt_data:
        raise ConvertError("gt_data must not contain a top-level 'gt' key")

    sweep_token_to_scene = maps["sweep_token_to_scene"]
    key_lidar_token_to_scene = maps["key_lidar_token_to_scene"]

    for scene_id, frames in track_data.items():
        for frame_id, tracks in frames.items():
            if sweep_token_to_scene.get(frame_id) != scene_id:
                raise ConvertError(f"bad tracker frame id/scene: {scene_id}/{frame_id}")
            for track_id, item in tracks.items():
                if set(item.keys()) != {"bbox", "class_name", "score"}:
                    raise ConvertError(f"bad track item keys at {scene_id}/{frame_id}/{track_id}: {item.keys()}")
                if len(item["bbox"]) != 7:
                    raise ConvertError(f"bad track bbox length at {scene_id}/{frame_id}/{track_id}")

    for scene_id, frames in gt_data.items():
        for frame_id, instances in frames.items():
            if key_lidar_token_to_scene.get(frame_id) != scene_id:
                raise ConvertError(f"bad GT frame id/scene: {scene_id}/{frame_id}")
            for instance_token, item in instances.items():
                if set(item.keys()) != {"bbox", "class_name"}:
                    raise ConvertError(f"bad GT item keys at {scene_id}/{frame_id}/{instance_token}: {item.keys()}")
                if len(item["bbox"]) != 7:
                    raise ConvertError(f"bad GT bbox length at {scene_id}/{frame_id}/{instance_token}")


def main() -> None:
    args = parse_args()
    nuscenes_root = Path(args.nuscenes_root).resolve()
    result_path = Path(args.mctrack_result_json).resolve()
    output_dir = Path(args.output_dir).resolve()

    maps = build_nuscenes_maps(nuscenes_root, args.version)
    result_json = load_json(result_path)
    track_data = convert_tracks(result_json, maps)
    gt_data = convert_keyframe_gt(nuscenes_root, args.version, maps)
    validate_flat_outputs(track_data, gt_data, maps)

    save_pickle(track_data, output_dir / TRACK_PKL_NAME)
    save_pickle(gt_data, output_dir / GT_PKL_NAME)

    print(f"[done] {output_dir / TRACK_PKL_NAME}")
    print(f"[done] {output_dir / GT_PKL_NAME}")
    print(f"[summary] scenes={len(track_data)}, track_frames={sum(len(v) for v in track_data.values())}")
    print(f"[summary] gt_scenes={len(gt_data)}, gt_frames={sum(len(v) for v in gt_data.values())}")


if __name__ == "__main__":
    try:
        main()
    except ConvertError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(2)
