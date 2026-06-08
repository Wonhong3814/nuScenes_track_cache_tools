#!/usr/bin/env python3
"""Convert sweep-token detection JSON to MCTrack base-version nuScenes input."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from pyquaternion import Quaternion


SWEEP_PREFIX = "sweeps/LIDAR_TOP/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuscenes_root", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--det_json", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--detector", default="largekernel")
    parser.add_argument("--split", default="sweep_trainval")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r") as f:
        return json.load(f)


def transform_matrix(translation: list[float], rotation: list[float], inverse: bool = False) -> np.ndarray:
    tm = np.eye(4)
    tm[:3, :3] = Quaternion(rotation).rotation_matrix
    tm[:3, 3] = np.array(translation)
    if inverse:
        tm = np.linalg.inv(tm)
    return tm


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(rotation: list[float]) -> float:
    return wrap_angle(float(Quaternion(rotation).yaw_pitch_roll[0]))


def convert_bbox(bbox: dict[str, Any]) -> dict[str, Any]:
    w, l, h = [float(v) for v in bbox["size"]]
    return {
        "detection_score": float(bbox.get("detection_score", 0.0)),
        "category": str(bbox.get("detection_name", "")),
        "global_xyz": [float(v) for v in bbox["translation"]],
        "global_orientation": [float(v) for v in bbox["rotation"]],
        "global_yaw": yaw_from_quat(bbox["rotation"]),
        "lwh": [l, w, h],
        "global_velocity": [float(v) for v in bbox.get("velocity", [0.0, 0.0])[:2]],
        "global_acceleration": [float(v) for v in bbox.get("acceleration", [0.0, 0.0])[:2]],
        "bbox_image": {
            "camera_type": None,
            "x1y1x2y2": [0.0, 0.0, 0.0, 0.0],
        },
    }


def build_scene_sweeps(nuscenes_root: Path, version: str) -> dict[str, list[dict[str, Any]]]:
    table_root = nuscenes_root / version
    sample_data = load_json(table_root / "sample_data.json")
    samples = load_json(table_root / "sample.json")
    scenes = load_json(table_root / "scene.json")

    sample_by_token = {item["token"]: item for item in samples}
    scene_name_by_token = {item["token"]: item["name"] for item in scenes}
    scene_sweeps: dict[str, list[dict[str, Any]]] = {}
    for sd in sample_data:
        if not sd.get("filename", "").startswith(SWEEP_PREFIX):
            continue
        sample = sample_by_token[sd["sample_token"]]
        scene_name = scene_name_by_token[sample["scene_token"]]
        scene_sweeps.setdefault(scene_name, []).append(sd)

    for sweeps in scene_sweeps.values():
        sweeps.sort(key=lambda item: item["timestamp"])
    return scene_sweeps


def main() -> None:
    args = parse_args()
    nuscenes_root = Path(args.nuscenes_root).resolve()
    det_json_path = Path(args.det_json).resolve()
    save_root = Path(args.save_path).resolve()
    table_root = nuscenes_root / args.version

    det_json = load_json(det_json_path)
    det_results = det_json.get("results")
    if not isinstance(det_results, dict):
        raise ValueError("det_json must contain dict field 'results'")

    ego_pose = {item["token"]: item for item in load_json(table_root / "ego_pose.json")}
    calibrated_sensor = {item["token"]: item for item in load_json(table_root / "calibrated_sensor.json")}
    scene_sweeps = build_scene_sweeps(nuscenes_root, args.version)

    all_datas: dict[str, list[dict[str, Any]]] = {}
    total_frames = 0
    total_boxes = 0
    missing_detection_frames = 0

    for scene_name, sweeps in scene_sweeps.items():
        scene_datas = []
        for frame_index, sd in enumerate(sweeps):
            pose = ego_pose[sd["ego_pose_token"]]
            calib = calibrated_sensor[sd["calibrated_sensor_token"]]
            global2ego = transform_matrix(pose["translation"], pose["rotation"], inverse=True)
            ego2global = transform_matrix(pose["translation"], pose["rotation"], inverse=False)
            lidar2ego = transform_matrix(calib["translation"], calib["rotation"], inverse=False)
            ego2lidar = transform_matrix(calib["translation"], calib["rotation"], inverse=True)
            lidar2global = ego2global.dot(lidar2ego)
            global2lidar = ego2lidar.dot(global2ego)

            raw_dets = det_results.get(sd["token"])
            if raw_dets is None:
                missing_detection_frames += 1
                raw_dets = []

            bboxes = [convert_bbox(bbox) for bbox in raw_dets]
            total_boxes += len(bboxes)
            scene_datas.append(
                {
                    "frame_id": frame_index,
                    "cur_sample_token": sd["token"],
                    "timestamp": int(sd["timestamp"]),
                    "bboxes": bboxes,
                    "transform_matrix": {
                        "global2ego": global2ego.tolist(),
                        "ego2lidar": ego2lidar.tolist(),
                        "global2lidar": global2lidar.tolist(),
                        "lidar2global": lidar2global.tolist(),
                        "cameras_transform_matrix": {},
                    },
                }
            )
            total_frames += 1
        all_datas[scene_name] = scene_datas

    output_dir = save_root / args.detector
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}.json"
    with output_path.open("w") as f:
        json.dump(all_datas, f)

    print(f"[done] {output_path}")
    print(f"[summary] scenes={len(all_datas)} frames={total_frames} boxes={total_boxes}")
    print(f"[summary] missing_detection_frames={missing_detection_frames}")


if __name__ == "__main__":
    main()
