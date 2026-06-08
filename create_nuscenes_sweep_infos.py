#!/usr/bin/env python3
"""Create CenterPoint-style nuScenes infos for every LIDAR_TOP sweep frame."""

from __future__ import annotations

import argparse
import json
import pickle
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np
from pyquaternion import Quaternion


SWEEP_PREFIX = "sweeps/LIDAR_TOP/"
DEFAULT_NSWEEPS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuscenes_root", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_table(root: Path, name: str) -> list[dict[str, Any]]:
    path = root / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r") as f:
        return json.load(f)


def make_tf(translation: list[float], rotation: list[float], inverse: bool) -> Any:
    tm = np.eye(4)
    tm[:3, :3] = Quaternion(rotation).rotation_matrix
    tm[:3, 3] = np.array(translation)
    if inverse:
        tm = np.linalg.inv(tm)
    return tm


def duplicate_sweep(ref_path: Path, sd_token: str) -> dict[str, Any]:
    return {
        "lidar_path": str(ref_path),
        "sample_data_token": sd_token,
        "transform_matrix": None,
        "time_lag": 0.0,
    }


def build_history(
    *,
    sd: dict[str, Any],
    sample_data_by_token: dict[str, dict[str, Any]],
    ego_pose_by_token: dict[str, dict[str, Any]],
    calib_by_token: dict[str, dict[str, Any]],
    nuscenes_root: Path,
    ref_from_car: Any,
    car_from_global: Any,
    ref_time: float,
    nsweeps: int,
) -> list[dict[str, Any]]:
    sweeps: list[dict[str, Any]] = []
    curr = sd

    while len(sweeps) < nsweeps - 1:
        prev_token = curr.get("prev", "")
        if not prev_token:
            if sweeps:
                sweeps.append(dict(sweeps[-1]))
            else:
                sweeps.append(duplicate_sweep(nuscenes_root / sd["filename"], sd["token"]))
            continue

        curr = sample_data_by_token[prev_token]
        pose = ego_pose_by_token[curr["ego_pose_token"]]
        calib = calib_by_token[curr["calibrated_sensor_token"]]
        global_from_car = make_tf(pose["translation"], pose["rotation"], inverse=False)
        car_from_current = make_tf(calib["translation"], calib["rotation"], inverse=False)
        tm = reduce(lambda a, b: a.dot(b), [ref_from_car, car_from_global, global_from_car, car_from_current])
        time_lag = ref_time - 1e-6 * float(curr["timestamp"])

        sweeps.append(
            {
                "lidar_path": str(nuscenes_root / curr["filename"]),
                "sample_data_token": curr["token"],
                "transform_matrix": tm,
                "global_from_car": global_from_car,
                "car_from_current": car_from_current,
                "time_lag": time_lag,
            }
        )

    return sweeps


def main() -> None:
    args = parse_args()
    nsweeps = DEFAULT_NSWEEPS

    nuscenes_root = Path(args.nuscenes_root).resolve()
    table_root = nuscenes_root / args.version
    output = Path(args.output).resolve()

    sample_data = load_table(table_root, "sample_data")
    samples = load_table(table_root, "sample")
    scenes = load_table(table_root, "scene")
    ego_pose = load_table(table_root, "ego_pose")
    calibrated_sensor = load_table(table_root, "calibrated_sensor")

    sample_by_token = {item["token"]: item for item in samples}
    scene_by_token = {item["token"]: item for item in scenes}
    sample_data_by_token = {item["token"]: item for item in sample_data}
    ego_pose_by_token = {item["token"]: item for item in ego_pose}
    calib_by_token = {item["token"]: item for item in calibrated_sensor}

    sweep_sds = [
        sd for sd in sample_data
        if sd.get("filename", "").startswith(SWEEP_PREFIX)
    ]
    sweep_sds.sort(key=lambda item: (sample_by_token[item["sample_token"]]["scene_token"], item["timestamp"]))

    infos: list[dict[str, Any]] = []
    missing_files = 0
    for sd in sweep_sds:
        lidar_path = nuscenes_root / sd["filename"]
        if not lidar_path.exists():
            missing_files += 1
            continue

        sample = sample_by_token[sd["sample_token"]]
        scene = scene_by_token[sample["scene_token"]]
        calib = calib_by_token[sd["calibrated_sensor_token"]]
        pose = ego_pose_by_token[sd["ego_pose_token"]]
        ref_from_car = make_tf(calib["translation"], calib["rotation"], inverse=True)
        car_from_global = make_tf(pose["translation"], pose["rotation"], inverse=True)
        ref_time = 1e-6 * float(sd["timestamp"])

        info = {
            "lidar_path": str(lidar_path),
            "token": sd["token"],
            "sample_token": sd["sample_token"],
            "scene_token": sample["scene_token"],
            "scene_name": scene["name"],
            "sweeps": build_history(
                sd=sd,
                sample_data_by_token=sample_data_by_token,
                ego_pose_by_token=ego_pose_by_token,
                calib_by_token=calib_by_token,
                nuscenes_root=nuscenes_root,
                ref_from_car=ref_from_car,
                car_from_global=car_from_global,
                ref_time=ref_time,
                nsweeps=nsweeps,
            ),
            "ref_from_car": ref_from_car,
            "car_from_global": car_from_global,
            "timestamp": ref_time,
        }
        infos.append(info)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(infos, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[done] wrote {len(infos)} sweep infos: {output}")
    print(f"[summary] selected_sweeps={len(sweep_sds)} missing_files={missing_files} nsweeps={nsweeps}")
    print("[summary] anchor frames are sweeps/LIDAR_TOP; detector input uses the current anchor sweep only")


if __name__ == "__main__":
    main()
