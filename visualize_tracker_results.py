#!/usr/bin/env python3
"""Visualize nuScenes MCTrack tracker results over LIDAR_TOP point clouds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from pyquaternion import Quaternion


LIDAR_TOP_PREFIXES = ("sweeps/LIDAR_TOP/", "samples/LIDAR_TOP/")
BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuscenes_root", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tracker_result_json", help="MCTrack nuScenes-style results.json")
    group.add_argument("--track_pkl", help="Flat tracker cache pkl from convert_official_results.py")
    parser.add_argument("--gt_pkl", help="Optional flat keyframe GT pkl to overlay when frame ids match")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene_id", help="Visualize only one scene. Default: all scenes in tracker results.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional cap after scene filtering.")
    parser.add_argument("--image_size", type=int, default=1400)
    parser.add_argument("--point_size", type=float, default=0.18)
    parser.add_argument("--line_width", type=float, default=1.4)
    parser.add_argument("--no_points", action="store_true", help="Draw boxes only.")
    parser.add_argument("--no_labels", action="store_true", help="Do not draw track labels.")
    parser.add_argument("--no_sample_gt", action="store_true", help="Do not overlay keyframe GT from each frame's sample_token.")
    parser.add_argument("--make_3d_html", action="store_true", help="Also write per-frame interactive 3D Plotly HTML.")
    parser.add_argument("--max_3d_frames", type=int, default=None, help="Optional cap for 3D HTML frames after frame filtering.")
    parser.add_argument("--max_3d_points", type=int, default=50000)
    parser.add_argument("--plotlyjs", choices=["inline", "cdn"], default="inline")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def make_tf(translation: list[float], rotation: list[float], inverse: bool = False) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Quaternion(rotation).rotation_matrix
    mat[:3, 3] = np.asarray(translation, dtype=np.float64)
    if inverse:
        mat = np.linalg.inv(mat)
    return mat


def transform_points(points_xyz: np.ndarray, mat: np.ndarray) -> np.ndarray:
    if points_xyz.size == 0:
        return points_xyz.reshape(-1, 3)
    homo = np.concatenate([points_xyz[:, :3], np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)], axis=1)
    return (homo @ mat.T)[:, :3]


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_quat(yaw: float) -> list[float]:
    return [float(v) for v in Quaternion(axis=[0.0, 0.0, 1.0], radians=wrap_angle(yaw)).elements]


def box_corners_global_from_quat(
    center: list[float],
    size_wlh: list[float],
    rotation: list[float],
) -> np.ndarray:
    w, l, h = [float(v) for v in size_wlh]
    x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dtype=np.float64)
    y_corners = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dtype=np.float64)
    z_corners = np.array([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], dtype=np.float64)
    corners = np.vstack([x_corners, y_corners, z_corners])
    rotated = Quaternion(rotation).rotation_matrix @ corners
    return rotated.T + np.asarray(center, dtype=np.float64).reshape(1, 3)


def box_corners_global_from_lwh_yaw(bbox: list[float]) -> np.ndarray:
    if len(bbox) != 7:
        raise ValueError(f"expected [x,y,z,l,w,h,yaw], got {bbox}")
    x, y, z, l, w, h, yaw = [float(v) for v in bbox]
    return box_corners_global_from_quat([x, y, z], [w, l, h], yaw_to_quat(yaw))


def bev_corners_from_lidar_corners(corners_lidar: np.ndarray) -> np.ndarray:
    return corners_lidar[[0, 1, 2, 3], :2]


def track_color(track_id: str) -> str:
    digest = hashlib.md5(str(track_id).encode("utf-8")).hexdigest()
    return f"#{digest[:6]}"


def class_color(class_name: str) -> str:
    palette = {
        "car": "#ff6b35",
        "truck": "#c77dff",
        "bus": "#ffd166",
        "trailer": "#9d4edd",
        "construction_vehicle": "#f4a261",
        "pedestrian": "#06d6a0",
        "motorcycle": "#118ab2",
        "bicycle": "#00b4d8",
        "traffic_cone": "#ef476f",
        "barrier": "#8d99ae",
    }
    return palette.get(class_name, "#ffffff")


def short_class_name(class_name: str) -> str:
    mapping = {
        "vehicle.car": "car",
        "vehicle.truck": "truck",
        "vehicle.bus.rigid": "bus",
        "vehicle.bus.bendy": "bus",
        "vehicle.trailer": "trailer",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "vehicle.bicycle": "bicycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
        "movable_object.barrier": "barrier",
    }
    return mapping.get(class_name, class_name)


def build_nuscenes_maps(nuscenes_root: Path, version: str) -> dict[str, Any]:
    table_root = nuscenes_root / version
    sample_data = load_json(table_root / "sample_data.json")
    samples = load_json(table_root / "sample.json")
    scenes = load_json(table_root / "scene.json")
    ego_pose = load_json(table_root / "ego_pose.json")
    calibrated_sensor = load_json(table_root / "calibrated_sensor.json")
    annotations = load_json(table_root / "sample_annotation.json")
    instances = load_json(table_root / "instance.json")
    categories = load_json(table_root / "category.json")

    scene_name_by_token = {scene["token"]: scene["name"] for scene in scenes}
    sample_by_token = {sample["token"]: sample for sample in samples}
    sample_to_scene = {
        sample["token"]: scene_name_by_token[sample["scene_token"]]
        for sample in samples
    }
    ego_pose_by_token = {item["token"]: item for item in ego_pose}
    calib_by_token = {item["token"]: item for item in calibrated_sensor}
    instance_by_token = {item["token"]: item for item in instances}
    category_by_token = {item["token"]: item for item in categories}

    frame_meta: dict[str, dict[str, Any]] = {}
    for sd in sample_data:
        filename = sd.get("filename", "")
        if not filename.startswith(LIDAR_TOP_PREFIXES):
            continue
        sample = sample_by_token[sd["sample_token"]]
        pose = ego_pose_by_token[sd["ego_pose_token"]]
        calib = calib_by_token[sd["calibrated_sensor_token"]]
        global_from_ego = make_tf(pose["translation"], pose["rotation"])
        ego_from_lidar = make_tf(calib["translation"], calib["rotation"])
        global_from_lidar = global_from_ego @ ego_from_lidar
        lidar_from_global = np.linalg.inv(global_from_lidar)
        frame_meta[sd["token"]] = {
            "frame_id": sd["token"],
            "sample_token": sd["sample_token"],
            "scene_id": sample_to_scene[sd["sample_token"]],
            "scene_token": sample["scene_token"],
            "filename": filename,
            "timestamp": int(sd["timestamp"]),
            "is_key_frame": bool(sd.get("is_key_frame", False)),
            "lidar_from_global": lidar_from_global,
        }

    sample_gt: dict[str, list[dict[str, Any]]] = {}
    for ann in annotations:
        instance = instance_by_token.get(ann["instance_token"], {})
        category = category_by_token.get(instance.get("category_token", ""), {})
        class_name = short_class_name(str(category.get("name", "unknown")))
        sample_gt.setdefault(ann["sample_token"], []).append(
            {
                "instance_token": str(ann["instance_token"]),
                "class_name": class_name,
                "corners_global": box_corners_global_from_quat(
                    center=ann["translation"],
                    size_wlh=ann["size"],
                    rotation=ann["rotation"],
                ),
                "source": "sample_keyframe_gt",
            }
        )

    return {
        "frame_meta": frame_meta,
        "sample_to_scene": sample_to_scene,
        "sample_gt": sample_gt,
    }


def load_tracker_from_json(result_path: Path, maps: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result_json = load_json(result_path)
    results = result_json.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"{result_path} must contain a dict field named 'results'")
    frame_meta = maps["frame_meta"]
    by_scene: dict[str, dict[str, list[dict[str, Any]]]] = {}
    bad_frame_ids: list[str] = []
    for frame_id, entries in results.items():
        meta = frame_meta.get(frame_id)
        if meta is None:
            bad_frame_ids.append(frame_id)
            continue
        normalized: list[dict[str, Any]] = []
        for entry in entries:
            normalized.append(
                {
                    "track_id": str(entry["tracking_id"]),
                    "class_name": str(entry["tracking_name"]),
                    "score": float(entry.get("tracking_score", 0.0)),
                    "corners_global": box_corners_global_from_quat(
                        center=entry["translation"],
                        size_wlh=entry["size"],
                        rotation=entry["rotation"],
                    ),
                }
            )
        by_scene.setdefault(meta["scene_id"], {})[frame_id] = normalized
    if bad_frame_ids:
        preview = ", ".join(bad_frame_ids[:5])
        raise ValueError(f"tracker result contains frame ids not found in LIDAR_TOP sample_data: {preview}")
    return by_scene


def load_tracker_from_pkl(track_pkl: Path, maps: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    raw = load_pickle(track_pkl)
    frame_meta = maps["frame_meta"]
    by_scene: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for scene_id, frames in raw.items():
        for frame_id, tracks in frames.items():
            if frame_id not in frame_meta:
                raise ValueError(f"track pkl frame id not found in LIDAR_TOP sample_data: {scene_id}/{frame_id}")
            items: list[dict[str, Any]] = []
            for track_id, entry in tracks.items():
                items.append(
                    {
                        "track_id": str(track_id),
                        "class_name": str(entry["class_name"]),
                        "score": float(entry.get("score", 0.0)),
                        "corners_global": box_corners_global_from_lwh_yaw(entry["bbox"]),
                    }
                )
            by_scene.setdefault(scene_id, {})[frame_id] = items
    return by_scene


def load_gt(gt_pkl: Path | None, maps: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    if gt_pkl is None:
        return {}
    raw = load_pickle(gt_pkl)
    frame_meta = maps["frame_meta"]
    by_scene: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for scene_id, frames in raw.items():
        for frame_id, instances in frames.items():
            if frame_id not in frame_meta:
                continue
            items: list[dict[str, Any]] = []
            for instance_token, entry in instances.items():
                items.append(
                    {
                        "instance_token": str(instance_token),
                        "class_name": str(entry["class_name"]),
                        "corners_global": box_corners_global_from_lwh_yaw(entry["bbox"]),
                        "source": "exact_frame_gt",
                    }
                )
            by_scene.setdefault(scene_id, {})[frame_id] = items
    return by_scene


def select_gt_for_frame(
    *,
    scene_id: str,
    frame_id: str,
    meta: dict[str, Any],
    exact_gt_by_scene: dict[str, dict[str, list[dict[str, Any]]]],
    sample_gt_by_sample: dict[str, list[dict[str, Any]]],
    use_sample_gt: bool,
) -> tuple[list[dict[str, Any]], str]:
    exact = exact_gt_by_scene.get(scene_id, {}).get(frame_id, [])
    if exact:
        return exact, "exact_frame_gt"
    if use_sample_gt:
        sample_gt = sample_gt_by_sample.get(meta["sample_token"], [])
        if sample_gt:
            return sample_gt, "sample_keyframe_gt"
    return [], "none"


def read_lidar_points(nuscenes_root: Path, filename: str) -> np.ndarray:
    path = nuscenes_root / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size % 5 != 0:
        raise ValueError(f"bad nuScenes lidar bin shape: {path} has {arr.size} floats")
    return arr.reshape(-1, 5)


def draw_bev_frame(
    *,
    nuscenes_root: Path,
    frame_id: str,
    frame_index: int,
    meta: dict[str, Any],
    tracks: list[dict[str, Any]],
    gt_items: list[dict[str, Any]],
    output_path: Path,
    image_size: int,
    point_size: float,
    line_width: float,
    draw_points: bool,
    draw_labels: bool,
) -> dict[str, Any]:
    points = read_lidar_points(nuscenes_root, meta["filename"])
    lidar_from_global = meta["lidar_from_global"]

    fig_size = image_size / 100.0
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=100)
    fig.patch.set_facecolor("#080a0f")
    ax.set_facecolor("#080a0f")

    if draw_points and points.size:
        intensity = points[:, 3] if points.shape[1] > 3 else np.zeros((points.shape[0],), dtype=np.float32)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=point_size,
            c=intensity,
            cmap="gray",
            alpha=0.65,
            linewidths=0,
        )

    all_xy: list[np.ndarray] = []
    if points.size:
        all_xy.append(points[:, :2])

    for item in gt_items:
        corners_lidar = transform_points(item["corners_global"], lidar_from_global)
        bev = bev_corners_from_lidar_corners(corners_lidar)
        all_xy.append(bev)
        closed = np.vstack([bev, bev[0]])
        ax.plot(closed[:, 0], closed[:, 1], color="#00ff66", linewidth=line_width + 0.5)
        if draw_labels:
            center = corners_lidar.mean(axis=0)
            ax.text(center[0], center[1], f"GT {item['class_name']}", color="#00ff66", fontsize=6)

    for item in tracks:
        corners_lidar = transform_points(item["corners_global"], lidar_from_global)
        bev = bev_corners_from_lidar_corners(corners_lidar)
        all_xy.append(bev)
        closed = np.vstack([bev, bev[0]])
        color = track_color(item["track_id"])
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=line_width)
        heading = (bev[0] + bev[1]) * 0.5
        center = corners_lidar.mean(axis=0)
        ax.plot([center[0], heading[0]], [center[1], heading[1]], color=color, linewidth=line_width * 0.8)
        if draw_labels:
            label = f"{item['track_id']} {item['class_name']} {item['score']:.2f}"
            ax.text(center[0], center[1], label, color=class_color(item["class_name"]), fontsize=6)

    if all_xy:
        xy = np.concatenate(all_xy, axis=0)
        finite = np.isfinite(xy).all(axis=1)
        xy = xy[finite]
        if xy.size:
            x_min, y_min = xy.min(axis=0)
            x_max, y_max = xy.max(axis=0)
            pad = max(5.0, 0.04 * max(x_max - x_min, y_max - y_min))
            ax.set_xlim(float(x_min - pad), float(x_max + pad))
            ax.set_ylim(float(y_min - pad), float(y_max + pad))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x lidar (m)", color="#c7d0df")
    ax.set_ylabel("y lidar (m)", color="#c7d0df")
    ax.tick_params(colors="#8e99aa", labelsize=7)
    ax.grid(color="#263142", linewidth=0.35, alpha=0.55)
    title = (
        f"{meta['scene_id']} frame={frame_index} token={frame_id} "
        f"tracks={len(tracks)} points={points.shape[0]}"
    )
    ax.set_title(title, color="#eef3ff", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    return {
        "scene_id": meta["scene_id"],
        "frame_index": frame_index,
        "frame_id": frame_id,
        "timestamp": meta["timestamp"],
        "filename": meta["filename"],
        "num_points": int(points.shape[0]),
        "num_tracks": int(len(tracks)),
        "num_gt": int(len(gt_items)),
        "gt_source": gt_items[0].get("source", "none") if gt_items else "none",
        "image": output_path.name,
    }


def _subsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
    return points[idx]


def add_box_trace_3d(fig: go.Figure, corners_lidar: np.ndarray, color: str, name: str, label: str) -> None:
    x_vals: list[float | None] = []
    y_vals: list[float | None] = []
    z_vals: list[float | None] = []
    for start, end in BOX_EDGES:
        x_vals.extend([float(corners_lidar[start, 0]), float(corners_lidar[end, 0]), None])
        y_vals.extend([float(corners_lidar[start, 1]), float(corners_lidar[end, 1]), None])
        z_vals.extend([float(corners_lidar[start, 2]), float(corners_lidar[end, 2]), None])
    fig.add_trace(
        go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode="lines",
            name=name,
            line={"color": color, "width": 5},
            hovertemplate=label + "<extra></extra>",
            showlegend=False,
        )
    )


def write_3d_frame_html(
    *,
    nuscenes_root: Path,
    frame_id: str,
    frame_index: int,
    meta: dict[str, Any],
    tracks: list[dict[str, Any]],
    gt_items: list[dict[str, Any]],
    output_path: Path,
    max_points: int,
    draw_points: bool,
    draw_labels: bool,
    plotlyjs: str,
) -> None:
    points = read_lidar_points(nuscenes_root, meta["filename"])
    points = _subsample_points(points, max_points)
    lidar_from_global = meta["lidar_from_global"]
    fig = go.Figure()

    if draw_points and points.size:
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                name="LiDAR points",
                marker={
                    "size": 1.2,
                    "color": points[:, 3],
                    "colorscale": "Viridis",
                    "opacity": 0.72,
                },
                hoverinfo="skip",
            )
        )

    for item in gt_items:
        corners_lidar = transform_points(item["corners_global"], lidar_from_global)
        label = f"GT {item['class_name']} {item.get('instance_token', '')}"
        add_box_trace_3d(fig, corners_lidar, "#00ff66", "GT", label)
        if draw_labels:
            center = corners_lidar.mean(axis=0)
            fig.add_trace(
                go.Scatter3d(
                    x=[float(center[0])],
                    y=[float(center[1])],
                    z=[float(center[2])],
                    mode="text",
                    text=[label],
                    textfont={"color": "#00ff66", "size": 10},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    for item in tracks:
        corners_lidar = transform_points(item["corners_global"], lidar_from_global)
        label = f"track={item['track_id']} {item['class_name']} score={item['score']:.3f}"
        color = track_color(item["track_id"])
        add_box_trace_3d(fig, corners_lidar, color, "track", label)
        if draw_labels:
            center = corners_lidar.mean(axis=0)
            fig.add_trace(
                go.Scatter3d(
                    x=[float(center[0])],
                    y=[float(center[1])],
                    z=[float(center[2])],
                    mode="text",
                    text=[f"{item['track_id']} {item['class_name']} {item['score']:.2f}"],
                    textfont={"color": class_color(item["class_name"]), "size": 10},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title=(
            f"{meta['scene_id']} frame={frame_index} token={frame_id}<br>"
            f"tracks={len(tracks)} gt={len(gt_items)} gt_source={gt_items[0].get('source', 'none') if gt_items else 'none'}"
        ),
        scene={
            "xaxis_title": "x lidar (m)",
            "yaxis_title": "y lidar (m)",
            "zaxis_title": "z lidar (m)",
            "aspectmode": "data",
            "bgcolor": "#080a0f",
        },
        paper_bgcolor="#080a0f",
        plot_bgcolor="#080a0f",
        font={"color": "#e6edf7"},
        margin={"l": 0, "r": 0, "b": 0, "t": 58},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(output_path),
        include_plotlyjs=(True if plotlyjs == "inline" else "cdn"),
        full_html=True,
        auto_open=False,
    )


def write_scene_index(scene_dir: Path, scene_id: str, rows: list[dict[str, Any]]) -> None:
    manifest_json = json.dumps(rows, ensure_ascii=True)
    options = "\n".join(
        f'<option value="{idx}">{idx:06d} {html.escape(row["frame_id"])} tracks={row["num_tracks"]}</option>'
        for idx, row in enumerate(rows)
    )
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(scene_id)} tracker visualization</title>
  <style>
    body {{ margin: 0; background: #080a0f; color: #e6edf7; font-family: Arial, sans-serif; }}
    header {{ position: sticky; top: 0; z-index: 10; display: flex; gap: 12px; align-items: center; padding: 10px 14px; background: #111722; border-bottom: 1px solid #263142; }}
    button, select {{ background: #192232; color: #e6edf7; border: 1px solid #3a465a; padding: 6px 9px; border-radius: 4px; }}
    input[type=range] {{ width: 360px; }}
    #meta {{ font-size: 13px; color: #bac7d8; }}
    main {{ display: flex; justify-content: center; padding: 14px; }}
    img {{ max-width: 98vw; max-height: calc(100vh - 80px); object-fit: contain; border: 1px solid #263142; background: #080a0f; }}
    a {{ color: #79c0ff; text-decoration: none; }}
  </style>
</head>
<body>
  <header>
    <strong>{html.escape(scene_id)}</strong>
    <button id="prev">Prev</button>
    <button id="next">Next</button>
    <button id="play">Play</button>
    <select id="frameSelect">{options}</select>
    <input id="slider" type="range" min="0" max="{max(len(rows) - 1, 0)}" value="0">
    <a id="html3d" href="#" target="_blank">3D HTML</a>
    <span id="meta"></span>
  </header>
  <main><img id="frameImage" src="" alt="tracker frame"></main>
  <script>
    const frames = {manifest_json};
    let index = 0;
    let timer = null;
    const img = document.getElementById('frameImage');
    const meta = document.getElementById('meta');
    const html3d = document.getElementById('html3d');
    const slider = document.getElementById('slider');
    const select = document.getElementById('frameSelect');
    function show(i) {{
      if (!frames.length) return;
      index = Math.max(0, Math.min(frames.length - 1, i));
      const f = frames[index];
      img.src = f.image;
      if (f.html_3d) {{
        html3d.style.display = 'inline';
        html3d.href = f.html_3d;
      }} else {{
        html3d.style.display = 'none';
      }}
      slider.value = index;
      select.value = index;
      meta.textContent = `frame ${{f.frame_index}} | token=${{f.frame_id}} | tracks=${{f.num_tracks}} | points=${{f.num_points}} | gt=${{f.num_gt}} | gt_source=${{f.gt_source}}`;
    }}
    document.getElementById('prev').onclick = () => show(index - 1);
    document.getElementById('next').onclick = () => show(index + 1);
    slider.oninput = () => show(parseInt(slider.value));
    select.onchange = () => show(parseInt(select.value));
    document.getElementById('play').onclick = () => {{
      if (timer) {{
        clearInterval(timer);
        timer = null;
        document.getElementById('play').textContent = 'Play';
      }} else {{
        timer = setInterval(() => show((index + 1) % frames.length), 180);
        document.getElementById('play').textContent = 'Pause';
      }}
    }};
    document.addEventListener('keydown', (ev) => {{
      if (ev.key === 'ArrowLeft') show(index - 1);
      if (ev.key === 'ArrowRight') show(index + 1);
    }});
    show(0);
  </script>
</body>
</html>
"""
    (scene_dir / "index.html").write_text(page)


def write_top_index(output_dir: Path, scene_rows: dict[str, list[dict[str, Any]]]) -> None:
    links = "\n".join(
        f'<li><a href="{html.escape(scene_id)}/index.html">{html.escape(scene_id)}</a> ({len(rows)} frames)</li>'
        for scene_id, rows in sorted(scene_rows.items())
    )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Tracker visualizations</title>
<style>body{{font-family:Arial,sans-serif;background:#080a0f;color:#e6edf7;padding:24px}}a{{color:#79c0ff}}</style>
</head><body><h1>Tracker visualizations</h1><ul>{links}</ul></body></html>
"""
    (output_dir / "index.html").write_text(page)


def main() -> None:
    args = parse_args()
    nuscenes_root = Path(args.nuscenes_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    maps = build_nuscenes_maps(nuscenes_root, args.version)
    if args.tracker_result_json:
        tracker_by_scene = load_tracker_from_json(Path(args.tracker_result_json).resolve(), maps)
    else:
        tracker_by_scene = load_tracker_from_pkl(Path(args.track_pkl).resolve(), maps)
    gt_by_scene = load_gt(Path(args.gt_pkl).resolve() if args.gt_pkl else None, maps)
    sample_gt_by_sample = maps["sample_gt"]

    frame_meta = maps["frame_meta"]
    scene_rows: dict[str, list[dict[str, Any]]] = {}
    total_frames = 0
    for scene_id in sorted(tracker_by_scene):
        if args.scene_id and scene_id != args.scene_id:
            continue
        scene_frames = tracker_by_scene[scene_id]
        ordered_frame_ids = sorted(scene_frames, key=lambda token: frame_meta[token]["timestamp"])
        if args.max_frames is not None:
            ordered_frame_ids = ordered_frame_ids[: args.max_frames]

        scene_dir = output_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for frame_index, frame_id in enumerate(ordered_frame_ids):
            meta = frame_meta[frame_id]
            gt_items, gt_source = select_gt_for_frame(
                scene_id=scene_id,
                frame_id=frame_id,
                meta=meta,
                exact_gt_by_scene=gt_by_scene,
                sample_gt_by_sample=sample_gt_by_sample,
                use_sample_gt=not args.no_sample_gt,
            )
            image_path = scene_dir / f"frame_{frame_index:06d}_{frame_id}_bev.png"
            row = draw_bev_frame(
                nuscenes_root=nuscenes_root,
                frame_id=frame_id,
                frame_index=frame_index,
                meta=meta,
                tracks=scene_frames[frame_id],
                gt_items=gt_items,
                output_path=image_path,
                image_size=args.image_size,
                point_size=args.point_size,
                line_width=args.line_width,
                draw_points=not args.no_points,
                draw_labels=not args.no_labels,
            )
            row["gt_source"] = gt_source
            if args.make_3d_html and (args.max_3d_frames is None or frame_index < args.max_3d_frames):
                html_path = scene_dir / f"frame_{frame_index:06d}_{frame_id}_3d.html"
                write_3d_frame_html(
                    nuscenes_root=nuscenes_root,
                    frame_id=frame_id,
                    frame_index=frame_index,
                    meta=meta,
                    tracks=scene_frames[frame_id],
                    gt_items=gt_items,
                    output_path=html_path,
                    max_points=args.max_3d_points,
                    draw_points=not args.no_points,
                    draw_labels=not args.no_labels,
                    plotlyjs=args.plotlyjs,
                )
                row["html_3d"] = html_path.name
            else:
                row["html_3d"] = ""
            rows.append(row)
            total_frames += 1
            if frame_index % 25 == 0:
                print(f"[viz] {scene_id} frame {frame_index + 1}/{len(ordered_frame_ids)} tracks={row['num_tracks']}")

        with (scene_dir / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "scene_id",
                    "frame_index",
                    "frame_id",
                    "timestamp",
                    "filename",
                    "num_points",
                    "num_tracks",
                    "num_gt",
                    "gt_source",
                    "image",
                    "html_3d",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        with (scene_dir / "manifest.json").open("w") as f:
            json.dump(rows, f, indent=2)
        write_scene_index(scene_dir, scene_id, rows)
        scene_rows[scene_id] = rows

    if not scene_rows:
        raise ValueError(f"no scenes rendered; requested scene_id={args.scene_id!r}")
    write_top_index(output_dir, scene_rows)

    print(f"[done] {output_dir / 'index.html'}")
    print(f"[summary] scenes={len(scene_rows)} frames={total_frames}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(2)
