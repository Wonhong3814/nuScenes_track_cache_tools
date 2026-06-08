#!/usr/bin/env python3
"""Run LargeKernel3D/CenterPoint inference on custom sweep infos without nuScenes eval."""

from __future__ import annotations

import argparse
import collections
import collections.abc
import json
import pickle
import sys
import types
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centerpoint_root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--nuscenes_root", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--sweep_info_pkl", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def add_centerpoint_to_path(centerpoint_root: Path) -> None:
    if not centerpoint_root.exists():
        raise FileNotFoundError(centerpoint_root)
    sys.path.insert(0, str(centerpoint_root))


def patch_python310_collections() -> None:
    for name in ("Iterable", "Mapping", "MutableMapping", "Sequence"):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))


def patch_torchvision_models_utils() -> None:
    if "torchvision.models.utils" in sys.modules:
        return
    try:
        import torch.hub
        import torchvision.models
    except Exception:
        return
    module = types.ModuleType("torchvision.models.utils")
    module.load_state_dict_from_url = torch.hub.load_state_dict_from_url
    sys.modules["torchvision.models.utils"] = module


def to_cpu_output(output: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in output.items():
        if key == "metadata":
            out[key] = value
        elif hasattr(value, "detach"):
            out[key] = value.detach().cpu()
        else:
            out[key] = value
    return out


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def infer_nsweeps_from_info(info_path: Path) -> int:
    with info_path.open("rb") as f:
        infos = pickle.load(f)
    if not infos:
        raise RuntimeError(f"empty sweep info pkl: {info_path}")
    return len(infos[0].get("sweeps", [])) + 1


def mapped_class_names(dataset: Any) -> list[str]:
    mapping = getattr(dataset, "_name_mapping", {})
    return [mapping.get(name, name) for name in dataset._class_names]


def write_sweep_detection_json(
    *,
    predictions: dict[str, dict[str, Any]],
    dataset: Any,
    nuscenes_root: Path,
    version: str,
    output_path: Path,
) -> None:
    from nuscenes.nuscenes import NuScenes
    from det3d.datasets.nuscenes.nusc_common import _lidar_nusc_box_to_global, _second_det_to_nusc_box, cls_attr_dist

    nusc = NuScenes(version=version, dataroot=str(nuscenes_root), verbose=True)
    class_names = mapped_class_names(dataset)

    result = {
        "results": {},
        "meta": {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        },
    }

    for token, det in predictions.items():
        annos = []
        boxes = _second_det_to_nusc_box(det)
        boxes = _lidar_nusc_box_to_global(nusc, boxes, token)
        for box in boxes:
            label = int(box.label)
            if label < 0 or label >= len(class_names):
                continue
            name = class_names[label]
            attr = max(cls_attr_dist[name].items(), key=lambda item: item[1])[0] if name in cls_attr_dist else ""
            annos.append(
                {
                    "sample_token": token,
                    "translation": [float(v) for v in box.center.tolist()],
                    "size": [float(v) for v in box.wlh.tolist()],
                    "rotation": [float(v) for v in box.orientation.elements.tolist()],
                    "velocity": [float(v) for v in box.velocity[:2].tolist()],
                    "detection_name": str(name),
                    "detection_score": float(box.score),
                    "attribute_name": attr,
                }
            )
        result["results"][token] = annos

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result, f)


def load_checkpoint_with_spconv_layout_fix(model: Any, checkpoint_path: Path) -> None:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()
    converted: dict[str, Any] = {}
    converted_count = 0

    for key, value in state_dict.items():
        target = model_state.get(key)
        if (
            target is not None
            and hasattr(value, "shape")
            and len(value.shape) == 5
            and len(target.shape) == 5
            and tuple(value.shape) != tuple(target.shape)
        ):
            expected_legacy = (
                int(target.shape[1]),
                int(target.shape[2]),
                int(target.shape[3]),
                int(target.shape[0]),
                int(target.shape[4]),
            )
            if tuple(value.shape) == expected_legacy:
                value = value.permute(3, 0, 1, 2, 4).contiguous()
                converted_count += 1
        converted[key] = value

    incompatible = model.load_state_dict(converted, strict=False)
    remaining_mismatch = []
    loaded_keys = set(converted.keys())
    for key, target in model_state.items():
        value = converted.get(key)
        if value is not None and hasattr(value, "shape") and tuple(value.shape) != tuple(target.shape):
            remaining_mismatch.append((key, tuple(target.shape), tuple(value.shape)))

    if remaining_mismatch:
        preview = remaining_mismatch[:10]
        raise RuntimeError(f"checkpoint still has mismatched tensor shapes after spconv layout fix: {preview}")

    unexpected = [key for key in incompatible.unexpected_keys if key in loaded_keys]
    print(f"[checkpoint] loaded {checkpoint_path}")
    print(f"[checkpoint] converted_spconv5d_weights={converted_count}")
    print(f"[checkpoint] missing_keys={len(incompatible.missing_keys)} unexpected_keys={len(unexpected)}")


def main() -> None:
    args = parse_args()
    centerpoint_root = Path(args.centerpoint_root).resolve()
    patch_python310_collections()
    patch_torchvision_models_utils()
    add_centerpoint_to_path(centerpoint_root)

    from det3d import torchie
    from det3d.datasets import build_dataloader, build_dataset
    from det3d.models import build_detector
    from det3d.torchie import Config
    from det3d.torchie.apis import batch_processor

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    nuscenes_root = Path(args.nuscenes_root).resolve()
    sweep_info_pkl = Path(args.sweep_info_pkl).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(str(config_path))
    cfg.local_rank = 0
    cfg.gpus = 1
    cfg.work_dir = str(work_dir)
    cfg.data.val.root_path = str(nuscenes_root)
    cfg.data.val.info_path = str(sweep_info_pkl)
    cfg.data.val.ann_file = str(sweep_info_pkl)
    cfg.data.val.test_mode = True
    cfg.data.val.nsweeps = infer_nsweeps_from_info(sweep_info_pkl)
    if args.batch_size is not None:
        cfg.data.samples_per_gpu = args.batch_size
    if args.workers is not None:
        cfg.data.workers_per_gpu = args.workers

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    load_checkpoint_with_spconv_layout_fix(model, checkpoint_path)
    model = model.cuda()
    model.eval()

    predictions: dict[str, dict[str, Any]] = {}
    prog_bar = torchie.ProgressBar(len(data_loader.dataset))
    with torch.no_grad():
        for data_batch in data_loader:
            outputs = batch_processor(model, data_batch, train_mode=False, local_rank=0)
            for output in outputs:
                token = output["metadata"]["token"]
                predictions[token] = to_cpu_output(output)
                prog_bar.update()

    prediction_pkl = work_dir / "sweep_prediction.pkl"
    detection_json = work_dir / "sweep_detections.json"
    save_pickle(predictions, prediction_pkl)
    write_sweep_detection_json(
        predictions=predictions,
        dataset=dataset,
        nuscenes_root=nuscenes_root,
        version=args.version,
        output_path=detection_json,
    )

    print(f"[done] {prediction_pkl}")
    print(f"[done] {detection_json}")
    print(f"[summary] frames={len(predictions)}")


if __name__ == "__main__":
    main()
