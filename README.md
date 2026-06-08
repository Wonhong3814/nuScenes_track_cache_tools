# nuScenes Sweep Detector-Tracker Pipeline

Standalone wrappers for the nuScenes sweep-anchor LargeKernel3D -> MCTrack pipeline.

This repo does not vendor or modify LargeKernel3D, FocalsConv/CenterPoint, MCTrack, checkpoints, or nuScenes data. It only keeps the wrapper scripts, converters, and exact run commands needed to reproduce the detector-tracker cache export.

## External Paths Used Locally

- nuScenes root: `/data1/nuScenes`
- LargeKernel3D repo: `/data1/wonhong/third_party/LargeKernel3D`
- FocalsConv/CenterPoint repo: `/data1/wonhong/third_party/FocalsConv/CenterPoint`
- MCTrack repo: `/data1/wonhong/third_party/MCTrack`
- LargeKernel3D checkpoint: `/data1/wonhong/checkpoints/largekernel3d/largekernel3d_tiny_val.pth`
- Output root: `/data1/wonhong/nuScenes_track_cache`

## Current Pipeline Semantics

- Anchor frames: every `/data1/nuScenes/sweeps/LIDAR_TOP/*.pcd.bin`
- Anchor key: sweep `sample_data_token`
- Detector input setting in the current script: one current anchor sweep (`DEFAULT_NSWEEPS = 1`)
- Detector output: `sweep_prediction.pkl`, `sweep_detections.json`
- MCTrack input: converted LargeKernel3D detection output, not raw point clouds
- MCTrack frame id inside its base json: sequential integer
- MCTrack `cur_sample_token`: sweep `sample_data_token`
- Final exported tracker frame id: sweep `sample_data_token`
- GT cache frame id: keyframe LIDAR_TOP `sample_data_token`
- MCTrack tracking core is CPU/numpy/lap/cv2 based. `CUDA_VISIBLE_DEVICES` affects detector inference, not the original MCTrack association logic.

## Full Commands

Create sweep-anchor CenterPoint infos:

```bash
cd /data1/wonhong

python3 /data1/wonhong/nuScenes_track_cache_tools/create_nuscenes_sweep_infos.py \
  --nuscenes_root /data1/nuScenes \
  --version v1.0-trainval \
  --output /data1/wonhong/nuScenes_track_cache/sweep_infos/infos_sweep_trainval_1sweep.pkl
```

Run LargeKernel3D detector on GPU 9:

```bash
CUDA_VISIBLE_DEVICES=9 python3 /data1/wonhong/nuScenes_track_cache_tools/run_largekernel3d_sweep_infer.py \
  --centerpoint_root /data1/wonhong/third_party/FocalsConv/CenterPoint \
  --config /data1/wonhong/third_party/LargeKernel3D/object-detection/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_largekernel3d_tiny.py \
  --checkpoint /data1/wonhong/checkpoints/largekernel3d/largekernel3d_tiny_val.pth \
  --nuscenes_root /data1/nuScenes \
  --version v1.0-trainval \
  --sweep_info_pkl /data1/wonhong/nuScenes_track_cache/sweep_infos/infos_sweep_trainval_1sweep.pkl \
  --work_dir /data1/wonhong/nuScenes_track_cache/largekernel3d_sweep_trainval_1sweep
```

Convert detector JSON into MCTrack base-version input:

```bash
python3 /data1/wonhong/nuScenes_track_cache_tools/convert_sweep_detections_to_mctrack_base.py \
  --nuscenes_root /data1/nuScenes \
  --version v1.0-trainval \
  --det_json /data1/wonhong/nuScenes_track_cache/largekernel3d_sweep_trainval_1sweep/sweep_detections.json \
  --save_path /data1/wonhong/third_party/MCTrack/data/base_version/nuscenes \
  --detector largekernel \
  --split sweep_trainval
```

Run MCTrack. This is CPU tracking; use `-p` for scene-level multiprocessing:

```bash
cd /data1/wonhong/third_party/MCTrack

python3 main.py --dataset nuscenes -p 8
```

Convert MCTrack results to the requested flat caches:

```bash
python3 /data1/wonhong/nuScenes_track_cache_tools/convert_official_results.py \
  --nuscenes_root /data1/nuScenes \
  --version v1.0-trainval \
  --mctrack_result_json /data1/wonhong/nuScenes_track_cache/mctrack_results/nuscenes/YYYYMMDD_HHMMSS/results.json \
  --output_dir /data1/wonhong/nuScenes_track_cache
```

Run only the first 100 frames of `scene-0240`:

```bash
python3 /data1/wonhong/nuScenes_track_cache_tools/run_mctrack_single_scene.py \
  --mctrack_root /data1/wonhong/third_party/MCTrack \
  --base_json /data1/wonhong/third_party/MCTrack/data/base_version/nuscenes/largekernel/sweep_trainval.json \
  --scene_id scene-0240 \
  --output_dir /data1/wonhong/nuScenes_track_cache/mctrack_results_scene0240_100frames \
  --max_frames 100
```

## Important MCTrack Config Values

`/data1/wonhong/third_party/MCTrack/config/nuscenes.yaml` should use:

```yaml
SPLIT: "sweep_trainval"
DETECTOR: largekernel
DATASET_ROOT: "/data1/nuScenes"
DETECTIONS_ROOT: "data/base_version/nuscenes/"
SAVE_PATH: "/data1/wonhong/nuScenes_track_cache/mctrack_results/"
FRAME_RATE: 10
```

`CACHE_BBOX_LENGTH: 30` is a per-category trajectory history/cache length. It is not the number of frames passed into `main.py` at once.

## Final Cache Schemas

Tracker cache:

```python
track_data[scene_id][frame_id][track_id] = {
    "bbox": [x, y, z, l, w, h, yaw],
    "class_name": str,
    "score": float,
}
```

GT cache:

```python
gt_data[scene_id][frame_id][instance_token] = {
    "bbox": [x, y, z, l, w, h, yaw],
    "class_name": str,
}
```

