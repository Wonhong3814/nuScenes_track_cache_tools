# nuScenes Sweep Detector-Tracker Pipeline

Reproducible nuScenes sweep-anchor LargeKernel3D -> MCTrack pipeline.

This repository tracks the wrapper/converter code and includes the upstream detector/tracker repositories as git submodules:

- `third_party/LargeKernel3D`
- `third_party/FocalsConv`
- `third_party/MCTrack`

It does not store nuScenes data, generated detector/tracker caches, or model checkpoint binaries in git.

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

## Clone

```bash
git clone --recursive <THIS_REPO_URL>
cd nuScenes_track_cache_tools
```

If the repo was cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

Install Python dependencies in the environment that will run CenterPoint/MCTrack:

```bash
python3 -m pip install -r requirements.txt
```

Download the LargeKernel3D tiny checkpoint:

```bash
python3 scripts/download_checkpoints.py \
  --checkpoint_dir checkpoints/largekernel3d
```

Prepare third-party compatibility patches and MCTrack config:

```bash
python3 scripts/bootstrap_third_party.py \
  --nuscenes_root /data1/nuScenes \
  --output_root /data1/wonhong/nuScenes_track_cache \
  --split sweep_trainval \
  --frame_rate 10
```

## One-Command Full Run

This runs sweep info creation, GPU detector inference, MCTrack input conversion, MCTrack tracking, and final cache conversion:

```bash
python3 scripts/run_full_sweep_pipeline.py \
  --nuscenes_root /data1/nuScenes \
  --output_root /data1/wonhong/nuScenes_track_cache \
  --checkpoint checkpoints/largekernel3d/largekernel3d_tiny_val.pth \
  --gpu 9 \
  --mctrack_processes 8
```

## Step-By-Step Commands

Create sweep-anchor CenterPoint infos:

```bash
cd nuScenes_track_cache_tools

python3 create_nuscenes_sweep_infos.py \
  --nuscenes_root /data1/nuScenes \
  --version v1.0-trainval \
  --output /data1/wonhong/nuScenes_track_cache/sweep_infos/infos_sweep_trainval_1sweep.pkl
```

Run LargeKernel3D detector on GPU 9:

```bash
CUDA_VISIBLE_DEVICES=9 python3 run_largekernel3d_sweep_infer.py \
  --centerpoint_root third_party/FocalsConv/CenterPoint \
  --config third_party/LargeKernel3D/object-detection/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_largekernel3d_tiny.py \
  --checkpoint checkpoints/largekernel3d/largekernel3d_tiny_val.pth \
  --nuscenes_root /data1/nuScenes \
  --version v1.0-trainval \
  --sweep_info_pkl /data1/wonhong/nuScenes_track_cache/sweep_infos/infos_sweep_trainval_1sweep.pkl \
  --work_dir /data1/wonhong/nuScenes_track_cache/largekernel3d_sweep_trainval_1sweep
```

Convert detector JSON into MCTrack base-version input:

```bash
python3 convert_sweep_detections_to_mctrack_base.py \
  --nuscenes_root /data1/nuScenes \
  --version v1.0-trainval \
  --det_json /data1/wonhong/nuScenes_track_cache/largekernel3d_sweep_trainval_1sweep/sweep_detections.json \
  --save_path third_party/MCTrack/data/base_version/nuscenes \
  --detector largekernel \
  --split sweep_trainval
```

Run MCTrack. This is CPU tracking; use `-p` for scene-level multiprocessing:

```bash
cd third_party/MCTrack

python3 main.py --dataset nuscenes -p 8

cd ../..
```

Convert MCTrack results to the requested flat caches:

```bash
python3 convert_official_results.py \
  --nuscenes_root /data1/nuScenes \
  --version v1.0-trainval \
  --mctrack_result_json /data1/wonhong/nuScenes_track_cache/mctrack_results/nuscenes/YYYYMMDD_HHMMSS/results.json \
  --output_dir /data1/wonhong/nuScenes_track_cache
```

Run only the first 100 frames of `scene-0240`:

```bash
python3 run_mctrack_single_scene.py \
  --mctrack_root third_party/MCTrack \
  --base_json third_party/MCTrack/data/base_version/nuscenes/largekernel/sweep_trainval.json \
  --scene_id scene-0240 \
  --output_dir /data1/wonhong/nuScenes_track_cache/mctrack_results_scene0240_100frames \
  --max_frames 100
```

## Important MCTrack Config Values

`third_party/MCTrack/config/nuscenes.yaml` should use:

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
