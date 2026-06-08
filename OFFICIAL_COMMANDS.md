# Official LargeKernel3D + MCTrack Commands

This file records the official commands/paths found in the cloned repos. It does
not define a new detector or tracker pipeline.

## Repos

- LargeKernel3D: `/data1/wonhong/third_party/LargeKernel3D`
- FocalsConv CenterPoint backend used by LargeKernel3D object detection: `/data1/wonhong/third_party/FocalsConv/CenterPoint`
- MCTrack: `/data1/wonhong/third_party/MCTrack`

LargeKernel3D object detection is not a standalone batch inference repo. Its
README says the object-detection part follows FocalsConv/CenterPoint and that
the configs/backbone files should be used there:

- `/data1/wonhong/third_party/LargeKernel3D/object-detection/README.md`
- `/data1/wonhong/third_party/FocalsConv/README.md`

## Important Compatibility Note

The official CenterPoint/LargeKernel3D nuScenes evaluation command is key-sample
based. It builds and consumes `infos_*_10sweeps...pkl`, where each item is a
nuScenes key sample with multiple sweeps aggregated as context.

The official MCTrack nuScenes converter is also key-sample based:

- `/data1/wonhong/third_party/MCTrack/preprocess/convert_nuscenes.py`

It walks `scene.first_sample_token -> sample.next` and writes integer
`frame_id = 0, 1, ...` plus `cur_sample_token = sample_token`.

Therefore, the official commands as-is do not make every
`/data1/nuScenes/sweeps/LIDAR_TOP/*.pcd.bin` an individual tracker frame. Making
every sweep an individual `frame_id = LIDAR_TOP sample_data_token` would require
a custom/modified data-info or converter path, which is outside the requested
"official command 그대로" constraint.

## LargeKernel3D Detection, Official CenterPoint Command

Downloaded official LargeKernel3D checkpoints:

```text
/data1/wonhong/checkpoints/largekernel3d/largekernel3d_tiny_val.pth
/data1/wonhong/checkpoints/largekernel3d/largekernel3d_multimodal_test.pth
/data1/wonhong/checkpoints/largekernel3d/largekernel3d_f_multimodal_test.pth
```

Downloaded auxiliary FocalsConv/CenterPoint multimodal pretrain:

```text
/data1/wonhong/third_party/FocalsConv/CenterPoint/checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth
```

Prepare the official CenterPoint nuScenes data layout:

```bash
cd /data1/wonhong/third_party/FocalsConv/CenterPoint
mkdir -p data
ln -sfn /data1/nuScenes data/nuScenes
```

Create official nuScenes info files:

```bash
cd /data1/wonhong/third_party/FocalsConv/CenterPoint
CUDA_VISIBLE_DEVICES=0 python3 tools/create_data.py nuscenes_data_prep \
  --root_path=/data1/nuScenes \
  --version="v1.0-trainval" \
  --nsweeps=10
```

Run LargeKernel3D config through official CenterPoint test command.

```bash
cd /data1/wonhong/third_party/FocalsConv/CenterPoint
CONFIG=nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_largekernel3d_tiny
CUDA_VISIBLE_DEVICES=0 python3 tools/dist_test.py \
  configs/nusc/voxelnet/${CONFIG}.py \
  --work_dir /data1/wonhong/nuScenes_track_cache/largekernel3d_val \
  --checkpoint /data1/wonhong/checkpoints/largekernel3d/largekernel3d_tiny_val.pth
```

Expected detector JSON from official CenterPoint evaluation:

```text
/data1/wonhong/nuScenes_track_cache/largekernel3d_val/infos_val_10sweeps_withvelo_filter_True.json
```

That JSON is key-sample detection output, not sweep-frame output.

## MCTrack Official Convert + Track Command

MCTrack official converter expects:

```text
data/nuScenes/detectors/{detector}/{split}.json
```

For the LargeKernel3D val detector JSON above:

```bash
cd /data1/wonhong/third_party/MCTrack
mkdir -p data/nuScenes/detectors/largekernel
cp /data1/wonhong/nuScenes_track_cache/largekernel3d_val/infos_val_10sweeps_withvelo_filter_True.json \
  data/nuScenes/detectors/largekernel/val.json
```

Run MCTrack official BaseVersion conversion:

```bash
cd /data1/wonhong/third_party/MCTrack
python3 preprocess/convert_nuscenes.py \
  --raw_data_path /data1/nuScenes \
  --dets_path data/nuScenes/detectors \
  --save_path data/base_version/nuscenes \
  --detector largekernel \
  --split val
```

Set MCTrack config values in `config/nuscenes.yaml` before tracking:

```yaml
SPLIT: "val"
DETECTOR: largekernel
DATASET_ROOT: "/data1/nuScenes"
DETECTIONS_ROOT: "data/base_version/nuscenes/"
SAVE_PATH: "/data1/wonhong/nuScenes_track_cache/mctrack_results/"
FRAME_RATE: 10
```

`FRAME_RATE: 10` corresponds to 0.1 sec. `CACHE_BBOX_LENGTH` is read in:

```text
/data1/wonhong/third_party/MCTrack/tracker/trajectory.py
```

The source assigns it to `self._cache_bbox_len` per category from:

```yaml
THRESHOLD:
  TRAJECTORY_THRE:
    CACHE_BBOX_LENGTH: ...
```

Run official MCTrack:

```bash
cd /data1/wonhong/third_party/MCTrack
CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset nuscenes -p 1
```

Expected MCTrack outputs under timestamped result dir:

```text
/data1/wonhong/nuScenes_track_cache/mctrack_results/nuscenes/YYYYMMDD_HHMMSS/results.json
/data1/wonhong/nuScenes_track_cache/mctrack_results/nuscenes/YYYYMMDD_HHMMSS/results_for_motion.json
```

Again, with official converter above these are key-sample results, not
sweep-frame sample_data-token results.

## MCTrack Mini-Sequence/Cache

In the cloned official MCTrack repo, no separate official command was found that
exports `bbox_seq`, `valid_mask`, and `score_seq` mini-sequence caches. The
README exposes:

- `python3 main.py --dataset nuscenes -p 1` for tracking.
- `python3 preprocess/motion_dataset/convert_nuscenes_result_to_pkl.py` for
  motion metric pkl conversion.

`CACHE_BBOX_LENGTH` is not a sliding-window dataset length in a mini-sequence
export script. It is consumed inside `tracker/trajectory.py` as a per-category
trajectory history cache length.

## Minimal Converter After Official Results

Only run this if the MCTrack result JSON is keyed by sweep LIDAR_TOP
`sample_data_token`. If it is the official key-sample output, the converter will
fail intentionally because that does not satisfy the requested schema.

```bash
python3 /data1/wonhong/nuScenes_track_cache_tools/convert_official_results.py \
  --nuscenes_root /data1/nuScenes \
  --version v1.0-trainval \
  --mctrack_result_json /data1/wonhong/nuScenes_track_cache/mctrack_results/nuscenes/YYYYMMDD_HHMMSS/results.json \
  --output_dir /data1/wonhong/nuScenes_track_cache
```

Outputs:

```text
/data1/wonhong/nuScenes_track_cache/mctrack_largekernel3d_sweep_tracks.pkl
/data1/wonhong/nuScenes_track_cache/nuscenes_keyframe_gt.pkl
```
