# Sweep-Anchor LargeKernel3D to MCTrack Commands

This path uses every `/data1/nuScenes/sweeps/LIDAR_TOP/*.pcd.bin` as an anchor
frame. The output key is the current anchor frame's LIDAR_TOP
`sample_data_token`.

Detector input uses one anchor sweep only. This is faster than the official
10-sweep input, while keeping the output key as the anchor sweep
`sample_data_token`.

```bash
cd /data1/wonhong

python3 /data1/wonhong/nuScenes_track_cache_tools/create_nuscenes_sweep_infos.py \
  --nuscenes_root /data1/nuScenes \
  --version v1.0-trainval \
  --output /data1/wonhong/nuScenes_track_cache/sweep_infos/infos_sweep_trainval_1sweep.pkl
```

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

```bash
python3 /data1/wonhong/nuScenes_track_cache_tools/convert_sweep_detections_to_mctrack_base.py \
  --nuscenes_root /data1/nuScenes \
  --version v1.0-trainval \
  --det_json /data1/wonhong/nuScenes_track_cache/largekernel3d_sweep_trainval_1sweep/sweep_detections.json \
  --save_path /data1/wonhong/third_party/MCTrack/data/base_version/nuscenes \
  --detector largekernel \
  --split sweep_trainval
```

`/data1/wonhong/third_party/MCTrack/config/nuscenes.yaml` is set to:

```yaml
SPLIT: "sweep_trainval"
DETECTOR: largekernel
DATASET_ROOT: "/data1/nuScenes"
DETECTIONS_ROOT: "data/base_version/nuscenes/"
SAVE_PATH: "/data1/wonhong/nuScenes_track_cache/mctrack_results/"
FRAME_RATE: 10
```

```bash
cd /data1/wonhong/third_party/MCTrack
CUDA_VISIBLE_DEVICES=9 python3 main.py --dataset nuscenes -p 1
```

After MCTrack prints the timestamped result directory, convert to the requested
flat pkl caches:

```bash
python3 /data1/wonhong/nuScenes_track_cache_tools/convert_official_results.py \
  --nuscenes_root /data1/nuScenes \
  --version v1.0-trainval \
  --mctrack_result_json /data1/wonhong/nuScenes_track_cache/mctrack_results/nuscenes/YYYYMMDD_HHMMSS/results.json \
  --output_dir /data1/wonhong/nuScenes_track_cache
```
