# Multi-Camera Self-Calibration in Sports Motion Capture: Leveraging Human and Stick Poses

## Overview

The codes evaluate multi-camera self-calibration using a stick (known-length rigid object) as the only calibration target. All experiments run on synthetic data with ground-truth camera poses, enabling exact error measurement.

---

## Dataset (`data/`)

### Structure

Each file follows the naming convention:

```
trajectory_<type>_noise<level>_cam<N>.pkl
```

| Field | Values |
|---|---|
| `<type>` | `circle_golf`, `figure8_baseball`, `square_hockey`, `triangle_kendo` |
| `<level>` | 0.1, 0.2, 0.3, 0.4, 0.5 (pixel std of 2D observation noise) |
| `<N>` | 3, 4, 5, 6, 7, 8, 9, 10 (number of cameras) |

**Total: 160 files** (4 trajectory types × 5 noise levels × 8 camera counts)

### Trajectory Types

| Name | Motion Path | Stick | Grip | Stick Length |
|---|---|---|---|---|
| `circle_golf` | Circle | Golf club | Right hand | 1.00 m |
| `figure8_baseball` | Figure-8 | Baseball bat | Left hand | 0.86 m |
| `square_hockey` | Square | Hockey stick | Both hands | 1.60 m |
| `triangle_kendo` | Triangle | Kendo stick | Both hands | 1.20 m |

Cameras are distributed on a hemisphere around the scene. Each sequence has **50 frames**.

### Pickle File Contents

| Key | Type | Description |
|---|---|---|
| `num_cameras` | int | Number of valid cameras |
| `n_frames` | int | Number of frames (50) |
| `K` | list `[3,3]` | Shared camera intrinsic matrix |
| `Rs_gt` | list of `[3,3]` | Ground-truth rotation matrices |
| `ts_gt` | list of `[3,]` | Ground-truth translation vectors |
| `cameras` | list of dicts | Camera objects (`K`, `R`, `t`, `name`) |
| `stick_length` | float | Known stick length in metres |
| `stick_3d_gt` | `[F, 2, 3]` | Ground-truth 3D stick endpoints per frame |
| `pose_3d_gt` | `[F, 19, 3]` | Ground-truth 3D human joint positions per frame |
| `all_observed_points_2d` | list of `[F, 2, 2]` | Noisy 2D stick endpoint projections per camera |
| `all_pose_projections_2d` | list of `[F, 19, 2]` | 2D human pose projections per camera |
| `all_projection_masks` | list of `[F, 19]` | Visibility mask for human joints |
| `motion_type` | str | Trajectory type |
| `noise_std` | float | Observation noise standard deviation (pixels) |
| `two_handed` | bool | Whether stick is held with both hands |
| `hand_indices` | list | Joint indices of the gripping hand(s) |

---

## Visualization (`data_vis_result/`)

GIF animations of the 3D scene (human skeleton + stick + camera frustums) are saved here.

### Generate GIFs

```bash
# single file
python visualize_data.py --data data/trajectory_circle_golf_noise0.1_cam3.pkl

# all 160 files in data/
python visualize_data.py

# custom fps and output directory
python visualize_data.py --data data/trajectory_square_hockey_noise0.3_cam5.pkl \
                         --fps 12 --output_dir data_vis_result
```

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `--data` | — | Path to a single `.pkl` file (also accepts a quoted glob) |
| `--data_dir` | `data` | Directory scanned for `trajectory_*.pkl` when `--data` is omitted |
| `--output_dir` | `data_vis_result` | Directory for output GIFs |
| `--fps` | 10 | Frames per second |

Each GIF is named after the source pkl: `<stem>.gif`.

---

## Calibration Output (`outputs/`)

Estimated camera parameters are stored here as JSON files, one per pkl.


### JSON File Contents

```json
{
  "data_file": "/abs/path/to/trajectory_*.pkl",
  "approach": "stick_only",
  "Rs_final": [ [[...3x3...]], ... ],
  "ts_final": [ [tx, ty, tz], ... ],
  "stick_3d_final": [ [[x0,y0,z0],[x1,y1,z1]], ... ]
}
```

---

## Evaluation (`evaluate_calibration.py`)

Compares an estimated calibration JSON against the ground truth in a `.pkl` file. Applies **Procrustes alignment** to remove the reference-frame ambiguity before computing errors.

### Metrics

| Metric | Description |
|---|---|
| Rotation error (°) | Geodesic angle between estimated and GT rotation matrices |
| Translation error (m) | Euclidean distance between estimated and GT camera centres |
| Stick length error (m) | Difference between recovered and GT mean stick length |

### Usage

```bash
# single JSON, infer matching pkl from data/
python evaluate_calibration.py --calib outputs/trajectory_circle_golf_noise0.1_cam4_calib .json

# explicit pair
python evaluate_calibration.py \
    --calib outputs/trajectory_circle_golf_noise0.1_cam4_calib.json \
    --data  data/trajectory_circle_golf_noise0.1_cam4.pkl

# batch: all outputs/*_calib.json matched against data/*.pkl
python evaluate_calibration.py

# save aggregated results to JSON
python evaluate_calibration.py --output eval_results.json
```

**Arguments**

| Argument | Default | Description |
|---|---|---|
| `--calib` | — | Path to a single calibration JSON |
| `--data` | — | Path to the matching ground-truth pkl |
| `--calib_dir` | `outputs` | Directory scanned for `*_calib.json` in batch mode |
| `--data_dir` | `data` | Directory used to find matching `.pkl` files |
| `--output` | — | Save aggregated evaluation results to this JSON path |

### 4 Examples Case Output

```
============================================================
OVERALL SUMMARY  (4 files, 17 cameras total)
============================================================
  Rotation error    mean=0.016°  std=0.009°  median=0.017°  max=0.034°
  Translation error mean=0.0008m  std=0.0004m  median=0.0007m  max=0.0018m
  Stick length error mean=0.0000m  std=0.0000m  max=0.0000m
============================================================
```

---
