#!/usr/bin/env python3
"""
Visualize synthetic data from a .pkl file and save an animated GIF to data_vis_result/.

Usage:
    # single file
    python visualize_data.py --data data/trajectory_circle_golf_noise0.1_cam3.pkl

    # all trajectory_*.pkl in data/
    python visualize_data.py

    # custom fps / output dir
    python visualize_data.py --data data/trajectory_circle_golf_noise0.1_cam3.pkl --fps 12 --output_dir my_vis
"""

import os
import glob
import argparse
import pickle
import numpy as np

from multicam_3d_pose_helper import Camera, replay_synthetic_data


def _rebuild_cameras(cameras_raw):
    """Reconstruct Camera objects from the serialised dict format saved by save_synthetic_data."""
    cameras = []
    for cam in cameras_raw:
        if isinstance(cam, dict):
            cameras.append(Camera(
                np.array(cam["K"]),
                np.array(cam["R"]),
                np.array(cam["t"]),
                name=cam.get("name", "camera"),
            ))
        else:
            cameras.append(cam)
    return cameras


def visualize(data_path, output_dir="data_vis_result", fps=10):
    """Load a .pkl file and save a GIF to output_dir.

    Returns the output path on success, None on failure.
    """
    print(f"Visualizing: {data_path}")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    if data.get("num_cameras", 0) == 0:
        print(f"  Skipping — no valid cameras in {data_path}")
        return None

    # Cameras are serialised as dicts; reconstruct proper Camera objects
    data = dict(data)
    data["cameras"] = _rebuild_cameras(data["cameras"])

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(data_path))[0]
    output_path = os.path.join(output_dir, stem + ".gif")

    replay_synthetic_data(data, output_path=output_path, fps=fps)
    print(f"  Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate GIF visualizations from synthetic .pkl data files"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to a single .pkl file (or a glob pattern in quotes)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory to scan for trajectory_*.pkl files when --data is not given (default: data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_vis_result",
        help="Directory to save GIFs (default: data_vis_result)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the GIF (default: 10)",
    )
    args = parser.parse_args()

    if args.data:
        data_files = sorted(glob.glob(args.data)) or [args.data]
    else:
        data_files = sorted(glob.glob(os.path.join(args.data_dir, "trajectory_*.pkl")))

    if not data_files:
        print("No data files found.")
        return 1

    print(f"Found {len(data_files)} file(s) to visualize")

    success_count = 0
    for data_file in data_files:
        result = visualize(data_file, output_dir=args.output_dir, fps=args.fps)
        if result is not None:
            success_count += 1

    print(f"\nDone: {success_count}/{len(data_files)} GIFs saved to '{args.output_dir}/'")
    return 0 if success_count == len(data_files) else 1


if __name__ == "__main__":
    exit(main())
