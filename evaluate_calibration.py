#!/usr/bin/env python3
"""
Evaluate estimated camera parameters (JSON) against ground truth from a .pkl data file.

Usage:
    python evaluate_calibration.py --calib data/trajectory_circle_golf_noise0.1_cam4_calib.json \
                                   --data  data/trajectory_circle_golf_noise0.1_cam4.pkl

Or evaluate all JSON files that have a matching .pkl:
    python evaluate_calibration.py --data_dir data
"""

import os
import json
import argparse
import glob
import pickle
import numpy as np


# ---------------------------------------------------------------------------
# Helpers (mirrors multi_cam_self_calib.py)
# ---------------------------------------------------------------------------

def procrustes_alignment(X, Y):
    """Find R, t such that R @ X[i] + t ≈ Y[i] (least-squares)."""
    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)
    H = X_c.T @ Y_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Y.mean(axis=0) - R @ X.mean(axis=0)
    return R, t


def rotation_error_deg(R_gt, R_est):
    R_rel = np.array(R_gt).T @ np.array(R_est)
    angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0))
    return float(np.degrees(angle))


def translation_error(c_gt, c_est):
    return float(np.linalg.norm(np.array(c_gt) - np.array(c_est)))


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(calib_path, data_path, verbose=True):
    """Compare a calibration JSON to ground truth in a .pkl file.

    Returns a dict with per-camera and summary metrics, or None on failure.
    """
    # --- load calibration result ---
    with open(calib_path) as f:
        calib = json.load(f)

    Rs_est = [np.array(R) for R in calib["Rs_final"]]
    ts_est = [np.array(t) for t in calib["ts_final"]]
    stick_3d_est = (
        np.array(calib["stick_3d_final"])
        if calib.get("stick_3d_final") is not None
        else None
    )

    # --- load ground truth ---
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    Rs_gt_raw = data.get("Rs_gt")
    ts_gt_raw = data.get("ts_gt")
    if Rs_gt_raw is None or ts_gt_raw is None:
        print(f"  No ground truth found in {data_path}")
        return None

    Rs_gt = [np.array(R) for R in Rs_gt_raw]
    ts_gt = [np.array(t) for t in ts_gt_raw]

    num_cameras = min(len(Rs_est), len(Rs_gt))

    # --- Procrustes alignment of camera centres ---
    est_centers = np.array([-R.T @ t for R, t in zip(Rs_est, ts_est)])
    gt_centers  = np.array([-R.T @ t for R, t in zip(Rs_gt,  ts_gt)])

    R_proc, t_proc = procrustes_alignment(est_centers, gt_centers)

    # Align estimated poses into GT frame
    Rs_aligned, ts_aligned, centers_aligned = [], [], []
    for R, t in zip(Rs_est, ts_est):
        C = -R.T @ t
        C_a = R_proc @ C + t_proc
        R_a = R @ R_proc.T
        t_a = -R_a @ C_a
        Rs_aligned.append(R_a)
        ts_aligned.append(t_a)
        centers_aligned.append(C_a)

    # --- per-camera errors ---
    per_camera = []
    for i in range(num_cameras):
        rot_err  = rotation_error_deg(Rs_gt[i], Rs_aligned[i])
        trans_err = translation_error(gt_centers[i], centers_aligned[i])
        per_camera.append({"camera": i + 1, "rotation_error_deg": rot_err, "translation_error_m": trans_err})
        if verbose:
            print(f"  [Camera {i+1}]  rot_err={rot_err:.3f}°  trans_err={trans_err:.4f}m")

    rot_errors   = [c["rotation_error_deg"]  for c in per_camera]
    trans_errors = [c["translation_error_m"] for c in per_camera]

    summary = {
        "mean_rotation_error_deg":     float(np.mean(rot_errors)),
        "std_rotation_error_deg":      float(np.std(rot_errors)),
        "mean_translation_error_m":    float(np.mean(trans_errors)),
        "std_translation_error_m":     float(np.std(trans_errors)),
    }

    # --- stick length error (if available) ---
    stick_3d_gt_raw = data.get("stick_3d_gt")
    if stick_3d_est is not None and stick_3d_gt_raw is not None:
        stick_3d_gt = np.array(stick_3d_gt_raw)

        # Align estimated stick 3D to GT frame
        stick_3d_aligned = np.zeros_like(stick_3d_est)
        for fi in range(stick_3d_est.shape[0]):
            for pi in range(stick_3d_est.shape[1]):
                stick_3d_aligned[fi, pi] = R_proc @ stick_3d_est[fi, pi] + t_proc

        est_len = float(np.mean(np.linalg.norm(
            stick_3d_aligned[:, 0] - stick_3d_aligned[:, 1], axis=1)))
        gt_len  = float(np.mean(np.linalg.norm(
            stick_3d_gt[:, 0] - stick_3d_gt[:, 1], axis=1)))
        summary["stick_length_error_m"] = abs(est_len - gt_len)
        summary["stick_length_estimated_m"] = est_len
        summary["stick_length_gt_m"] = gt_len

        if verbose:
            print(f"  Stick length: est={est_len:.4f}m  gt={gt_len:.4f}m  "
                  f"err={summary['stick_length_error_m']:.4f}m")

    if verbose:
        print(f"  Mean rot error:   {summary['mean_rotation_error_deg']:.3f}° "
              f"(±{summary['std_rotation_error_deg']:.3f}°)")
        print(f"  Mean trans error: {summary['mean_translation_error_m']:.4f}m "
              f"(±{summary['std_translation_error_m']:.4f}m)")

    return {"per_camera": per_camera, "summary": summary}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate calibration JSON against ground truth .pkl"
    )
    parser.add_argument("--calib", type=str,
                        help="Path to a single calibration JSON file")
    parser.add_argument("--data", type=str,
                        help="Path to the matching .pkl ground-truth file")
    parser.add_argument("--calib_dir", type=str, default="outputs",
                        help="Directory to scan for *_calib.json files "
                             "when --calib is not given (default: outputs)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory to look for matching .pkl files "
                             "when --data is not given (default: data)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save aggregated evaluation results to this JSON path")
    args = parser.parse_args()

    pairs = []

    if args.calib and args.data:
        pairs.append((args.calib, args.data))
    elif args.calib:
        # Infer matching pkl: look in data_dir for the same stem
        stem = os.path.basename(args.calib).replace("_calib.json", "")
        data_path = os.path.join(args.data_dir, stem + ".pkl")
        if not os.path.exists(data_path):
            print(f"Could not find matching data file at {data_path}. "
                  "Please provide --data explicitly.")
            return 1
        pairs.append((args.calib, data_path))
    else:
        # Auto-discover all *_calib.json in calib_dir, match .pkl in data_dir
        for calib_path in sorted(glob.glob(os.path.join(args.calib_dir, "*_calib.json"))):
            stem = os.path.basename(calib_path).replace("_calib.json", "")
            data_path = os.path.join(args.data_dir, stem + ".pkl")
            if os.path.exists(data_path):
                pairs.append((calib_path, data_path))
            else:
                print(f"Skipping {calib_path} — no matching .pkl found in {args.data_dir}")

    if not pairs:
        print("No calibration JSON / data pairs found.")
        return 1

    all_results = []
    for calib_path, data_path in pairs:
        print(f"\nEvaluating: {os.path.basename(calib_path)}")
        metrics = evaluate(calib_path, data_path, verbose=True)
        if metrics is not None:
            all_results.append({
                "calib_file": calib_path,
                "data_file": data_path,
                **metrics,
            })

    if args.output and all_results:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved evaluation results to: {args.output}")

    # --- overall summary across all evaluated files ---
    if len(all_results) > 1:
        all_rot   = [c["rotation_error_deg"]  for r in all_results for c in r["per_camera"]]
        all_trans = [c["translation_error_m"] for r in all_results for c in r["per_camera"]]
        all_stick = [r["summary"]["stick_length_error_m"]
                     for r in all_results if "stick_length_error_m" in r["summary"]]

        print(f"\n{'='*60}")
        print(f"OVERALL SUMMARY  ({len(all_results)} files, {len(all_rot)} cameras total)")
        print(f"{'='*60}")
        print(f"  Rotation error    mean={np.mean(all_rot):.3f}°  "
              f"std={np.std(all_rot):.3f}°  "
              f"median={np.median(all_rot):.3f}°  "
              f"max={np.max(all_rot):.3f}°")
        print(f"  Translation error mean={np.mean(all_trans):.4f}m  "
              f"std={np.std(all_trans):.4f}m  "
              f"median={np.median(all_trans):.4f}m  "
              f"max={np.max(all_trans):.4f}m")
        if all_stick:
            print(f"  Stick length error mean={np.mean(all_stick):.4f}m  "
                  f"std={np.std(all_stick):.4f}m  "
                  f"max={np.max(all_stick):.4f}m")
        print(f"{'='*60}")

    success = len(all_results) == len(pairs)
    print(f"\nEvaluated {len(all_results)}/{len(pairs)} pair(s) successfully")
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
