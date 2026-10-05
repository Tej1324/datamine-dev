#!/usr/bin/env python3
"""Fit diagnostic image-to-floor-plan homographies from point-picker output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def fit(points, camera, image_size, threshold):
    iw, ih = image_size
    source = np.float32([[p["picks"][camera]["x"] * iw,
                          p["picks"][camera]["y"] * ih] for p in points])
    # The target is the shared, unitless 2D floor-plan coordinate system.
    # It is deliberately not a CH10/CH11 pixel coordinate system.
    target = np.float32([[p["picks"]["layout"]["x"],
                          p["picks"]["layout"]["y"]] for p in points])
    matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, threshold)
    if matrix is None or mask is None:
        raise RuntimeError(f"could not fit {camera} homography")
    projected = cv2.perspectiveTransform(source[:, None, :], matrix)[:, 0]
    errors = np.linalg.norm(projected - target, axis=1)
    inliers = mask[:, 0].astype(bool)
    return {
        "image_to_layout": matrix.tolist(),
        "layout_to_image": np.linalg.inv(matrix).tolist(),
        "points_total": len(points),
        "inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "ransac_threshold_common_xy": threshold,
        "median_inlier_error_common_xy": float(np.median(errors[inliers])) if inliers.any() else None,
        "max_inlier_error_common_xy": float(np.max(errors[inliers])) if inliers.any() else None,
        "points": [{"point_id": p["point_id"], "inlier": bool(inliers[i]),
                    "error_layout_pixels": float(errors[i])}
                   for i, p in enumerate(points)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=Path, required=True)
    ap.add_argument("--layout", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--exclude-point", action="append", default=[])
    ap.add_argument("--filtered-points-output", type=Path)
    ap.add_argument("--image-width", type=int, default=1920)
    ap.add_argument("--image-height", type=int, default=1080)
    ap.add_argument("--ransac-threshold", type=float, default=0.05,
                    help="RANSAC radius in normalized common floor-plan XY")
    args = ap.parse_args()
    all_points = json.loads(args.points.read_text())
    excluded = set(args.exclude_point)
    points = [p for p in all_points if p["point_id"] not in excluded]
    if len(points) < 4:
        raise SystemExit("need at least four points after exclusions")
    layout = cv2.imread(str(args.layout), cv2.IMREAD_UNCHANGED)
    if layout is None:
        raise SystemExit(f"cannot read layout: {args.layout}")
    lh, lw = layout.shape[:2]
    result = {
        "type": "diagnostic_floorplan_homography",
        "source_points": str(args.points.resolve()),
        "layout_path": str(args.layout.resolve()),
        "image_size": [args.image_width, args.image_height],
        "layout_size": [lw, lh],
        "coordinate_system": "normalized_floorplan_xy",
        "excluded_points": sorted(excluded),
        "used_point_ids": [p["point_id"] for p in points],
        "cameras": {},
        "warning": "Diagnostic only. Do not replace NVIDIA/AMC production calibration without validation.",
    }
    for camera in ("ch10", "ch11"):
        result["cameras"][camera] = fit(
            points, camera, (args.image_width, args.image_height), args.ransac_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.filtered_points_output:
        args.filtered_points_output.parent.mkdir(parents=True, exist_ok=True)
        args.filtered_points_output.write_text(json.dumps(points, indent=2) + "\n")
    for camera, report in result["cameras"].items():
        rejected = [x["point_id"] for x in report["points"] if not x["inlier"]]
        print(f"{camera}: {report['inliers']}/{report['points_total']} inliers; "
              f"median common-XY error {report['median_inlier_error_common_xy']:.4f}; "
              f"rejected {','.join(rejected) or 'none'}")
    print(args.output)


if __name__ == "__main__":
    main()
