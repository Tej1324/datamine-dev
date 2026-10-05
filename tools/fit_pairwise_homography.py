#!/usr/bin/env python3
"""Fit a diagnostic CH10->CH11 image-plane homography from verified pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-time-delta", type=float, default=0.5)
    parser.add_argument("--min-similarity", type=float, default=0.90)
    parser.add_argument("--ransac-pixels", type=float, default=40.0)
    args = parser.parse_args()

    selected = []
    with args.pairs.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("same_person", "").lower() not in {"true", "1", "yes"}:
                continue
            if float(row["aligned_timestamp_delta_s"]) > args.max_time_delta:
                continue
            if float(row["solider_similarity"]) < args.min_similarity:
                continue
            selected.append(row)
    if len(selected) < 4:
        raise SystemExit(f"need at least 4 usable pairs; found {len(selected)}")

    source = np.float32([[float(r["left_footpoint_x_px"]), float(r["left_footpoint_y_px"])] for r in selected])
    target = np.float32([[float(r["right_footpoint_x_px"]), float(r["right_footpoint_y_px"])] for r in selected])
    matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, args.ransac_pixels)
    if matrix is None or mask is None:
        raise SystemExit("homography estimation failed")
    projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    residuals = np.linalg.norm(projected - target, axis=1)
    inliers = mask.reshape(-1).astype(bool)
    document = {
        "type": "diagnostic_pairwise_homography",
        "source": "verified_live_crop_footpoints",
        "pairs_total": len(selected),
        "inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "ransac_threshold_pixels": args.ransac_pixels,
        "max_aligned_time_delta_seconds": args.max_time_delta,
        "min_solider_similarity": args.min_similarity,
        "median_inlier_residual_pixels": float(np.median(residuals[inliers])) if inliers.any() else None,
        "median_all_residual_pixels": float(np.median(residuals)),
        "homography_ch10_to_ch11": matrix.tolist(),
        "pairs": [
            {"reference_id": r["reference_id"], "inlier": bool(inliers[i]),
             "residual_pixels": float(residuals[i])}
            for i, r in enumerate(selected)
        ],
        "warning": "Diagnostic only; moving-person footpoints are not metric floor landmarks.",
    }
    output = args.output or args.pairs.with_name("pairwise_homography_diagnostic.json")
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    summary = {k: document[k] for k in ("pairs_total", "inliers", "inlier_ratio", "median_inlier_residual_pixels", "median_all_residual_pixels")}
    summary["output"] = str(output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
