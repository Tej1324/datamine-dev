#!/usr/bin/env python3
"""OpenCV planar CH18<->CH20 calibration.

This tool produces ordinary planar homographies only. It never emits an
MV3DT projection matrix. Candidate points are deliberately rejected when
their geometry is too narrow or unstable under leave-one-out validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
LANDMARKS = ROOT / "landmarks.yaml"
FRAME_DIR = ROOT / "frames"
HOMOGRAPHY_DIR = ROOT / "outputs/homography"
VALIDATION_DIR = ROOT / "outputs/validation"


def norm_h(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    scale = matrix[2, 2]
    if abs(scale) < 1e-12:
        raise ValueError("homography has an invalid normalization term")
    return matrix / scale


def fit_homography(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(src) < 4:
        raise ValueError("at least four points are required")
    matrix, inliers = cv2.findHomography(
        src.astype(np.float64), dst.astype(np.float64), cv2.RANSAC, 12.0
    )
    if matrix is None or inliers is None:
        raise ValueError("OpenCV findHomography failed")
    return norm_h(matrix), inliers.reshape(-1).astype(bool)


def project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        points.astype(np.float64).reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)


def errors(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return np.linalg.norm(expected - actual, axis=1)


def hull_fraction(points: np.ndarray, width: float, height: float) -> float:
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull) / (width * height))


def leave_one_out(src: np.ndarray, dst: np.ndarray) -> list[float | None]:
    values: list[float | None] = []
    for index in range(len(src)):
        keep = np.ones(len(src), dtype=bool)
        keep[index] = False
        try:
            matrix, _ = fit_homography(src[keep], dst[keep])
            values.append(float(errors(dst[index:index + 1], project(src[index:index + 1], matrix))[0]))
        except (ValueError, cv2.error):
            values.append(None)
    return values


def draw_points(image: np.ndarray, points: np.ndarray, labels: list[str]) -> np.ndarray:
    output = image.copy()
    for label, (x, y) in zip(labels, points):
        center = (int(round(x)), int(round(y)))
        cv2.circle(output, center, 8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(output, label, (center[0] + 10, center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return output


def world_canvas(points: np.ndarray, scale: float = 180.0, margin: int = 40) -> tuple[np.ndarray, tuple[int, int]]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    size = np.ceil((maximum - minimum) * scale + 2 * margin).astype(int)
    transform = np.array(
        [[scale, 0.0, margin - minimum[0] * scale],
         [0.0, -scale, margin + maximum[1] * scale],
         [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return transform, (int(size[0]), int(size[1]))


def main() -> int:
    config = yaml.safe_load(LANDMARKS.read_text())
    records = config.get("landmarks", [])
    if len(records) < 4:
        raise ValueError("at least four paired landmarks are required")
    labels = [record["id"] for record in records]
    ch18 = np.array([record["ch18_pixel"] for record in records], dtype=np.float64)
    ch20 = np.array([record["ch20_pixel"] for record in records], dtype=np.float64)
    world_values = [record.get("floor_xy_arbitrary") for record in records]
    has_world = all(value is not None for value in world_values)
    mapping_mode = "world_and_pixel" if has_world else "pixel_only"

    h18_world = None
    h20_world = None
    if has_world:
        world = np.array(world_values, dtype=np.float64)
        h18_world, in18 = fit_homography(ch18, world)
        h20_world, in20 = fit_homography(ch20, world)
        h18_20 = norm_h(np.linalg.inv(h20_world) @ h18_world)
        h20_18 = norm_h(np.linalg.inv(h18_world) @ h20_world)
        world18 = project(ch18, h18_world)
        world20 = project(ch20, h20_world)
        e18 = errors(world, world18)
        e20 = errors(world, world20)
    else:
        h18_20, inliers = fit_homography(ch18, ch20)
        h20_18 = norm_h(np.linalg.inv(h18_20))
        in18 = inliers
        in20 = inliers
        e18 = None
        e20 = None

    cross20 = project(ch18, h18_20)
    cross18 = project(ch20, h20_18)
    ecross20 = errors(ch20, cross20)
    ecross18 = errors(ch18, cross18)
    loo = leave_one_out(ch18, ch20)
    numeric_loo = np.array([v for v in loo if v is not None], dtype=float)

    source_fraction = min(hull_fraction(ch18, 1280, 720), hull_fraction(ch20, 1280, 720))
    inlier_count = int(np.count_nonzero(in18 & in20))
    reasons: list[str] = []
    if len(records) < 8:
        reasons.append("at least 8 manual pairs are required")
    if inlier_count < 8:
        reasons.append("fewer than 8 RANSAC inliers")
    if source_fraction < 0.05:
        reasons.append("point hull covers too little of the source image")
    if float(np.median(ecross20)) > 12.0:
        reasons.append("cross-camera median residual exceeds 12 px")
    if float(np.max(ecross20)) > 30.0:
        reasons.append("cross-camera maximum residual exceeds 30 px")
    if len(numeric_loo) < 4:
        reasons.append("too few valid leave-one-out fits")
    elif float(np.median(numeric_loo)) > 30.0:
        reasons.append("leave-one-out median residual exceeds 30 px")
    elif float(np.max(numeric_loo)) > 100.0:
        reasons.append("leave-one-out maximum residual exceeds 100 px")
    stable = not reasons

    HOMOGRAPHY_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    matrices = {
        "accepted": stable,
        "type": "planar_homography_only",
        "mapping_space": "ch18_pixels_to_ch20_pixels",
        "source_geometry": [1280, 720],
        "coordinate_system": config["world_coordinate_system"],
        "H18_to_world": h18_world.tolist() if h18_world is not None else None,
        "H20_to_world": h20_world.tolist() if h20_world is not None else None,
        "H18_to_20": h18_20.tolist(),
        "H20_to_18": h20_18.tolist(),
    }
    (HOMOGRAPHY_DIR / "candidate_matrices.json").write_text(json.dumps(matrices, indent=2) + "\n")

    ch18_img = cv2.imread(str(FRAME_DIR / "ch18_1280x720.jpg"))
    ch20_img = cv2.imread(str(FRAME_DIR / "ch20_1280x720.jpg"))
    if ch18_img is None or ch20_img is None:
        raise FileNotFoundError("working CH18/CH20 frames are missing")
    cv2.imwrite(str(VALIDATION_DIR / "ch18_points.jpg"), draw_points(ch18_img, ch18, labels))
    cv2.imwrite(str(VALIDATION_DIR / "ch20_points.jpg"), draw_points(ch20_img, ch20, labels))
    warped18_to_20 = cv2.warpPerspective(ch18_img, h18_20, (ch20_img.shape[1], ch20_img.shape[0]))
    warped20_to_18 = cv2.warpPerspective(ch20_img, h20_18, (ch18_img.shape[1], ch18_img.shape[0]))
    cv2.imwrite(str(VALIDATION_DIR / "ch18_warped_to_ch20_candidate.jpg"), warped18_to_20)
    cv2.imwrite(str(VALIDATION_DIR / "ch20_warped_to_ch18_candidate.jpg"), warped20_to_18)
    cv2.imwrite(
        str(VALIDATION_DIR / "camera_overlay_candidate.jpg"),
        cv2.addWeighted(warped18_to_20, 0.5, ch20_img, 0.5, 0.0),
    )
    cv2.imwrite(
        str(VALIDATION_DIR / "camera_overlay_reverse_candidate.jpg"),
        cv2.addWeighted(warped20_to_18, 0.5, ch18_img, 0.5, 0.0),
    )

    if has_world:
        world_to_px, canvas_size = world_canvas(world)
        h18_canvas = world_to_px @ h18_world
        h20_canvas = world_to_px @ h20_world
        warped18 = cv2.warpPerspective(ch18_img, h18_canvas, canvas_size)
        warped20 = cv2.warpPerspective(ch20_img, h20_canvas, canvas_size)
        cv2.imwrite(str(VALIDATION_DIR / "ch18_warped_candidate.jpg"), warped18)
        cv2.imwrite(str(VALIDATION_DIR / "ch20_warped_candidate.jpg"), warped20)
        cv2.imwrite(
            str(VALIDATION_DIR / "floor_overlay_candidate.jpg"),
            cv2.addWeighted(warped18, 0.5, warped20, 0.5, 0.0),
        )

    report = {
        "accepted": stable,
        "rejection_reasons": reasons,
        "mapping_mode": mapping_mode,
        "camera_order": ["ch18", "ch20"],
        "landmark_count": len(records),
        "inlier_count_both_camera_fits": inlier_count,
        "source_hull_fraction_min": source_fraction,
        "per_landmark": [
            {
                "id": label,
                "ch18_to_world_residual": float(e18[i]) if e18 is not None else None,
                "ch20_to_world_residual": float(e20[i]) if e20 is not None else None,
                "ch18_to_ch20_residual_px": float(ecross20[i]),
                "ch20_to_ch18_residual_px": float(ecross18[i]),
                "leave_one_out_ch18_to_ch20_px": loo[i],
            }
            for i, label in enumerate(labels)
        ],
        "aggregate": {
            "ch18_to_world_rms": (
                float(np.sqrt(np.mean(e18 ** 2))) if e18 is not None else None
            ),
            "ch20_to_world_rms": (
                float(np.sqrt(np.mean(e20 ** 2))) if e20 is not None else None
            ),
            "ch18_to_ch20_rms_px": float(np.sqrt(np.mean(ecross20 ** 2))),
            "ch18_to_ch20_median_px": float(np.median(ecross20)),
            "ch18_to_ch20_max_px": float(np.max(ecross20)),
            "leave_one_out_median_px": float(np.median(numeric_loo)) if len(numeric_loo) else None,
            "leave_one_out_max_px": float(np.max(numeric_loo)) if len(numeric_loo) else None,
        },
        "metric_scale_verified": False,
        "mv3dt_projection_matrix_written": False,
    }
    (VALIDATION_DIR / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (VALIDATION_DIR / "validation_report.md").write_text(
        "# CH18/CH20 planar homography validation\n\n"
        f"- Accepted: **{stable}**\n"
        f"- Mapping mode: **{mapping_mode}**\n"
        f"- Landmarks: **{len(records)}**\n"
        f"- Cross-camera RMS: **{report['aggregate']['ch18_to_ch20_rms_px']:.3f} px**\n"
        f"- Cross-camera median: **{report['aggregate']['ch18_to_ch20_median_px']:.3f} px**\n"
        f"- Cross-camera maximum: **{report['aggregate']['ch18_to_ch20_max_px']:.3f} px**\n"
        f"- Leave-one-out median/max: **{report['aggregate']['leave_one_out_median_px']} / {report['aggregate']['leave_one_out_max_px']} px**\n\n"
        "The matrices are candidate planar homographies only. Metric scale is unverified.\n"
        "No MV3DT projection matrix was generated.\n"
    )
    print(json.dumps(report, indent=2))
    return 0 if stable else 2


if __name__ == "__main__":
    raise SystemExit(main())
