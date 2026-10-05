"""Read-only ground-plane projection for the isolated SOLIDER experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


class GroundProjector:
    """Maps a person's bbox bottom-centre pixel to calibrated world X/Y."""

    def __init__(self, camera_model_dir: Path, image_size=None, calibration_size=(1920, 1080)):
        self.inverse: dict[int, np.ndarray] = {}
        for source in (0, 1):
            document = yaml.safe_load((camera_model_dir / f"cam_{source:02d}.yml").read_text())
            projection = np.asarray(document["projectionMatrix_3x4_w2p"], dtype=np.float64).reshape(3, 4)
            if image_size is not None:
                projection = projection.copy()
                projection[0] *= image_size[0] / calibration_size[0]
                projection[1] *= image_size[1] / calibration_size[1]
            homography = projection[:, (0, 1, 3)]  # world ground plane Z=0
            self.inverse[source] = np.linalg.inv(homography)

    def world_point(self, source: int, bbox: list[float]) -> tuple[float, float] | None:
        inverse = self.inverse.get(int(source))
        if inverse is None:
            return None
        left, top, width, height = (float(value) for value in bbox)
        pixel = np.asarray([left + width / 2.0, top + height, 1.0], dtype=np.float64)
        world = inverse @ pixel
        if not np.isfinite(world).all() or abs(world[2]) < 1e-9:
            return None
        return float(world[0] / world[2]), float(world[1] / world[2])


class FloorPlan2DProjector:
    """Project image footpoints into one shared normalized 2D floor-plan frame."""

    def __init__(self, calibration_file: Path):
        document = yaml.safe_load(calibration_file.read_text()) if calibration_file.suffix in {".yml", ".yaml"} else None
        if document is None:
            import json
            document = json.loads(calibration_file.read_text())
        self.width, self.height = (float(v) for v in document["image_size"])
        self.matrices = {
            (0 if camera == "ch10" else 1 if camera == "ch11" else int(camera.replace("ch", ""))):
            np.asarray(data["image_to_layout"], dtype=np.float64)
            for camera, data in document["cameras"].items()
        }

    def world_point(self, source: int, bbox: list[float]) -> tuple[float, float] | None:
        matrix = self.matrices.get(int(source))
        if matrix is None:
            return None
        left, top, width, height = (float(value) for value in bbox)
        pixel = np.asarray([left + width / 2.0, top + height, 1.0], dtype=np.float64)
        point = matrix @ pixel
        if not np.isfinite(point).all() or abs(point[2]) < 1e-9:
            return None
        return float(point[0] / point[2]), float(point[1] / point[2])
