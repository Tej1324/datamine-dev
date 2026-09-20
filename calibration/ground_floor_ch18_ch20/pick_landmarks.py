#!/usr/bin/env python3
"""Loopback browser picker for CH18/CH20 manual floor correspondences."""

from __future__ import annotations

import argparse
import json
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import yaml


ROOT = Path(__file__).resolve().parent
FRAMES = ROOT / "frames"
LAYOUT = ROOT / "layout/layout.png"
LANDMARKS = ROOT / "landmarks.yaml"
HTML = ROOT / "picker.html"


def load_points() -> list[dict]:
    if not LANDMARKS.exists():
        return []
    try:
        data = yaml.safe_load(LANDMARKS.read_text()) or {}
        if not str(data.get("status", "")).startswith("manual"):
            return []
        result = []
        for index, item in enumerate(data.get("landmarks", []), 1):
            left, right = item.get("ch18_pixel"), item.get("ch20_pixel")
            if left and right:
                result.append({
                    "id": item.get("id", f"p{index:02d}"),
                    "ch18_pixel": [float(left[0]), float(left[1])],
                    "ch20_pixel": [float(right[0]), float(right[1])],
                })
        return result
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return []


def save_points(points: list[dict]) -> None:
    landmarks = []
    for index, point in enumerate(points, 1):
        landmarks.append({
            "id": point.get("id", f"p{index:02d}"),
            "type": "manual_floor_plane_point",
            "description": "operator-selected permanent floor-plane or floor-contact point",
            "ch18_pixel": [round(float(point["ch18_pixel"][0]), 3), round(float(point["ch18_pixel"][1]), 3)],
            "ch20_pixel": [round(float(point["ch20_pixel"][0]), 3), round(float(point["ch20_pixel"][1]), 3)],
            "floor_xy_arbitrary": None,
            "floor_plane_asserted": True,
            "verified": False,
        })
    payload = {
        "schema": "ground-floor-planar-landmarks-v1",
        "project": "ground_floor_ch18_ch20",
        "camera_order": ["ch18", "ch20"],
        "coordinate_space": "production_1280x720",
        "source_resolution": [1280, 720],
        "status": "manual_points_saved_pending_validation",
        "overlap_observation": {
            "status": "operator_selected",
            "region": "operator-defined CH18/CH20 common floor region",
            "evidence": "manual paired clicks on permanent floor-plane/floor-contact features",
        },
        "world_coordinate_system": {
            "name": "not_provided",
            "units": "unknown",
            "metric_scale_verified": False,
            "note": "World/floor coordinates were intentionally not fabricated by the picker.",
        },
        "landmarks": landmarks,
    }
    LANDMARKS.write_text(yaml.safe_dump(payload, sort_keys=False))


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self.send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", HTML.read_bytes())
            return
        if path == "/layout/layout.png":
            self.send_bytes(HTTPStatus.OK, "image/png", LAYOUT.read_bytes())
            return
        if path == "/state":
            self.send_bytes(HTTPStatus.OK, "application/json", json.dumps({"points": self.server.points}).encode())
            return
        if path.startswith("/frames/"):
            name = Path(path.removeprefix("/frames/")).name
            if name not in {"ch18_1280x720.jpg", "ch20_1280x720.jpg"}:
                self.send_bytes(HTTPStatus.NOT_FOUND, "text/plain", b"not found")
                return
            self.send_bytes(HTTPStatus.OK, "image/jpeg", (FRAMES / name).read_bytes())
            return
        self.send_bytes(HTTPStatus.NOT_FOUND, "text/plain", b"not found")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            path = urlparse(self.path).path
            if path == "/save":
                points = payload.get("points", [])
                if not isinstance(points, list) or any(
                    "ch18_pixel" not in point or "ch20_pixel" not in point for point in points
                ):
                    raise ValueError("invalid point payload")
                save_points(points)
                self.server.points = points
                result = {"ok": True, "message": f"Saved {len(points)} manual pairs to landmarks.yaml."}
            elif path == "/validate":
                run = subprocess.run(
                    ["python3", str(ROOT / "calibrate.py")],
                    capture_output=True, text=True, timeout=120, check=False,
                )
                result = {"ok": run.returncode == 0, "message": (run.stdout or run.stderr)[-4000:]}
            else:
                self.send_bytes(HTTPStatus.NOT_FOUND, "application/json", b'{"ok":false}')
                return
            self.send_bytes(HTTPStatus.OK, "application/json", json.dumps(result).encode())
        except (OSError, ValueError, TypeError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
            self.send_bytes(HTTPStatus.BAD_REQUEST, "application/json", json.dumps({"ok": False, "message": str(exc)}).encode())

    def log_message(self, *_args) -> None:
        return


class Server(ThreadingHTTPServer):
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = Server((args.host, args.port), Handler)
    server.points = load_points()
    print(f"CH18/CH20 picker: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
