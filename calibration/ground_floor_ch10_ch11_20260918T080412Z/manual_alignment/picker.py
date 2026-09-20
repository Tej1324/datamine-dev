#!/usr/bin/env python3
"""Native-resolution CH10/CH11 manual correspondence picker."""
from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parent
HTML = ROOT / "picker.html"
FRAMES = ROOT / "frames"
LAYOUT = ROOT / "layout/layout.png"
LANDMARKS = ROOT / "landmarks.yaml"


def load_points() -> list[dict]:
    if not LANDMARKS.exists():
        return []
    try:
        data = yaml.safe_load(LANDMARKS.read_text()) or {}
        return [
            {
                "id": item.get("id", f"p{index:02d}"),
                "ch10_pixel": [float(item["ch10_pixel"][0]), float(item["ch10_pixel"][1])],
                "ch11_pixel": [float(item["ch11_pixel"][0]), float(item["ch11_pixel"][1])],
                "layout_pixel": ([float(item["layout_pixel"][0]), float(item["layout_pixel"][1])] if item.get("layout_pixel") else None),
            }
            for index, item in enumerate(data.get("landmarks", []), 1)
            if item.get("ch10_pixel") and item.get("ch11_pixel")
        ]
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return []


def save_points(points: list[dict]) -> None:
    landmarks = []
    for index, point in enumerate(points, 1):
        landmarks.append({
            "id": point.get("id", f"p{index:02d}"),
            "type": "manual_floor_plane_point",
            "description": "operator-selected permanent floor-plane or floor-contact point",
            "ch10_pixel": [round(float(point["ch10_pixel"][0]), 3), round(float(point["ch10_pixel"][1]), 3)],
            "ch11_pixel": [round(float(point["ch11_pixel"][0]), 3), round(float(point["ch11_pixel"][1]), 3)],
            "layout_pixel": ([round(float(point["layout_pixel"][0]), 3), round(float(point["layout_pixel"][1]), 3)] if point.get("layout_pixel") else None),
            "floor_plane_asserted": True,
            "verified": False,
        })
    payload = {
        "schema": "ground-floor-amc-manual-alignment-v1",
        "project": "ground_floor_ch10_ch11_20260918T080412Z",
        "camera_order": ["ch10", "ch11"],
        "coordinate_space": "native_calibration_video_1920x1080",
        "source_resolution": [1920, 1080],
        "status": "manual_points_saved_pending_validation",
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
        elif path == "/state":
            self.send_bytes(HTTPStatus.OK, "application/json", json.dumps({"points": self.server.points}).encode())
        elif path == "/layout/layout.png":
            self.send_bytes(HTTPStatus.OK, "image/png", LAYOUT.read_bytes())
        elif path == "/frames/ch10_1920x1080.jpg":
            self.send_bytes(HTTPStatus.OK, "image/jpeg", (FRAMES / "ch10_1920x1080.jpg").read_bytes())
        elif path == "/frames/ch11_1920x1080.jpg":
            self.send_bytes(HTTPStatus.OK, "image/jpeg", (FRAMES / "ch11_1920x1080.jpg").read_bytes())
        else:
            self.send_bytes(HTTPStatus.NOT_FOUND, "text/plain", b"not found")

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if urlparse(self.path).path != "/save":
                raise ValueError("unknown endpoint")
            points = payload.get("points", [])
            if not isinstance(points, list) or any("ch10_pixel" not in p or "ch11_pixel" not in p for p in points):
                raise ValueError("invalid point payload")
            save_points(points)
            self.server.points = points
            body = {"ok": True, "message": f"Saved {len(points)} CH10/CH11 pairs to landmarks.yaml."}
            self.send_bytes(HTTPStatus.OK, "application/json", json.dumps(body).encode())
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_bytes(HTTPStatus.BAD_REQUEST, "application/json", json.dumps({"ok": False, "message": str(exc)}).encode())

    def log_message(self, *_args) -> None:
        return


class Server(ThreadingHTTPServer):
    allow_reuse_address = True


def main() -> None:
    server = Server(("127.0.0.1", 8766), Handler)
    server.points = load_points()
    print("CH10/CH11 picker: http://127.0.0.1:8766/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
