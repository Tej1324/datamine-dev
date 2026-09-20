"""Lightweight loopback HTTP bridge for six DeepStream camera MJPEG streams."""

import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template

ROOT = Path(__file__).resolve().parents[1]
DETECTION_FILE = ROOT / "runs/detection_live.json"
TCP_HOST = "127.0.0.1"
app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent / "templates"))


class SystemMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.value = {"cpu_percent": None, "gpu_percent": None,
                      "vram_used_mib": None, "vram_total_mib": None,
                      "nvdec_percent": None, "nvenc_percent": None,
                      "updated_at": None}
        self.previous_cpu = self._cpu_sample()
        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _cpu_sample():
        fields = Path("/proc/stat").read_text().splitlines()[0].split()
        values = [int(value) for value in fields[1:]]
        return sum(values), values[3] + values[4]

    def _run(self):
        while True:
            try:
                current = self._cpu_sample()
                total_delta = current[0] - self.previous_cpu[0]
                idle_delta = current[1] - self.previous_cpu[1]
                self.previous_cpu = current
                cpu = 100.0 * (1.0 - idle_delta / total_delta) if total_delta else 0.0
                query = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.encoder,"
                     "utilization.decoder,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1, check=False)
                fields = [item.strip() for item in query.stdout.splitlines()[0].split(",")]
                gpu, enc, dec, used, total = (float(item) for item in fields)
                snapshot = {"cpu_percent": round(cpu, 1), "gpu_percent": round(gpu, 1),
                            "vram_used_mib": round(used, 1), "vram_total_mib": round(total, 1),
                            "nvdec_percent": round(dec, 1), "nvenc_percent": round(enc, 1),
                            "updated_at": time.time()}
                with self.lock:
                    self.value = snapshot
            except (OSError, IndexError, ValueError, subprocess.SubprocessError):
                pass
            time.sleep(2)

    def read(self):
        with self.lock:
            return dict(self.value)


class StreamHub:
    def __init__(self, port):
        self.port = port
        self.lock = threading.Condition()
        self.latest = None
        self.sequence = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    listener.bind((TCP_HOST, self.port))
                    listener.listen(1)
                    listener.settimeout(1)
                    while True:
                        try:
                            sock, _ = listener.accept()
                            break
                        except socket.timeout:
                            continue
                    with sock:
                        buffer = b""
                        while True:
                            data = sock.recv(64 * 1024)
                            if not data:
                                break
                            buffer += data
                            while True:
                                start = buffer.find(b"\xff\xd8")
                                end = buffer.find(b"\xff\xd9", start + 2)
                                if start < 0:
                                    buffer = buffer[-2:]
                                    break
                                if end < 0:
                                    buffer = buffer[start:]
                                    break
                                frame = buffer[start:end + 2]
                                buffer = buffer[end + 2:]
                                with self.lock:
                                    self.latest = frame
                                    self.sequence += 1
                                    self.lock.notify_all()
            except OSError:
                time.sleep(0.5)

    def subscribe(self):
        with self.lock:
            return self.sequence


CAMERA_IDS = ("ground_01", "ground_02", "ground_03", "ground_04", "ground_05", "ground_06")
CAMERA_HUBS = {camera_id: StreamHub(7001 + index) for index, camera_id in enumerate(CAMERA_IDS)}
SYSTEM_METRICS = SystemMetrics()


@app.get("/")
@app.get("/detection")
def index():
    return render_template("index.html")


def _mjpeg_response(hub):
    initial_sequence = hub.subscribe()

    def chunks():
        sequence = initial_sequence
        try:
            while True:
                with hub.lock:
                    while hub.sequence <= sequence:
                        hub.lock.wait(timeout=5)
                    image = hub.latest
                    sequence = hub.sequence
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(image)).encode() + b"\r\n\r\n" + image + b"\r\n"
        finally:
            pass

    return Response(chunks(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/stream/<camera_id>.jpg")
def camera_stream_jpeg(camera_id):
    hub = CAMERA_HUBS.get(camera_id)
    if hub is None:
        return jsonify(error="unknown camera"), 404
    return _mjpeg_response(hub)


def read_json(path):
    try:
        return jsonify(json.loads(path.read_text()))
    except (OSError, ValueError):
        return jsonify(error="metadata unavailable", stale=True), 503


@app.get("/api/detection/live")
def detection_live():
    return read_json(DETECTION_FILE)


@app.get("/api/preview/metrics")
def preview_metrics():
    return jsonify({
        "mode": "native_camera_mjpeg",
        "cameras": {camera_id: f"/stream/{camera_id}.jpg" for camera_id in CAMERA_IDS},
    })


@app.get("/api/system/metrics")
def system_metrics():
    return jsonify(SYSTEM_METRICS.read())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("VIEWER_PORT", "6000")),
            debug=False, threaded=True)
