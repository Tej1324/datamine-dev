"""Low-latency loopback MJPEG dashboard for the DeepStream overview."""

import os
import socket
import subprocess
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template

ROOT = Path(__file__).resolve().parents[1]
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
    """Retains one completed GPU-encoded JPEG for all connected browsers."""

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
                            client, _ = listener.accept()
                            break
                        except socket.timeout:
                            continue
                    with client:
                        pending = b""
                        while True:
                            chunk = client.recv(64 * 1024)
                            if not chunk:
                                break
                            pending += chunk
                            while True:
                                start = pending.find(b"\xff\xd8")
                                end = pending.find(b"\xff\xd9", start + 2)
                                if start < 0:
                                    pending = pending[-2:]
                                    break
                                if end < 0:
                                    pending = pending[start:]
                                    break
                                with self.lock:
                                    self.latest = pending[start:end + 2]
                                    self.sequence += 1
                                    self.lock.notify_all()
                                pending = pending[end + 2:]
            except OSError:
                time.sleep(0.5)


SYSTEM_METRICS = SystemMetrics()
OVERVIEW_HUB = StreamHub(7007)


@app.get("/")
@app.get("/detection")
def index():
    return render_template("index.html")


@app.get("/api/preview/metrics")
def preview_metrics():
    return jsonify(mode="latest-jpeg", overview_jpeg="/api/overview/frame")


@app.get("/api/overview/frame")
def overview_latest_frame():
    """Return only the newest complete JPEG; never queue old frames."""
    with OVERVIEW_HUB.lock:
        image = OVERVIEW_HUB.latest
        sequence = OVERVIEW_HUB.sequence
    if not image:
        return Response("No overview frame available", status=503)
    return Response(
        image,
        mimetype="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "X-Overview-Sequence": str(sequence),
        },
    )


@app.get("/stream/overview.jpg")
def overview_stream_jpeg():
    """One persistent MJPEG response; every yielded image is the newest one."""
    with OVERVIEW_HUB.lock:
        initial_sequence = OVERVIEW_HUB.sequence

    def chunks():
        sequence = initial_sequence
        while True:
            with OVERVIEW_HUB.lock:
                while OVERVIEW_HUB.sequence <= sequence:
                    OVERVIEW_HUB.lock.wait(timeout=5)
                image = OVERVIEW_HUB.latest
                sequence = OVERVIEW_HUB.sequence
            if image:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                       str(len(image)).encode() + b"\r\n\r\n" + image + b"\r\n")

    return Response(
        chunks(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/api/system/metrics")
def system_metrics():
    return jsonify(SYSTEM_METRICS.read())


@app.get("/api/reid-experiment/latest")
def reid_experiment_latest():
    """Read-only browser overlay state from the isolated SOLIDER worker."""
    state = ROOT / "runs/reid_experiment/live_state.json"
    if not state.exists():
        return jsonify(enabled=False, tracks=[], stats={})
    try:
        import json
        document = json.loads(state.read_text(encoding="utf-8"))
        document["enabled"] = True
        return jsonify(document)
    except (OSError, ValueError):
        return jsonify(enabled=False, tracks=[], stats={})


@app.get("/api/reid-experiment/journeys")
def reid_experiment_journeys():
    """Return the current Global-ID tracklet history for journey analytics."""
    state = ROOT / "runs/reid_experiment/live_state.json"
    if not state.exists():
        return jsonify(enabled=False, journeys=[])
    try:
        import json
        document = json.loads(state.read_text(encoding="utf-8"))
        return jsonify(enabled=True, updated_at=document.get("updated_at"),
                       journeys=document.get("journeys", []))
    except (OSError, ValueError):
        return jsonify(enabled=False, journeys=[])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("VIEWER_PORT", "6000")),
            debug=False, threaded=True)
