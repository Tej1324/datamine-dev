"""Minimal DeepStream runtime: RTSP -> YOLO person detection -> NvDCF -> footfall."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import quote

import yaml
from pyservicemaker import Pipeline, Probe

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ROOT / "config/cameras.yaml"
TRACKER_LIBRARY = "/opt/nvidia/deepstream/deepstream-9.1/lib/libnvds_nvmultiobjecttracker.so"


def env_file():
    values = {}
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({key: value for key, value in os.environ.items() if key in values})
    return values


def load_cameras():
    cameras = (yaml.safe_load(CAMERAS.read_text()) or {}).get("cameras", [])
    requested = [item.strip() for item in os.getenv("FOOTFALL_CAMERA_IDS", "ground_01").split(",") if item.strip()]
    by_id = {camera["id"]: camera for camera in cameras}
    missing = [camera_id for camera_id in requested if camera_id not in by_id]
    if missing:
        raise ValueError(f"unknown camera IDs: {','.join(missing)}")
    result = [by_id[camera_id] for camera_id in requested]
    subtype = os.getenv("STREAM_SUBTYPE_OVERRIDE", "").strip()
    return [{**camera, "subtype": int(subtype)} for camera in result] if subtype else result


def make_url(camera, values):
    user = quote(values["CAMERA_USERNAME"], safe="")
    password = quote(values["CAMERA_PASSWORD"], safe="")
    return (f"rtsp://{user}:{password}@{values['NVR_225_HOST']}:{values['NVR_225_PORT']}"
            f"/cam/realmonitor?channel={int(camera['channel'])}&subtype={int(camera['subtype'])}")


def main():
    cameras = load_cameras()
    values = env_file()
    offline_dir = os.getenv("OFFLINE_VIDEO_DIR", "").strip()
    width = int(os.getenv("FOOTFALL_WIDTH", "1280"))
    height = int(os.getenv("FOOTFALL_HEIGHT", "720"))
    camera_ids = [camera["id"] for camera in cameras]
    if offline_dir:
        root = Path(offline_dir)
        urls = [f"file://{root / f'cam_{index:02d}.mp4'}" for index in range(len(cameras))]
    else:
        urls = [make_url(camera, values) for camera in cameras]

    pipeline = Pipeline("footfall-yolo-nvdcf")
    pipeline.add("nvmultiurisrcbin", "sources", {
        "uri-list": ",".join(urls), "sensor-id-list": ",".join(camera_ids),
        "sensor-name-list": ",".join(camera_ids), "mode": 0,
        "max-batch-size": len(cameras), "width": width, "height": height,
        "live-source": not bool(offline_dir), "batched-push-timeout": 100000,
        "drop-pipeline-eos": not bool(offline_dir), "select-rtp-protocol": 4,
        "latency": 100, "drop-on-latency": True,
        "init-rtsp-reconnect-interval": 5, "rtsp-reconnect-interval": 10,
        "rtsp-reconnect-attempts": -1, "cudadec-memtype": 0,
        "disable-audio": True, "leaky": 2, "max-size-buffers": 1, "port": "0",
    })
    detector = Path(os.getenv("DETECTOR_CONFIG", str(ROOT / "config/detector_b6.txt")))
    tracker = Path(os.getenv("TRACKER_CONFIG", str(ROOT / "config/tracker.yml")))
    pipeline.add("nvinfer", "yolo_person", {"config-file-path": str(detector)})
    pipeline.add("nvtracker", "nvdcf", {
        "ll-lib-file": TRACKER_LIBRARY, "ll-config-file": str(tracker),
        "tracker-width": width, "tracker-height": height, "gpu-id": 0,
        "display-tracking-id": True, "operate-on-class-ids": "0",
    })

    from footfall_counter import FootfallCounter
    counter = FootfallCounter()
    pipeline.attach("nvdcf", Probe("footfall_counter", counter), tips="src")

    pipeline.add("tee", "display_tee")
    pipeline.add("queue", "display_queue", {"leaky": 2, "max-size-buffers": 1})
    rows = 1 if len(cameras) <= 2 else 2
    columns = 1 if len(cameras) == 1 else 2 if len(cameras) == 2 else 3
    pipeline.add("nvmultistreamtiler", "tiler", {
        "rows": rows, "columns": columns, "width": width * columns,
        "height": height * rows, "gpu-id": 0, "nvbuf-memory-type": 2,
    })
    pipeline.add("nvvideoconvert", "rgba", {"nvbuf-memory-type": 2, "disable-passthrough": True})
    pipeline.add("capsfilter", "rgba_caps", {"caps": "video/x-raw(memory:NVMM), format=RGBA"})
    pipeline.add("nvdsosd", "osd", {"process-mode": 1, "display-text": True})
    pipeline.add("nvvideoconvert", "rgb", {"nvbuf-memory-type": 2, "disable-passthrough": True})
    pipeline.add("capsfilter", "rgb_caps", {"caps": "video/x-raw(memory:NVMM), format=RGB"})
    pipeline.add("nvimageenc", "jpeg", {"quality": int(os.getenv("LIVE_JPEG_QUALITY", "90"))})
    pipeline.add("queue", "jpeg_queue", {"leaky": 2, "max-size-buffers": 1})
    pipeline.add("tcpclientsink", "dashboard_tcp", {
        "host": "127.0.0.1", "port": int(os.getenv("OVERVIEW_TCP_PORT", "7010")),
        "sync": False, "async": False,
    })
    pipeline.link("sources", "yolo_person", "nvdcf", "display_tee")
    pipeline.link("display_queue", "tiler", "rgba", "rgba_caps", "osd", "rgb",
                  "rgb_caps", "jpeg", "jpeg_queue", "dashboard_tcp")
    pipeline.link(("display_tee", "display_queue"), ("src_%u", ""))
    print(f"Footfall: cameras={camera_ids} detector=YOLO tracker=NvDCF", flush=True)
    try:
        pipeline.start().wait()
    finally:
        pipeline.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
