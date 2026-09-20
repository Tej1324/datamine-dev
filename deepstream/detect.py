"""Six-camera DeepStream display pipeline using NVIDIA's standard multi-URI bin."""

import os
import sys
from pathlib import Path
from urllib.parse import quote

import yaml
from pyservicemaker import Pipeline


ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ROOT / "config/cameras.yaml"
DETECTOR = ROOT / "config/detector.txt"
TRACKER_CONFIG = ROOT / "config/tracker.yml"
STAFF_FILTER_CONFIG = ROOT / "config/staff_filter.yml"
TRACKER_LIBRARY = "/opt/nvidia/deepstream/deepstream-9.1/lib/libnvds_nvmultiobjecttracker.so"


def env_file():
    values = {}
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({key: value for key, value in os.environ.items() if key in values})
    return values


def load_cameras():
    cameras = (yaml.safe_load(CAMERAS.read_text()) or {}).get("cameras", [])
    expected = [7, 16, 10, 18, 20, 11]
    if len(cameras) != len(expected) or [camera["channel"] for camera in cameras] != expected:
        raise ValueError("requires ordered NVR-225 channels 7,16,10,18,20,11")
    if {camera["nvr"] for camera in cameras} != {225}:
        raise ValueError("requires NVR-225 sources")
    return cameras


def make_url(camera, values):
    user = quote(values["CAMERA_USERNAME"], safe="")
    password = quote(values["CAMERA_PASSWORD"], safe="")
    return (f"rtsp://{user}:{password}@{values['NVR_225_HOST']}:{values['NVR_225_PORT']}"
            f"/cam/realmonitor?channel={int(camera['channel'])}"
            f"&subtype={int(camera['subtype'])}")


def staff_filter_settings():
    settings = yaml.safe_load(STAFF_FILTER_CONFIG.read_text()) or {}
    profile = ROOT / settings.get("profile_path", "data/staff_filter/profile.yml")
    return {
        "enabled": bool(settings.get("enabled", True)),
        "staff-color-threshold": float(settings.get("staff_color_threshold", 0.62)),
        "profile-path": str(profile),
    }


def main():
    cameras = load_cameras()
    values = env_file()
    urls = [make_url(camera, values) for camera in cameras]
    camera_ids = [camera["id"] for camera in cameras]
    staff_settings = staff_filter_settings()

    pipeline = Pipeline("ground-floor-deepstream")
    pipeline.add("nvmultiurisrcbin", "ground_floor_sources", {
        "uri-list": ",".join(urls), "sensor-id-list": ",".join(camera_ids),
        "sensor-name-list": ",".join(camera_ids), "mode": 0, "max-batch-size": 6,
        "width": 1280, "height": 720, "live-source": True,
        # The six RTSP feeds run at approximately 10 FPS.  NVIDIA's live
        # streammux guidance is to allow roughly one source frame period for
        # batch formation; 30 ms created partial/jittered batches here.
        "batched-push-timeout": 100000, "drop-pipeline-eos": True,
        "select-rtp-protocol": 4, "latency": 100, "drop-on-latency": True,
        "cudadec-memtype": 0, "disable-audio": True, "leaky": 2,
        "max-size-buffers": 1, "port": "0",
    })
    # Keep the validated static-batch-5 YOLO26s engine. nvinfer handles the
    # six-source mux in inference sub-batches without changing the model.
    pipeline.add("nvinfer", "yolo26s", {"config-file-path": str(DETECTOR), "batch-size": 5})
    pipeline.add("staffcolorfilter", "staff_color_filter", staff_settings)
    pipeline.add("nvtracker", "nvdcf_local", {
        "ll-lib-file": TRACKER_LIBRARY, "ll-config-file": str(TRACKER_CONFIG),
        "tracker-width": 960, "tracker-height": 544, "gpu-id": 0,
        "display-tracking-id": True, "operate-on-class-ids": "0",
    })
    # Keep the video path source-specific after tracking.  nvstreamdemux
    # exposes one source buffer per camera; there is deliberately no tee,
    # overview tiler, or shared batched OSD after this point.  Each source
    # gets its own GPU RGBA conversion and OSD, so drawing cannot race another
    # camera's encoder or a tiled overview branch.
    pipeline.add("nvstreamdemux", "camera_demux")
    for index in range(len(camera_ids)):
        pipeline.add("queue", f"camera_{index}_queue", {"leaky": 2, "max-size-buffers": 1})
        pipeline.add("nvvideoconvert", f"camera_{index}_rgba", {
            "nvbuf-memory-type": 2, "disable-passthrough": True,
        })
        pipeline.add("capsfilter", f"camera_{index}_rgba_caps", {
            "caps": "video/x-raw(memory:NVMM), format=RGBA",
        })
        pipeline.add("nvdsosd", f"camera_{index}_osd", {
            "process-mode": 1, "display-text": True,
        })
        pipeline.add("nvvideoconvert", f"camera_{index}_convert", {
            "nvbuf-memory-type": 2, "disable-passthrough": True,
        })
        pipeline.add("capsfilter", f"camera_{index}_rgb", {"caps": "video/x-raw(memory:NVMM), format=RGB"})
        pipeline.add("nvimageenc", f"camera_{index}_jpeg", {"quality": 95})
        pipeline.add("tcpclientsink", f"camera_{index}_tcp", {
            "host": "127.0.0.1", "port": 7001 + index, "sync": False,
        })

    pipeline.link("ground_floor_sources", "yolo26s", "staff_color_filter", "nvdcf_local", "camera_demux")
    for index in range(len(camera_ids)):
        pipeline.link(("camera_demux", f"camera_{index}_queue"), (f"src_{index}", ""))
        pipeline.link(
            f"camera_{index}_queue", f"camera_{index}_rgba", f"camera_{index}_rgba_caps",
            f"camera_{index}_osd",
            f"camera_{index}_convert", f"camera_{index}_rgb",
            f"camera_{index}_jpeg", f"camera_{index}_tcp",
        )
    print("NVIDIA DeepStream: ordered six-camera NVR-225 source list [7,16,10,18,20,11]", flush=True)
    print("nvmultiurisrcbin(NVDEC/NVMM, source batch=6) -> YOLO26s FP16 -> staff filter -> NvDCF/Re-ID -> demux -> per-camera GPU RGBA -> per-camera GPU OSD -> 1280x720 RGB -> JPEG", flush=True)
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
