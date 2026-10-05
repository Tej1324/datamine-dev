"""Six-camera DeepStream display pipeline using NVIDIA's standard multi-URI bin."""

import os
import sys
from pathlib import Path
from urllib.parse import quote

import yaml
from pyservicemaker import Pipeline


ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ROOT / "config/cameras.yaml"
DETECTOR = Path(os.getenv("DETECTOR_CONFIG", str(ROOT / "config/detector.txt")))
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
    configured_channels = [camera["channel"] for camera in cameras]
    if len(cameras) != len(expected) or configured_channels != expected:
        raise ValueError("requires the authoritative NVR-225 camera order 7,16,10,18,20,11")
    if {camera["nvr"] for camera in cameras} != {225}:
        raise ValueError("requires NVR-225 sources")
    selected = os.getenv("MV3DT_CAMERA_IDS", "").strip()
    if selected:
        requested = [item.strip() for item in selected.split(",") if item.strip()]
        by_id = {camera["id"]: camera for camera in cameras}
        missing = [camera_id for camera_id in requested if camera_id not in by_id]
        if missing:
            raise ValueError(f"unknown MV3DT camera IDs: {','.join(missing)}")
        cameras = [by_id[camera_id] for camera_id in requested]
    subtype_override = os.getenv("STREAM_SUBTYPE_OVERRIDE", "").strip()
    if subtype_override:
        subtype = int(subtype_override)
        cameras = [{**camera, "subtype": subtype} for camera in cameras]
    return cameras


def make_url(camera, values):
    user = quote(values["CAMERA_USERNAME"], safe="")
    password = quote(values["CAMERA_PASSWORD"], safe="")
    return (f"rtsp://{user}:{password}@{values['NVR_225_HOST']}:{values['NVR_225_PORT']}"
            f"/cam/realmonitor?channel={int(camera['channel'])}"
            f"&subtype={int(camera['subtype'])}")


def env_bool(name, default=False):
    return os.getenv(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


def staff_color_settings():
    settings = yaml.safe_load(STAFF_FILTER_CONFIG.read_text()) or {}
    return {
        "profile_path": str(ROOT / settings.get("color_profile_path", "data/staff_filter/profile.yml")),
        "threshold": float(settings.get("color_threshold", 0.615878)),
    }


def main():
    cameras = load_cameras()
    values = env_file()
    offline_video_dir = os.getenv("OFFLINE_VIDEO_DIR", "").strip()
    if offline_video_dir:
        # Opt-in diagnostic mode only. Production remains on the NVR RTSP
        # sources when this variable is unset.
        video_root = Path(offline_video_dir)
        urls = [f"file://{video_root / f'cam_{index:02d}.mp4'}"
                for index, _ in enumerate(cameras)]
    else:
        urls = [make_url(camera, values) for camera in cameras]
    camera_ids = [camera["id"] for camera in cameras]
    staff_settings = staff_color_settings()
    os.environ.setdefault("STAFF_ENROLLMENT_OUTPUT", "/workspace/runs/staff_enrollment_live.json")
    source_width = int(os.getenv("SOURCE_WIDTH", os.getenv("TRACKER_WIDTH", "1280")))
    source_height = int(os.getenv("SOURCE_HEIGHT", os.getenv("TRACKER_HEIGHT", "720")))

    pipeline = Pipeline("ground-floor-deepstream")
    pipeline.add("nvmultiurisrcbin", "ground_floor_sources", {
        "uri-list": ",".join(urls), "sensor-id-list": ",".join(camera_ids),
        "sensor-name-list": ",".join(camera_ids), "mode": 0,
        "max-batch-size": int(os.getenv("SOURCE_MAX_BATCH_SIZE", str(len(cameras)))),
        "width": source_width, "height": source_height,
        "live-source": not bool(offline_video_dir),
        # The six RTSP feeds run at approximately 10 FPS.  NVIDIA's live
        # streammux guidance is to allow roughly one source frame period for
        # batch formation; 30 ms created partial/jittered batches here.
        "batched-push-timeout": 100000,
        "drop-pipeline-eos": not bool(offline_video_dir),
        "select-rtp-protocol": 4, "latency": 100, "drop-on-latency": True,
        # Keep a transient NVR/RTSP interruption from terminating the native
        # MV3DT process.  This is especially important for a live pair: one
        # dead source must reconnect instead of leaving the peer matcher with
        # only one camera.
        "init-rtsp-reconnect-interval": 5,
        "rtsp-reconnect-interval": 10,
        "rtsp-reconnect-attempts": -1,
        "cudadec-memtype": 0, "disable-audio": True, "leaky": 2,
        "max-size-buffers": 1, "port": "0",
    })
    # The detector config owns the engine's static batch size.  This keeps the
    # validated batch-5 engine selectable while allowing a separately built
    # batch-6 engine/config to be tested without editing production assets.
    pipeline.add("nvinfer", "yolo26s", {"config-file-path": str(DETECTOR)})
    stage_diagnostics = env_bool("STAGE_DIAGNOSTICS", False)
    experimental_crop_bridge = env_bool("REID_EXPERIMENT_CROP_BRIDGE", False)
    footfall_enabled = env_bool("FOOTFALL_ENABLE", False)
    if stage_diagnostics:
        # GstIdentity exposes a monotonic num-buffers statistic and, with
        # silent=false, emits a per-buffer trace under GST_DEBUG=identity:6.
        # These probes do not copy or modify video and are only enabled for
        # source-stage debugging.
        for name in ("decoded_frames", "post_yolo_frames", "post_tracker_frames", "jpeg_input_frames"):
            pipeline.add("identity", name, {"silent": False})
    tracker_config = Path(os.getenv("TRACKER_CONFIG", str(TRACKER_CONFIG)))
    pipeline.add("nvtracker", "nvdcf_local", {
        "ll-lib-file": TRACKER_LIBRARY, "ll-config-file": str(tracker_config),
        "tracker-width": int(os.getenv("TRACKER_WIDTH", str(source_width))),
        "tracker-height": int(os.getenv("TRACKER_HEIGHT", str(source_height))), "gpu-id": 0,
        # Keep tracker IDs in NvDsObjectMeta for backend/Re-ID/Global-ID
        # processing, but suppress NvTracker's built-in local-ID text when
        # the Global-ID-only display mode is requested.
        "display-tracking-id": (
            os.getenv("GLOBAL_ID_OSD", "0") != "1"
            and os.getenv("FOOTFALL_LOGICAL_ID_OSD", "0") != "1"
            and not experimental_crop_bridge
        ),
        "operate-on-class-ids": "0",
    })
    if stage_diagnostics:
        from pyservicemaker import Probe
        from stage_counter import StageCounter

        stage_counters = {
            element: StageCounter(label)
            for element, label in (
                ("ground_floor_sources", "decoded/source"),
                ("yolo26s", "post-yolo"),
                ("nvdcf_local", "post-tracker"),
            )
        }
        for element, counter in stage_counters.items():
            pipeline.attach(element, Probe(f"stage_counter_{element}", counter), tips="src")
    if env_bool("MV3DT_GLOBAL_ID_VERIFY", False):
        from pyservicemaker import Probe
        from global_id_verifier import GlobalIdVerifier

        verifier = GlobalIdVerifier()
        pipeline.attach("nvdcf_local", Probe("mv3dt_global_id_verifier", verifier), tips="src")
        print("MV3DT global-ID metadata verification enabled", flush=True)
    footfall_operator = None
    if footfall_enabled:
        from footfall_counter import FootfallCounter

        footfall_operator = FootfallCounter()
    if env_bool("MV3DT_ASSOC_DIAGNOSTIC", False):
        from pyservicemaker import Probe
        from mv3dt_association_diagnostic import Mv3dtAssociationDiagnostic

        association_diagnostic = Mv3dtAssociationDiagnostic()
        pipeline.attach(
            "nvdcf_local",
            Probe("mv3dt_association_diagnostic", association_diagnostic),
            tips="src",
        )
        print("MV3DT association diagnostic enabled", flush=True)
        pipeline.add("mv3dtdiagnostic", "mv3dt_world_diagnostic", {
            "output-path": os.getenv("MV3DT_WORLD_DIAGNOSTIC_OUTPUT", "/workspace/runs/mv3dt_world.jsonl"),
        })
    world_identity = env_bool("MV3DT_WORLD_IDENTITY_ENABLE", False)
    if world_identity:
        pipeline.add("mv3dtworldidentity", "mv3dt_world_identity", {})
        print("MV3DT calibrated world-identity matcher enabled", flush=True)
    if experimental_crop_bridge:
        # Tracker output is CUDA device memory on this dGPU installation.
        # The isolated bridge needs a CPU-readable surface only while the
        # experiment is enabled, so nvvideoconvert produces CUDA-unified
        # memory before the selected person crops are copied asynchronously.
        pipeline.add("nvvideoconvert", "reid_experiment_surface_copy", {
            "nvbuf-memory-type": 3,
            "gpu-id": 0,
            "output-buffers": 2,
        })
        # This is an opt-in, passthrough branch. The bridge copies only
        # selected object crops into its own bounded worker queue; it never
        # changes production metadata or waits for SOLIDER inference.
        pipeline.add("reidcropbridge", "reid_experiment_crop_bridge", {})
        print("Experimental SOLIDER crop bridge enabled", flush=True)
    color_staff_filter = env_bool("STAFF_COLOR_FILTER_ENABLE", False)
    if color_staff_filter:
        # This is deliberately downstream of NvDCF: the detector/tracker keep
        # the local object ID, while only the downstream display metadata is
        # removed for a color-matched staff track. No Re-ID or Global ID is
        # consulted by this filter.
        pipeline.add("staffcolorfilter", "staff_color_filter", {
            "profile-path": staff_settings["profile_path"],
            "staff-color-threshold": staff_settings["threshold"],
            "enabled": True,
        })
        print("Local NvDCF-ID staff color filter enabled", flush=True)
    tao_staff_classifier = env_bool("STAFF_CLASSIFIER_ENABLE", False)
    if tao_staff_classifier:
        classifier_config = Path(os.getenv(
            "STAFF_CLASSIFIER_CONFIG", str(ROOT / "config/staff_classifier_infer.txt")))
        pipeline.add("nvinfer", "staff_classifier", {
            "config-file-path": str(classifier_config),
        })
        from pyservicemaker import Probe
        from staff_classifier_suppressor import StaffClassifierSuppressor

        classifier_suppressor = StaffClassifierSuppressor()
        pipeline.attach("staff_classifier", Probe("staff_classifier_suppressor", classifier_suppressor), tips="src")
        print("TAO staff/customer classifier enabled; staff OSD suppression enabled", flush=True)
    cpp_global_identity = env_bool("GLOBAL_ID_CPP_ENABLE", False)
    if cpp_global_identity:
        pipeline.add("globalidentity", "global_identity", {})
        print("C++ asynchronous Global-ID worker enabled", flush=True)
    diagnostic = None
    continuity = None
    if env_bool("STAFF_ENROLLMENT_EXPORT_ENABLE", False):
        print("Staff enrollment: live NvDCF/Re-ID snapshot export enabled", flush=True)
    if os.getenv("NVCDF_CONTINUITY_DIAGNOSTIC", "0") == "1":
        from pyservicemaker import Probe
        from nvdcf_continuity import NvdcfContinuityDiagnostic

        continuity = NvdcfContinuityDiagnostic()
        pipeline.attach(
            "nvdcf_local",
            Probe("nvdcf_continuity_diagnostic", continuity),
            tips="src",
        )
        print("NvDCF local-ID continuity diagnostic enabled", flush=True)
    if os.getenv("REID_DIAGNOSTIC", "0") == "1" and not cpp_global_identity:
        from pyservicemaker import Probe
        from reid_diagnostic import ReIdMetadataDiagnostic

        source_labels = {index: camera_id for index, camera_id in enumerate(camera_ids)}
        identity_manager = None
        if os.getenv("GLOBAL_ID_ENABLE", "0") == "1":
            from global_identity import GlobalIdentityManager

            identity_manager = GlobalIdentityManager()
        tracklet_store = None
        if os.getenv("TRACKLET_EMBED_DIAGNOSTIC", "0") == "1" or os.getenv("TRACKLET_REPRESENTATION_DIAGNOSTIC", "0") == "1":
            from tracklet_embeddings import LocalTrackletEmbeddingStore
            representation = None
            if os.getenv("TRACKLET_REPRESENTATION_DIAGNOSTIC", "0") == "1":
                from tracklet_representation import TrackletRepresentation

                representation = TrackletRepresentation()
            tracklet_store = LocalTrackletEmbeddingStore(representation)
        diagnostic = ReIdMetadataDiagnostic(source_labels, identity_manager, tracklet_store)
        pipeline.attach(
            "nvdcf_local",
            Probe("reid_metadata_diagnostic", diagnostic),
            tips="src",
        )
        print("Phase 2A Re-ID metadata diagnostic enabled", flush=True)
    # NVIDIA's tiled-display topology is tracker -> tee -> tiler -> OSD.
    # The tiler first transforms each source's metadata into tile coordinates;
    # the single GPU OSD then draws against that transformed metadata.  This
    # prevents objects from one source being rendered in another tile.
    pipeline.add("tee", "display_tee")
    pipeline.add("queue", "overview_queue", {"leaky": 2, "max-size-buffers": 1})
    overview_width = int(os.getenv("LIVE_OVERVIEW_WIDTH", "1280"))
    # Keep each camera at its native 16:9 shape. Two-camera calibration runs
    # use a 1x2 strip (1280x360); the dashboard letterboxes that GPU JPEG into
    # its 1280x736 stage. Six-camera runs retain the 3x2 layout.
    overview_height = int(os.getenv("LIVE_OVERVIEW_HEIGHT", "480"))
    jpeg_quality = int(os.getenv("LIVE_JPEG_QUALITY", "92"))
    live_transport = os.getenv("LIVE_TRANSPORT", "mjpeg").lower()
    # Tile only the streams actually selected for this process.  Using the
    # full camera-config count here leaves a single-camera run in the first
    # tile of a 2x3 canvas, which makes the rest of the dashboard black and
    # misaligns any line drawn over the visible image.
    active_camera_count = len(camera_ids)
    if active_camera_count <= 1:
        tile_rows, tile_columns = 1, 1
    elif active_camera_count == 2:
        tile_rows, tile_columns = 1, 2
    else:
        tile_rows, tile_columns = 2, 3
    pipeline.add("nvmultistreamtiler", "overview_tiler", {
        "rows": tile_rows, "columns": tile_columns, "width": overview_width, "height": overview_height,
        "gpu-id": 0, "nvbuf-memory-type": 2,
    })
    pipeline.add("nvvideoconvert", "display_rgba", {
        "nvbuf-memory-type": 2, "disable-passthrough": True,
    })
    pipeline.add("capsfilter", "display_rgba_caps", {
        "caps": "video/x-raw(memory:NVMM), format=RGBA",
    })
    pipeline.add("nvdsosd", "display_osd", {
        "process-mode": 1, "display-text": True,
    })
    pipeline.add("nvvideoconvert", "overview_convert", {
        "nvbuf-memory-type": 2, "disable-passthrough": True,
    })
    if live_transport != "mjpeg":
        raise ValueError("LIVE_TRANSPORT must be mjpeg")
    pipeline.add("capsfilter", "overview_rgb", {
        "caps": "video/x-raw(memory:NVMM), format=RGB",
    })
    # NVIDIA nvimageenc encodes this one overview on the GPU. The Flask
    # process only relays completed JPEG byte strings to the browser.
    pipeline.add("nvimageenc", "overview_jpeg", {"quality": jpeg_quality})
    # Keep the network relay from applying a sink clock or back-pressuring the
    # GPU encoder.  The viewer already keeps only the newest JPEG, so a
    # one-buffer leaky queue is the correct live-stream policy here.
    pipeline.add("queue", "overview_jpeg_queue", {
        "leaky": 2, "max-size-buffers": 1,
    })
    pipeline.add("tcpclientsink", "overview_tcp", {
        "host": "127.0.0.1", "port": int(os.getenv("OVERVIEW_TCP_PORT", "7007")),
        "sync": False, "async": False,
    })

    if stage_diagnostics:
        pipeline.link("ground_floor_sources", "decoded_frames", "yolo26s", "post_yolo_frames", "nvdcf_local")
    else:
        pipeline.link("ground_floor_sources", "yolo26s", "nvdcf_local")
    tracker_output = "nvdcf_local"
    if env_bool("MV3DT_ASSOC_DIAGNOSTIC", False):
        pipeline.link(tracker_output, "mv3dt_world_diagnostic")
        tracker_output = "mv3dt_world_diagnostic"
    if world_identity:
        pipeline.link(tracker_output, "mv3dt_world_identity")
        tracker_output = "mv3dt_world_identity"
    if experimental_crop_bridge:
        pipeline.link(tracker_output, "reid_experiment_surface_copy", "reid_experiment_crop_bridge")
        tracker_output = "reid_experiment_crop_bridge"
    if color_staff_filter:
        pipeline.link(tracker_output, "staff_color_filter")
        tracker_output = "staff_color_filter"
    if tao_staff_classifier:
        pipeline.link(tracker_output, "staff_classifier")
        tracker_output = "staff_classifier"
    if footfall_operator is not None:
        from pyservicemaker import Probe

        pipeline.attach(tracker_output, Probe("footfall_counter", footfall_operator), tips="src")
        print("Footfall tracker counter enabled", flush=True)
    if cpp_global_identity:
        if stage_diagnostics:
            pipeline.link(tracker_output, "post_tracker_frames", "global_identity", "display_tee")
        else:
            pipeline.link(tracker_output, "global_identity", "display_tee")
    else:
        if stage_diagnostics:
            pipeline.link(tracker_output, "post_tracker_frames", "display_tee")
        else:
            pipeline.link(tracker_output, "display_tee")
    pipeline.link(
        "overview_queue", "overview_tiler", "display_rgba", "display_rgba_caps",
        "display_osd", "overview_convert", "overview_rgb", "overview_jpeg",
        *( ("jpeg_input_frames",) if stage_diagnostics else () ),
        "overview_jpeg_queue", "overview_tcp",
    )
    if experimental_crop_bridge:
        from pyservicemaker import Probe
        from solider_display import SoliderDisplay
        pipeline.attach('overview_tiler', Probe('solider_current_frame_labels', SoliderDisplay()), tips='src')
    pipeline.link(("display_tee", "overview_queue"), ("src_%u", ""))
    print(f"NVIDIA DeepStream: source list {camera_ids}, subtype override={os.getenv('STREAM_SUBTYPE_OVERRIDE', 'config')}, source={source_width}x{source_height}", flush=True)
    print(f"nvmultiurisrcbin(NVDEC/NVMM) -> YOLO26s FP16 (engine input is resized internally) -> NvDCF/Re-ID -> GPU OSD -> GPU tiled {live_transport} overview", flush=True)
    try:
        pipeline.start().wait()
    finally:
        pipeline.stop()
        if continuity is not None:
            continuity.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
