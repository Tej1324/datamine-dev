# Datamine DeepStream viewer

Six-camera NVIDIA DeepStream pipeline for channels 7, 16, 10, 18, 20, and 11.
The runtime path is:

```text
RTSP/NVDEC → YOLO26s FP16 → staff filter → NvDCF/Re-ID → per-camera OSD → GPU JPEG → viewer
```

Global Identity is intentionally not part of this project. The dashboard uses
local tracker IDs only.

## Prerequisites

- NVIDIA GPU and driver with Docker runtime support
- Docker image `datamine-deepstream:9.1-gpuviewer`
- DeepStream 9.1 runtime
- YOLO engine/parser and NvDCF/Re-ID engine placed under `models/`
- Runtime credentials copied from `.env.example` to an untracked `.env`

Never commit `.env`, camera credentials, model weights, TensorRT engines, or
recorded video. These are ignored by design.

## Build and run

```bash
docker build -f Dockerfile.deepstream -t datamine-deepstream:9.1-gpuviewer .
bash start_ground_floor.sh
```

The viewer is available at `http://127.0.0.1:8080/?v=mjpeg`. When using SSH,
forward port 8080 to the local machine.

## Staff classifier dataset

The current production filter remains the legacy color filter until a trained
classifier is validated. Run the offline labeler on the host:

```bash
python3 tools/staff_filter_labeler.py
python3 tools/prepare_staff_classifier_dataset.py
```

The dataset validator requires both `staff` and `customer` examples from all
six cameras. It keeps samples from one source frame in the same split.

## Plugin build

The custom staff filter is a DeepStream GStreamer plugin. Build it inside the
DeepStream container with:

```bash
bash scripts/build_plugins.sh
```

The output is written to the ignored `build/` directory.
