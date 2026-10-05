# Datamine DeepStream viewer

Six-camera NVIDIA DeepStream pipeline for channels 7, 16, 10, 18, 20, and 11.
The runtime path is:

```text
RTSP/NVDEC → YOLO26s FP16 → NvDCF/Re-ID → optional staff gate → GPU OSD → NVIDIA GPU JPEG → loopback MJPEG browser viewer
```

Every YOLO person reaches NvDCF. Staff filtering is a post-tracker,
positive-only NVIDIA Re-ID gallery decision, so uncertain people remain
visible and local tracking is preserved.

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

The viewer is available at `http://127.0.0.1:8080/?v=mjpeg`. The dashboard
uses one NVIDIA `nvimageenc` GPU JPEG overview and a lightweight loopback
multipart-MJPEG relay. When using SSH, forward only the dashboard port:

```bash
ssh -N \
  -L 18080:127.0.0.1:8080 \
  datamine-l4
```

Open `http://127.0.0.1:18080/?v=mjpeg`.

## Staff enrollment

Run the live enrollment exporter and UI together. The exporter is downstream
of NvDCF and publishes tracked boxes plus full NVIDIA Re-ID embeddings. The UI
shows one camera at a time from each six-camera capture; click every staff box,
choose `NO STAFF` when appropriate, and continue. You never enter or remember
local tracker IDs, and customers are never used as a training class.

Terminal 1:

```bash
STAFF_ENROLLMENT_EXPORT_ENABLE=1 bash start_ground_floor.sh
```

Terminal 2:

```bash
python3 tools/staff_filter_labeler.py
```

Open `http://127.0.0.1:8780`. Each saved positive selection automatically
updates `data/staff_filter/reid_gallery.tsv`. Collect front, side, back, near,
and distant views, then enable the pooled gallery gate:

```bash
STAFF_REID_FILTER_ENABLE=1 bash start_ground_floor.sh
```

The gate requires three high-quality matches within five sampled observations.
Confirmed staff objects are removed only from downstream display/output
metadata; NvDCF local tracking and Global ID remain intact. Uncertain people
remain visible.

## Plugin build

The custom staff filter is a DeepStream GStreamer plugin. Build it inside the
DeepStream container with:

```bash
bash scripts/build_plugins.sh
```

The output is written to the ignored `build/` directory. A running process
must be restarted to load a newly built shared library; building alone does
not replace the library loaded by an existing process.
