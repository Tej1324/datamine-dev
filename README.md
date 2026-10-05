# Datamine footfall runtime

Small NVIDIA DeepStream runtime for entrance footfall analytics:

```text
RTSP → NVDEC → YOLO person detector → NvDCF tracker → two-line counter → dashboard
```

The repository intentionally keeps only the YOLO/person-detection, NvDCF,
footfall-counter, and dashboard path. TAO staff classification, SOLIDER/Re-ID,
MV3DT/global IDs, calibration, crop review, and experiment code are not part of
this runtime.

## Requirements

- NVIDIA GPU and Docker with the NVIDIA runtime
- DeepStream image `datamine-deepstream:9.1-gpuviewer`
- YOLO/NvDCF assets under `models/` and the configs under `config/`
- An untracked `.env` containing the RTSP values used by `config/cameras.yaml`:
  `CAMERA_USERNAME`, `CAMERA_PASSWORD`, `NVR_225_HOST`, and `NVR_225_PORT`

`.env`, video, model-weight, TensorRT-engine, log, and runtime-state files are
ignored by Git.

## Run

Start the CH7 entrance pipeline and dashboard:

```bash
bash scripts/start_footfall_ch7.sh
```

Open the dashboard at [http://127.0.0.1:18082](http://127.0.0.1:18082).

For a remote machine, create the tunnel from your laptop:

```bash
ssh -N -L 18082:127.0.0.1:18082 datamine-l4
```

Then open the same local URL in the browser.

Useful commands:

```bash
docker logs -f footfall-ch7
curl http://127.0.0.1:18082/api/state
docker rm -f footfall-ch7
```

## Configure the entrance

The dashboard uses two clicks for each line. Draw and save:

- line 1 → line 2: entry
- line 2 → line 1: exit

The footpoint is the bottom-center of each person box. A crossing is counted
only when the same NvDCF track crosses the two lines in order within the
configured transition window. State is stored in `runs/footfall/` at runtime.

The launcher defaults to camera `ground_01` and stream subtype `1`. Override
these without editing code, for example:

```bash
FOOTFALL_CAMERA_IDS=ground_01 FOOTFALL_SUBTYPE=1 \
  bash scripts/start_footfall_ch7.sh
```

The dashboard supports one camera or a small camera set. For multiple cameras,
the counter keeps source ID and NvDCF track ID separate.

## Runtime files

- `deepstream/detect.py` — DeepStream source, YOLO, NvDCF, OSD, and JPEG stream
- `deepstream/footfall_counter.py` — two-line state machine and persistence
- `tools/footfall_dashboard.py` — line editor, live MJPEG view, and API
- `scripts/start_footfall_ch7.sh` — dashboard/container launcher
- `config/cameras.yaml` — camera channel definitions
- `config/detector_b6.txt` — YOLO detector configuration
- `config/tracker.yml` — NvDCF tracker configuration
