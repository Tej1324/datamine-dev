#!/usr/bin/env bash
set -Eeuo pipefail

# Live CH10/CH11 dashboard using NvDCF plus the repository's calibrated
# cross-camera Global-ID matcher. NVIDIA MV3DT association and MQTT are off.

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"

calibration_dir="${project_dir}/${MV3DT_CALIBRATION_DIR:-calibration/ground_floor_ch10_ch11_20260928T064948Z}"
experiment_dir="${calibration_dir}/mv3dt/experiment"
dataset_dir="${calibration_dir}/mv3dt/dataset"
container_name="ground-floor-calibrated-reid"
viewer_log="/tmp/datamine-mv3dt-viewer.log"
live_width="${MV3DT_LIVE_WIDTH:-1280}"
live_height="${MV3DT_LIVE_HEIGHT:-720}"
live_subtype="${MV3DT_STREAM_SUBTYPE:-}"
live_caminfo_dir="${experiment_dir}/camInfo_live_${live_width}x${live_height}"

for required in \
  "${dataset_dir}/camInfo/cam_00.yml" \
  "${dataset_dir}/camInfo/cam_01.yml"; do
  [[ -f "${required}" ]] || { echo "Missing MV3DT file: ${required}" >&2; exit 2; }
done

# This launcher intentionally does not start NVIDIA MV3DT or MQTT. The
# calibrated matrices are consumed by our own global-identity matcher below.
docker rm -f "${container_name}" 2>/dev/null || true

# The completed VGGT calibration is in 1920x1080 camera pixels. The default
# path scales projection rows for the production substream. Set
# MV3DT_STREAM_SUBTYPE=0 and MV3DT_LIVE_WIDTH=1920 MV3DT_LIVE_HEIGHT=1080
# to decode the main stream and consume the calibration without scaling.
mkdir -p "${live_caminfo_dir}"
python3 - "${dataset_dir}/camInfo" "${live_caminfo_dir}" "${live_width}" "${live_height}" <<'PY'
from pathlib import Path
import sys
import yaml

source_dir, target_dir = map(Path, sys.argv[1:3])
live_width, live_height = map(float, sys.argv[3:5])
sx, sy = live_width / 1920.0, live_height / 1080.0
for index in (0, 1):
    source = source_dir / f"cam_{index:02d}.yml"
    target = target_dir / source.name
    document = yaml.safe_load(source.read_text())
    matrix = document["projectionMatrix_3x4_w2p"]
    document["projectionMatrix_3x4_w2p"] = [
        value * (sx if row == 0 else sy if row == 1 else 1.0)
        for row in range(3) for value in matrix[row * 4:(row + 1) * 4]
    ]
    target.write_text(yaml.safe_dump(document, sort_keys=False))
PY
# Stop the previous six-camera producer so this calibrated two-camera pipeline
# is the single live producer.
docker rm -f ground-floor-gpu-viewer 2>/dev/null || true

if ! curl -fsS http://127.0.0.1:8080/ >/dev/null 2>&1; then
  setsid nohup env VIEWER_PORT=8080 LIVE_TRANSPORT=mjpeg \
    /usr/bin/python3 viewer/server.py >"${viewer_log}" 2>&1 </dev/null &
  for _ in {1..20}; do
    curl -fsS http://127.0.0.1:8080/ >/dev/null 2>&1 && break
    sleep 0.5
  done
fi
curl -fsS http://127.0.0.1:8080/ >/dev/null 2>&1 || {
  echo "Dashboard viewer is not available; see ${viewer_log}" >&2
  exit 1
}

# The two-camera pipeline renumbers its sources to 0 and 1. The VGGT files
# are still cam_00/cam_01; register them against these runtime IDs.
docker run -d --name "${container_name}" \
  --runtime nvidia --gpus all --network host \
  --env-file .env \
  -e MV3DT_CAMERA_IDS="${MV3DT_CAMERA_IDS:-ground_03,ground_06}" \
  -e DETECTOR_CONFIG=/workspace/config/detector_b6.txt \
  -e SOURCE_MAX_BATCH_SIZE=2 \
  -e SOURCE_WIDTH="${live_width}" \
  -e SOURCE_HEIGHT="${live_height}" \
  -e TRACKER_WIDTH="${live_width}" \
  -e TRACKER_HEIGHT="${live_height}" \
  -e STREAM_SUBTYPE_OVERRIDE="${live_subtype}" \
  -e TRACKER_CONFIG=/workspace/config/tracker.yml \
  -e GLOBAL_REID_CALIBRATION_ENABLE=1 \
  -e GLOBAL_REID_CALIBRATION_DIR="/workspace/experiments/$(basename "${live_caminfo_dir}")" \
  -e GLOBAL_REID_CALIBRATION_SOURCES=0,1 \
  -e GLOBAL_REID_IDENTITY_CAMERAS=0,1 \
  -e GLOBAL_REID_SIMULTANEOUS_CAMERA_PAIRS=0-1 \
  -e GLOBAL_REID_MAX_GAP_SECONDS=60 \
  -e GLOBAL_REID_CROSS_CAMERA_MATCH_THRESHOLD=0.78 \
  -e GLOBAL_REID_CROSS_CAMERA_MIN_MARGIN=0.10 \
  -e GLOBAL_REID_CROSS_CAMERA_CONFIRMATIONS=5 \
  -e GLOBAL_REID_CROSS_CAMERA_MIN_QUALITY=0.55 \
  -e GLOBAL_REID_ID_MERGE_THRESHOLD=0.80 \
  -e GLOBAL_REID_ID_MERGE_MARGIN=0.03 \
  -e GLOBAL_REID_ID_MERGE_MAX_GAP_SECONDS=3.0 \
  -e GLOBAL_REID_CALIBRATION_MAX_DISTANCE="${GLOBAL_REID_CALIBRATION_MAX_DISTANCE:-1.5}" \
  -e GLOBAL_REID_CALIBRATION_PAIR_MAX_DISTANCE="${GLOBAL_REID_CALIBRATION_PAIR_MAX_DISTANCE:-0.60}" \
  -e GLOBAL_REID_CALIBRATION_MAX_GAP_SECONDS="${GLOBAL_REID_CALIBRATION_MAX_GAP_SECONDS:-2.5}" \
  -e GLOBAL_REID_CALIBRATION_SCORE_WEIGHT="${GLOBAL_REID_CALIBRATION_SCORE_WEIGHT:-0.55}" \
  -e STAFF_REID_FILTER_ENABLE=0 \
  -e STAFF_COLOR_FILTER_ENABLE=0 \
  -e GLOBAL_ID_CPP_ENABLE=1 \
  -e GLOBAL_ID_ENABLE=0 \
  -e REID_DIAGNOSTIC=0 \
  -e LIVE_TRANSPORT=mjpeg \
  -e LIVE_OVERVIEW_WIDTH="${LIVE_OVERVIEW_WIDTH:-1280}" \
  -e LIVE_OVERVIEW_HEIGHT="${LIVE_OVERVIEW_HEIGHT:-360}" \
  -e LIVE_JPEG_QUALITY="${LIVE_JPEG_QUALITY:-95}" \
  -e GST_PLUGIN_PATH=/workspace/build \
  -e LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream-9.1/lib:/root/.local/lib:/opt/tritonserver/lib:/opt/tritonclient/lib:/usr/src/tensorrt/lib:/opt/riva/lib:/usr/local/cuda-13/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.local/lib \
  -v "${project_dir}:/workspace" \
  -v "${dataset_dir}:/workspace/inputs" \
  -v "${experiment_dir}:/workspace/experiments" \
  -w /workspace \
  --entrypoint /bin/bash \
  datamine-deepstream:9.1-gpuviewer \
  -lc 'bash scripts/build_plugins.sh && exec python3 -u deepstream/detect.py'

echo "Calibrated Re-ID live dashboard: http://127.0.0.1:8080/?v=mjpeg"
echo "Container: ${container_name}"
echo "DeepStream log: docker logs -f ${container_name}"
echo "Stop: docker rm -f ${container_name}"
