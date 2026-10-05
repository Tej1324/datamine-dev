#!/usr/bin/env bash
set -Eeuo pipefail

# Native NVIDIA MV3DT live path.
# The older custom C++ Global-ID path remains in
# scripts/start_mv3dt_live_ch10_ch11.sh and is intentionally not modified.

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"

calibration_dir="${project_dir}/${MV3DT_CALIBRATION_DIR:-calibration/ground_floor_ch10_ch11_20260928T064948Z}"
experiment_dir="${calibration_dir}/mv3dt/experiment"
dataset_dir="${calibration_dir}/mv3dt/dataset"
container_name="native-mv3dt-ch10-ch11"
mqtt_name="datamine-mv3dt-mqtt"
live_width="${MV3DT_LIVE_WIDTH:-1280}"
live_height="${MV3DT_LIVE_HEIGHT:-720}"
live_subtype="${MV3DT_STREAM_SUBTYPE:-}"
live_caminfo_dir="${experiment_dir}/camInfo_live_${live_width}x${live_height}"
viewer_log="/tmp/datamine-mv3dt-viewer.log"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
world_identity_output="${MV3DT_WORLD_IDENTITY_OUTPUT:-/workspace/runs/mv3dt_world_identity_${run_stamp}.jsonl}"
# The default live path is NVIDIA's native MV3DT peer-tracklet associator.
# The older custom world-identity matcher remains available only by explicitly
# setting MV3DT_WORLD_IDENTITY_ENABLE=1 and selecting its config.
tracker_config_container="${NATIVE_TRACKER_CONFIG:-/workspace/experiments/config_tracker_live.yml}"
if [[ "${tracker_config_container}" != /* ]]; then
  tracker_config_container="/workspace/experiments/${tracker_config_container}"
fi

# The native launcher is the only production Global-ID owner.  Do not let a
# stale diagnostic/custom config silently turn this into a local-ID viewer.
# Experimental runs can still opt in explicitly from their dedicated wrapper.
if [[ "${MV3DT_ALLOW_NONPRODUCTION_CONFIG:-0}" != "1" ]]; then
  [[ "${tracker_config_container}" == "/workspace/experiments/config_tracker_live.yml" ]] || {
    echo "Refusing non-production tracker config: ${tracker_config_container}" >&2
    echo "Use scripts/start_reid_experiment_ch10_ch11.sh or set MV3DT_ALLOW_NONPRODUCTION_CONFIG=1 for diagnostics." >&2
    exit 2
  }
fi

using_default_tracker_config=0
if [[ "${tracker_config_container}" == "/workspace/experiments/config_tracker_live.yml" ]]; then
  using_default_tracker_config=1
fi

for required in \
  "${dataset_dir}/camInfo/cam_00.yml" \
  "${dataset_dir}/camInfo/cam_01.yml" \
  "${experiment_dir}/config_tracker_live.yml" \
  "${experiment_dir}/pub_sub_info_config_sensor_names.yml" \
  "${experiment_dir}/config_mqtt.txt"; do
  [[ -f "${required}" ]] || { echo "Missing MV3DT file: ${required}" >&2; exit 2; }
done

if [[ "${MV3DT_ALLOW_NONPRODUCTION_CONFIG:-0}" != "1" ]]; then
  grep -Eq '^  enableMsgSync:[[:space:]]*1([[:space:]]|$)' "${experiment_dir}/config_tracker_live.yml" || {
    echo "Production config must enable MV3DT message synchronization." >&2; exit 2;
  }
  grep -Eq '^  communicatorType:[[:space:]]*2([[:space:]]|$)' "${experiment_dir}/config_tracker_live.yml" || {
    echo "Production config must use the MQTT communicator." >&2; exit 2;
  }
  grep -Eq '^  reidType:[[:space:]]*2([[:space:]]|$)' "${experiment_dir}/config_tracker_live.yml" || {
    echo "Production NvDCF config must use Re-ID target re-association (reidType 2)." >&2; exit 2;
  }
fi

if [[ "${MV3DT_ALLOW_EXPERIMENTAL:-0}" != "1" ]] && {
  [[ "${GLOBAL_ID_CPP_ENABLE:-0}" == "1" ]] ||
  [[ "${MV3DT_WORLD_IDENTITY_ENABLE:-0}" == "1" ]] ||
  [[ "${REID_EXPERIMENT_CROP_BRIDGE:-0}" == "1" ]];
}; then
  echo "Refusing experimental/custom Global-ID overlay in the production launcher." >&2
  echo "Use the dedicated experimental launcher or set MV3DT_ALLOW_EXPERIMENTAL=1." >&2
  exit 2
fi

docker rm -f "${container_name}" 2>/dev/null || true

# The completed VGGT calibration is in 1920x1080 camera pixels. The default
# path scales projection rows for the production substream. Set
# MV3DT_STREAM_SUBTYPE=0 and MV3DT_LIVE_WIDTH=1920 MV3DT_LIVE_HEIGHT=1080
# to decode the main stream and consume the calibration without scaling.
mkdir -p "${live_caminfo_dir}"
generated_tracker_config_host="${experiment_dir}/config_tracker_live_${live_width}x${live_height}.yml"
generated_tracker_config_container="/workspace/experiments/config_tracker_live_${live_width}x${live_height}.yml"
python3 - "${dataset_dir}/camInfo" "${live_caminfo_dir}" "${live_width}" "${live_height}" \
  "${experiment_dir}/config_tracker_live.yml" "${generated_tracker_config_host}" <<'PY'
from pathlib import Path
import sys
import yaml

source_dir, target_dir = map(Path, sys.argv[1:3])
live_width, live_height = map(float, sys.argv[3:5])
source_config, target_config = map(Path, sys.argv[5:7])
sx, sy = live_width / 1920.0, live_height / 1080.0
for index in range(2):
    source = source_dir / f"cam_{index:02d}.yml"
    target = target_dir / source.name
    document = yaml.safe_load(source.read_text())
    matrix = document["projectionMatrix_3x4_w2p"]
    document["projectionMatrix_3x4_w2p"] = [
        value * (sx if row == 0 else sy if row == 1 else 1.0)
        for row in range(3) for value in matrix[row * 4:(row + 1) * 4]
    ]
    target.write_text(yaml.safe_dump(document, sort_keys=False))
config_text = source_config.read_text()
config_text = config_text.replace(
    "/workspace/experiments/camInfo_live/",
    f"/workspace/experiments/{target_dir.name}/",
)
target_config.write_text(config_text)
PY

if [[ "${using_default_tracker_config}" == "1" ]]; then
  tracker_config_container="${generated_tracker_config_container}"
fi

# Stop only the custom producer. The dashboard viewer is shared by both paths.
docker rm -f ground-floor-calibrated-reid 2>/dev/null || true

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

# Native MV3DT uses MQTT for peer tracklets and global-ID propagation.
docker rm -f "${mqtt_name}" 2>/dev/null || true
docker run -d --restart unless-stopped --name "${mqtt_name}" --network host \
  -v "${project_dir}/config/mosquitto-mv3dt.conf:/mosquitto/config/mv3dt.conf:ro" \
  eclipse-mosquitto:2.0.20 mosquitto -c /mosquitto/config/mv3dt.conf -v >/dev/null

docker run -d --restart unless-stopped --name "${container_name}" \
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
  -e TRACKER_CONFIG="${tracker_config_container}" \
  -e GLOBAL_ID_CPP_ENABLE=0 \
  -e GLOBAL_ID_ENABLE=0 \
  -e GLOBAL_ID_OSD="${GLOBAL_ID_OSD:-0}" \
  -e STAFF_REID_FILTER_ENABLE=0 \
  -e STAFF_COLOR_FILTER_ENABLE=0 \
  -e LIVE_TRANSPORT=mjpeg \
  -e LIVE_OVERVIEW_WIDTH="${LIVE_OVERVIEW_WIDTH:-1280}" \
  -e LIVE_OVERVIEW_HEIGHT="${LIVE_OVERVIEW_HEIGHT:-360}" \
  -e LIVE_JPEG_QUALITY="${LIVE_JPEG_QUALITY:-95}" \
  -e STAGE_DIAGNOSTICS="${STAGE_DIAGNOSTICS:-0}" \
  -e MV3DT_GLOBAL_ID_VERIFY="${MV3DT_GLOBAL_ID_VERIFY:-1}" \
  -e MV3DT_GLOBAL_ID_OSD="${MV3DT_GLOBAL_ID_OSD:-1}" \
  -e MV3DT_JOURNEY_OUTPUT="${MV3DT_JOURNEY_OUTPUT:-/workspace/runs/mv3dt_journeys_${run_stamp}.jsonl}" \
  -e MV3DT_ASSOC_DIAGNOSTIC="${MV3DT_ASSOC_DIAGNOSTIC:-0}" \
  -e REID_DIAGNOSTIC="${REID_DIAGNOSTIC:-0}" \
  -e REID_DIAGNOSTIC_OUTPUT="${REID_DIAGNOSTIC_OUTPUT:-/workspace/runs/reid_diagnostic_current.jsonl}" \
  -e MV3DT_WORLD_IDENTITY_OUTPUT="${world_identity_output}" \
  -e MV3DT_WORLD_IDENTITY_ENABLE="${MV3DT_WORLD_IDENTITY_ENABLE:-0}" \
  -e MV3DT_WORLD_IDENTITY_MAX_DISTANCE="${MV3DT_WORLD_IDENTITY_MAX_DISTANCE:-8.00}" \
  -e MV3DT_WORLD_IDENTITY_MAX_GAP_SECONDS="${MV3DT_WORLD_IDENTITY_MAX_GAP_SECONDS:-8.00}" \
  -e MV3DT_WORLD_IDENTITY_SYNC_WINDOW_SECONDS="${MV3DT_WORLD_IDENTITY_SYNC_WINDOW_SECONDS:-0.75}" \
  -e MV3DT_WORLD_IDENTITY_REID_WEIGHT="${MV3DT_WORLD_IDENTITY_REID_WEIGHT:-0.80}" \
  -e MV3DT_WORLD_IDENTITY_MIN_REID_SIMILARITY="${MV3DT_WORLD_IDENTITY_MIN_REID_SIMILARITY:-0.70}" \
  -e MV3DT_WORLD_IDENTITY_MERGE_REID_SIMILARITY="${MV3DT_WORLD_IDENTITY_MERGE_REID_SIMILARITY:-0.75}" \
  -e MV3DT_WORLD_IDENTITY_CANDIDATE_MARGIN="${MV3DT_WORLD_IDENTITY_CANDIDATE_MARGIN:-0.05}" \
  -e MV3DT_WORLD_IDENTITY_MERGE_CANDIDATE_MARGIN="${MV3DT_WORLD_IDENTITY_MERGE_CANDIDATE_MARGIN:-0.08}" \
  -e MV3DT_WORLD_IDENTITY_MATCH_CONFIRMATIONS="${MV3DT_WORLD_IDENTITY_MATCH_CONFIRMATIONS:-3}" \
  -e MV3DT_WORLD_IDENTITY_MERGE_CONFIRMATIONS="${MV3DT_WORLD_IDENTITY_MERGE_CONFIRMATIONS:-5}" \
  -e MV3DT_WORLD_IDENTITY_BIRTH_CONFIRMATIONS="${MV3DT_WORLD_IDENTITY_BIRTH_CONFIRMATIONS:-3}" \
  -e MV3DT_WORLD_IDENTITY_BIRTH_TIMEOUT_SECONDS="${MV3DT_WORLD_IDENTITY_BIRTH_TIMEOUT_SECONDS:-1.50}" \
  -e MV3DT_WORLD_IDENTITY_STRONG_GEOMETRY_DISTANCE="${MV3DT_WORLD_IDENTITY_STRONG_GEOMETRY_DISTANCE:-0.75}" \
  -e REID_EXPERIMENT_CROP_BRIDGE="${REID_EXPERIMENT_CROP_BRIDGE:-0}" \
  -e REID_EXPERIMENT_SOCKET="${REID_EXPERIMENT_SOCKET:-/workspace/runs/reid_experiment/crops.sock}" \
  -e REID_EXPERIMENT_INTERVAL_FRAMES="${REID_EXPERIMENT_INTERVAL_FRAMES:-10}" \
  -e REID_EXPERIMENT_MIN_CROP_WIDTH="${REID_EXPERIMENT_MIN_CROP_WIDTH:-32}" \
  -e REID_EXPERIMENT_MIN_CROP_HEIGHT="${REID_EXPERIMENT_MIN_CROP_HEIGHT:-64}" \
  -e REID_EXPERIMENT_MIN_DETECTION_CONFIDENCE="${REID_EXPERIMENT_MIN_DETECTION_CONFIDENCE:-0.50}" \
  -e REID_EXPERIMENT_QUEUE_SIZE="${REID_EXPERIMENT_QUEUE_SIZE:-128}" \
  -e GST_PLUGIN_PATH=/workspace/build \
  -e GST_DEBUG="${GST_DEBUG:-}" \
  -e LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream-9.1/lib:/root/.local/lib:/opt/tritonserver/lib:/opt/tritonclient/lib:/usr/src/tensorrt/lib:/opt/riva/lib:/usr/local/cuda-13/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.local/lib \
  -v "${project_dir}:/workspace" \
  -v "${dataset_dir}:/workspace/inputs" \
  -v "${experiment_dir}:/workspace/experiments" \
  -w /workspace \
  --entrypoint /bin/bash \
  datamine-deepstream:9.1-gpuviewer \
  -lc 'if [[ "${MV3DT_ASSOC_DIAGNOSTIC}" == "1" || "${MV3DT_WORLD_IDENTITY_ENABLE}" == "1" || "${GLOBAL_ID_CPP_ENABLE}" == "1" ]]; then bash scripts/build_plugins.sh; fi; if [[ "${REID_EXPERIMENT_CROP_BRIDGE}" == "1" ]]; then bash scripts/build_reid_experiment_bridge.sh; fi; exec python3 -u deepstream/detect.py'

echo "Native NVIDIA MV3DT dashboard: http://127.0.0.1:8080/?v=mjpeg"
echo "Container: ${container_name}"
echo "MQTT: ${mqtt_name} on 127.0.0.1:1883"
echo "DeepStream log: docker logs -f ${container_name}"
echo "Stop: docker rm -f ${container_name} ${mqtt_name}"
