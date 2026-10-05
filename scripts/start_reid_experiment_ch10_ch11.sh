#!/usr/bin/env bash
set -euo pipefail

# Explicit live SOLIDER experiment. It leaves NVIDIA MV3DT's own Global-ID
# assignment intact and adds only the opt-in crop bridge + browser E-ID layer.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"
run_dir="${project_dir}/runs/reid_experiment"
dataset_id="$(date -u +%Y%m%dT%H%M%SZ)"
dataset_dir="${run_dir}/datasets/${dataset_id}"
socket_host="${run_dir}/crops.sock"
worker_log="${run_dir}/worker.log"
pid_file="${run_dir}/worker.pid"
camera_models="${project_dir}/${MV3DT_CALIBRATION_DIR:-calibration/ground_floor_ch10_ch11_20260928T064948Z}/mv3dt/dataset/camInfo"
floorplan_calibration="${REID_EXPERIMENT_2D_CALIBRATION:-}"
[[ "${MV3DT_CAMERA_IDS:-ground_03,ground_06}" == "ground_03,ground_06" ]] || { echo 'VGGT mapping requires CH10 then CH11'; exit 2; }
for camera in cam_00 cam_01; do
  [[ -s "${camera_models}/${camera}.yml" ]] || { echo "Missing VGGT camera: ${camera}"; exit 2; }
done
venv_python="/home/ubuntu/venvs/reid-experiment/bin/python"

[[ -x "${venv_python}" ]] || { echo "Missing isolated SOLIDER venv: ${venv_python}" >&2; exit 2; }
mkdir -p "${run_dir}/crops/CH10" "${run_dir}/crops/CH11"
rm -f "${socket_host}" "${run_dir}/live_state.json"
if [[ -f "${pid_file}" ]] && kill -0 "$(<"${pid_file}")" 2>/dev/null; then
  kill "$(<"${pid_file}")"
fi

cuda_libs="/home/ubuntu/venvs/reid-experiment/lib/python3.12/site-packages/nvidia"
# 0.74 increased recall but allowed visually similar shoppers to steal an
# existing identity. Use a stricter store-calibrated default and require
# repeated evidence before a cross-camera assignment.
setsid env \
  PYTHONPATH="${project_dir}:${project_dir}/vendor/reid_experiment/SOLIDER-REID" \
  LD_LIBRARY_PATH="${cuda_libs}/cudnn/lib:${cuda_libs}/cublas/lib:${cuda_libs}/cuda_runtime/lib:${cuda_libs}/cuda_nvrtc/lib:${cuda_libs}/cuda_cupti/lib:${cuda_libs}/nvtx/lib:${cuda_libs}/nccl/lib:${LD_LIBRARY_PATH:-}" \
  REID_EXPERIMENT_SIMILARITY_THRESHOLD="${REID_EXPERIMENT_SIMILARITY_THRESHOLD:-0.90}" \
  REID_EXPERIMENT_MIN_MARGIN="${REID_EXPERIMENT_MIN_MARGIN:-0.05}" \
  REID_EXPERIMENT_MATCH_CONFIRMATIONS="${REID_EXPERIMENT_MATCH_CONFIRMATIONS:-3}" \
  REID_EXPERIMENT_MAX_WORLD_DISTANCE="${REID_EXPERIMENT_MAX_WORLD_DISTANCE:-1.25}" \
  REID_EXPERIMENT_MAX_GAP_SECONDS="${REID_EXPERIMENT_MAX_GAP_SECONDS:-3600}" \
  REID_EXPERIMENT_DATASET_DIR="${REID_EXPERIMENT_DATASET_DIR:-${dataset_dir}}" \
  REID_EXPERIMENT_DATASET_LIMIT="${REID_EXPERIMENT_DATASET_LIMIT:-20000}" \
  REID_EXPERIMENT_DISABLE_WORLD_GATE="${REID_EXPERIMENT_DISABLE_WORLD_GATE:-0}" \
  REID_EXPERIMENT_VGGT=1 \
  REID_EXPERIMENT_IMAGE_WIDTH="${MV3DT_LIVE_WIDTH:-1280}" \
  REID_EXPERIMENT_IMAGE_HEIGHT="${MV3DT_LIVE_HEIGHT:-720}" \
  REID_EXPERIMENT_2D_CALIBRATION="${floorplan_calibration}" \
  REID_EXPERIMENT_CH11_ANCHORED="${REID_EXPERIMENT_CH11_ANCHORED:-0}" \
  REID_EXPERIMENT_ANCHOR_SOURCE="${REID_EXPERIMENT_ANCHOR_SOURCE:-1}" \
  REID_EXPERIMENT_MATCH_THRESHOLD="${REID_EXPERIMENT_MATCH_THRESHOLD:-0.90}" \
  "${venv_python}" -u deepstream/reid_experiment_crop_worker.py \
    --socket "${socket_host}" --output "${run_dir}/results.jsonl" \
    --sample-dir "${run_dir}/crops" --camera-model-dir "${camera_models}" \
    --live-state "${run_dir}/live_state.json" --queue-size "${REID_EXPERIMENT_QUEUE_SIZE:-128}" \
    --batch-size "${REID_BATCH_SIZE:-1}" --device cuda >"${worker_log}" 2>&1 < /dev/null &
echo $! >"${pid_file}"

for _ in {1..40}; do
  [[ -S "${socket_host}" ]] && break
  sleep 0.25
done
[[ -S "${socket_host}" ]] || { tail -40 "${worker_log}" >&2; exit 1; }

REID_EXPERIMENT_CROP_BRIDGE=1 \
MV3DT_ALLOW_EXPERIMENTAL=1 \
MV3DT_ALLOW_NONPRODUCTION_CONFIG=1 \
GLOBAL_ID_OSD=1 \
MV3DT_GLOBAL_ID_VERIFY=0 \
MV3DT_GLOBAL_ID_OSD=0 \
REID_EXPERIMENT_SOCKET=/workspace/runs/reid_experiment/crops.sock \
REID_EXPERIMENT_INTERVAL_FRAMES="${REID_EXPERIMENT_INTERVAL_FRAMES:-10}" \
REID_EXPERIMENT_MIN_DETECTION_CONFIDENCE="${REID_EXPERIMENT_MIN_DETECTION_CONFIDENCE:-0.50}" \
NATIVE_TRACKER_CONFIG="${NATIVE_TRACKER_CONFIG:-config_tracker_live_nomsgsync.yml}" \
bash scripts/start_native_mv3dt_live_ch10_ch11.sh

echo "SOLIDER live dashboard: http://127.0.0.1:8080/?v=mjpeg"
echo "SOLIDER worker log: ${worker_log}"
