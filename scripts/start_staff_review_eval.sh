#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"
run_dir="${STAFF_REVIEW_RUN:-${project_dir}/runs/staff_review_eval}"
mkdir -p "${run_dir}/crops" "${run_dir}/transport"
log="${run_dir}/run.log"
worker_log="${run_dir}/worker.log"
worker_pid="${run_dir}/worker.pid"
collector_pid="${run_dir}/collector.pid"
dashboard_pid="${run_dir}/dashboard.pid"
camera_ids="ground_01,ground_03,ground_06,ground_04,ground_05"
camera_labels="CH7,CH10,CH11,CH18,CH20"

# This evaluation owns the GPU detector while it is running. Stop only the
# known project workers, not unrelated OS services or containers.
docker rm -f footfall-ch7 2>/dev/null || true
if [[ -f runs/footfall/dashboard.pid ]]; then
  kill "$(<runs/footfall/dashboard.pid)" 2>/dev/null || true
fi

rm -f "${run_dir}/crops.sock" "${worker_pid}" "${collector_pid}" "${dashboard_pid}"
export REID_CAMERA_LABELS="${camera_labels}"
export REID_EXPERIMENT_DATASET_DIR="${run_dir}/crops"
export REID_EXPERIMENT_DATASET_LIMIT="${STAFF_REVIEW_CROP_LIMIT:-20000}"
export REID_EXPERIMENT_SAMPLE_LIMIT=0

setsid nohup env REID_CAMERA_LABELS="${camera_labels}" python3 -u deepstream/crop_collector.py \
  --socket "${run_dir}/crops.sock" --output "${run_dir}/crops" \
  --limit "${STAFF_REVIEW_CROP_LIMIT:-20000}" >"${worker_log}" 2>&1 < /dev/null &
echo $! >"${worker_pid}"
for _ in {1..60}; do [[ -S "${run_dir}/crops.sock" ]] && break; sleep 0.25; done
[[ -S "${run_dir}/crops.sock" ]] || { tail -80 "${worker_log}" >&2; exit 1; }

# The DeepStream graph always publishes an overview TCP stream. A temporary
# relay keeps that graph healthy even though this evaluation dashboard is
# intentionally offline until the five-minute capture is complete.
setsid nohup python3 tools/footfall_dashboard.py --port 18084 --tcp-port 7011 \
  --root "${run_dir}/transport" >"${run_dir}/transport.log" 2>&1 < /dev/null &
transport_pid=$!

docker rm -f staff-review-detector 2>/dev/null || true
docker run -d --name staff-review-detector --runtime nvidia --gpus all --network host \
  --env-file .env \
  -e MV3DT_CAMERA_IDS="${camera_ids}" \
  -e STREAM_SUBTYPE_OVERRIDE="${STAFF_REVIEW_SUBTYPE:-1}" \
  -e SOURCE_MAX_BATCH_SIZE=5 -e SOURCE_WIDTH=1280 -e SOURCE_HEIGHT=720 \
  -e TRACKER_WIDTH=1280 -e TRACKER_HEIGHT=720 \
  -e DETECTOR_CONFIG=/workspace/config/detector_b6.txt \
  -e TRACKER_CONFIG=/workspace/config/tracker.yml \
  -e REID_EXPERIMENT_CROP_BRIDGE=1 \
  -e REID_EXPERIMENT_SOCKET="/workspace/${run_dir#${project_dir}/}/crops.sock" \
  -e REID_EXPERIMENT_INTERVAL_FRAMES="${STAFF_REVIEW_INTERVAL_FRAMES:-15}" \
  -e REID_EXPERIMENT_MIN_DETECTION_CONFIDENCE=0.35 \
  -e LIVE_TRANSPORT=mjpeg -e OVERVIEW_TCP_PORT=7011 \
  -e LIVE_OVERVIEW_WIDTH=1280 -e LIVE_OVERVIEW_HEIGHT=480 \
  -e GLOBAL_ID_CPP_ENABLE=0 -e GLOBAL_ID_ENABLE=0 \
  -e STAFF_COLOR_FILTER_ENABLE=0 -e FOOTFALL_ENABLE=0 \
  -e GST_PLUGIN_PATH=/workspace/build \
  -v "${project_dir}:/workspace" -w /workspace \
  --entrypoint /bin/bash datamine-deepstream:9.1-gpuviewer \
  -lc 'exec python3 -u deepstream/detect.py' >>"${log}" 2>&1
echo "$(date -u +%FT%TZ) collection started for ${camera_labels}" >>"${log}"

setsid bash -c '
  set -Eeuo pipefail
  sleep 300
  docker rm -f staff-review-detector 2>/dev/null || true
  if [[ -f "'"${worker_pid}"'" ]]; then kill "$(<"'"${worker_pid}"'")" 2>/dev/null || true; fi
  if [[ -f "'"${dashboard_pid}"'" ]]; then kill "$(<"'"${dashboard_pid}"'")" 2>/dev/null || true; fi
  kill "'"${transport_pid}"'" 2>/dev/null || true
  /home/ubuntu/venvs/reid-experiment/bin/python tools/staff_review_dashboard.py \
    >"'"${run_dir}"'/dashboard.log" 2>&1 &
  echo $! >"'"${dashboard_pid}"'"
  echo "$(date -u +%FT%TZ) collection complete; review dashboard ready on port 18083" >>"'"${log}"'"
' >"${run_dir}/finalizer.log" 2>&1 < /dev/null &
echo $! >"${collector_pid}"
echo "Collection started. Five-minute finalizer PID: $(<"${collector_pid}")"
