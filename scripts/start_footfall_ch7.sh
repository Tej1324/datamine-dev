#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"
run_dir="${project_dir}/runs/footfall"
mkdir -p "${run_dir}"
dashboard_port="${FOOTFALL_DASHBOARD_PORT:-18082}"
tcp_port="${FOOTFALL_TCP_PORT:-7010}"
track_ttl_seconds="${FOOTFALL_TRACK_TTL_SECONDS:-60}"
tracker_fps="${FOOTFALL_TRACKER_FPS:-10}"
case "${track_ttl_seconds}" in (*[!0-9]*|'') track_ttl_seconds=60;; esac
case "${tracker_fps}" in (*[!0-9]*|'') tracker_fps=10;; esac
track_ttl_frames=$((track_ttl_seconds * tracker_fps))
tracker_runtime="${run_dir}/tracker_runtime.yml"
awk -v shadow="${track_ttl_frames}" -v search="${track_ttl_frames}" '
  /^[[:space:]]*maxShadowTrackingAge:/ {
    sub(/[0-9]+[[:space:]]*#/, shadow "    #")
  }
  /^[[:space:]]*maxTrackletMatchingTimeSearchRange:/ {
    sub(/[0-9]+[[:space:]]*#/, search "    #")
  }
  { print }
' "${project_dir}/config/tracker.yml" >"${tracker_runtime}"
dashboard_pid="${run_dir}/dashboard.pid"
if [[ -f "${dashboard_pid}" ]] && kill -0 "$(<"${dashboard_pid}")" 2>/dev/null; then
  kill "$(<"${dashboard_pid}")" || true
fi
setsid nohup python3 tools/footfall_dashboard.py --port "${dashboard_port}" \
  --tcp-port "${tcp_port}" --root "${run_dir}" >"${run_dir}/dashboard.log" 2>&1 </dev/null &
echo $! >"${dashboard_pid}"

docker rm -f footfall-ch7 2>/dev/null || true
docker run -d --name footfall-ch7 --runtime nvidia --gpus all --network host --env-file .env \
  -e MV3DT_CAMERA_IDS=ground_01 \
  -e STREAM_SUBTYPE_OVERRIDE="${FOOTFALL_SUBTYPE:-1}" \
  -e SOURCE_MAX_BATCH_SIZE=1 \
  -e SOURCE_WIDTH="${FOOTFALL_WIDTH:-1280}" -e SOURCE_HEIGHT="${FOOTFALL_HEIGHT:-720}" \
  -e TRACKER_WIDTH="${FOOTFALL_WIDTH:-1280}" -e TRACKER_HEIGHT="${FOOTFALL_HEIGHT:-720}" \
  -e DETECTOR_CONFIG=/workspace/config/detector_b6.txt \
  -e TRACKER_CONFIG=/workspace/runs/footfall/tracker_runtime.yml \
  -e FOOTFALL_TRACK_TTL_SECONDS="${track_ttl_seconds}" \
  -e FOOTFALL_TRACK_REID_THRESHOLD="${FOOTFALL_TRACK_REID_THRESHOLD:-0.80}" \
  -e FOOTFALL_STAFF_MIN_VOTES="${FOOTFALL_STAFF_MIN_VOTES:-1}" \
  -e FOOTFALL_LOGICAL_ID_OSD=1 \
  -e STAFF_CLASSIFIER_ENABLE=1 \
  -e STAFF_CLASSIFIER_CONFIG=/workspace/config/staff_classifier_infer.txt \
  -e FOOTFALL_ENABLE=1 -e OVERVIEW_TCP_PORT="${tcp_port}" \
  -e LIVE_OVERVIEW_WIDTH="${FOOTFALL_WIDTH:-1280}" -e LIVE_OVERVIEW_HEIGHT="${FOOTFALL_HEIGHT:-720}" \
  -e LIVE_TRANSPORT=mjpeg -e GLOBAL_ID_CPP_ENABLE=0 -e GLOBAL_ID_ENABLE=0 \
  -e FOOTFALL_LINES=/workspace/runs/footfall/lines.json \
  -e FOOTFALL_STATE=/workspace/runs/footfall/state.json \
  -v "${project_dir}:/workspace" -w /workspace \
  --entrypoint /bin/bash datamine-deepstream:9.1-gpuviewer \
  -lc 'exec python3 -u deepstream/detect.py'

echo "Footfall dashboard: http://127.0.0.1:${dashboard_port}"
echo "Container: footfall-ch7"
echo "Logs: docker logs -f footfall-ch7"
