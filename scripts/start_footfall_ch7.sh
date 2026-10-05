#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"
run_dir="${project_dir}/runs/footfall"
mkdir -p "${run_dir}"
dashboard_port="${FOOTFALL_DASHBOARD_PORT:-18082}"
tcp_port="${FOOTFALL_TCP_PORT:-7010}"

if [[ -f "${run_dir}/dashboard.pid" ]] && kill -0 "$(<"${run_dir}/dashboard.pid")" 2>/dev/null; then
  kill "$(<"${run_dir}/dashboard.pid")" || true
fi
docker rm -f footfall-ch7 2>/dev/null || true

setsid nohup python3 tools/footfall_dashboard.py --port "${dashboard_port}" \
  --tcp-port "${tcp_port}" --root "${run_dir}" \
  >"${run_dir}/dashboard.log" 2>&1 </dev/null &
echo $! >"${run_dir}/dashboard.pid"

docker run -d --name footfall-ch7 --runtime nvidia --gpus all --network host --env-file .env \
  -e FOOTFALL_CAMERA_IDS="${FOOTFALL_CAMERA_IDS:-ground_01}" \
  -e STREAM_SUBTYPE_OVERRIDE="${FOOTFALL_SUBTYPE:-1}" \
  -e SOURCE_MAX_BATCH_SIZE="${FOOTFALL_MAX_BATCH_SIZE:-1}" \
  -e FOOTFALL_WIDTH="${FOOTFALL_WIDTH:-1280}" -e FOOTFALL_HEIGHT="${FOOTFALL_HEIGHT:-720}" \
  -e TRACKER_WIDTH="${FOOTFALL_WIDTH:-1280}" -e TRACKER_HEIGHT="${FOOTFALL_HEIGHT:-720}" \
  -e DETECTOR_CONFIG=/workspace/config/detector_b6.txt \
  -e TRACKER_CONFIG=/workspace/config/tracker.yml \
  -e FOOTFALL_ROOT=/workspace/runs/footfall \
  -e FOOTFALL_LINES=/workspace/runs/footfall/lines.json \
  -e FOOTFALL_STATE=/workspace/runs/footfall/state.json \
  -e OVERVIEW_TCP_PORT="${tcp_port}" \
  -e LIVE_JPEG_QUALITY="${LIVE_JPEG_QUALITY:-90}" \
  -v "${project_dir}:/workspace" -w /workspace \
  --entrypoint /bin/bash datamine-deepstream:9.1-gpuviewer \
  -lc 'exec python3 -u deepstream/detect.py'

echo "Footfall dashboard: http://127.0.0.1:${dashboard_port}"
echo "Container: footfall-ch7"
echo "Logs: docker logs -f footfall-ch7"
