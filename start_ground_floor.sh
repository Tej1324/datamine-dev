#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

viewer_log="/tmp/datamine-viewer.log"
container_name="ground-floor-gpu-viewer"

# Stop only this project's known processes/container before starting a clean pair.
pkill -f '[v]iewer/server.py' 2>/dev/null || true
pkill -f '[d]eepstream/detect.py' 2>/dev/null || true
docker rm -f "$container_name" 2>/dev/null || true

setsid nohup env VIEWER_PORT=8080 /usr/bin/python3 viewer/server.py \
  >"$viewer_log" 2>&1 </dev/null &

for _ in {1..20}; do
  if curl -fsS http://127.0.0.1:8080/ >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if ! curl -fsS http://127.0.0.1:8080/ >/dev/null 2>&1; then
  echo "viewer failed to start; see $viewer_log" >&2
  exit 1
fi

docker run -d --name "$container_name" \
  --runtime nvidia --gpus all --network host \
  --env-file .env \
  -e GST_PLUGIN_PATH=/workspace/build \
  -e LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream-9.1/lib:/root/.local/lib:/opt/tritonserver/lib:/opt/tritonclient/lib:/usr/src/tensorrt/lib:/opt/riva/lib:/usr/local/cuda-13/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.local/lib \
  -v "$project_dir:/workspace" -w /workspace \
  --entrypoint /bin/bash \
  datamine-deepstream:9.1-gpuviewer \
  -lc 'exec python3 -u deepstream/detect.py'

echo "viewer: http://127.0.0.1:8080/?v=mjpeg"
echo "container: $container_name"
echo "viewer log: $viewer_log"
echo "DeepStream log: docker logs -f $container_name"
