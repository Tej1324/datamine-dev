#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
calibration_dir="${project_dir}/${MV3DT_CALIBRATION_DIR:-calibration/ground_floor_ch10_ch11_20260928T064948Z}"
mv3dt_dir="${calibration_dir}/mv3dt"
broker_dir="${project_dir}/calibration/ground_floor_ch10_ch11_20260918T080412Z/mv3dt"
dataset_dir="${mv3dt_dir}/dataset"
experiment_dir="${mv3dt_dir}/experiment"

for required in "${dataset_dir}/camInfo/cam_00.yml" "${dataset_dir}/camInfo/cam_01.yml"; do
  [[ -f "${required}" ]] || {
    echo "Missing VGGT MV3DT calibration output: ${required}" >&2
    echo "Run AMC and export the MV3DT package before starting this test." >&2
    exit 2
  }
done
[[ -f "${experiment_dir}/config_deepstream.txt" ]] || {
  echo "Missing isolated MV3DT experiment. Run scripts/prepare_mv3dt_ch10_ch11.sh first." >&2
  exit 2
}

docker compose -f "${broker_dir}/compose.yaml" up -d
trap 'docker compose -f "${broker_dir}/compose.yaml" down' EXIT

docker run --rm --gpus all --network host \
  -v "${project_dir}:/workspace" \
  -v "${experiment_dir}:/workspace/experiments" \
  -v "${dataset_dir}:/workspace/inputs" \
  -w /workspace/experiments \
  datamine-deepstream:9.1-gpuviewer \
  deepstream-test5-app -c config_deepstream.txt
