#!/usr/bin/env bash
set -Eeuo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
amc_dir="${project_dir}/calibration/auto-magic-calib/tools/auto-magic-calib"
env_file="${project_dir}/calibration/ground_floor_ch10_ch11_20260918T080412Z/amc_runtime.env"
[[ -f "${env_file}" && -f "${amc_dir}/compose/compose.yml" ]] || { echo "AMC setup is incomplete" >&2; exit 2; }
mkdir -p "${project_dir}/calibration/ground_floor_ch16_ch18/amc-projects"
docker compose --env-file "${env_file}" -f "${amc_dir}/compose/compose.yml" -p datamine-amc-ch10-ch11 up -d
echo "AMC UI: http://127.0.0.1:5100"
echo "AMC API: http://127.0.0.1:8100/v1"
echo "Existing project: 20260918_011215_6295"
