#!/usr/bin/env bash
set -Eeuo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
amc_dir="${project_dir}/calibration/auto-magic-calib/tools/auto-magic-calib"
env_file="${project_dir}/calibration/ground_floor_ch10_ch11_20260918T080412Z/amc_runtime.env"
docker compose --env-file "${env_file}" -f "${amc_dir}/compose/compose.yml" -p datamine-amc-ch10-ch11 down
