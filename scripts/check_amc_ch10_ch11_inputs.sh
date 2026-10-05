#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
calib_dir="${project_dir}/calibration/ground_floor_ch10_ch11_20260918T080412Z"

for video in "${calib_dir}/inputs/videos/cam_00.mp4" "${calib_dir}/inputs/videos/cam_01.mp4"; do
  [[ -f "${video}" ]] || { echo "Missing ${video}" >&2; exit 2; }
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,r_frame_rate,codec_name \
    -of default=noprint_wrappers=1 "${video}"
done

echo "Camera order: cam_00=ch10, cam_01=ch11"
echo "AMC project: 20260918_011215_6295"
echo "Required thresholds: min_length=90 min_moving_dist=2.0 min_mean_moving_dist=0.01"
echo "Capture log synchronization status: unknown; do not start AMC until the shared time window is verified."
