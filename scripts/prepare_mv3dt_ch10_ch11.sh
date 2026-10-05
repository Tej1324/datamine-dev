#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
reference_dir="${project_dir}/vendor/deepstream-tracker-3d-multi-view"
calibration_dir="${project_dir}/${MV3DT_CALIBRATION_DIR:-calibration/ground_floor_ch10_ch11_20260928T064948Z}"
dataset_dir="${calibration_dir}/mv3dt/dataset"
experiment_dir="${calibration_dir}/mv3dt/experiment"
caminfo_export_dir="${calibration_dir}/mv3dt_vggt/camInfo"

for required in \
  "${calibration_dir}/inputs/videos/cam_00.mp4" \
  "${calibration_dir}/inputs/videos/cam_01.mp4" \
  "${caminfo_export_dir}/camInfo_00.yml" \
  "${caminfo_export_dir}/camInfo_01.yml" \
  "${calibration_dir}/mv3dt_vggt/transforms.yml"; do
  [[ -f "${required}" ]] || {
    echo "Missing new VGGT MV3DT input: ${required}" >&2
    exit 2
  }
done

if [[ ! -d "${reference_dir}" ]]; then
  mkdir -p "${project_dir}/vendor"
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/NVIDIA/DeepStream.git "${project_dir}/vendor/deepstream-reference"
  git -C "${project_dir}/vendor/deepstream-reference" sparse-checkout set \
    src/apps/reference_apps/deepstream-tracker-3d-multi-view
  ln -s "${project_dir}/vendor/deepstream-reference/src/apps/reference_apps/deepstream-tracker-3d-multi-view" \
    "${reference_dir}"
fi

mkdir -p "${dataset_dir}/videos" "${dataset_dir}/camInfo" "${experiment_dir}"
# Copy rather than symlink: the dataset directory is mounted at /workspace/inputs
# inside Docker, so host-absolute symlink targets are not visible there.
rm -f \
  "${dataset_dir}/videos/cam_00.mp4" \
  "${dataset_dir}/videos/cam_01.mp4" \
  "${dataset_dir}/map.png" \
  "${dataset_dir}/camInfo/cam_00.yml" \
  "${dataset_dir}/camInfo/cam_01.yml" \
  "${dataset_dir}/transforms.yml"
cp -f "${calibration_dir}/inputs/videos/cam_00.mp4" \
  "${dataset_dir}/videos/cam_00.mp4"
cp -f "${calibration_dir}/inputs/videos/cam_01.mp4" \
  "${dataset_dir}/videos/cam_01.mp4"
cp -f "${calibration_dir}/mv3dt_vggt/map.png" \
  "${dataset_dir}/map.png"
cp -f "${caminfo_export_dir}/camInfo_00.yml" \
  "${dataset_dir}/camInfo/cam_00.yml"
cp -f "${caminfo_export_dir}/camInfo_01.yml" \
  "${dataset_dir}/camInfo/cam_01.yml"
cp -f "${calibration_dir}/mv3dt_vggt/transforms.yml" \
  "${dataset_dir}/transforms.yml"

(
  cd "${reference_dir}"
  configurator_args=(
    "--dataset-dir=${dataset_dir}"
    "--output-dir=${experiment_dir}"
  )
  if [[ "${MV3DT_ENABLE_OSD:-0}" == "1" ]]; then
    configurator_args+=(--enable-osd)
  fi
  if [[ "${MV3DT_ENABLE_MSG_BROKER:-0}" == "1" ]]; then
    configurator_args+=(--enable-msg-broker)
  fi
  python3 utils/deepstream_auto_configurator.py "${configurator_args[@]}"
)

# The NVIDIA configurator currently invokes its helper as `python`, while this
# host exposes Python as `python3`. Generate the broker config explicitly when
# the configurator did not create it; this remains inside the isolated test.
if [[ ! -f "${experiment_dir}/pub_sub_info_config_0.yml" ]]; then
  (
    cd "${reference_dir}"
    python3 utils/generate_pub_sub_configs.py \
      --cam_info_path "${dataset_dir}/camInfo" \
      --neighbor_criteria top_N:1 \
      --output_path "${experiment_dir}" \
      --deployment_config_path "${experiment_dir}/deployment_config.yml"
  )
fi

# The stock configurator emits a PeopleNet config. This repository's isolated
# test uses its existing YOLO26 detector unless another PGIE config is given.
pgie_config="${project_dir}/${MV3DT_PGIE_CONFIG:-config/detector_mv3dt_yolo26_b1.txt}"
if [[ -f "${pgie_config}" ]]; then
  cp "${pgie_config}" "${experiment_dir}/config_pgie.txt"
  if [[ "${pgie_config}" == *"detector_mv3dt_yolo26_b1.txt" ]]; then
    sed -i \
      's#^model-engine-file=.*#model-engine-file=/workspace/models/yolo26s/yolo26s_b1_736x1280_l4_fp16.engine#' \
      "${experiment_dir}/config_deepstream.txt"
    sed -i \
      '/^\[primary-gie\]/,/^\[/ s/^batch-size=.*/batch-size=1/' \
      "${experiment_dir}/config_deepstream.txt"
  fi
fi

# Keep the existing detector/tracker model path and NvDCF/Re-ID settings for
# this isolated test; only the MV3DT sections are added in the test copy.
# Copy instead of symlinking: Docker cannot resolve a symlink whose target is
# an absolute host path.
rm -f "${experiment_dir}/config_tracker.yml"
cp -f "${project_dir}/config/tracker_mv3dt_ch10_ch11.yml" \
  "${experiment_dir}/config_tracker.yml"

echo "Generated isolated MV3DT experiment at ${experiment_dir}"
echo "Calibration source: ${calibration_dir}/mv3dt_vggt (VGGT)"
