#!/usr/bin/env bash
set -Eeuo pipefail

# Build only the isolated crop bridge inside the DeepStream image. This is not
# called by the production startup path.
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deepstream_root="${DEEPSTREAM_ROOT:-/opt/nvidia/deepstream/deepstream-9.1}"
output_dir="${project_dir}/build"
mkdir -p "${output_dir}"

cxxflags=(-std=c++17 -fPIC -shared -O2
  "-I${deepstream_root}/sources/includes"
  "-I${deepstream_root}/sources/gst-plugins/gst-nvdsmeta")
ldflags=("-L${deepstream_root}/lib" -lnvds_meta -lnvdsgst_meta -lnvbufsurface)
pkg_flags=( $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-base-1.0) )

g++ "${cxxflags[@]}" "${project_dir}/src/reid_experiment/gst_reid_crop_bridge.cpp" \
  -o "${output_dir}/libgstreidcropbridge.so" "${pkg_flags[@]}" "${ldflags[@]}" -pthread

echo "Built ${output_dir}/libgstreidcropbridge.so"
