#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deepstream_root="${DEEPSTREAM_ROOT:-/opt/nvidia/deepstream/deepstream-9.1}"
output_dir="${project_dir}/build"

mkdir -p "${output_dir}"

cxxflags=(
  -std=c++17 -fPIC -shared -O2
  "-I${deepstream_root}/sources/includes"
  "-I${deepstream_root}/sources/gst-plugins/gst-nvdsmeta"
)
ldflags=(
  "-L${deepstream_root}/lib"
  -lnvds_meta -lnvdsgst_meta -lnvbufsurface
  -lyaml-cpp
)

pkg_flags=( $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-base-1.0) )

g++ "${cxxflags[@]}" "${project_dir}/src/staff_filter/gst_staff_color_filter.cpp" \
  -o "${output_dir}/libgststaffcolorfilter.so" \
  "${pkg_flags[@]}" "${ldflags[@]}"

echo "Built ${output_dir}/libgststaffcolorfilter.so"
