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

g++ "${cxxflags[@]}" "${project_dir}/src/global_identity/gst_global_identity.cpp" \
  -o "${output_dir}/libgstglobalidentity.so" \
  "${pkg_flags[@]}" "${ldflags[@]}" -pthread

echo "Built ${output_dir}/libgstglobalidentity.so"

g++ "${cxxflags[@]}" "${project_dir}/src/mv3dt_diagnostic/gst_mv3dt_diagnostic.cpp" \
  -o "${output_dir}/libgstmv3dtdiagnostic.so" \
  "${pkg_flags[@]}" "${ldflags[@]}" -pthread

echo "Built ${output_dir}/libgstmv3dtdiagnostic.so"

g++ "${cxxflags[@]}" "${project_dir}/src/mv3dt_world_identity/gst_mv3dt_world_identity.cpp" \
  -o "${output_dir}/libgstmv3dtworldidentity.so" \
  "${pkg_flags[@]}" "${ldflags[@]}" -pthread

echo "Built ${output_dir}/libgstmv3dtworldidentity.so"
