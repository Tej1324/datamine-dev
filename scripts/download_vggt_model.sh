#!/usr/bin/env bash
set -Eeuo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${project_dir}/calibration/auto-magic-calib/tools/auto-magic-calib/models/vggt"
model_path="${model_dir}/vggt_1B_commercial.pt"
if [[ -f "${model_path}" ]]; then echo "VGGT model already present: ${model_path}"; exit 0; fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "Set HF_TOKEN after accepting the VGGT model license." >&2; exit 2; }
hf_bin="${project_dir}/venv/bin/hf"
[[ -x "${hf_bin}" ]] || { echo "Missing ${hf_bin}; install huggingface_hub first." >&2; exit 2; }
mkdir -p "${model_dir}"
HF_TOKEN="${HF_TOKEN}" "${hf_bin}" download facebook/VGGT-1B-Commercial --local-dir "${model_dir}"
[[ -s "${model_path}" ]] || { echo "VGGT model was not downloaded" >&2; exit 2; }
