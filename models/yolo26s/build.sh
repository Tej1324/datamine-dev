#!/usr/bin/env bash
# Run inside DeepStream 9.1 on the target L4. Rebuild for Jetson.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/models/yolo26s"
TRACKER_DIR="$PROJECT_ROOT/models/tracker"
DS_ROOT=/opt/nvidia/deepstream/deepstream-9.1
UPSTREAM_REV=2894babce8e75c49115dbe0c7b516289ed853565
mkdir -p "$MODEL_DIR" "$TRACKER_DIR"
cd "$MODEL_DIR"
if [[ ! -d DeepStream-Yolo/.git ]]; then
    git clone https://github.com/marcoslucianops/DeepStream-Yolo.git DeepStream-Yolo
    git -C DeepStream-Yolo checkout "$UPSTREAM_REV"
fi
if [[ "$(git -C DeepStream-Yolo rev-parse HEAD)" != "$UPSTREAM_REV" ]]; then
    echo "DeepStream-Yolo revision differs from the validated build; inspect before rebuilding." >&2
    exit 1
fi
# Compatibility fix: upstream leaves the DS 9.1 rotation field uninitialized.
python3 - <<'PYFIX'
from pathlib import Path
p=Path("DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp")
p.write_text(p.read_text().replace("NvDsInferParseObjectInfo b;", "NvDsInferParseObjectInfo b{};"))
PYFIX
CUDA_VER=13.2 make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo -j2 > parser-build.log 2>&1
python3 -m venv --without-pip .venv
python3 -m pip --python .venv/bin/python install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cpu > export-setup.log 2>&1
python3 -m pip --python .venv/bin/python install --no-cache-dir -r requirements.lock.txt \
    >> export-setup.log 2>&1
if [[ ! -s yolo26s.pt ]]; then
    curl -fL --retry 2 https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt -o yolo26s.pt.partial
    mv yolo26s.pt.partial yolo26s.pt
fi
mkdir -p .ultralytics .matplotlib
export YOLO_CONFIG_DIR="$MODEL_DIR/.ultralytics" MPLCONFIGDIR="$MODEL_DIR/.matplotlib"
.venv/bin/python DeepStream-Yolo/utils/export_yolo26.py \
    -w yolo26s.pt -s 736 1280 --batch 1 --simplify > export.log 2>&1
/usr/src/tensorrt/bin/trtexec --onnx=yolo26s.onnx \
    --saveEngine=yolo26s_b1_736x1280_l4_fp16.engine --fp16 --skipInference > engine-build.log 2>&1
/usr/src/tensorrt/bin/trtexec --loadEngine=yolo26s_b1_736x1280_l4_fp16.engine \
    --warmUp=500 --duration=3 > engine-validation.log 2>&1
if [[ ! -s "$TRACKER_DIR/resnet50_market1501.etlt" ]]; then
    curl -fL --retry 2 \
        https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/versions/deployable_v1.0/files/resnet50_market1501.etlt \
        -o "$TRACKER_DIR/resnet50_market1501.etlt.partial"
    mv "$TRACKER_DIR/resnet50_market1501.etlt.partial" "$TRACKER_DIR/resnet50_market1501.etlt"
fi
g++ -std=c++17 "$TRACKER_DIR/prepare_tracker.cpp" \
    -I"$DS_ROOT/sources/includes" -I/usr/local/cuda-13.2/include \
    $(pkg-config --cflags glib-2.0) -L"$DS_ROOT/lib" -Wl,-rpath,"$DS_ROOT/lib" \
    -lnvds_nvmultiobjecttracker -o "$TRACKER_DIR/prepare_tracker"
"$TRACKER_DIR/prepare_tracker" "$PROJECT_ROOT/config/config_tracker_NvDCF_accuracy.yml" \
    > "$TRACKER_DIR/prepare.log" 2>&1
python3 "$PROJECT_ROOT/deepstream/entrance_counter.py" --check
echo "YOLO26s FP16 and NvDCF accuracy assets ready for this L4."
