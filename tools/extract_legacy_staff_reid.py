#!/usr/bin/env python3
"""Extract legacy staff crops with the existing NVIDIA Re-ID TensorRT engine.

This is an offline migration tool. It does not change DeepStream, NvDCF, the
live gallery, or the production filter. It prepares the exact tracker input
shape, runs the existing batch-100 engine through the DeepStream container's
TensorRT executable, and writes reviewable candidate embeddings separately.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


INPUT_SHAPE = (256, 128)  # height, width
BATCH_SIZE = 100
OFFSETS = np.asarray([123.675, 116.280, 103.530], dtype=np.float32)
SCALE = np.float32(0.01735207)


def letterbox_rgb(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    target_height, target_width = INPUT_SHAPE
    ratio = min(target_width / width, target_height / height)
    resized_width = max(1, round(width * ratio))
    resized_height = max(1, round(height * ratio))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    top = (target_height - resized_height) // 2
    left = (target_width - resized_width) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def preprocess(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unable to read crop: {path}")
    rgb = letterbox_rgb(image).astype(np.float32)
    chw = np.transpose((rgb - OFFSETS) * SCALE, (2, 0, 1))
    return chw


def read_manifest(root: Path) -> list[dict]:
    manifest = root / "data/staff_filter/staff_manifest.jsonl"
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("label") != "staff":
            continue
        item["path"] = str(root / item["path"])
        rows.append(item)
    return rows


def run_trtexec(container: str, engine: str, input_path: Path, output_path: Path, batch: int) -> None:
    command = [
        "docker", "exec", container, "trtexec",
        f"--loadEngine={engine}",
        f"--shapes=input:{batch}x3x256x128",
        f"--loadInputs=input:/workspace/{input_path.relative_to(Path.cwd())}",
        f"--exportOutput=/workspace/{output_path.relative_to(Path.cwd())}",
        "--iterations=1", "--warmUp=0", "--duration=0.1",
    ]
    subprocess.run(command, check=True)


def load_outputs(path: Path) -> np.ndarray:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        output = next((item.get("values") for item in value
                       if item.get("name") == "fc_pred"), None)
    else:
        output = value.get("fc_pred")
        if output is None:
            output = value.get("outputs", {}).get("fc_pred")
    if output is None:
        raise RuntimeError(f"fc_pred missing from TensorRT output: {path}")
    array = np.asarray(output, dtype=np.float32)
    if array.size % 256 != 0:
        raise RuntimeError(f"unexpected Re-ID output shape: {array.shape}")
    array = array.reshape(-1, 256)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--container", default="ground-floor-gpu-viewer")
    parser.add_argument("--engine", default="/workspace/models/tracker/resnet50_market1501.etlt_b100_gpu0_fp16.engine")
    parser.add_argument("--output", type=Path,
                        default=Path("data/staff_filter/reid_legacy_candidates.jsonl"))
    args = parser.parse_args()
    root = args.root.resolve()
    rows = read_manifest(root)
    if not rows:
        raise SystemExit("no staff rows found in staff_manifest.jsonl")

    output_rows = []
    with tempfile.TemporaryDirectory(dir=root / "runs") as temporary:
        temporary_path = Path(temporary)
        for start in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[start:start + BATCH_SIZE]
            tensors = [preprocess(Path(item["path"])) for item in batch_rows]
            if len(tensors) < BATCH_SIZE:
                tensors.extend([np.zeros((3, 256, 128), dtype=np.float32)] * (BATCH_SIZE - len(tensors)))
            input_path = temporary_path / f"batch_{start:05d}.bin"
            output_path = temporary_path / f"batch_{start:05d}.json"
            np.asarray(tensors, dtype=np.float32).tofile(input_path)
            run_trtexec(args.container, args.engine, input_path, output_path, BATCH_SIZE)
            embeddings = load_outputs(output_path)
            for item, embedding in zip(batch_rows, embeddings):
                output_rows.append({
                    "path": item["path"], "camera": item.get("camera"),
                    "source_frame": item.get("source_frame"), "split": item.get("split"),
                    "embedding": [float(value) for value in embedding],
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for item in output_rows:
            output.write(json.dumps(item, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    print(f"extracted={len(output_rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
