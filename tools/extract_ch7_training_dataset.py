"""Extract sampled person crops from a recorded CH7 video for review."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("CH7_TRAINING_RUN", str(ROOT / "runs/ch7_training_latest")))
VIDEO = Path(os.environ.get("CH7_TRAINING_VIDEO", str(RUN / "video/ch7_latest.mkv")))
DETECTOR = ROOT / "models/yolo26s/yolo26s.onnx"
CLASSIFIER = Path(os.environ.get(
    "STAFF_REVIEW_MODEL",
    str(ROOT / "runs/tao_staff_classifier_reviewed_v2/staff_customer_reviewed_dynamic.onnx"),
))
REID_MODEL = ROOT / "models/tracker/swin_tiny_market1501_aicity156_featuredim256.onnx"


def detect(session, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    canvas = np.zeros((736, 1280, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    canvas[:height, :width] = rgb
    tensor = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]
    raw = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
    boxes, scores = [], []
    for x1, y1, x2, y2, score, cls in raw:
        if int(cls) != 0 or float(score) < 0.35:
            continue
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width, int(x2)), min(height, int(y2))
        if x2 - x1 < 20 or y2 - y1 < 40:
            continue
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(score))
    keep = cv2.dnn.NMSBoxes(boxes, scores, 0.35, 0.45)
    return [tuple(boxes[int(i)]) for i in (keep.flatten() if len(keep) else [])]


def main() -> None:
    crop_dir = RUN / "review/crops/CH7"
    crop_dir.mkdir(parents=True, exist_ok=True)
    det = ort.InferenceSession(str(DETECTOR), providers=["CPUExecutionProvider"])
    cls = ort.InferenceSession(str(CLASSIFIER), providers=["CPUExecutionProvider"])
    reid_options = ort.SessionOptions()
    reid_options.intra_op_num_threads = int(os.environ.get("CH7_REID_THREADS", "4"))
    reid = ort.InferenceSession(str(REID_MODEL), reid_options, providers=["CPUExecutionProvider"])
    existing = sorted(crop_dir.glob("*.jpg"))
    if existing:
        paths = existing
        frame_no = 0
    else:
        cap = cv2.VideoCapture(str(VIDEO))
        sample_every = int(os.environ.get("CH7_SAMPLE_EVERY", "15"))
        paths = []
        frame_no = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_no % sample_every == 0:
                for box in detect(det, frame):
                    x, y, w, h = box
                    path = crop_dir / f"crop_{len(paths):06d}.jpg"
                    crop = frame[y:y + h, x:x + w]
                    if crop.size and cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                        paths.append(path)
            frame_no += 1
        cap.release()

    from staff_review_dashboard import classify_batch

    manifest = []
    reid_vectors = []
    reid_input = reid.get_inputs()[0].name
    embedding_batch_size = int(os.environ.get("CH7_REID_BATCH", "256"))
    for offset in range(0, len(paths), embedding_batch_size):
        batch = []
        for path in paths[offset:offset + embedding_batch_size]:
            image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (128, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            image = (image - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
            batch.append(np.transpose(image, (2, 0, 1)))
        values = reid.run(None, {reid_input: np.stack(batch)})[0]
        for vector in values:
            vector = np.asarray(vector, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            reid_vectors.append((vector / norm).round(6).tolist() if norm else [])
    for offset in range(0, len(paths), 64):
        batch = paths[offset:offset + 64]
        predictions = classify_batch(cls, batch)
        for index, (path, (label, probability)) in enumerate(zip(batch, predictions), offset):
            manifest.append({
                "id": len(manifest),
                "path": f"crops/CH7/{path.name}",
                "camera": "CH7",
                "label": label,
                "model_label": label,
                "staff_probability": round(float(probability), 6),
                "embedding": reid_vectors[index],
            })
    review = RUN / "review"
    (review / "review_manifest.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in manifest),
        encoding="utf-8",
    )
    (review / "corrections.jsonl").touch()
    print(json.dumps({"frames": frame_no, "crops": len(manifest), "review_run": str(review)}))


if __name__ == "__main__":
    main()
