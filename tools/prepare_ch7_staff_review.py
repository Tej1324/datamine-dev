"""Prepare the CH7 footfall review crops for the two-column TAO labeler."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/ch7_staff_review"
SOURCE = ROOT / "runs/footfall"
MODEL = Path(os.environ.get(
    "STAFF_REVIEW_MODEL",
    str(ROOT / "runs/tao_staff_classifier_reviewed_v2/staff_customer_reviewed_dynamic.onnx"),
))


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    from staff_review_dashboard import classify_batch
    import onnxruntime as ort

    source_rows = []
    review_path = SOURCE / "staff_review.jsonl"
    for line in review_path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        crop = SOURCE / "staff_crops" / str(item.get("crop", ""))
        if crop.is_file():
            source_rows.append((item, crop))

    RUN.mkdir(parents=True, exist_ok=True)
    crops = RUN / "crops/CH7"
    crops.mkdir(parents=True, exist_ok=True)
    # The dashboard serves through RUN, so expose the existing crop files
    # without duplicating the image data.
    for item, crop in source_rows:
        target = crops / crop.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(crop)

    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    paths = [crop for _, crop in source_rows]
    manifest = []
    for offset in range(0, len(paths), 64):
        batch = paths[offset:offset + 64]
        predictions = classify_batch(session, batch)
        for (item, crop), (label, staff_probability) in zip(source_rows[offset:offset + 64], predictions):
            manifest.append({
                "id": len(manifest),
                "path": f"crops/CH7/{crop.name}",
                "camera": "CH7",
                "local_track_id": item.get("local_track_id"),
                "source_id": item.get("source_id"),
                "label": label,
                "model_label": label,
                "staff_probability": round(float(staff_probability), 6),
            })
    (RUN / "review_manifest.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in manifest),
        encoding="utf-8",
    )
    (RUN / "corrections.jsonl").touch()
    print(f"Prepared {len(manifest)} CH7 crops in {RUN}")


if __name__ == "__main__":
    main()
