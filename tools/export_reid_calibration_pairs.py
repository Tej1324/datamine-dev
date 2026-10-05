#!/usr/bin/env python3
"""Export manually verified live-crop pairs for geometry analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def cosine(left, right):
    if not left or not right:
        return ""
    denominator = math.sqrt(sum(x * x for x in left) * sum(x * x for x in right))
    return sum(x * y for x, y in zip(left, right)) / denominator if denominator else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    manifest = {}
    starts = {}
    with (dataset / "manifest.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            manifest[int(row["index"])] = row
            source = int(row["source_id"])
            starts[source] = min(float(row["timestamp"]), starts.get(source, float("inf")))

    annotations = [json.loads(line) for line in (dataset / "annotations.jsonl").open(encoding="utf-8") if line.strip()]
    rows = []
    for annotation in annotations:
        left = manifest[int(annotation["left_index"])]
        right = manifest[int(annotation["right_index"])]
        def footpoint(row):
            x, y, w, h = (float(v) for v in row["bbox"])
            return x + w / 2.0, y + h
        lx, ly = footpoint(left)
        rx, ry = footpoint(right)
        lw = left.get("world") or ["", ""]
        rw = right.get("world") or ["", ""]
        rows.append({
            "reference_id": annotation.get("reference_id", ""),
            "same_person": bool(annotation.get("same_person")),
            "left_index": left["index"], "right_index": right["index"],
            "left_local_track": left.get("local_track_id", ""),
            "right_local_track": right.get("local_track_id", ""),
            "left_global_id": left.get("experimental_global_id", ""),
            "right_global_id": right.get("experimental_global_id", ""),
            "left_footpoint_x_px": lx, "left_footpoint_y_px": ly,
            "right_footpoint_x_px": rx, "right_footpoint_y_px": ry,
            "left_projected_x": lw[0], "left_projected_y": lw[1],
            "right_projected_x": rw[0], "right_projected_y": rw[1],
            "projected_distance_m": math.hypot(float(lw[0]) - float(rw[0]), float(lw[1]) - float(rw[1])) if lw[0] != "" and rw[0] != "" else "",
            "solider_similarity": cosine(left.get("embedding"), right.get("embedding")),
            "left_timestamp_s": left["timestamp"], "right_timestamp_s": right["timestamp"],
            "raw_timestamp_delta_s": abs(float(left["timestamp"]) - float(right["timestamp"])),
            "aligned_timestamp_delta_s": abs((float(left["timestamp"]) - starts[int(left["source_id"])]) - (float(right["timestamp"]) - starts[int(right["source_id"])])),
        })
    output = args.output or dataset / "calibration_pairs.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["reference_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output), "pairs": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
