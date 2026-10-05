#!/usr/bin/env python3
"""Score manually labelled live-crop pairs and sweep appearance thresholds."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def cosine(a, b):
    if not a or not b:
        return None
    den = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / den if den else None


def load_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def score(dataset: Path, annotations: list[dict], threshold: float | None = None):
    manifest_rows = load_jsonl(dataset / "manifest.jsonl")
    starts = {}
    for row in manifest_rows:
        source = int(row["source_id"])
        timestamp = float(row.get("timestamp_seconds", row["timestamp"]))
        if timestamp > 1_000_000:
            timestamp /= 1_000_000_000.0
        row["review_timestamp"] = timestamp
        starts[source] = min(timestamp, starts.get(source, float("inf")))
    for row in manifest_rows:
        row["aligned_timestamp"] = row["review_timestamp"] - starts[int(row["source_id"])]
    manifest = {int(r["index"]): r for r in manifest_rows}
    counts = Counter()
    results = []
    for number, annotation in enumerate(annotations, 1):
        left = manifest.get(int(annotation["left_index"]))
        right = manifest.get(int(annotation["right_index"]))
        if left is None or right is None:
            counts["unresolved"] += 1
            continue
        similarity = cosine(left.get("embedding", []), right.get("embedding", []))
        world_distance = None
        if left.get("world") and right.get("world"):
            world_distance = math.hypot(
                float(left["world"][0]) - float(right["world"][0]),
                float(left["world"][1]) - float(right["world"][1]),
            )
        if similarity is None:
            # The live-crop manifest intentionally omits embeddings by default;
            # system-ID scoring still works, while threshold sweeps require a
            # future manifest with embeddings enabled.
            predicted = (left.get("experimental_global_id") is not None and
                         left.get("experimental_global_id") == right.get("experimental_global_id"))
        else:
            predicted = similarity >= threshold if threshold is not None else (
                left.get("experimental_global_id") is not None and
                left.get("experimental_global_id") == right.get("experimental_global_id"))
        expected = bool(annotation.get("same_person"))
        counts["tp" if expected and predicted else "fn" if expected else "fp" if predicted else "tn"] += 1
        results.append({"number": number, "left_index": left["index"], "right_index": right["index"],
                        "expected_same": expected, "predicted_same": predicted,
                        "system_left": left.get("experimental_global_id"),
                        "system_right": right.get("experimental_global_id"),
                        "similarity": similarity,
                        "world_distance": world_distance,
                        "raw_timestamp_delta": abs(float(left["review_timestamp"]) - float(right["review_timestamp"])),
                        "aligned_timestamp_delta": abs(float(left["aligned_timestamp"]) - float(right["aligned_timestamp"])),
                        "left_decision": left.get("decision"), "right_decision": right.get("decision")})
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {"counts": dict(counts), "precision": precision, "recall": recall, "f1": f1, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    annotations = load_jsonl(args.dataset / "annotations.jsonl") if (args.dataset / "annotations.jsonl").exists() else []
    report = score(args.dataset, annotations)
    document = {"dataset": str(args.dataset), "annotations": len(annotations), "system_id_metrics": report}
    print(json.dumps(document, indent=2))
    if args.output:
        args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
