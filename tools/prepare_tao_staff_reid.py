#!/usr/bin/env python3
"""Prepare identity-labelled staff crops for NVIDIA TAO ReIdentificationNet.

The input JSONL must contain one record per crop, for example:
{"path":"data/staff_filter/staff/staff_0001.jpg",
 "identity_id":"staff_001", "camera":"ground_01"}

This deliberately refuses category-only data ("staff", "uniform", etc.).
TAO ReIdentificationNet is an identity metric-learning task and needs several
images of each individual, not one class containing every employee.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    rows = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{number}: record must be an object")
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True,
                        help="identity-labelled staff JSONL")
    parser.add_argument("--output", type=Path, required=True,
                        help="new empty TAO/Market-1501 dataset directory")
    parser.add_argument("--min-images", type=int, default=3,
                        help="minimum crops required per identity (default: 3)")
    parser.add_argument("--project-root", type=Path, default=Path("."),
                        help="base directory for relative crop paths")
    args = parser.parse_args()

    rows = read_rows(args.manifest)
    if not rows:
        raise SystemExit("manifest is empty")
    if args.min_images < 3:
        raise SystemExit("--min-images must be at least 3")

    by_identity: dict[str, list[tuple[dict, Path]]] = defaultdict(list)
    missing = []
    for row in rows:
        identity = str(row.get("identity_id", "")).strip()
        if not identity:
            missing.append(row)
            continue
        source = Path(str(row.get("path", "")))
        if not source.is_absolute():
            source = args.project_root / source
        if not source.is_file():
            raise SystemExit(f"crop does not exist: {source}")
        by_identity[identity].append((row, source))
    if missing:
        raise SystemExit(
            f"{len(missing)} records have no identity_id. Add stable employee IDs; "
            "a category such as 'staff' or 'uniform' is not sufficient for TAO Re-ID."
        )

    counts = Counter({identity: len(items) for identity, items in by_identity.items()})
    too_small = {identity: count for identity, count in counts.items()
                 if count < args.min_images}
    if too_small:
        details = ", ".join(f"{key}={value}" for key, value in sorted(too_small.items()))
        raise SystemExit(
            f"identities need at least {args.min_images} crops: {details}"
        )
    if len(by_identity) < 2:
        raise SystemExit("at least two distinct employee identities are required")

    output = args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output must be new or empty: {output}")
    for split in ("bounding_box_train", "query", "bounding_box_test"):
        (output / split).mkdir(parents=True, exist_ok=True)

    written = Counter()
    # Deterministic split per identity: one query, one gallery/test image, and
    # the remaining views for training. This keeps every identity represented
    # in both retrieval sides while preserving cross-view examples.
    for class_index, identity in enumerate(sorted(by_identity), 1):
        items = sorted(by_identity[identity], key=lambda pair: str(pair[0].get("path")))
        person_id = f"{class_index:04d}"
        for view_index, (row, source) in enumerate(items):
            if view_index == 0:
                split = "query"
            elif view_index == 1:
                split = "bounding_box_test"
            else:
                split = "bounding_box_train"
            camera = str(row.get("camera", "cam"))
            destination = output / split / f"{person_id}_c{camera}_s1_{view_index:04d}.jpg"
            shutil.copy2(source, destination)
            written[split] += 1

    summary = {
        "identities": len(by_identity),
        "input_crops": len(rows),
        "written": dict(written),
        "identity_counts": dict(sorted(counts.items())),
        "format": "Market-1501-compatible TAO ReIdentificationNet dataset",
    }
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                   encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
