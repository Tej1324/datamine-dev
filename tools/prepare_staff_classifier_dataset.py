"""Validate and prepare the staff/customer crop dataset for classifier training.

This tool is offline-only. It does not touch the DeepStream pipeline or alter
the live staff filter. Samples from the same source frame stay in one split so
near-identical crops cannot leak between train and validation sets.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "data/staff_filter/labels.yaml"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=ROOT / "data/staff_filter/classifier_manifest.jsonl")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    items = yaml.safe_load(args.labels.read_text()) or []
    valid = []
    for item in items:
        label = item.get("class", "staff").lower()
        crop = ROOT / item["crop"]
        if label not in {"staff", "customer"} or not crop.is_file():
            continue
        valid.append({
            "path": str(crop.relative_to(ROOT)),
            "label": label,
            "camera": item.get("camera", "unknown"),
            "source_frame": item.get("frame", "unknown"),
        })

    counts = Counter(item["label"] for item in valid)
    if not counts["staff"] or not counts["customer"]:
        raise SystemExit(
            f"Need both classes before training: staff={counts['staff']} customer={counts['customer']}"
        )
    if len({item["camera"] for item in valid}) < 6:
        raise SystemExit("Need samples from all six cameras before training")

    groups = defaultdict(list)
    for item in valid:
        groups[(item["camera"], item["source_frame"])].append(item)
    keys = sorted(groups)
    random.Random(20260919).shuffle(keys)
    val_keys = set(keys[:max(1, round(len(keys) * args.val_fraction))])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for key in sorted(keys):
            split = "val" if key in val_keys else "train"
            for item in groups[key]:
                stream.write(json.dumps({**item, "split": split}) + "\n")

    train = sum(len(groups[key]) for key in keys if key not in val_keys)
    val = sum(len(groups[key]) for key in keys if key in val_keys)
    print(json.dumps({"counts": counts, "train": train, "val": val, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
