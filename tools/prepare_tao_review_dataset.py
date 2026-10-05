#!/usr/bin/env python3
"""Build a group-disjoint TAO dataset from the reviewed crop manifest."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("runs/staff_review_eval/review_manifest.jsonl"))
    parser.add_argument("--root", type=Path, default=Path("runs/staff_review_eval"))
    parser.add_argument("--output", type=Path, default=Path("data/tao_staff_classifier_reviewed"))
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("group", f"crop:{row['id']}")].append(row)
    rng = random.Random(args.seed)
    group_values = list(groups.values())
    rng.shuffle(group_values)
    split_groups = {"train": [], "val": [], "test": []}
    for index, group in enumerate(group_values):
        split = "test" if index % 10 == 0 else "val" if index % 10 == 1 else "train"
        split_groups[split].append(group)
    summary = {"seed": args.seed, "source_rows": len(rows), "classes": {}, "group_disjoint": True}
    for split, split_group_list in split_groups.items():
        for group in split_group_list:
            for row in group:
                label = row["label"]
                source = args.root / row["path"]
                if not source.is_file():
                    continue
                destination = args.output / split / label
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination / f"{label}_{int(row['id']):06d}{source.suffix.lower()}")
                summary["classes"].setdefault(label, {}).setdefault(split, 0)
                summary["classes"][label][split] += 1
    for label in ("staff", "customer"):
        summary["classes"].setdefault(label, {})
        for split in ("train", "val", "test"):
            summary["classes"][label].setdefault(split, 0)
    (args.output / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
