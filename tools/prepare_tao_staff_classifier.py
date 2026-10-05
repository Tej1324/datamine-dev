#!/usr/bin/env python3
"""Create a small TAO image-classification dataset from staff/customer crops."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staff", type=Path, default=Path("data/staff_filter/staff"))
    parser.add_argument("--customer", type=Path, default=Path("data/staff_filter/customer"))
    parser.add_argument("--output", type=Path, default=Path("data/tao_staff_classifier"))
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--clean", action="store_true", help="remove only the selected output directory first")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        if not args.clean:
            raise SystemExit(f"output is not empty: {args.output}; use --clean explicitly")
        shutil.rmtree(args.output)

    rng = random.Random(args.seed)
    rows = {}
    for label, directory in (("staff", args.staff), ("customer", args.customer)):
        files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            raise SystemExit(f"no images found in {directory}")
        rng.shuffle(files)
        n = len(files)
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))
        rows[label] = {
            "train": files[: n - n_val - n_test],
            "val": files[n - n_val - n_test: n - n_test],
            "test": files[n - n_test:],
        }

    summary = {"seed": args.seed, "classes": {}, "format": "TAO image classification"}
    for split in ("train", "val", "test"):
        for label in ("staff", "customer"):
            destination = args.output / split / label
            destination.mkdir(parents=True, exist_ok=True)
            files = rows[label][split]
            summary["classes"].setdefault(label, {})[split] = len(files)
            for index, source in enumerate(files):
                shutil.copy2(source, destination / f"{label}_{index:05d}{source.suffix.lower()}")
    (args.output / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
