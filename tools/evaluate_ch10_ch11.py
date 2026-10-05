#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 NVIDIA
# SPDX-License-Identifier: Apache-2.0
"""Evaluate manually verified CH10/CH11 cross-camera identity pairs.

The annotation file can refer to displayed global IDs (``id_type: global``)
or local NvTracker IDs (``id_type: local``).  Local IDs are resolved through
the native world-identity JSONL log.  Global-ID annotations are useful when
the operator is labelling a dashboard screenshot; adding a timestamp or frame
number lets the evaluator verify that the ID existed in the recorded run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ID_RE = re.compile(r"^[gG](\d+)$")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if value.get("type") == "world_identity_assignment":
                rows.append(value)
    return rows


def load_csv_annotations(path: Path) -> list[dict[str, Any]]:
    """Load the operator-friendly one-row-per-second annotation format.

    Required columns are ``second``, ``left_id``, ``right_id`` and ``same``.
    ``reference_id`` is optional and is useful when the same person is marked
    at several seconds.  IDs are the labels visible in the dashboard, e.g.
    ``G67`` and ``G4``.
    """
    annotations: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"CSV annotation file is empty: {path}")
        required = {"second", "left_id", "right_id", "same"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"CSV annotations missing columns: {', '.join(missing)}; "
                "use second,reference_id,left_id,right_id,same"
            )
        for line_number, row in enumerate(reader, 2):
            if not any((value or "").strip() for value in row.values()):
                continue
            same = str(row.get("same", "")).strip().lower()
            if same not in {"yes", "y", "true", "1", "no", "n", "false", "0"}:
                raise ValueError(
                    f"CSV row {line_number}: same must be yes/no, got {same!r}"
                )
            left_id = (row.get("left_id") or "").strip()
            right_id = (row.get("right_id") or "").strip()
            if not left_id or not right_id:
                raise ValueError(
                    f"CSV row {line_number}: left_id and right_id are required"
                )
            second_text = (row.get("second") or "").strip()
            try:
                second = float(second_text)
            except ValueError as error:
                raise ValueError(
                    f"CSV row {line_number}: second must be a number"
                ) from error
            if second < 0:
                raise ValueError(f"CSV row {line_number}: second cannot be negative")
            reference_id = (row.get("reference_id") or "").strip()
            if not reference_id:
                reference_id = f"second_{second:g}_row_{line_number}"
            annotations.append({
                "reference_id": reference_id,
                "sample_second": second,
                "same_person": same in {"yes", "y", "true", "1"},
                "source_a": 0,
                "id_a": left_id,
                "source_b": 1,
                "id_b": right_id,
                "id_type": "global",
            })
    return annotations


def load_annotations(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return load_csv_annotations(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        document = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("PyYAML is required for YAML annotations; use .json instead") from error
        document = yaml.safe_load(text)
    if isinstance(document, list):
        annotations = document
    elif isinstance(document, dict):
        annotations = document.get("annotations", [])
    else:
        annotations = []
    if not isinstance(annotations, list) or not all(isinstance(item, dict) for item in annotations):
        raise ValueError("Annotations must be a list or an object containing an annotations list")
    return annotations


def parse_id(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid identity value: {value!r}")
    if isinstance(value, int):
        return value
    match = ID_RE.match(str(value).strip())
    if match:
        return int(match.group(1))
    return int(str(value).strip())


def source_value(annotation: dict[str, Any], side: str) -> int:
    value = annotation.get(f"source_{side}", annotation.get(f"camera_{side}"))
    if value is None:
        raise ValueError(f"Missing source_{side}/camera_{side}")
    return int(value)


def side_id(annotation: dict[str, Any], side: str) -> int:
    value = annotation.get(f"id_{side}")
    if value is None:
        value = annotation.get(f"global_id_{side}")
    if value is None:
        value = annotation.get(f"local_id_{side}")
    if value is None:
        raise ValueError(f"Missing id_{side}/global_id_{side}/local_id_{side}")
    return parse_id(value)


def side_type(annotation: dict[str, Any], side: str) -> str:
    value = annotation.get(f"id_type_{side}", annotation.get("id_type", "global"))
    value = str(value).lower()
    if value not in {"global", "local"}:
        raise ValueError(f"id_type must be global or local, got {value!r}")
    return value


def nearest_record(
    rows: list[dict[str, Any]],
    source: int,
    identity: int,
    identity_type: str,
    frame: int | None,
    timestamp_ns: int | None,
    tolerance_frames: int,
    tolerance_ns: int,
) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        if int(row.get("source_id", -1)) != source:
            continue
        field = "global_id" if identity_type == "global" else "local_object_id"
        if int(row.get(field, -1)) != identity:
            continue
        frame_delta = abs(int(row.get("frame", 0)) - frame) if frame is not None else 0
        time_delta = abs(int(row.get("timestamp", 0)) - timestamp_ns) if timestamp_ns is not None else 0
        if frame is not None and frame_delta > tolerance_frames:
            continue
        if timestamp_ns is not None and time_delta > tolerance_ns:
            continue
        candidates.append((frame_delta, time_delta, row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def evaluate(rows: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    results = []
    reference_ids: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    unresolved = []

    for index, annotation in enumerate(annotations, 1):
        try:
            source_a = source_value(annotation, "a")
            source_b = source_value(annotation, "b")
            id_a = side_id(annotation, "a")
            id_b = side_id(annotation, "b")
            type_a = side_type(annotation, "a")
            type_b = side_type(annotation, "b")
            frame_a = annotation.get("frame_a", annotation.get("frame"))
            frame_b = annotation.get("frame_b", annotation.get("frame"))
            timestamp_a = annotation.get("timestamp_a_ns", annotation.get("timestamp_ns"))
            timestamp_b = annotation.get("timestamp_b_ns", annotation.get("timestamp_ns"))
            tolerance_frames = int(annotation.get("tolerance_frames", 5))
            tolerance_seconds = float(annotation.get("tolerance_seconds", 2.0))
            tolerance_ns = int(tolerance_seconds * 1_000_000_000)
            record_a = nearest_record(rows, source_a, id_a, type_a,
                                      int(frame_a) if frame_a is not None else None,
                                      int(timestamp_a) if timestamp_a is not None else None,
                                      tolerance_frames, tolerance_ns)
            record_b = nearest_record(rows, source_b, id_b, type_b,
                                      int(frame_b) if frame_b is not None else None,
                                      int(timestamp_b) if timestamp_b is not None else None,
                                      tolerance_frames, tolerance_ns)
            # A dashboard annotation names the displayed G ID directly. If no
            # matching log row exists, preserve that value but flag the row as
            # unverified rather than silently dropping the operator's label.
            global_a = int(record_a["global_id"]) if record_a else (id_a if type_a == "global" else None)
            global_b = int(record_b["global_id"]) if record_b else (id_b if type_b == "global" else None)
            expected_same = bool(annotation.get("same_person", True))
            predicted_same = global_a is not None and global_b is not None and global_a == global_b
            resolved = record_a is not None and record_b is not None
            correct = predicted_same == expected_same if global_a is not None and global_b is not None else None
            if correct is None:
                counts["unresolved"] += 1
                unresolved.append(index)
            else:
                counts["evaluated"] += 1
                counts["correct" if correct else "incorrect"] += 1
                if expected_same:
                    counts["positive" if predicted_same else "missed_match"] += 1
                elif predicted_same:
                    counts["false_merge"] += 1
                else:
                    counts["negative"] += 1
            reference_id = str(annotation.get("reference_id", f"annotation_{index}"))
            if global_a is not None:
                reference_ids[reference_id][source_a].add(global_a)
            if global_b is not None:
                reference_ids[reference_id][source_b].add(global_b)
            results.append({
                "index": index,
                "reference_id": reference_id,
                "expected_same_person": expected_same,
                "predicted_same_global_id": predicted_same,
                "correct": correct,
                "resolved_from_log": resolved,
                "source_a": source_a,
                "source_b": source_b,
                "input_a": f"G{id_a}" if type_a == "global" else f"L{id_a}",
                "input_b": f"G{id_b}" if type_b == "global" else f"L{id_b}",
                "resolved_global_a": global_a,
                "resolved_global_b": global_b,
                "log_record_a": record_a,
                "log_record_b": record_b,
            })
        except (TypeError, ValueError) as error:
            counts["invalid_annotations"] += 1
            results.append({"index": index, "error": str(error)})

    tp = counts["positive"]
    fn = counts["missed_match"]
    fp = counts["false_merge"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
    switches = 0
    for sources in reference_ids.values():
        for identities in sources.values():
            switches += max(0, len(identities) - 1)
    return {
        "annotations": len(annotations),
        "counts": dict(counts),
        "positive_precision": precision,
        "positive_recall": recall,
        "positive_f1": f1,
        "identity_switches_in_annotations": switches,
        "unresolved_annotation_indexes": unresolved,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=Path("runs/mv3dt_world_identity.jsonl"))
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    try:
        report = evaluate(load_jsonl(args.assignments), load_annotations(args.annotations))
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    encoded = json.dumps(report, indent=2 if args.pretty else None, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
