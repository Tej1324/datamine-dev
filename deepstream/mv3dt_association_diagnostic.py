"""Read-only MV3DT association diagnostics.

This probe never assigns, merges, or changes an ID.  It records the ID exposed
by NvMultiObjectTracker, source/frame timestamps, image-space feet, and any
world-foot coordinates exposed by the Python metadata wrapper.  It also emits
nearest cross-camera candidates so calibration and timing errors can be
distinguished from tracker decisions.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

from pyservicemaker import BatchMetadataOperator


def _float(value):
    try:
        value = float(value)
        return value if value == value else None
    except (TypeError, ValueError):
        return None


def _timestamp(raw: int) -> float:
    value = float(raw or 0)
    return value / 1_000_000_000.0 if value > 1_000_000_000 else value


class Mv3dtAssociationDiagnostic(BatchMetadataOperator):
    """Collect association evidence without participating in matching."""

    def __init__(self):
        super().__init__()
        self.output = Path(os.getenv("MV3DT_ASSOC_DIAGNOSTIC_OUTPUT", "/workspace/runs/mv3dt_association.jsonl"))
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.max_time_delta = float(os.getenv("MV3DT_ASSOC_DIAGNOSTIC_MAX_TIME_DELTA", "0.75"))
        self.max_candidates = max(1, int(os.getenv("MV3DT_ASSOC_DIAGNOSTIC_MAX_CANDIDATES", "5")))
        self.report_seconds = float(os.getenv("MV3DT_ASSOC_DIAGNOSTIC_REPORT_SECONDS", "10"))
        self._queue: queue.Queue[dict | None] = queue.Queue(maxsize=2048)
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._writer.start()
        self._started = time.monotonic()
        self._last_report = self._started
        self._frames = 0
        self._objects = 0
        self._latest: dict[str, list[dict]] = {}
        self._shared: set[tuple[int, str, str]] = set()

    @staticmethod
    def _world(object_meta):
        # Wrapper versions expose tracker fields with either spelling.  Keep
        # null when unavailable rather than inventing a world coordinate.
        for name in ("pt_world_feet", "ptWorldFeet", "world_feet", "worldFeet"):
            value = getattr(object_meta, name, None)
            if value is not None:
                try:
                    x = _float(value[0])
                    y = _float(value[1])
                    if x is not None and y is not None:
                        return [x, y]
                except (IndexError, TypeError, KeyError):
                    pass
        return None

    @staticmethod
    def _feet(object_meta):
        rect = object_meta.rect_params
        left, top = float(rect.left), float(rect.top)
        return [left + float(rect.width) / 2.0, top + float(rect.height)]

    @staticmethod
    def _distance(left, right):
        if left is None or right is None:
            return None
        dx = left[0] - right[0]
        dy = left[1] - right[1]
        return (dx * dx + dy * dy) ** 0.5

    def _enqueue(self, record):
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            pass

    def _write_loop(self):
        with self.output.open("a", encoding="utf-8") as output:
            while True:
                record = self._queue.get()
                if record is None:
                    return
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
                output.flush()

    def _report(self):
        now = time.monotonic()
        if now - self._last_report < self.report_seconds:
            return
        self._last_report = now
        print(
            "MV3DT_ASSOC_DIAGNOSTIC: "
            f"frames={self._frames} objects={self._objects} "
            f"avg_fps={self._frames / max(now - self._started, 1e-6):.2f} "
            f"shared_reported_ids={sorted(self._shared)[-20:]}",
            flush=True,
        )

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            source = str(frame_meta.source_id)
            frame = int(frame_meta.frame_number)
            raw_timestamp = int(frame_meta.ntp_timestamp or frame_meta.buffer_pts or 0)
            seconds = _timestamp(raw_timestamp)
            current = []
            for object_meta in frame_meta.object_items:
                if int(object_meta.class_id) != 0:
                    continue
                reported_id = int(object_meta.object_id)
                if reported_id == 0xFFFFFFFFFFFFFFFF:
                    continue
                item = {
                    "source_id": source,
                    "frame_number": frame,
                    "timestamp": raw_timestamp,
                    "timestamp_seconds": seconds,
                    "reported_tracker_id": reported_id,
                    "image_feet": self._feet(object_meta),
                    "world_feet": self._world(object_meta),
                    "bbox": [float(object_meta.rect_params.left), float(object_meta.rect_params.top),
                             float(object_meta.rect_params.width), float(object_meta.rect_params.height)],
                    "detector_confidence": float(object_meta.confidence),
                    "tracker_confidence": float(object_meta.tracker_confidence),
                }
                candidates = []
                for other_source, previous in self._latest.items():
                    if other_source == source:
                        continue
                    for other in previous:
                        delta = abs(seconds - other["timestamp_seconds"])
                        if delta <= self.max_time_delta:
                            world_distance = self._distance(item["world_feet"], other["world_feet"])
                            image_distance = self._distance(item["image_feet"], other["image_feet"])
                            candidates.append({
                                "source_id": other_source,
                                "reported_tracker_id": other["reported_tracker_id"],
                                "time_delta_seconds": delta,
                                "world_distance": world_distance,
                                "image_feet_distance": image_distance,
                            })
                            if item["reported_tracker_id"] == other["reported_tracker_id"]:
                                self._shared.add((reported_id, min(source, other_source), max(source, other_source)))
                candidates.sort(key=lambda value: (
                    value["world_distance"] is None,
                    value["world_distance"] if value["world_distance"] is not None else value["image_feet_distance"],
                ))
                item["peer_candidates"] = candidates[: self.max_candidates]
                current.append(item)
                self._enqueue({"type": "track_observation", **item})
                self._objects += 1
            self._latest[source] = current
            self._frames += 1
        self._report()

    def close(self):
        self._enqueue({"type": "summary", "frames": self._frames, "objects": self._objects,
                       "shared_reported_ids": sorted(self._shared)})
        self._queue.put(None)
        self._writer.join(timeout=2.0)
