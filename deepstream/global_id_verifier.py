"""Optional verifier for native MV3DT cross-camera object IDs."""

from __future__ import annotations

import threading
import time
import os
import json
from pathlib import Path

from pyservicemaker import BatchMetadataOperator


class GlobalIdVerifier(BatchMetadataOperator):
    """Report and optionally display native MV3DT cross-camera object IDs."""

    def __init__(self, report_seconds: float = 10.0, max_frame_gap: int = 30):
        super().__init__()
        self.report_seconds = report_seconds
        self.max_frame_gap = max_frame_gap
        self.display_ids = os.getenv("MV3DT_GLOBAL_ID_OSD", "0").lower() in {"1", "true", "yes", "on"}
        self.started = time.monotonic()
        self.last_report = self.started
        self.frames = 0
        self.objects = 0
        self.sources: dict[str, dict[int, int]] = {}
        self.last_seen: dict[int, dict[str, int]] = {}
        self.confirmed: set[tuple[int, str, str]] = set()
        configured_sources = [
            item.strip()
            for item in os.getenv("MV3DT_CAMERA_IDS", "").split(",")
            if item.strip()
        ]
        self.source_labels = dict(enumerate(configured_sources))
        self.journey_output = Path(os.getenv(
            "MV3DT_JOURNEY_OUTPUT", "/workspace/runs/mv3dt_journeys.jsonl"
        ))
        self.journey_output.parent.mkdir(parents=True, exist_ok=True)
        self.journeys: dict[int, dict] = {}
        self.lock = threading.Lock()

    @staticmethod
    def _event_epoch(frame_meta) -> float:
        """Use camera/NTP time for dwell, not delayed processing time."""
        raw = int(getattr(frame_meta, "ntp_timestamp", 0) or 0)
        if raw > 10_000_000_000:
            return raw / 1_000_000_000.0
        return time.time()

    def _write_journey_snapshot(self, now: float) -> None:
        """Persist compact dwell/journey state without changing tracker IDs."""
        snapshot = {
            "type": "journey_snapshot",
            "written_at": time.time(),
            "monotonic_seconds": now,
            "identities": [
                {
                    **journey,
                    "cameras": sorted(journey["cameras"]),
                    "dwell_seconds": max(
                        0.0,
                        journey["last_seen_epoch"] - journey["first_seen_epoch"],
                    ),
                }
                for journey in sorted(self.journeys.values(), key=lambda item: item["global_id"])
            ],
        }
        with self.journey_output.open("a", encoding="utf-8") as output:
            output.write(json.dumps(snapshot, separators=(",", ":")) + "\n")

    def handle_metadata(self, batch_meta) -> None:
        now = time.monotonic()
        with self.lock:
            self.frames += int(batch_meta.n_frames)
            for frame_meta in batch_meta.frame_items:
                source_index = int(frame_meta.source_id)
                source = self.source_labels.get(source_index, str(source_index))
                frame_number = int(frame_meta.frame_number)
                event_epoch = self._event_epoch(frame_meta)
                ids = self.sources.setdefault(source, {})
                for object_meta in frame_meta.object_items:
                    if int(object_meta.class_id) != 0:
                        continue
                    object_id = int(object_meta.object_id)
                    if object_id == 0xFFFFFFFFFFFFFFFF:
                        continue
                    if self.display_ids:
                        text_params = object_meta.text_params
                        text_params.display_text = f"G{object_id}"
                        object_meta.text_params = text_params
                    self.objects += 1
                    ids[object_id] = ids.get(object_id, 0) + 1
                    journey = self.journeys.setdefault(object_id, {
                        "global_id": object_id,
                        "first_seen_monotonic": now,
                        "last_seen_monotonic": now,
                        "first_seen_epoch": event_epoch,
                        "last_seen_epoch": event_epoch,
                        "cameras": set(),
                        "observations": 0,
                        "cross_camera": False,
                    })
                    journey["last_seen_monotonic"] = now
                    journey["last_seen_epoch"] = event_epoch
                    journey["cameras"].add(source)
                    journey["observations"] += 1
                    journey["cross_camera"] = len(journey["cameras"]) > 1
                    seen = self.last_seen.setdefault(object_id, {})
                    for other_source, other_frame in seen.items():
                        if other_source != source and abs(frame_number - other_frame) <= self.max_frame_gap:
                            pair = (object_id, min(other_source, source), max(other_source, source))
                            self.confirmed.add(pair)
                    seen[source] = frame_number
            if now - self.last_report < self.report_seconds:
                return
            elapsed = max(now - self.started, 1e-6)
            source_counts = {
                source: len(ids) for source, ids in sorted(self.sources.items())
            }
            confirmed = sorted(self.confirmed)
            self.last_report = now
            self._write_journey_snapshot(now)
        print(
            "GLOBAL_ID_VERIFY: "
            f"frames={self.frames} objects={self.objects} "
            f"avg_fps={self.frames / elapsed:.2f} "
            f"unique_ids_by_source={source_counts} "
            f"cross_source_overlaps={confirmed[-20:]}",
            flush=True,
        )
