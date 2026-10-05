"""Metadata-only Global Identity association for the six-camera pipeline.

NvDCF remains responsible for short-term local tracking and NVIDIA remains
responsible for producing the normalized 256-D Re-ID features. This module
only owns copied metadata and copied feature vectors; it never accesses video
surfaces or GstBuffers.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _csv_env(name: str, default: str) -> set[str]:
    value = os.getenv(name, default)
    return {item.strip() for item in value.split(",") if item.strip()}


@dataclass(frozen=True)
class TrackObservation:
    source_id: str
    local_object_id: int
    timestamp: int
    timestamp_seconds: float
    bbox: tuple[float, float, float, float]
    detector_confidence: float
    tracker_confidence: float
    embedding: np.ndarray
    visibility: float | None = None


@dataclass(frozen=True)
class GalleryEntry:
    vector: np.ndarray
    source_id: str
    timestamp_seconds: float
    quality: float


@dataclass
class LocalIdentityState:
    source_id: str
    local_object_id: int
    first_seen_seconds: float
    last_seen_seconds: float
    assigned_global_id: int | None = None
    valid_observations: int = 0
    good_observations: int = 0
    pending: deque[tuple[np.ndarray, float, float]] = field(default_factory=deque)
    last_gallery_update_seconds: float | None = None


@dataclass
class GlobalPerson:
    global_id: int
    first_seen_seconds: float
    last_seen_seconds: float
    last_source_id: str
    cameras_seen: set[str] = field(default_factory=set)
    active_tracks: dict[str, tuple[int, float]] = field(default_factory=dict)
    gallery: deque[GalleryEntry] = field(default_factory=deque)
    confidence_history: deque[float] = field(default_factory=lambda: deque(maxlen=20))


class GlobalIdentityManager:
    """Conservative 60-minute, multi-view Global ID manager.

    ``ground_01``/``ground_03``/``ground_06`` (CH7/CH10/CH11) are identity
    birth cameras by default. The other cameras are observer cameras: they
    may recover an existing identity but can never create a new one.
    """

    def __init__(self):
        self.match_threshold = _float_env("GLOBAL_REID_MATCH_THRESHOLD", 0.62)
        self.min_margin = _float_env("GLOBAL_REID_MIN_MARGIN", 0.05)
        self.max_gap_seconds = _float_env("GLOBAL_REID_MAX_GAP_SECONDS", 3600.0)
        self.gallery_size = _int_env("GLOBAL_REID_GALLERY_SIZE", 24)
        self.gallery_min_interval = _float_env("GLOBAL_REID_GALLERY_MIN_INTERVAL_SECONDS", 0.5)
        self.min_good_observations = _int_env("GLOBAL_REID_MIN_GOOD_OBSERVATIONS", 3)
        self.max_pending_observations = _int_env("GLOBAL_REID_MAX_PENDING_OBSERVATIONS", 8)
        self.gallery_top_k = _int_env("GLOBAL_REID_GALLERY_TOP_K", 3)
        self.gallery_redundant_similarity = _float_env("GLOBAL_REID_GALLERY_REDUNDANT_SIMILARITY", 0.995)
        self.same_camera_active_gap = _float_env("GLOBAL_REID_SAME_CAMERA_ACTIVE_GAP_SECONDS", 3.0)
        self.min_detector_confidence = _float_env("GLOBAL_REID_MIN_DETECTOR_CONFIDENCE", 0.30)
        self.min_tracker_confidence = _float_env("GLOBAL_REID_MIN_TRACKER_CONFIDENCE", 0.30)
        self.min_visibility = _float_env("GLOBAL_REID_MIN_VISIBILITY", 0.60)
        self.min_bbox_width = _float_env("GLOBAL_REID_MIN_BBOX_WIDTH", 24.0)
        self.min_bbox_height = _float_env("GLOBAL_REID_MIN_BBOX_HEIGHT", 64.0)
        self.identity_cameras = _csv_env(
            "GLOBAL_REID_IDENTITY_CAMERAS", "ground_01,ground_03,ground_06"
        )
        self.observer_cameras = _csv_env(
            "GLOBAL_REID_OBSERVER_CAMERAS", "ground_02,ground_04,ground_05"
        )
        self._source_camera_aliases = {
            "0": "ground_01", "1": "ground_02", "2": "ground_03",
            "3": "ground_04", "4": "ground_05", "5": "ground_06",
        }
        self.output_path = Path(os.getenv("GLOBAL_REID_OUTPUT", "runs/global_identity.jsonl"))
        self._lock = threading.Lock()
        self._next_global_id = _int_env("GLOBAL_REID_START_ID", 1)
        self._persons: dict[int, GlobalPerson] = {}
        self._local_tracks: dict[tuple[str, int], LocalIdentityState] = {}
        self._queue: queue.Queue[dict | None] = queue.Queue(
            maxsize=_int_env("GLOBAL_REID_QUEUE_SIZE", 512)
        )
        self._dropped_events = 0
        self._association_ns: list[int] = []
        self._match_count = 0
        self._created_count = 0
        self._ambiguous_count = 0
        self._same_camera_rejections = 0
        self._unassigned_count = 0
        self._gallery_updates = 0
        self._writer = threading.Thread(target=self._write_loop, name="global-id-writer", daemon=True)
        self._writer.start()
        self._closed = False
        atexit.register(self.close)

    @staticmethod
    def _timestamp_seconds(raw: int) -> float:
        value = float(raw)
        return value / 1_000_000_000.0 if value > 1_000_000_000.0 else value

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray | None:
        vector = np.array(vector, dtype=np.float32, copy=True).reshape(-1)
        if vector.size != 256 or not np.isfinite(vector).all():
            return None
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 0.0:
            return None
        return np.array(vector / norm, dtype=np.float32, copy=True)

    @staticmethod
    def _cosine(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.dot(left, right))

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    def _write_loop(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as output:
            while True:
                event = self._queue.get()
                if event is None:
                    return
                output.write(json.dumps(event, separators=(",", ":")) + "\n")
                output.flush()

    def _emit(self, event: dict) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._dropped_events += 1

    def _camera_mode(self, source_id: str) -> str:
        camera_names = {source_id, self._source_camera_aliases.get(source_id, source_id)}
        if camera_names & self.identity_cameras:
            return "identity_birth"
        # Unknown sources are safe by default: they cannot create identities.
        return "observer_only"

    def _quality(self, observation: TrackObservation) -> tuple[bool, float, str]:
        detector = self._clamp(float(observation.detector_confidence))
        tracker = self._clamp(float(observation.tracker_confidence))
        visibility = 1.0 if observation.visibility is None else self._clamp(float(observation.visibility))
        width = float(observation.bbox[2])
        height = float(observation.bbox[3])
        if detector < self.min_detector_confidence:
            return False, 0.0, "LOW_DETECTOR_CONFIDENCE"
        if tracker < self.min_tracker_confidence:
            return False, 0.0, "LOW_TRACKER_CONFIDENCE"
        if visibility < self.min_visibility:
            return False, 0.0, "LOW_VISIBILITY"
        if width < self.min_bbox_width or height < self.min_bbox_height:
            return False, 0.0, "SMALL_BBOX"
        size_score = self._clamp(min(
            width / max(self.min_bbox_width * 2.0, 1.0),
            height / max(self.min_bbox_height * 2.0, 1.0),
        ))
        quality = 0.35 * detector + 0.35 * tracker + 0.20 * visibility + 0.10 * size_score
        return True, quality, "QUALITY_GATE_PASSED"

    def _purge_expired(self, now: float) -> None:
        for global_id in [
            global_id for global_id, person in self._persons.items()
            if now - person.last_seen_seconds > self.max_gap_seconds
        ]:
            del self._persons[global_id]
            self._emit({"type": "GLOBAL_ID_EXPIRED", "timestamp": now, "global_id": global_id})
        for key in [
            key for key, state in self._local_tracks.items()
            if now - state.last_seen_seconds > self.max_gap_seconds
        ]:
            del self._local_tracks[key]

    def _gallery_similarity(self, probes: list[np.ndarray], person: GlobalPerson) -> tuple[float, float]:
        values = sorted(
            (self._cosine(probe, entry.vector) for probe in probes for entry in person.gallery),
            reverse=True,
        )
        if not values:
            return 0.0, 0.0
        top = values[: self.gallery_top_k]
        return float(0.70 * top[0] + 0.30 * float(np.mean(top))), float(top[0])

    def _candidate_search(self, observation: TrackObservation, probes: list[np.ndarray]):
        candidates = []
        same_camera_rejections = []
        for person in self._persons.values():
            gap = observation.timestamp_seconds - person.last_seen_seconds
            if gap < 0.0 or gap > self.max_gap_seconds:
                continue
            active = person.active_tracks.get(observation.source_id)
            if active is not None:
                active_local_id, active_time = active
                if (
                    active_local_id != observation.local_object_id
                    and observation.timestamp_seconds - active_time <= self.same_camera_active_gap
                ):
                    same_camera_rejections.append((person, gap))
                    continue
            score, raw_best = self._gallery_similarity(probes, person)
            candidates.append((score, raw_best, person, gap))
        return candidates, same_camera_rejections

    def _append_gallery(
        self, person: GlobalPerson, observation: TrackObservation,
        normalized: np.ndarray, quality: float,
    ) -> tuple[bool, str]:
        if person.gallery and max(
            self._cosine(normalized, entry.vector) for entry in person.gallery
        ) >= self.gallery_redundant_similarity:
            return False, "REDUNDANT_VIEW"
        entry = GalleryEntry(
            np.array(normalized, copy=True), observation.source_id,
            observation.timestamp_seconds, quality,
        )
        same_source_indices = [
            index for index, item in enumerate(person.gallery)
            if item.source_id == observation.source_id
        ]
        per_source_limit = max(1, self.gallery_size // max(len(self.identity_cameras), 1))
        if len(person.gallery) < self.gallery_size:
            person.gallery.append(entry)
        elif len(same_source_indices) < per_source_limit:
            weakest = min(range(len(person.gallery)), key=lambda index: person.gallery[index].quality)
            person.gallery[weakest] = entry
        else:
            weakest = min(same_source_indices, key=lambda index: person.gallery[index].quality)
            if person.gallery[weakest].quality >= quality:
                return False, "LOWER_QUALITY_THAN_EXISTING_VIEW"
            person.gallery[weakest] = entry
        self._gallery_updates += 1
        return True, "QUALITY_DIVERSE_VIEW"

    @staticmethod
    def _event(observation: TrackObservation, result: dict) -> dict:
        return {
            "type": "assignment", "timestamp": observation.timestamp,
            "source_id": observation.source_id, "local_object_id": observation.local_object_id,
            **result,
        }

    def observe(self, observation: TrackObservation) -> dict:
        started = time.perf_counter_ns()
        normalized = self._normalize(observation.embedding)
        if normalized is None:
            return {"assigned_global_id": None, "assignment_reason": "INVALID_EMBEDDING"}
        quality_ok, quality, quality_reason = self._quality(observation)
        with self._lock:
            self._purge_expired(observation.timestamp_seconds)
            key = (observation.source_id, observation.local_object_id)
            state = self._local_tracks.get(key)
            if state is None:
                state = LocalIdentityState(
                    observation.source_id, observation.local_object_id,
                    observation.timestamp_seconds, observation.timestamp_seconds,
                    pending=deque(maxlen=self.max_pending_observations),
                )
                self._local_tracks[key] = state
            state.last_seen_seconds = observation.timestamp_seconds
            state.valid_observations += 1

            if not quality_ok:
                result = {
                    "assigned_global_id": state.assigned_global_id,
                    "assignment_reason": "ASSIGNED_POOR_QUALITY" if state.assigned_global_id else quality_reason,
                    "candidate_count": 0, "best_similarity": None,
                    "second_best_similarity": None, "margin": None,
                    "camera_mode": self._camera_mode(observation.source_id),
                    "quality_score": quality,
                }
                self._association_ns.append(time.perf_counter_ns() - started)
                return result

            state.good_observations += 1
            state.pending.append((normalized, quality, observation.timestamp_seconds))
            if state.assigned_global_id is not None:
                person = self._persons.get(state.assigned_global_id)
                if person is not None:
                    previous_camera = person.last_source_id
                    previous_seen = person.last_seen_seconds
                    person.last_seen_seconds = observation.timestamp_seconds
                    person.last_source_id = observation.source_id
                    person.cameras_seen.add(observation.source_id)
                    person.active_tracks[observation.source_id] = (
                        observation.local_object_id, observation.timestamp_seconds
                    )
                    person.confidence_history.append(quality)
                    should_update = (
                        state.last_gallery_update_seconds is None
                        or observation.timestamp_seconds - state.last_gallery_update_seconds >= self.gallery_min_interval
                    )
                    inserted, insert_reason = (False, "GALLERY_INTERVAL")
                    if should_update:
                        inserted, insert_reason = self._append_gallery(person, observation, normalized, quality)
                        if inserted:
                            state.last_gallery_update_seconds = observation.timestamp_seconds
                            self._emit({
                                "type": "gallery_update", "timestamp": observation.timestamp,
                                "global_id": person.global_id, "source_id": observation.source_id,
                                "local_object_id": observation.local_object_id,
                                "quality_score": quality, "update_reason": insert_reason,
                            })
                    result = {
                        "assigned_global_id": person.global_id,
                        "assignment_reason": "SAME_LOCAL_TRACK_GALLERY_UPDATED" if inserted else "SAME_LOCAL_TRACK",
                        "candidate_count": 0, "best_similarity": None,
                        "second_best_similarity": None, "margin": None,
                        "previous_camera": previous_camera,
                        "time_gap": observation.timestamp_seconds - previous_seen,
                        "camera_mode": self._camera_mode(observation.source_id),
                        "quality_score": quality,
                    }
                    self._association_ns.append(time.perf_counter_ns() - started)
                    return result
                state.assigned_global_id = None

            if state.good_observations < self.min_good_observations:
                result = {
                    "assigned_global_id": None, "assignment_reason": "PENDING_GOOD_EMBEDDINGS",
                    "candidate_count": 0, "best_similarity": None,
                    "second_best_similarity": None, "margin": None,
                    "camera_mode": self._camera_mode(observation.source_id),
                    "good_observations": state.good_observations,
                    "required_good_observations": self.min_good_observations,
                    "quality_score": quality,
                }
                self._association_ns.append(time.perf_counter_ns() - started)
                return result

            probes = [item[0] for item in state.pending]
            candidates, same_camera_rejections = self._candidate_search(observation, probes)
            for person, gap in same_camera_rejections:
                self._same_camera_rejections += 1
                self._emit({
                    "type": "rejection", "timestamp": observation.timestamp,
                    "source_id": observation.source_id, "local_object_id": observation.local_object_id,
                    "candidate_global_id": person.global_id,
                    "reason": "SAME_CAMERA_ACTIVE_TRACK", "previous_camera": person.last_source_id,
                    "time_gap": gap,
                })
            candidates.sort(key=lambda item: item[0], reverse=True)
            best = candidates[0] if candidates else None
            second = candidates[1][0] if len(candidates) > 1 else None
            margin = best[0] - second if best and second is not None else (best[0] if best else None)
            camera_mode = self._camera_mode(observation.source_id)
            if best and best[0] >= self.match_threshold and margin >= self.min_margin:
                person = best[2]
                previous_camera = person.last_source_id
                time_gap = observation.timestamp_seconds - person.last_seen_seconds
                state.assigned_global_id = person.global_id
                person.last_seen_seconds = observation.timestamp_seconds
                person.last_source_id = observation.source_id
                person.cameras_seen.add(observation.source_id)
                person.active_tracks[observation.source_id] = (
                    observation.local_object_id, observation.timestamp_seconds
                )
                person.confidence_history.append(quality)
                inserted, insert_reason = self._append_gallery(person, observation, normalized, quality)
                if inserted:
                    state.last_gallery_update_seconds = observation.timestamp_seconds
                self._match_count += 1
                result = {
                    "assigned_global_id": person.global_id,
                    "assignment_reason": "CROSS_CAMERA_REID_MATCH" if previous_camera != observation.source_id else "REID_RECOVERY",
                    "candidate_count": len(candidates), "best_similarity": best[0],
                    "best_gallery_similarity": best[1], "second_best_similarity": second,
                    "margin": margin, "previous_camera": previous_camera,
                    "time_gap": time_gap, "camera_mode": camera_mode,
                    "quality_score": quality, "gallery_update_reason": insert_reason,
                }
            elif best and best[0] >= self.match_threshold and margin < self.min_margin:
                self._ambiguous_count += 1
                self._unassigned_count += 1
                result = {
                    "assigned_global_id": None, "assignment_reason": "UNASSIGNED_LOW_MARGIN",
                    "candidate_count": len(candidates), "best_similarity": best[0],
                    "best_gallery_similarity": best[1], "second_best_similarity": second,
                    "margin": margin, "previous_camera": best[2].last_source_id,
                    "time_gap": best[3], "camera_mode": camera_mode,
                    "quality_score": quality,
                }
            elif camera_mode == "observer_only":
                self._unassigned_count += 1
                result = {
                    "assigned_global_id": None,
                    "assignment_reason": "UNASSIGNED_LOW_SIMILARITY" if best else "UNASSIGNED_NO_MATCH",
                    "candidate_count": len(candidates),
                    "best_similarity": best[0] if best else None,
                    "best_gallery_similarity": best[1] if best else None,
                    "second_best_similarity": second, "margin": margin,
                    "previous_camera": best[2].last_source_id if best else None,
                    "time_gap": best[3] if best else None,
                    "camera_mode": camera_mode, "quality_score": quality,
                }
            else:
                global_id = self._next_global_id
                self._next_global_id += 1
                person = GlobalPerson(
                    global_id=global_id,
                    first_seen_seconds=observation.timestamp_seconds,
                    last_seen_seconds=observation.timestamp_seconds,
                    last_source_id=observation.source_id,
                    cameras_seen={observation.source_id},
                    active_tracks={observation.source_id: (
                        observation.local_object_id, observation.timestamp_seconds
                    )},
                    gallery=deque(maxlen=self.gallery_size),
                )
                self._persons[global_id] = person
                state.assigned_global_id = global_id
                self._append_gallery(person, observation, normalized, quality)
                state.last_gallery_update_seconds = observation.timestamp_seconds
                self._created_count += 1
                result = {
                    "assigned_global_id": global_id,
                    "assignment_reason": "NEW_GLOBAL_PERSON_AFTER_CONFIRMATION",
                    "candidate_count": len(candidates),
                    "best_similarity": best[0] if best else None,
                    "best_gallery_similarity": best[1] if best else None,
                    "second_best_similarity": second, "margin": margin,
                    "previous_camera": best[2].last_source_id if best else None,
                    "time_gap": best[3] if best else None,
                    "camera_mode": camera_mode, "quality_score": quality,
                }
            self._emit(self._event(observation, result))
            self._association_ns.append(time.perf_counter_ns() - started)
            return result

    def summary(self) -> dict:
        with self._lock:
            latencies = list(self._association_ns)
            return {
                "global_persons": len(self._persons),
                "gallery_vectors": sum(len(person.gallery) for person in self._persons.values()),
                "cross_camera_matches": self._match_count,
                "new_global_persons": self._created_count,
                "ambiguous_cases": self._ambiguous_count,
                "unassigned_observations": self._unassigned_count,
                "same_camera_active_track_rejections": self._same_camera_rejections,
                "gallery_updates": self._gallery_updates,
                "dropped_events": self._dropped_events,
                "association_latency": {
                    "count": len(latencies),
                    "mean_us": sum(latencies) / len(latencies) / 1000.0 if latencies else None,
                    "max_us": max(latencies) / 1000.0 if latencies else None,
                },
                "settings": {
                    "match_threshold": self.match_threshold,
                    "min_margin": self.min_margin,
                    "max_gap_seconds": self.max_gap_seconds,
                    "gallery_size": self.gallery_size,
                    "gallery_min_interval_seconds": self.gallery_min_interval,
                    "min_good_observations": self.min_good_observations,
                    "identity_cameras": sorted(self.identity_cameras),
                    "observer_cameras": sorted(self.observer_cameras),
                },
            }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._emit({"type": "summary", "timestamp": time.time(), "stats": self.summary()})
        try:
            self._queue.put_nowait(None)
            self._writer.join(timeout=2.0)
        except Exception:
            pass
