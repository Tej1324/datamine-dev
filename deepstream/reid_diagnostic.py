"""Phase 2A-only diagnostic for NvDCF object Re-ID metadata."""

from __future__ import annotations

import atexit
import json
import math
import os
import queue
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from pyservicemaker import BatchMetadataOperator


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


class ReIdMetadataDiagnostic(BatchMetadataOperator):
    """Sample/validate per-object Re-ID metadata without touching video."""

    EXPECTED_FEATURE_SIZE = 256

    def __init__(self, source_labels: dict[int, str] | None = None, identity_manager=None, tracklet_store=None):
        super().__init__()
        self.source_labels = source_labels or {}
        self.identity_manager = identity_manager
        self.tracklet_store = tracklet_store
        self.osd_enabled = os.getenv("GLOBAL_ID_OSD", "0") == "1"
        # NvDCF/Re-ID extraction remains controlled by tracker.yml
        # (reidExtractionInterval=2).  Global-ID/gallery insertion is sampled
        # independently at roughly 0.5 s by default.
        self.sample_every_frames = _env_int(
            "GLOBAL_REID_SAMPLE_EVERY_FRAMES",
            _env_int("REID_DIAGNOSTIC_SAMPLE_EVERY", 5),
        )
        self.history_size = _env_int("REID_DIAGNOSTIC_HISTORY_SIZE", 10)
        self.include_vector = os.getenv("REID_DIAGNOSTIC_INCLUDE_VECTOR", "0") == "1"
        self.output_path = Path(os.getenv("REID_DIAGNOSTIC_OUTPUT", "runs/reid_diagnostic.jsonl"))
        self._queue: queue.Queue[dict | None] = queue.Queue(maxsize=_env_int("REID_DIAGNOSTIC_QUEUE_SIZE", 512))
        self._writer = threading.Thread(target=self._write_loop, name="reid-diagnostic-writer", daemon=True)
        self._writer.start()
        self._closed = False
        self._last_sampled_frame: dict[tuple[str, int], int] = {}
        self._history: dict[tuple[str, int], deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )
        self._latest_by_source: dict[str, dict[tuple[str, int], np.ndarray]] = defaultdict(dict)
        # Display-only cache.  It does not participate in matching; it keeps
        # the most recent assigned Global ID visible between diagnostic/Re-ID
        # samples and expires it using the existing Global-ID gap setting.
        self._display_global_ids: dict[tuple[str, int], tuple[int, float]] = {}
        self._display_global_id_ttl = float(os.getenv("GLOBAL_REID_MAX_GAP_SECONDS", "3600"))
        self.stats = {
            "total_frames": 0, "total_objects_seen": 0, "objects_with_reid": 0,
            "objects_without_reid": 0, "total_reid_vectors": 0,
            "valid_vectors": 0, "invalid_vectors": 0, "dropped_log_records": 0,
            "sampled_tracks": set(), "sources": set(),
            "same_track_similarity": [], "different_track_similarity": [],
            "extraction_ns": [], "copy_ns": [], "similarity_ns": [], "callback_ns": [],
        }
        self._lock = threading.Lock()
        atexit.register(self.close)

    @staticmethod
    def _bbox(rect) -> list[float]:
        return [float(rect.left), float(rect.top), float(rect.width), float(rect.height)]

    @staticmethod
    def _cosine(left: np.ndarray, right: np.ndarray) -> float:
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm <= 0.0 or right_norm <= 0.0:
            return float("nan")
        return float(np.dot(left, right) / (left_norm * right_norm))

    def _enqueue(self, record: dict) -> None:
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            with self._lock:
                self.stats["dropped_log_records"] += 1

    def _write_loop(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as output:
            while True:
                record = self._queue.get()
                if record is None:
                    return
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
                output.flush()

    def summary(self) -> dict:
        with self._lock:
            result = dict(self.stats)
            result["sampled_tracks"] = len(result["sampled_tracks"])
            result["sources"] = sorted(result["sources"])
            for key in ("same_track_similarity", "different_track_similarity"):
                values = result[key]
                result[key] = {
                    "count": len(values),
                    "min": min(values) if values else None,
                    "mean": sum(values) / len(values) if values else None,
                    "max": max(values) if values else None,
                }
            for key in ("extraction_ns", "copy_ns", "similarity_ns", "callback_ns"):
                values = result[key]
                result[key] = {
                    "count": len(values),
                    "mean_us": sum(values) / len(values) / 1000.0 if values else None,
                    "max_us": max(values) / 1000.0 if values else None,
                }
            return result

    def _emit_summary(self, timestamp: float) -> None:
        self._enqueue({"type": "summary", "timestamp": timestamp, "stats": self.summary()})

    def _set_global_id_osd(self, object_meta, key, timestamp_seconds: float) -> None:
        if not self.osd_enabled:
            return
        assigned = self._display_global_ids.get(key)
        global_id = None
        if assigned is not None and timestamp_seconds - assigned[1] <= self._display_global_id_ttl:
            global_id = assigned[0]
        text_params = object_meta.text_params
        text_params.display_text = f"G{global_id}" if global_id is not None else ""
        object_meta.text_params = text_params

    def handle_metadata(self, batch_meta) -> None:
        callback_started = time.perf_counter_ns()
        last_timestamp_seconds = None
        with self._lock:
            self.stats["total_frames"] += batch_meta.n_frames
        for frame_meta in batch_meta.frame_items:
            source_id = str(frame_meta.source_id)
            try:
                source_label = self.source_labels.get(int(frame_meta.source_id), source_id)
            except (TypeError, ValueError):
                source_label = source_id
            frame_number = int(frame_meta.frame_number)
            timestamp = int(frame_meta.ntp_timestamp or frame_meta.buffer_pts or 0)
            last_timestamp_seconds = self.tracklet_store._timestamp_seconds(timestamp) if self.tracklet_store is not None else None
            for object_meta in frame_meta.object_items:
                if int(object_meta.class_id) != 0:
                    continue
                object_id = int(object_meta.object_id)
                key = (source_id, object_id)
                timestamp_seconds = self.identity_manager._timestamp_seconds(timestamp) if self.identity_manager is not None else (
                    float(timestamp) / 1_000_000_000.0 if timestamp > 1_000_000_000 else float(timestamp)
                )
                with self._lock:
                    self.stats["total_objects_seen"] += 1
                    self.stats["sources"].add(source_id)
                self._set_global_id_osd(object_meta, key, timestamp_seconds)
                reid_items = list(object_meta.obj_reid_items)
                if not reid_items:
                    with self._lock:
                        self.stats["objects_without_reid"] += 1
                    continue
                with self._lock:
                    self.stats["objects_with_reid"] += 1
                    self.stats["total_reid_vectors"] += len(reid_items)
                if frame_number - self._last_sampled_frame.get(key, -self.sample_every_frames) < self.sample_every_frames:
                    continue
                self._last_sampled_frame[key] = frame_number
                for user_meta in reid_items:
                    extraction_started = time.perf_counter_ns()
                    reid_meta = user_meta.as_obj_reid()
                    feature_size = int(reid_meta.feature_size)
                    copy_started = time.perf_counter_ns()
                    vector = np.array(reid_meta.feature_vector, dtype=np.float32, copy=True).reshape(-1)
                    copy_finished = time.perf_counter_ns()
                    finite = bool(np.isfinite(vector).all())
                    norm = float(np.linalg.norm(vector)) if vector.size else 0.0
                    valid = feature_size == self.EXPECTED_FEATURE_SIZE and vector.size == feature_size and finite and norm > 0.0
                    visibility_value = getattr(object_meta, "visibility", None)
                    try:
                        visibility = float(visibility_value) if visibility_value is not None else None
                    except (TypeError, ValueError):
                        visibility = None
                    with self._lock:
                        self.stats["extraction_ns"].append(copy_finished - extraction_started)
                        self.stats["copy_ns"].append(copy_finished - copy_started)
                        self.stats["sampled_tracks"].add(key)
                        self.stats["valid_vectors" if valid else "invalid_vectors"] += 1
                    similarity_started = time.perf_counter_ns()
                    previous = self._history[key][-1] if self._history[key] else None
                    same_similarity = self._cosine(previous, vector) if previous is not None and valid else None
                    different_values = []
                    for other_key, other_vector in self._latest_by_source[source_id].items():
                        if other_key != key and valid:
                            value = self._cosine(other_vector, vector)
                            if math.isfinite(value):
                                different_values.append(value)
                    similarity_finished = time.perf_counter_ns()
                    with self._lock:
                        self.stats["similarity_ns"].append(similarity_finished - similarity_started)
                        if same_similarity is not None and math.isfinite(same_similarity):
                            self.stats["same_track_similarity"].append(same_similarity)
                        self.stats["different_track_similarity"].extend(different_values)
                    if valid:
                        self._history[key].append(vector)
                        self._latest_by_source[source_id][key] = vector
                    record = {
                        "type": "observation", "timestamp": timestamp,
                        "source_id": source_id, "source_label": source_label,
                        "local_object_id": object_id, "frame_number": frame_number,
                        "bbox": self._bbox(object_meta.rect_params),
                        "detector_confidence": float(object_meta.confidence),
                        "tracker_confidence": float(object_meta.tracker_confidence),
                        "visibility": visibility,
                        "feature_size": feature_size, "embedding_norm": norm,
                        "embedding_min": float(vector.min()) if vector.size else None,
                        "embedding_max": float(vector.max()) if vector.size else None,
                        "embedding_finite": finite, "valid": valid,
                        "embedding_sample": vector[:8].tolist(),
                        "same_track_cosine": same_similarity,
                        "different_track_cosines": different_values,
                    }
                    if self.identity_manager is not None and valid:
                        try:
                            from .global_identity import TrackObservation
                        except ImportError:
                            from global_identity import TrackObservation

                        assignment = self.identity_manager.observe(TrackObservation(
                            source_id=source_id,
                            local_object_id=object_id,
                            timestamp=timestamp,
                            timestamp_seconds=self.identity_manager._timestamp_seconds(timestamp),
                            bbox=tuple(record["bbox"]),
                            detector_confidence=record["detector_confidence"],
                            tracker_confidence=record["tracker_confidence"],
                            embedding=vector,
                            visibility=record["visibility"],
                        ))
                        assigned_global_id = assignment.get("assigned_global_id")
                        if assigned_global_id is not None:
                            if len(self._display_global_ids) >= 4096 and key not in self._display_global_ids:
                                self._display_global_ids.pop(next(iter(self._display_global_ids)))
                            self._display_global_ids[key] = (assigned_global_id, timestamp_seconds)
                        record["global_person_id"] = assignment.get("assigned_global_id")
                        record["global_assignment_reason"] = assignment.get("assignment_reason")
                        record["global_candidate_count"] = assignment.get("candidate_count")
                        record["global_best_similarity"] = assignment.get("best_similarity")
                        record["global_second_best_similarity"] = assignment.get("second_best_similarity")
                        record["global_margin"] = assignment.get("margin")
                        self._set_global_id_osd(object_meta, key, timestamp_seconds)
                    if self.tracklet_store is not None:
                        try:
                            from .tracklet_embeddings import TrackletEmbeddingObservation
                        except ImportError:
                            from tracklet_embeddings import TrackletEmbeddingObservation
                        self.tracklet_store.observe(TrackletEmbeddingObservation(
                            source_id=source_id,
                            local_object_id=object_id,
                            timestamp=timestamp,
                            timestamp_seconds=self.tracklet_store._timestamp_seconds(timestamp),
                            bbox=tuple(record["bbox"]),
                            detector_confidence=record["detector_confidence"],
                            tracker_confidence=record["tracker_confidence"],
                            embedding=vector,
                        ))
                    if self.include_vector:
                        record["embedding"] = vector.tolist()
                    self._enqueue(record)
        if self.tracklet_store is not None and last_timestamp_seconds is not None:
            self.tracklet_store.cleanup(last_timestamp_seconds)
        with self._lock:
            self.stats["callback_ns"].append(time.perf_counter_ns() - callback_started)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._emit_summary(time.time())
        if self.identity_manager is not None:
            self.identity_manager.close()
        if self.tracklet_store is not None:
            self.tracklet_store.close()
        try:
            self._queue.put_nowait(None)
            self._writer.join(timeout=2.0)
        except Exception:
            pass
