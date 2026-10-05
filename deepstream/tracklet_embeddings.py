"""Phase 2C-1 bounded local-track Re-ID embedding collector."""

from __future__ import annotations

import atexit
import itertools
import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class TrackletEmbeddingObservation:
    source_id: str
    local_object_id: int
    timestamp: int
    timestamp_seconds: float
    bbox: tuple[float, float, float, float]
    detector_confidence: float
    tracker_confidence: float
    embedding: np.ndarray


@dataclass
class LocalTracklet:
    source_id: str
    local_object_id: int
    first_seen_timestamp: int
    first_seen_seconds: float
    last_seen_timestamp: int
    last_seen_seconds: float
    observation_count: int = 0
    valid_embedding_count: int = 0
    rejected_embedding_count: int = 0
    embeddings: list[np.ndarray] = field(default_factory=list)
    embedding_timestamps: list[int] = field(default_factory=list)
    detector_confidences: list[float] = field(default_factory=list)
    tracker_confidences: list[float] = field(default_factory=list)
    bbox_sizes: list[tuple[float, float]] = field(default_factory=list)
    embedding_norms: list[float] = field(default_factory=list)
    last_accepted_seconds: float | None = None


class LocalTrackletEmbeddingStore:
    """Bounded diagnostic memory for one local NvDCF track at a time."""

    def __init__(self, representation=None):
        self.sample_every = _int_env("TRACKLET_EMBED_SAMPLE_EVERY", 5)
        self.max_per_track = _int_env("TRACKLET_EMBED_MAX_PER_TRACK", 12)
        self.min_gap_seconds = _float_env("TRACKLET_EMBED_MIN_GAP_MS", 250.0) / 1000.0
        self.min_det_conf = _float_env("TRACKLET_EMBED_MIN_DET_CONF", 0.0)
        self.min_bbox_height = _float_env("TRACKLET_EMBED_MIN_BBOX_HEIGHT", 0.0)
        self.min_bbox_width = _float_env("TRACKLET_EMBED_MIN_BBOX_WIDTH", 0.0)
        self.max_redundant_sim = _float_env("TRACKLET_EMBED_MAX_REDUNDANT_SIM", 0.995)
        self.ttl_seconds = _float_env("TRACKLET_MEMORY_TTL_SECONDS", 60.0)
        self.include_vector = os.getenv("TRACKLET_EMBED_INCLUDE_VECTOR", "0") == "1"
        self.output_path = Path(os.getenv("TRACKLET_EMBED_OUTPUT", "runs/tracklet_embeddings.jsonl"))
        self.representation = representation
        self._tracks: dict[tuple[str, int], LocalTracklet] = {}
        self._completed: dict[tuple[str, int], tuple[float, dict]] = {}
        self._queue: queue.Queue[dict | None] = queue.Queue(maxsize=_int_env("TRACKLET_EMBED_QUEUE_SIZE", 512))
        self._lock = threading.Lock()
        self._dropped_events = 0
        self._writer = threading.Thread(target=self._write_loop, name="tracklet-embedding-writer", daemon=True)
        self._writer.start()
        self._closed = False
        atexit.register(self.close)

    @staticmethod
    def _normalize(vector) -> tuple[np.ndarray | None, str | None]:
        copied = np.array(vector, dtype=np.float32, copy=True).reshape(-1)
        if copied.size != 256:
            return None, "INVALID_DIMENSION"
        if not np.isfinite(copied).all():
            return None, "NON_FINITE"
        norm = float(np.linalg.norm(copied))
        if not math.isfinite(norm) or norm <= 0.0:
            return None, "ZERO_NORM"
        return copied / norm, None

    @staticmethod
    def _cosine(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.dot(left, right))

    @staticmethod
    def _timestamp_seconds(raw: int) -> float:
        value = float(raw)
        return value / 1_000_000_000.0 if value > 1_000_000_000.0 else value

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

    def _new_track(self, observation: TrackletEmbeddingObservation) -> LocalTracklet:
        track = LocalTracklet(
            source_id=observation.source_id,
            local_object_id=observation.local_object_id,
            first_seen_timestamp=observation.timestamp,
            first_seen_seconds=observation.timestamp_seconds,
            last_seen_timestamp=observation.timestamp,
            last_seen_seconds=observation.timestamp_seconds,
        )
        self._tracks[(observation.source_id, observation.local_object_id)] = track
        self._emit({
            "type": "TRACK_STARTED", "timestamp": observation.timestamp,
            "source_id": observation.source_id, "local_object_id": observation.local_object_id,
        })
        return track

    def _summary(self, track: LocalTracklet) -> dict:
        vectors = track.embeddings
        norms = track.embedding_norms
        pairwise = [self._cosine(a, b) for a, b in itertools.combinations(vectors, 2)]
        prototype = None
        prototype_values = []
        if vectors:
            mean_vector = np.mean(np.stack(vectors), axis=0)
            mean_norm = float(np.linalg.norm(mean_vector))
            if mean_norm > 0.0 and math.isfinite(mean_norm):
                prototype = mean_vector / mean_norm
                prototype_values = [self._cosine(vector, prototype) for vector in vectors]
        duration = max(0.0, track.last_seen_seconds - track.first_seen_seconds)
        result = {
            "type": "TRACK_SUMMARY", "timestamp": track.last_seen_timestamp,
            "source_id": track.source_id, "local_object_id": track.local_object_id,
            "track_duration_seconds": duration,
            "observation_count": track.observation_count,
            "valid_embedding_count": track.valid_embedding_count,
            "rejected_embedding_count": track.rejected_embedding_count,
            "stored_embedding_count": len(vectors),
            "mean_embedding_norm": float(np.mean(norms)) if norms else None,
            "min_embedding_norm": min(norms) if norms else None,
            "max_embedding_norm": max(norms) if norms else None,
            "mean_pairwise_similarity": float(np.mean(pairwise)) if pairwise else None,
            "min_pairwise_similarity": min(pairwise) if pairwise else None,
            "max_pairwise_similarity": max(pairwise) if pairwise else None,
            "mean_prototype_similarity": float(np.mean(prototype_values)) if prototype_values else None,
            "min_prototype_similarity": min(prototype_values) if prototype_values else None,
            "max_prototype_similarity": max(prototype_values) if prototype_values else None,
            "std_prototype_similarity": float(np.std(prototype_values)) if prototype_values else None,
        }
        return result

    def _complete(self, key: tuple[str, int], track: LocalTracklet, now: float) -> None:
        summary = self._summary(track)
        if self.representation is not None:
            self.representation.complete(track, summary)
        self._completed[key] = (now, summary)
        self._emit({
            "type": "TRACK_COMPLETED", "timestamp": track.last_seen_timestamp,
            "source_id": track.source_id, "local_object_id": track.local_object_id,
            "track_duration_seconds": summary["track_duration_seconds"],
        })
        self._emit(summary)
        del self._tracks[key]

    def observe(self, observation: TrackletEmbeddingObservation) -> None:
        key = (observation.source_id, observation.local_object_id)
        with self._lock:
            track = self._tracks.get(key) or self._new_track(observation)
            track.last_seen_timestamp = observation.timestamp
            track.last_seen_seconds = observation.timestamp_seconds
            track.observation_count += 1
            track.detector_confidences.append(float(observation.detector_confidence))
            track.tracker_confidences.append(float(observation.tracker_confidence))
            track.bbox_sizes.append((float(observation.bbox[2]), float(observation.bbox[3])))
            observation_index = track.observation_count
            should_sample = observation_index == 1 or (observation_index - 1) % self.sample_every == 0
            if not should_sample:
                return
            if (
                track.last_accepted_seconds is not None
                and observation.timestamp_seconds - track.last_accepted_seconds < self.min_gap_seconds
            ):
                track.rejected_embedding_count += 1
                self._emit(self._rejection(observation, observation_index, "MIN_TIME_GAP"))
                return
            normalized, error = self._normalize(observation.embedding)
            if error:
                track.rejected_embedding_count += 1
                self._emit(self._rejection(observation, observation_index, error))
                return
            track.valid_embedding_count += 1
            if len(track.embeddings) >= self.max_per_track:
                track.rejected_embedding_count += 1
                self._emit(self._rejection(observation, observation_index, "MAX_PER_TRACK"))
                return
            if observation.detector_confidence < self.min_det_conf:
                track.rejected_embedding_count += 1
                self._emit(self._rejection(observation, observation_index, "LOW_DET_CONF"))
                return
            if observation.bbox[2] < self.min_bbox_width:
                track.rejected_embedding_count += 1
                self._emit(self._rejection(observation, observation_index, "SMALL_BBOX_WIDTH"))
                return
            if observation.bbox[3] < self.min_bbox_height:
                track.rejected_embedding_count += 1
                self._emit(self._rejection(observation, observation_index, "SMALL_BBOX_HEIGHT"))
                return
            similarity = None
            if track.embeddings:
                similarity = self._cosine(track.embeddings[-1], normalized)
                if similarity >= self.max_redundant_sim:
                    track.rejected_embedding_count += 1
                    self._emit(self._rejection(observation, observation_index, "REDUNDANT", similarity))
                    return
            track.embeddings.append(normalized)
            track.embedding_timestamps.append(observation.timestamp)
            track.embedding_norms.append(float(np.linalg.norm(np.array(observation.embedding, copy=True).reshape(-1))))
            track.last_accepted_seconds = observation.timestamp_seconds
            event = {
                "type": "EMBED_ACCEPTED", "timestamp": observation.timestamp,
                "source_id": observation.source_id, "local_object_id": observation.local_object_id,
                "sample_index": len(track.embeddings), "embedding_dim": 256,
                "embedding_norm": track.embedding_norms[-1],
                "detector_confidence": float(observation.detector_confidence),
                "tracker_confidence": float(observation.tracker_confidence),
                "bbox": list(observation.bbox),
                "elapsed_track_seconds": observation.timestamp_seconds - track.first_seen_seconds,
                "similarity_to_previous": similarity, "acceptance_reason": "VALID_DIVERSE",
            }
            if self.include_vector:
                event["embedding"] = normalized.tolist()
            self._emit(event)

    @staticmethod
    def _rejection(observation, sample_index, reason, similarity=None) -> dict:
        return {
            "type": "EMBED_REJECTED", "timestamp": observation.timestamp,
            "source_id": observation.source_id, "local_object_id": observation.local_object_id,
            "sample_index": sample_index, "reason": reason,
            "similarity_to_previous": similarity,
        }

    def cleanup(self, now_seconds: float) -> int:
        with self._lock:
            expired = [
                (key, track) for key, track in self._tracks.items()
                if now_seconds - track.last_seen_seconds >= self.ttl_seconds
            ]
            for key, track in expired:
                self._complete(key, track, now_seconds)
            old_completed = [
                key for key, (completed_at, _) in self._completed.items()
                if now_seconds - completed_at >= self.ttl_seconds
            ]
            for key in old_completed:
                del self._completed[key]
            return len(expired)

    def summaries(self) -> list[dict]:
        with self._lock:
            return [summary for _, summary in self._completed.values()]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
            self._writer.join(timeout=2.0)
        except Exception:
            pass
        if self.representation is not None:
            self.representation.close()
