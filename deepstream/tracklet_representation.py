"""Diagnostic prototype and exemplar representations for completed tracklets."""

from __future__ import annotations

import atexit
import json
import math
import os
import queue
import threading
from pathlib import Path

import numpy as np


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except ValueError:
        return default


class TrackletRepresentation:
    """Build bounded, owned representations when a local tracklet completes."""

    def __init__(self):
        self.exemplar_count = _int_env("TRACKLET_EXEMPLAR_COUNT", 4)
        self.min_diversity = _float_env("TRACKLET_EXEMPLAR_MIN_DIVERSITY", 0.03)
        self.include_vectors = os.getenv("TRACKLET_REPRESENTATION_INCLUDE_VECTORS", "0") == "1"
        self.output_path = Path(os.getenv(
            "TRACKLET_REPRESENTATION_OUTPUT", "runs/tracklet_representations.jsonl"
        ))
        self._queue: queue.Queue[dict | None] = queue.Queue(maxsize=_int_env(
            "TRACKLET_REPRESENTATION_QUEUE_SIZE", 512
        ))
        self._writer = threading.Thread(
            target=self._write_loop, name="tracklet-representation-writer", daemon=True
        )
        self._writer.start()
        self._closed = False
        self._dropped_events = 0
        atexit.register(self.close)

    @staticmethod
    def _owned_unit_vectors(vectors) -> list[np.ndarray]:
        result = []
        for vector in vectors or []:
            copied = np.array(vector, dtype=np.float32, copy=True).reshape(-1)
            if copied.size != 256 or not np.isfinite(copied).all():
                continue
            norm = float(np.linalg.norm(copied))
            if math.isfinite(norm) and norm > 0.0:
                result.append(np.array(copied / norm, dtype=np.float32, copy=True))
        return result

    @staticmethod
    def _cosine(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.dot(left, right))

    @classmethod
    def build_representation(cls, vectors, exemplar_count=4, min_diversity=0.03) -> dict:
        """Return a representation from vectors, safely handling empty input."""
        owned = cls._owned_unit_vectors(vectors)
        if not owned:
            return {
                "embedding_count": 0, "exemplar_count": 0, "prototype": None,
                "exemplars": [], "prototype_similarity_mean": None,
                "prototype_similarity_min": None, "prototype_similarity_max": None,
                "prototype_similarity_std": None, "best_exemplar_similarity": None,
                "mean_top2_exemplar_similarity": None,
                "mean_top3_exemplar_similarity": None,
                "pairwise_similarity_mean": None, "pairwise_similarity_min": None,
                "pairwise_similarity_max": None,
            }

        mean_vector = np.mean(np.stack(owned), axis=0).astype(np.float32)
        mean_norm = float(np.linalg.norm(mean_vector))
        if not math.isfinite(mean_norm) or mean_norm <= 0.0:
            return cls.build_representation([], exemplar_count, min_diversity)
        prototype = np.array(mean_vector / mean_norm, dtype=np.float32, copy=True)
        prototype_similarities = [cls._cosine(vector, prototype) for vector in owned]

        pairwise = [cls._cosine(left, right) for index, left in enumerate(owned)
                    for right in owned[index + 1:]]

        limit = min(max(1, int(exemplar_count)), len(owned))
        # Stable ordering makes equal/near-equal choices reproducible.
        first = min(range(len(owned)), key=lambda index: (-prototype_similarities[index], index))
        selected = [first]
        remaining = [index for index in range(len(owned)) if index != first]
        while remaining and len(selected) < limit:
            ranked = []
            for index in remaining:
                max_similarity = max(cls._cosine(owned[index], owned[item]) for item in selected)
                diversity = 1.0 - max_similarity
                ranked.append((diversity, prototype_similarities[index], -index, index))
            ranked.sort(reverse=True)
            diversity, _, _, choice = ranked[0]
            if diversity < float(min_diversity):
                break
            selected.append(choice)
            remaining.remove(choice)

        exemplars = [np.array(owned[index], dtype=np.float32, copy=True) for index in selected]
        query_metrics = []
        for query in owned:
            similarities = sorted((cls._cosine(query, exemplar) for exemplar in exemplars), reverse=True)
            query_metrics.append(similarities)

        def mean_top_k(k: int):
            values = [float(np.mean(metrics[:k])) for metrics in query_metrics if metrics]
            return float(np.mean(values)) if values else None

        return {
            "embedding_count": len(owned),
            "exemplar_count": len(exemplars),
            "prototype": prototype,
            "exemplars": exemplars,
            "prototype_similarity_mean": float(np.mean(prototype_similarities)),
            "prototype_similarity_min": float(np.min(prototype_similarities)),
            "prototype_similarity_max": float(np.max(prototype_similarities)),
            "prototype_similarity_std": float(np.std(prototype_similarities)),
            "best_exemplar_similarity": mean_top_k(1),
            "mean_top2_exemplar_similarity": mean_top_k(2),
            "mean_top3_exemplar_similarity": mean_top_k(3),
            "pairwise_similarity_mean": float(np.mean(pairwise)) if pairwise else None,
            "pairwise_similarity_min": float(np.min(pairwise)) if pairwise else None,
            "pairwise_similarity_max": float(np.max(pairwise)) if pairwise else None,
        }

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

    def complete(self, track, summary: dict | None = None) -> dict:
        """Consume a completed LocalTracklet without retaining its object/pointers."""
        representation = self.build_representation(
            track.embeddings, self.exemplar_count, self.min_diversity
        )
        event = {
            "type": "REPRESENTATION_SUMMARY",
            "timestamp": int(track.last_seen_timestamp),
            "source_id": track.source_id,
            "local_object_id": track.local_object_id,
            "track_duration_seconds": (
                float(summary["track_duration_seconds"]) if summary else
                max(0.0, track.last_seen_seconds - track.first_seen_seconds)
            ),
        }
        for field in (
            "embedding_count", "exemplar_count", "prototype_similarity_mean",
            "prototype_similarity_min", "prototype_similarity_max", "prototype_similarity_std",
            "best_exemplar_similarity", "mean_top2_exemplar_similarity",
            "mean_top3_exemplar_similarity", "pairwise_similarity_mean",
            "pairwise_similarity_min", "pairwise_similarity_max",
        ):
            event[field] = representation[field]
        if self.include_vectors:
            event["prototype"] = representation["prototype"].tolist() if representation["prototype"] is not None else None
            event["exemplars"] = [vector.tolist() for vector in representation["exemplars"]]
        self._emit(event)
        return representation

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
            self._writer.join(timeout=2.0)
        except Exception:
            pass
