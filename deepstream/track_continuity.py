"""Short-term appearance-linked identity continuity for one camera.

NvDCF owns frame-to-frame tracking and its local object IDs.  Those IDs are
allowed to change after a detector/tracker gap, so this module provides a
bounded logical ID layer for downstream OSD and footfall logic.  It is not a
replacement for NvDCF and deliberately expires state after a configurable
TTL.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


def _float_env(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _iou(left, right) -> float:
    if not left or not right:
        return 0.0
    ax, ay, aw, ah = left
    bx, by, bw, bh = right
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0.0 else 0.0


def _center_distance(left, right) -> float:
    if not left or not right:
        return float("inf")
    return math.hypot(
        left[0] + left[2] / 2.0 - right[0] - right[2] / 2.0,
        left[1] + left[3] / 2.0 - right[1] - right[3] / 2.0,
    )


def _normalise(values) -> list[float]:
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError):
        return []
    if len(vector) != 256:
        return []
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm > 0.0 else []


def frame_token(frame) -> int:
    """Return a stable per-buffer token across DeepStream Python wrappers."""
    try:
        frame_number = int(getattr(frame, "frame_num", 0))
        if frame_number > 0:
            return frame_number
    except (TypeError, ValueError):
        pass
    try:
        pts = int(getattr(frame, "buffer_pts", 0))
        if pts > 0:
            return pts
    except (TypeError, ValueError):
        pass
    return int(time.monotonic() * 1_000_000)


@dataclass
class _Alias:
    source_id: int
    logical_id: int
    last_seen: float
    last_frame: int
    last_bbox: tuple[float, float, float, float] | None = None
    local_ids: set[int] = field(default_factory=set)
    embeddings: list[list[float]] = field(default_factory=list)
    staff_votes: int = 0
    staff_confirmed: bool = False


class TrackContinuity:
    """Link short-lived NvDCF IDs using native 256-D Re-ID features."""

    def __init__(self):
        self.ttl_seconds = _float_env("FOOTFALL_TRACK_TTL_SECONDS", 60.0)
        self.similarity_threshold = _float_env(
            "FOOTFALL_TRACK_REID_THRESHOLD", 0.80
        )
        self.staff_min_votes = _int_env("FOOTFALL_STAFF_MIN_VOTES", 1)
        self.max_embeddings = _int_env("FOOTFALL_TRACK_REID_HISTORY", 8)
        self.gallery_threshold = _float_env("FOOTFALL_STAFF_THRESHOLD", 0.78)
        self.gallery_path = Path(os.getenv(
            "FOOTFALL_STAFF_GALLERY", "/workspace/runs/footfall/staff_gallery.jsonl"
        ))
        self.staff_path = Path(os.getenv(
            "FOOTFALL_STAFF", "/workspace/runs/footfall/staff_track_ids.json"
        ))
        self.gallery_mtime = 0.0
        self.staff_mtime = 0.0
        self.gallery: list[list[float]] = []
        self.staff_ids: set[int] = set()
        self.aliases: dict[tuple[int, int], _Alias] = {}
        self.local_to_alias: dict[tuple[int, int], tuple[int, int]] = {}
        self.used_ids: dict[int, set[int]] = {}
        self.next_ids: dict[int, int] = {}
        self.frame_cache: dict[tuple[int, int, int], dict] = {}
        self.frame_assignments: dict[tuple[int, int], set[tuple[int, int]]] = {}
        self.frame_assignment_times: dict[tuple[int, int], float] = {}

    @staticmethod
    def object_embedding(obj) -> list[float]:
        try:
            items = list(obj.obj_reid_items)
            if not items:
                return []
            meta = items[-1].as_obj_reid()
            if int(meta.feature_size) != 256:
                return []
            return _normalise(meta.feature_vector)
        except (AttributeError, TypeError, ValueError):
            return []

    @staticmethod
    def classifier_staff(obj) -> bool:
        for name in ("classifier_meta_items", "classifier_metas", "classifier_meta_list"):
            classifiers = getattr(obj, name, None)
            if classifiers is None:
                continue
            try:
                classifiers = list(classifiers)
            except TypeError:
                continue
            for classifier in classifiers:
                for label_name in ("label_info_items", "label_infos", "label_info_list"):
                    labels = getattr(classifier, label_name, None)
                    if labels is None:
                        continue
                    try:
                        labels = list(labels)
                    except TypeError:
                        continue
                    for label in labels:
                        text = str(
                            getattr(label, "result_label", None)
                            or getattr(label, "label", None)
                            or getattr(label, "result_label_name", None)
                            or ""
                        ).strip().lower().replace("_", " ")
                        if text == "staff" or "staff" in text:
                            return True
        return False

    def _reload_staff_sources(self) -> None:
        try:
            stamp = self.gallery_path.stat().st_mtime
            if stamp != self.gallery_mtime:
                gallery = []
                for line in self.gallery_path.read_text().splitlines():
                    try:
                        vector = _normalise(json.loads(line).get("embedding", []))
                        if vector:
                            gallery.append(vector)
                    except (OSError, TypeError, ValueError, json.JSONDecodeError):
                        continue
                self.gallery = gallery
                self.gallery_mtime = stamp
        except OSError:
            pass
        try:
            stamp = self.staff_path.stat().st_mtime
            if stamp != self.staff_mtime:
                value = json.loads(self.staff_path.read_text())
                self.staff_ids = {int(item) for item in value.get("track_ids", [])}
                self.staff_mtime = stamp
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    def _cleanup(self, now: float) -> None:
        expired = [key for key, alias in self.aliases.items()
                   if now - alias.last_seen > self.ttl_seconds]
        for key in expired:
            self.aliases.pop(key, None)
        if expired:
            expired_set = set(expired)
            self.local_to_alias = {
                key: value for key, value in self.local_to_alias.items()
                if value not in expired_set
            }
        # Do not allow the cache to retain old video frames indefinitely.
        self.frame_cache = {
            key: value for key, value in self.frame_cache.items()
            if now - value["wall_time"] <= 2.0
        }
        self.frame_assignments = {
            key: value for key, value in self.frame_assignments.items()
            if now - self.frame_assignment_times.get(key, now) <= 2.0
        }
        self.frame_assignment_times = {
            key: value for key, value in self.frame_assignment_times.items()
            if now - value <= 2.0
        }

    def _new_id(self, source_id: int, local_id: int) -> int:
        used = self.used_ids.setdefault(source_id, set())
        logical_id = max(self.next_ids.get(source_id, 1), 1)
        while logical_id in used:
            logical_id += 1
        used.add(logical_id)
        self.next_ids[source_id] = max(self.next_ids.get(source_id, 1), logical_id + 1)
        return logical_id

    def _gallery_match(self, embedding: list[float]) -> bool:
        return bool(embedding and self.gallery and max(
            (_cosine(embedding, candidate) for candidate in self.gallery),
            default=-1.0,
        ) >= self.gallery_threshold)

    def _best_alias(self, source_id: int, embedding: list[float], bbox, now: float,
                    frame_key: tuple[int, int]) -> tuple[_Alias | None, float]:
        assigned = self.frame_assignments.get(frame_key, set())
        best, best_score = None, -1.0
        for alias in self.aliases.values():
            if alias.source_id != source_id or (source_id, alias.logical_id) in assigned:
                continue
            if now - alias.last_seen > self.ttl_seconds:
                continue
            score = max((_cosine(embedding, item) for item in alias.embeddings), default=-1.0)
            if score > best_score:
                best, best_score = alias, score
        if best is not None and best_score >= self.similarity_threshold:
            return best, best_score

        # Re-ID metadata is emitted intermittently. For a short detector gap,
        # bridge with the last box; long gaps still require appearance.
        spatial_best, spatial_distance = None, float("inf")
        for alias in self.aliases.values():
            if alias.source_id != source_id or (source_id, alias.logical_id) in assigned:
                continue
            if now - alias.last_seen > 2.0 or alias.last_bbox is None or bbox is None:
                continue
            overlap = _iou(alias.last_bbox, bbox)
            distance = _center_distance(alias.last_bbox, bbox)
            max_distance = max(90.0, 0.75 * max(alias.last_bbox[3], bbox[3]))
            if (overlap >= 0.08 or distance <= max_distance) and distance < spatial_distance:
                spatial_best, spatial_distance = alias, distance
        if spatial_best is not None:
            return spatial_best, -spatial_distance
        return None, best_score

    def resolve(self, source_id: int, local_id: int, frame_number: int,
                embedding: list[float], bbox=None, *, classifier_staff: bool = False) -> dict:
        """Return the logical ID and persistent staff state for one object."""
        source_id, local_id, frame_number = int(source_id), int(local_id), int(frame_number)
        valid_local_id = 0 < local_id < (1 << 64) - 1
        cache_key = (source_id, local_id, frame_number) if valid_local_id else None
        if cache_key is not None:
            cached = self.frame_cache.get(cache_key)
            if cached is not None:
                return cached
        now = time.monotonic()
        self._reload_staff_sources()
        self._cleanup(now)
        frame_key = (source_id, frame_number)
        self.frame_assignments.setdefault(frame_key, set())
        self.frame_assignment_times[frame_key] = now
        alias_key = self.local_to_alias.get((source_id, local_id)) if valid_local_id else None
        alias = self.aliases.get(alias_key) if alias_key else None
        assigned = self.frame_assignments.get(frame_key, set())
        if alias is not None and (source_id, alias.logical_id) in assigned:
            # A stale local-ID mapping must not let two simultaneous objects
            # share one logical ID. The second object must use appearance,
            # short-gap geometry, or receive a new alias.
            alias = None
        similarity = 1.0 if alias is not None else -1.0
        if alias is None:
            alias, similarity = self._best_alias(source_id, embedding, bbox, now, frame_key)
            if alias is None:
                logical_id = self._new_id(source_id, local_id)
                alias = _Alias(source_id, logical_id, now, frame_number)
                self.aliases[(source_id, logical_id)] = alias
        alias.last_seen = now
        alias.last_frame = frame_number
        if bbox is not None:
            alias.last_bbox = tuple(float(value) for value in bbox)
        if valid_local_id:
            alias.local_ids.add(local_id)
            self.local_to_alias[(source_id, local_id)] = (source_id, alias.logical_id)
        self.frame_assignments[frame_key].add((source_id, alias.logical_id))
        if embedding:
            alias.embeddings.append(embedding)
            alias.embeddings = alias.embeddings[-self.max_embeddings:]
        staff_evidence = (
            bool(classifier_staff)
            or local_id in self.staff_ids
            or self._gallery_match(embedding)
        )
        if staff_evidence:
            alias.staff_votes += 1
            if alias.staff_votes >= self.staff_min_votes:
                alias.staff_confirmed = True
        result = {
            "source_id": source_id,
            "local_track_id": local_id,
            "logical_track_id": alias.logical_id,
            "staff": alias.staff_confirmed,
            "staff_votes": alias.staff_votes,
            "similarity": similarity,
            "wall_time": now,
        }
        if cache_key is not None:
            self.frame_cache[cache_key] = result
        return result


_INSTANCE: TrackContinuity | None = None


def get_track_continuity() -> TrackContinuity:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = TrackContinuity()
    return _INSTANCE
