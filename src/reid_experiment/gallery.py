"""Sticky, calibration-aware experimental gallery for SOLIDER observations."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else -1.0


@dataclass
class Identity:
    identity_id: int
    last_seen: float
    source_ids: set[int] = field(default_factory=set)
    positions: deque[tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=12))
    embeddings: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=12))
    tracklets: dict[str, dict] = field(default_factory=dict)
    # In anchored mode the gallery intentionally remains owned by CH11, so
    # source_ids alone cannot tell us that CH10 has been matched.
    cross_camera_confirmed: bool = False


@dataclass
class TrackState:
    identity_id: int
    first_seen: float
    last_seen: float
    confirmed: bool = False
    gallery_committed: bool = False
    confirmations: dict[int, int] = field(default_factory=lambda: defaultdict(int))


class ExperimentalGallery:
    """Association logic kept separate from production Global-ID state."""

    def __init__(self, output: Path, similarity_threshold=0.92, minimum_margin=0.08,
                 confirmations=4, max_gap_seconds=3.0, max_world_distance=1.25,
                 gallery_size=12, minimum_gallery_samples=1, ch11_only=False,
                 anchor_source=1, track_min_embeddings=1,
                 local_track_gap_seconds=3.0):
        self.output = output
        self.threshold = similarity_threshold
        self.minimum_margin = minimum_margin
        self.confirmations = confirmations
        self.max_gap = max_gap_seconds
        self.max_world_distance = max_world_distance
        self.identities: dict[int, Identity] = {}
        self.tracks: dict[tuple[int, int], TrackState] = {}
        self.next_id = 1
        self.gallery_size = gallery_size
        self.minimum_gallery_samples = minimum_gallery_samples
        self.ch11_only = bool(ch11_only)
        self.anchor_source = int(anchor_source)
        self.track_min_embeddings = max(1, int(track_min_embeddings))
        self.local_track_gap = max(0.0, float(local_track_gap_seconds))
        self.track_embeddings: dict[tuple[int, int], list[np.ndarray]] = {}
        # Prevent two simultaneously visible local tracks from consuming the
        # same cross-camera identity. Entries are released after max_gap.
        self.active_identity_tracks: dict[int, tuple[int, int]] = {}
        self.stats = defaultdict(int)

    def _same_source_track_active(self, identity_id: int, source: int,
                                  current_key: tuple[int, int], now: float) -> bool:
        """Prevent one identity from owning two simultaneous tracks in one view."""
        return any(
            key != current_key and key[0] == int(source) and
            state.identity_id == identity_id and state.confirmed and
            now - state.last_seen <= self.local_track_gap
            for key, state in self.tracks.items()
        )

    def _identity_has_conflicting_active_track(self, identity_id: int,
                                               source: int,
                                               current_key: tuple[int, int],
                                               now: float) -> bool:
        """Reject an association that would duplicate an identity in one view."""
        return self._same_source_track_active(
            identity_id, int(source), current_key, float(now))

    def _new_identity(self, source: int, now: float, embedding, world) -> int:
        identity_id = self.next_id
        self.next_id += 1
        identity = Identity(identity_id, now)
        identity.embeddings = deque(maxlen=self.gallery_size)
        identity.source_ids.add(source)
        identity.embeddings.append(embedding)
        if world is not None:
            identity.positions.append((now, world[0], world[1]))
        self.identities[identity_id] = identity
        self._touch_tracklet(identity, source, None, now, 1)
        return identity_id

    @staticmethod
    def _touch_tracklet(identity: Identity, source: int, local_track: int | None,
                        timestamp: float, samples: int = 1) -> None:
        if local_track is None:
            return
        key = f"{int(source)}:{int(local_track)}"
        item = identity.tracklets.setdefault(key, {
            "source_id": int(source), "local_track_id": int(local_track),
            "first_seen": float(timestamp), "last_seen": float(timestamp),
            "samples": 0,
        })
        item["last_seen"] = float(timestamp)
        item["samples"] += int(samples)

    def journey_snapshot(self) -> list[dict]:
        return [
            {"global_id": identity.identity_id,
             "first_seen": min((float(item["first_seen"]) for item in identity.tracklets.values()),
                                default=float(identity.last_seen)),
             "last_seen": float(identity.last_seen),
             "source_ids": sorted(int(value) for value in identity.source_ids),
             "cross_camera_confirmed": bool(identity.cross_camera_confirmed),
             "tracklets": sorted(identity.tracklets.values(),
                                 key=lambda item: (item["first_seen"], item["source_id"], item["local_track_id"]))}
            for identity in sorted(self.identities.values(), key=lambda item: item.identity_id)
        ]

    def _merge_identity(self, winner: Identity, loser: Identity) -> Identity:
        """Merge two provisional identities after later tracklet evidence."""
        if winner.identity_id == loser.identity_id:
            return winner
        if loser.identity_id < winner.identity_id:
            winner, loser = loser, winner
        winner.embeddings.extend(loser.embeddings)
        winner.source_ids.update(loser.source_ids)
        winner.tracklets.update(loser.tracklets)
        winner.last_seen = max(winner.last_seen, loser.last_seen)
        winner.cross_camera_confirmed = True
        for state in self.tracks.values():
            if state.identity_id == loser.identity_id:
                state.identity_id = winner.identity_id
                state.confirmed = True
        self.identities.pop(loser.identity_id, None)
        self.stats["retroactive_merges"] += 1
        return winner

    def observe(self, *, source: int, local_track: int, timestamp: float,
                embedding: list[float], world: tuple[float, float] | None,
                quality: float = 1.0, detector_confidence: float = 1.0) -> dict:
        now = float(timestamp)
        key = (int(source), int(local_track))
        state = self.tracks.get(key)
        vector = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if vector.ndim != 1 or not np.isfinite(vector).all() or norm <= 0 or quality < 0.55 or detector_confidence < 0.35:
            self.stats["rejected_quality"] += 1
            if state is not None and state.confirmed and state.identity_id in self.identities:
                identity = self.identities[state.identity_id]
                state.last_seen = max(state.last_seen, now)
                identity.last_seen = max(identity.last_seen, now)
                self._touch_tracklet(identity, source, local_track, now)
                return {"experimental_global_id": identity.identity_id,
                        "decision": "STICKY_LOW_QUALITY",
                        "cross_camera_confirmed": identity.cross_camera_confirmed}
            return {"experimental_global_id": None, "decision": "REJECT_QUALITY"}
        vector /= norm
        if self.ch11_only:
            return self._observe_ch11_only(
                source=int(source), local_track=int(local_track), timestamp=now,
                vector=vector)
        key = (int(source), int(local_track))
        track_gallery = self.track_embeddings.setdefault(key, [])
        track_gallery.append(np.array(vector, copy=True))
        del track_gallery[:-self.gallery_size]
        state = self.tracks.get(key)
        if state is not None and state.confirmed:
            identity = self.identities.get(state.identity_id)
            if identity is not None:
                # A track that initially received a provisional local Global
                # ID may later have enough evidence to merge with an identity
                # born in the other camera.
                alternatives = []
                for other in self.identities.values():
                    if other.identity_id == identity.identity_id or int(source) in other.source_ids:
                        continue
                    if now - other.last_seen > self.max_gap or not other.embeddings:
                        continue
                    if len(other.embeddings) < self.minimum_gallery_samples:
                        continue
                    if world is not None and other.positions and self.max_world_distance is not None:
                        _, x, y = other.positions[-1]
                        if math.hypot(world[0] - x, world[1] - y) > self.max_world_distance:
                            self.stats['retroactive_geometry_rejected'] += 1
                            continue
                    # A retroactive merge must obey the same one-track-per-
                    # camera rule as a new match. The previous check inspected
                    # the current identity's tracks instead of the candidate
                    # identity, allowing a false merge after track churn.
                    if self._identity_has_conflicting_active_track(
                            other.identity_id, int(source), key, now):
                        continue
                    scores = sorted((cosine(current, item)
                                     for current in track_gallery for item in other.embeddings), reverse=True)
                    if scores:
                        alternatives.append((float(np.mean(scores[:min(3, len(scores))])), other))
                alternatives.sort(key=lambda item: item[0], reverse=True)
                second = alternatives[1][0] if len(alternatives) > 1 else -1.0
                if (alternatives and len(track_gallery) >= self.track_min_embeddings and
                        alternatives[0][0] >= self.threshold and
                        alternatives[0][0] - second >= self.minimum_margin):
                    merge_similarity = alternatives[0][0]
                    merged_identity = alternatives[0][1]
                    candidate = merged_identity.identity_id
                    previous = state.confirmations.get(candidate, 0)
                    state.confirmations.clear()
                    state.confirmations[candidate] = previous + int(now > state.last_seen)
                    if state.confirmations[candidate] >= self.confirmations:
                        identity = self._merge_identity(identity, merged_identity)
                        state.identity_id = identity.identity_id
                        state.confirmations.clear()
                        merge_decision = "RETROACTIVE_MERGE"
                    else:
                        merge_decision = "STICKY_PENDING_MERGE"
                else:
                    state.confirmations.clear()
                    merge_similarity = None
                    merge_decision = "STICKY_LOCAL_TRACK"
                state.last_seen = now
                identity.last_seen = now
                identity.source_ids.add(int(source))
                identity.embeddings.append(vector)
                self._touch_tracklet(identity, source, local_track, now)
                if world is not None:
                    identity.positions.append((now, world[0], world[1]))
                self.stats["sticky_updates"] += 1
                return {"experimental_global_id": identity.identity_id, "decision": merge_decision,
                    "cross_camera_confirmed": identity.cross_camera_confirmed,
                    "similarity": merge_similarity,
                    "track_embedding_count": len(track_gallery),
                    "evidence": len(track_gallery)}
        candidates = []
        for identity in self.identities.values():
            if now - identity.last_seen > self.max_gap:
                continue
            if self._same_source_track_active(identity.identity_id, int(source), key, now):
                continue
            # Do not accept an identity from one lucky embedding.  The
            # gallery must contain several observations and the current crop
            # must agree with the best few of them.  This is deliberately
            # conservative for low-quality, partially occluded store views.
            if len(identity.embeddings) < self.minimum_gallery_samples:
                continue
            scores = sorted((cosine(current, item)
                             for current in track_gallery for item in identity.embeddings), reverse=True)
            appearance = float(np.mean(scores[:min(3, len(scores))])) if scores else -1.0
            distance = None
            if world is not None and identity.positions:
                # Compare against the latest calibrated position. Using the
                # minimum over the entire history lets a moving person match
                # an old position from several seconds ago.
                _, latest_x, latest_y = identity.positions[-1]
                distance = math.hypot(world[0] - latest_x, world[1] - latest_y)
                if self.max_world_distance is not None and distance > self.max_world_distance:
                    continue
            candidates.append((appearance, distance, identity))
        self.stats["cross_camera_candidates"] += len(candidates)
        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0] if candidates else None
        second = candidates[1][0] if len(candidates) > 1 else -1.0
        if best is None or best[0] < self.threshold or best[0] - second < self.minimum_margin:
            identity_id = self._new_identity(int(source), now, vector, world)
            self.tracks[key] = TrackState(identity_id, now, now, confirmed=True,
                                           gallery_committed=True)
            self.stats["new_id"] += 1
            self._touch_tracklet(self.identities[identity_id], source, local_track, now)
            return {"experimental_global_id": identity_id, "decision": "NEW_ID",
                    "similarity": best[0] if best else None, "world_distance": best[1] if best else None}
        candidate_id = best[2].identity_id
        matched_identity = best[2]
        match_evidence = {
            "matched_source_ids": sorted(int(value) for value in matched_identity.source_ids),
            "matched_last_seen": float(matched_identity.last_seen),
            "match_time_gap": float(now - matched_identity.last_seen),
            "matched_gallery_samples": len(matched_identity.embeddings),
            "track_embedding_count": len(track_gallery),
            "matched_gallery_similarity": float(best[0]),
        }
        pending = self.tracks.setdefault(key, TrackState(candidate_id, now, now))
        previous = pending.confirmations.get(candidate_id, 0)
        pending.confirmations.clear()
        pending.confirmations[candidate_id] = previous
        pending.confirmations[candidate_id] += 1
        pending.last_seen = now
        if pending.confirmations[candidate_id] < self.confirmations:
            self.stats["ambiguous"] += 1
            return {"experimental_global_id": None, "decision": "PENDING_CONFIRMATION",
                    "candidate_id": candidate_id, "similarity": best[0], "world_distance": best[1],
                    **match_evidence}
        pending.identity_id = candidate_id
        pending.last_seen = now
        pending.confirmed = True
        identity = self.identities[candidate_id]
        identity.source_ids.add(int(source))
        identity.last_seen = now
        identity.cross_camera_confirmed = True
        if not pending.gallery_committed:
            identity.embeddings.extend(track_gallery)
            pending.gallery_committed = True
        else:
            identity.embeddings.append(vector)
        self._touch_tracklet(identity, source, local_track, now)
        if world is not None:
            identity.positions.append((now, world[0], world[1]))
        self.stats["cross_camera_match"] += 1
        return {"experimental_global_id": candidate_id, "decision": "CROSS_CAMERA_MATCH",
                "cross_camera_confirmed": True,
                "similarity": best[0], "world_distance": best[1], **match_evidence}

    def _observe_ch11_only(self, *, source: int, local_track: int,
                           timestamp: float, vector: np.ndarray) -> dict:
        """Appearance-only experiment: CH11 anchors IDs; CH10 can only match."""
        key = (source, local_track)
        track_gallery = self.track_embeddings.setdefault(key, [])
        track_gallery.append(np.array(vector, copy=True))
        state = self.tracks.get(key)

        if source == self.anchor_source:
            if state is not None and state.identity_id is not None:
                identity = self.identities[state.identity_id]
                identity.embeddings.append(np.array(vector, copy=True))
                identity.last_seen = timestamp
                state.last_seen = timestamp
                self.stats["anchor_gallery_updates"] += 1
                return {"experimental_global_id": identity.identity_id,
                        "decision": "STICKY_CH11_GALLERY_APPEND",
                        "cross_camera_confirmed": identity.cross_camera_confirmed,
                        "similarity": None, "world_distance": None,
                        "gallery_size": len(identity.embeddings),
                        "track_embedding_count": len(track_gallery)}
            # Stitch a new CH11 local track to an existing anchor identity
            # before birthing a duplicate identity. Require a clear margin so
            # similar clothing does not merge two people.
            anchor_candidates = []
            for identity in self.identities.values():
                if self.anchor_source not in identity.source_ids or not identity.embeddings:
                    continue
                scores = sorted((float(np.dot(current, anchor))
                                 for current in track_gallery for anchor in identity.embeddings),
                                reverse=True)
                top = scores[:3]
                anchor_candidates.append((float(np.mean(top)), identity, len(scores)))
            anchor_candidates.sort(key=lambda item: item[0], reverse=True)
            second = anchor_candidates[1][0] if len(anchor_candidates) > 1 else -1.0
            if (anchor_candidates and anchor_candidates[0][0] >= self.threshold and
                    anchor_candidates[0][0] - second >= self.minimum_margin):
                similarity, identity, pair_count = anchor_candidates[0]
                owner = self.active_identity_tracks.get(identity.identity_id)
                if owner is None or owner == key:
                    self.tracks[key] = TrackState(identity.identity_id, timestamp, timestamp, True)
                    identity.embeddings.append(np.array(vector, copy=True))
                    identity.last_seen = timestamp
                    self.active_identity_tracks[identity.identity_id] = key
                    self.stats["anchor_gallery_stitches"] += 1
                    return {"experimental_global_id": identity.identity_id,
                            "decision": "CH11_ANCHOR_STITCH",
                            "cross_camera_confirmed": identity.cross_camera_confirmed,
                            "similarity": similarity, "world_distance": None,
                            "pairwise_comparisons": pair_count,
                            "track_embedding_count": len(track_gallery)}
            identity_id = self.next_id
            self.next_id += 1
            identity = Identity(identity_id, timestamp)
            identity.source_ids.add(self.anchor_source)
            identity.embeddings = deque([np.array(vector, copy=True)], maxlen=None)
            self.identities[identity_id] = identity
            self.tracks[key] = TrackState(identity_id, timestamp, timestamp, True)
            self.stats["new_id"] += 1
            return {"experimental_global_id": identity_id,
                    "decision": "CH11_NEW_GLOBAL_ID", "similarity": None,
                    "cross_camera_confirmed": False,
                    "world_distance": None, "gallery_size": 1,
                    "track_embedding_count": len(track_gallery)}

        # CH10 is never an identity anchor and never creates a new ID.
        if source != 0:
            return {"experimental_global_id": None, "decision": "UNMATCHED_NON_ANCHOR",
                    "similarity": None, "world_distance": None,
                    "track_embedding_count": len(track_gallery)}
        if state is not None and state.identity_id is not None and state.confirmed:
            state.last_seen = timestamp
            self.stats["sticky_updates"] += 1
            return {"experimental_global_id": state.identity_id,
                    "decision": "STICKY_CH10_MATCH", "cross_camera_confirmed": True,
                    "similarity": None,
                    "world_distance": None, "track_embedding_count": len(track_gallery)}
        if len(track_gallery) < self.track_min_embeddings:
            self.stats["unmatched"] += 1
            return {"experimental_global_id": None, "decision": "UNMATCHED_WAITING_EMBEDDINGS",
                    "similarity": None, "world_distance": None,
                    "track_embedding_count": len(track_gallery)}

        candidates = []
        for identity in self.identities.values():
            if self.anchor_source not in identity.source_ids or not identity.embeddings:
                continue
            owner = self.active_identity_tracks.get(identity.identity_id)
            if owner is not None and owner != key:
                owner_state = self.tracks.get(owner)
                if owner_state is not None and timestamp - owner_state.last_seen <= self.max_gap:
                    continue
                self.active_identity_tracks.pop(identity.identity_id, None)
            scores = [float(np.dot(current, anchor))
                      for current in track_gallery for anchor in identity.embeddings]
            scores.sort(reverse=True)
            top = scores[:3]
            candidates.append((float(np.mean(top)), identity, len(scores)))
        self.stats["cross_camera_candidates"] += len(candidates)
        candidates.sort(key=lambda item: item[0], reverse=True)
        second = candidates[1][0] if len(candidates) > 1 else -1.0
        if (not candidates or candidates[0][0] < self.threshold or
                candidates[0][0] - second < self.minimum_margin):
            self.stats["unmatched"] += 1
            return {"experimental_global_id": None, "decision": "UNMATCHED",
                    "similarity": candidates[0][0] if candidates else None,
                    "world_distance": None,
                    "track_embedding_count": len(track_gallery),
                    "candidate_count": len(candidates)}

        similarity, identity, pair_count = candidates[0]
        state = self.tracks.get(key)
        if state is None or state.identity_id != identity.identity_id:
            state = TrackState(identity.identity_id, timestamp, timestamp, False)
            self.tracks[key] = state
        state.confirmations[identity.identity_id] += 1
        if state.confirmations[identity.identity_id] < self.confirmations:
            self.stats["ambiguous"] += 1
            return {"experimental_global_id": None,
                    "decision": "PENDING_CONFIRMATION",
                    "similarity": similarity, "world_distance": None,
                    "candidate_count": len(candidates),
                    "track_embedding_count": len(track_gallery)}
        state.confirmed = True
        state.last_seen = timestamp
        self.tracks[key] = state
        identity.last_seen = timestamp
        identity.cross_camera_confirmed = True
        self.active_identity_tracks[identity.identity_id] = key
        self.stats["cross_camera_match"] += 1
        return {"experimental_global_id": identity.identity_id,
                "decision": "CROSS_CAMERA_MATCH",
                "similarity": similarity, "world_distance": None,
                "candidate_count": len(candidates), "pairwise_comparisons": pair_count,
                "top_k": min(3, pair_count), "track_embedding_count": len(track_gallery),
                "matched_source_ids": [self.anchor_source],
                "matched_last_seen": timestamp,
                "match_time_gap": 0.0,
                "matched_gallery_samples": len(identity.embeddings),
                "matched_gallery_similarity": similarity}

    def write_event(self, record: dict) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
