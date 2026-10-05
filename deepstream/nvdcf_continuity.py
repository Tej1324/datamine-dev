"""Metadata-only NvDCF local-ID continuity diagnostic."""

from __future__ import annotations

import atexit
import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    from pyservicemaker import BatchMetadataOperator
except ImportError:  # Allows pure-Python unit tests outside the DeepStream image.
    class BatchMetadataOperator:  # pragma: no cover - runtime base is provided by NVIDIA.
        def __init__(self):
            pass


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


def _timestamp_seconds(timestamp: int) -> float:
    value = float(timestamp)
    return value / 1_000_000_000.0 if value > 1_000_000_000.0 else value


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = left
    bx1, by1, bw, bh = right
    ax2, ay2 = ax1 + max(0.0, aw), ay1 + max(0.0, ah)
    bx2, by2 = bx1 + max(0.0, bw), by1 + max(0.0, bh)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    union += max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 0.0 else 0.0


def _center_distance(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    left_center = (left[0] + left[2] / 2.0, left[1] + left[3] / 2.0)
    right_center = (right[0] + right[2] / 2.0, right[1] + right[3] / 2.0)
    return math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1])


@dataclass
class _TrackState:
    source_id: str
    local_object_id: int
    first_timestamp: int
    first_frame: int
    last_timestamp: int
    last_frame: int
    last_seen_seconds: float
    last_bbox: tuple[float, float, float, float]
    observation_count: int = 0
    detector_gap_count: int = 0
    detector_confidences: list[float] = field(default_factory=list)
    tracker_confidences: list[float] = field(default_factory=list)
    last_detector_present: bool = False


class NvdcfContinuityDiagnostic(BatchMetadataOperator):
    """Records tracked-object timelines and conservative ID-switch candidates."""

    LOW_TRACKER_CONFIDENCE = 0.5
    UNTRACKED_OBJECT_ID = (1 << 64) - 1

    def __init__(self):
        super().__init__()
        self.max_gap_frames = _int_env("NVCDF_CONTINUITY_MAX_GAP_FRAMES", 5)
        self.iou_threshold = _float_env("NVCDF_CONTINUITY_IOU_THRESHOLD", 0.30)
        self.center_distance_threshold = _float_env(
            "NVCDF_CONTINUITY_CENTER_DISTANCE_THRESHOLD", 150.0
        )
        self.ttl_seconds = _float_env("NVCDF_CONTINUITY_STATE_TTL_SECONDS", 10.0)
        self.output_path = Path(os.getenv(
            "NVCDF_CONTINUITY_OUTPUT", "runs/nvdcf_continuity.jsonl"
        ))
        self._tracks: dict[tuple[str, int], _TrackState] = {}
        # Only the immediately preceding observed frame per source is kept for
        # continuity matching.  This prevents unrelated older tracks from
        # being paired with every new object in a crowded scene.
        self._previous_frames: dict[str, tuple[int, list[dict]]] = {}
        self._queue: queue.Queue[dict | None] = queue.Queue(maxsize=512)
        self._writer = threading.Thread(
            target=self._write_loop, name="nvdcf-continuity-writer", daemon=True
        )
        self._writer.start()
        self._closed = False
        self._dropped_events = 0
        atexit.register(self.close)

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

    @staticmethod
    def _bbox(rect) -> tuple[float, float, float, float]:
        return (float(rect.left), float(rect.top), float(rect.width), float(rect.height))

    @staticmethod
    def _confidence(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    @staticmethod
    def _detector_present(confidence: float) -> bool:
        return math.isfinite(confidence) and confidence >= 0.0

    def _summary(self, state: _TrackState) -> dict:
        return {
            "type": "TRACK_SUMMARY",
            "source_id": state.source_id,
            "local_object_id": state.local_object_id,
            "first_timestamp": state.first_timestamp,
            "last_timestamp": state.last_timestamp,
            "first_frame": state.first_frame,
            "last_frame": state.last_frame,
            "duration_frames": max(0, state.last_frame - state.first_frame + 1),
            "observation_count": state.observation_count,
            "detector_gap_count": state.detector_gap_count,
            "min_tracker_confidence": min(state.tracker_confidences) if state.tracker_confidences else None,
            "mean_tracker_confidence": (
                sum(state.tracker_confidences) / len(state.tracker_confidences)
                if state.tracker_confidences else None
            ),
            "min_detector_confidence": min(state.detector_confidences) if state.detector_confidences else None,
            "mean_detector_confidence": (
                sum(state.detector_confidences) / len(state.detector_confidences)
                if state.detector_confidences else None
            ),
        }

    def _finish(self, key: tuple[str, int]) -> None:
        state = self._tracks.pop(key, None)
        if state is not None:
            self._emit(self._summary(state))

    def _emit_switch(self, previous: dict, current: dict, gap: int,
                     overlap: float, distance: float) -> None:
        classifications = []
        if gap == 1 and previous["detector_present"] and current["detector_present"]:
            classifications.append("DETECTOR_CONTINUOUS_TRACKER_SWITCH")
        else:
            classifications.append("DETECTOR_GAP_NEW_TRACK")
        previous_tracker_confidence = previous["tracker_confidence"]
        new_tracker_confidence = current["tracker_confidence"]
        if (
            not math.isfinite(previous_tracker_confidence)
            or not math.isfinite(new_tracker_confidence)
            or previous_tracker_confidence < self.LOW_TRACKER_CONFIDENCE
            or new_tracker_confidence < self.LOW_TRACKER_CONFIDENCE
        ):
            classifications.append("LOW_TRACKER_CONFIDENCE_TRANSITION")
        previous_bbox = previous["bbox"]
        current_bbox = current["bbox"]
        # This method is only called after the continuity gate has passed;
        # spatial mismatch is therefore deliberately not emitted here.
        self._emit({
            "type": "LOCAL_ID_CONTINUITY_BREAK",
            "source_id": previous["source_id"],
            "previous_local_object_id": previous["local_object_id"],
            "new_local_object_id": current["local_object_id"],
            "previous_frame": previous["frame_number"],
            "new_frame": current["frame_number"],
            "frame_gap": gap,
            "bbox_iou": overlap,
            "center_distance": distance,
            "previous_tracker_confidence": previous_tracker_confidence,
            "new_tracker_confidence": new_tracker_confidence,
            "previous_detector_confidence": previous["detector_confidence"],
            "new_detector_confidence": current["detector_confidence"],
            "previous_bbox": {
                "left": previous_bbox[0], "top": previous_bbox[1],
                "width": previous_bbox[2], "height": previous_bbox[3],
            },
            "new_bbox": {
                "left": current_bbox[0], "top": current_bbox[1],
                "width": current_bbox[2], "height": current_bbox[3],
            },
            "classification": classifications[-1] if len(classifications) > 1 else classifications[0],
            "classifications": classifications,
        })

        # Preserve the old event name for consumers that already parse it, but
        # only as an alias of the new gated event.  It is no longer a broad
        # all-population pairing signal.
        self._emit({
            "type": "LOCAL_ID_SWITCH_CANDIDATE",
            "source_id": previous["source_id"],
            "previous_local_object_id": previous["local_object_id"],
            "new_local_object_id": current["local_object_id"],
            "previous_frame": previous["frame_number"],
            "new_frame": current["frame_number"],
            "frame_gap": gap,
            "bbox_iou": overlap,
            "center_distance": distance,
            "previous_tracker_confidence": previous_tracker_confidence,
            "new_tracker_confidence": new_tracker_confidence,
            "previous_detector_confidence": previous["detector_confidence"],
            "new_detector_confidence": current["detector_confidence"],
            "classification": classifications[-1] if len(classifications) > 1 else classifications[0],
            "classifications": classifications,
            "alias_of": "LOCAL_ID_CONTINUITY_BREAK",
        })

    def _match_adjacent_frames(self, source_id: str, frame_number: int, current: list[dict]) -> None:
        previous_entry = self._previous_frames.get(source_id)
        if previous_entry is None:
            return
        previous_frame, previous = previous_entry
        gap = frame_number - previous_frame
        if gap < 1 or gap > self.max_gap_frames:
            return

        # Greedy one-to-one assignment.  IoU is primary; center distance is
        # the tie-breaker.  This prevents all-to-all pairings during crossings.
        pairs = []
        for previous_index, previous_item in enumerate(previous):
            for current_index, current_item in enumerate(current):
                overlap = _iou(previous_item["bbox"], current_item["bbox"])
                distance = _center_distance(previous_item["bbox"], current_item["bbox"])
                pairs.append((overlap, -distance, previous_index, current_index, distance))
        pairs.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        used_previous = set()
        used_current = set()
        for overlap, _, previous_index, current_index, distance in pairs:
            if previous_index in used_previous or current_index in used_current:
                continue
            used_previous.add(previous_index)
            used_current.add(current_index)
            previous_item = previous[previous_index]
            current_item = current[current_index]
            if previous_item["local_object_id"] == current_item["local_object_id"]:
                continue
            if overlap < self.iou_threshold and distance > self.center_distance_threshold:
                continue
            self._emit_switch(previous_item, current_item, gap, overlap, distance)
            self._finish((source_id, previous_item["local_object_id"]))

    def handle_metadata(self, batch_meta) -> None:
        for frame_meta in batch_meta.frame_items:
            source_id = str(frame_meta.source_id)
            frame_number = int(frame_meta.frame_number)
            timestamp = int(frame_meta.ntp_timestamp or frame_meta.buffer_pts or 0)
            timestamp_seconds = _timestamp_seconds(timestamp)
            current_frame = []
            for object_meta in frame_meta.object_items:
                try:
                    if int(object_meta.class_id) != 0:
                        continue
                    local_id = int(object_meta.object_id)
                    # NvDsObjectMeta uses UINT64_MAX for an object that has
                    # not received a valid tracker ID yet.  It is not a
                    # person track and must not participate in continuity
                    # comparisons.
                    if local_id == self.UNTRACKED_OBJECT_ID:
                        continue
                    bbox = self._bbox(object_meta.rect_params)
                    detector_confidence = self._confidence(object_meta.confidence)
                    tracker_confidence = self._confidence(object_meta.tracker_confidence)
                except (AttributeError, TypeError, ValueError):
                    continue
                detector_present = self._detector_present(detector_confidence)
                current_frame.append({
                    "source_id": source_id,
                    "local_object_id": local_id,
                    "frame_number": frame_number,
                    "bbox": bbox,
                    "detector_confidence": detector_confidence,
                    "tracker_confidence": tracker_confidence,
                    "detector_present": detector_present,
                })

            self._match_adjacent_frames(source_id, frame_number, current_frame)
            for current in current_frame:
                local_id = current["local_object_id"]
                key = (source_id, local_id)
                bbox = current["bbox"]
                detector_confidence = current["detector_confidence"]
                tracker_confidence = current["tracker_confidence"]
                detector_present = current["detector_present"]

                state = self._tracks.get(key)
                if state is None:
                    state = _TrackState(
                        source_id, local_id, timestamp, frame_number,
                        timestamp, frame_number, timestamp_seconds, bbox,
                    )
                    self._tracks[key] = state
                elif frame_number > state.last_frame + 1:
                    state.detector_gap_count += frame_number - state.last_frame - 1
                state.last_timestamp = timestamp
                state.last_frame = frame_number
                state.last_seen_seconds = timestamp_seconds
                state.last_bbox = bbox
                state.last_detector_present = detector_present
                state.observation_count += 1
                state.detector_confidences.append(detector_confidence)
                state.tracker_confidences.append(tracker_confidence)
                self._emit({
                    "type": "TRACK_OBSERVATION", "timestamp": timestamp,
                    "source_id": source_id, "frame_number": frame_number,
                    "local_object_id": local_id,
                    "bbox": {"left": bbox[0], "top": bbox[1], "width": bbox[2], "height": bbox[3]},
                    "detector_confidence": detector_confidence,
                    "tracker_confidence": tracker_confidence,
                })
            self._previous_frames[source_id] = (frame_number, current_frame)
            self.cleanup(timestamp_seconds)

    def cleanup(self, now_seconds: float) -> int:
        expired = [
            key for key, state in self._tracks.items()
            if now_seconds - state.last_seen_seconds >= self.ttl_seconds
        ]
        for key in expired:
            self._finish(key)
        return len(expired)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for key in list(self._tracks):
            self._finish(key)
        try:
            self._queue.put_nowait(None)
            self._writer.join(timeout=2.0)
        except Exception:
            pass
