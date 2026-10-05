"""Tracker-based CH7 footfall counter using two ordered virtual lines."""

from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path

from pyservicemaker import BatchMetadataOperator

from track_continuity import frame_token, get_track_continuity


def _side(line, point):
    (x1, y1), (x2, y2) = line
    return (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)


def _crosses(old_point, new_point, line_start, line_end):
    old_side = _side((line_start, line_end), old_point)
    new_side = _side((line_start, line_end), new_point)
    return old_side == 0 or new_side == 0 or (old_side < 0) != (new_side < 0)


class FootfallCounter(BatchMetadataOperator):
    """Count line-1 -> line-2 as entry and line-2 -> line-1 as exit."""

    def __init__(self):
        super().__init__()
        self.lines_path = Path(os.getenv("FOOTFALL_LINES", "/workspace/runs/footfall/lines.json"))
        self.state_path = Path(os.getenv("FOOTFALL_STATE", "/workspace/runs/footfall/state.json"))
        self.staff_path = Path(os.getenv("FOOTFALL_STAFF", "/workspace/runs/footfall/staff_track_ids.json"))
        self.staff_gallery_path = Path(os.getenv("FOOTFALL_STAFF_GALLERY", "/workspace/runs/footfall/staff_gallery.jsonl"))
        self.candidates_path = Path(os.getenv("FOOTFALL_CANDIDATES", "/workspace/runs/footfall/staff_candidates.json"))
        self.embedding_path = Path(os.getenv(
            "FOOTFALL_EMBEDDINGS", "/workspace/runs/footfall/embeddings.jsonl"))
        self.embedding_limit = int(os.getenv("FOOTFALL_EMBEDDING_LIMIT", "5000"))
        self.embedding_interval = int(os.getenv("FOOTFALL_EMBEDDING_INTERVAL", "10"))
        self.embedding_count = 0
        self.last_embedding_frame = {}
        self.track_ttl_seconds = float(os.getenv("FOOTFALL_TRACK_TTL_SECONDS", "60"))
        self.continuity = get_track_continuity()
        self.alias_events = {}
        self.retracted_staff_aliases = set()
        try:
            self.embedding_count = sum(1 for line in self.embedding_path.open(encoding="utf-8") if line.strip())
        except OSError:
            pass
        self.lines_mtime = 0.0
        self.staff_mtime = 0.0
        self.gallery_mtime = 0.0
        self.staff_gallery = []
        self.last_candidates_write = 0.0
        self.lines = None
        self.staff_ids = set()
        self.tracks = {}
        self.state = {"day": date.today().isoformat(), "entries": 0, "exits": 0}
        self._load_state()

    def _load_state(self):
        try:
            self.state.update(json.loads(self.state_path.read_text()))
        except (OSError, ValueError):
            pass
        if self.state.get("day") != date.today().isoformat():
            self.state = {"day": date.today().isoformat(), "entries": 0, "exits": 0}

    def _reload(self):
        try:
            stamp = self.lines_path.stat().st_mtime
            if stamp != self.lines_mtime:
                value = json.loads(self.lines_path.read_text())
                if value.get("line1") and value.get("line2"):
                    self.lines, self.lines_mtime = value, stamp
        except (OSError, ValueError, TypeError):
            pass
        try:
            stamp = self.staff_path.stat().st_mtime
            if stamp != self.staff_mtime:
                value = json.loads(self.staff_path.read_text())
                self.staff_ids = {int(item) for item in value.get("track_ids", [])}
                self.staff_mtime = stamp
        except (OSError, ValueError, TypeError):
            pass
        try:
            stamp = self.staff_gallery_path.stat().st_mtime
            if stamp != self.gallery_mtime:
                gallery = []
                for line in self.staff_gallery_path.read_text().splitlines():
                    try:
                        item = json.loads(line)
                        vector = [float(value) for value in item.get("embedding", [])]
                        norm = sum(value * value for value in vector) ** 0.5
                        if len(vector) == 256 and norm > 0:
                            item["embedding"] = [value / norm for value in vector]
                            gallery.append(item)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                self.staff_gallery, self.gallery_mtime = gallery, stamp
        except OSError:
            pass

    def _write_state(self):
        self.state["footfall"] = max(0, int(self.state["entries"]) - int(self.state["exits"]))
        self.state["updated_at"] = time.time()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, separators=(",", ":")))
        temp.replace(self.state_path)

    def _save_embedding(self, frame, obj, track_id):
        if self.embedding_count >= self.embedding_limit:
            return
        frame_number = frame_token(frame)
        key = (int(frame.source_id), track_id)
        if frame_number - self.last_embedding_frame.get(key, -self.embedding_interval) < self.embedding_interval:
            return
        items = list(obj.obj_reid_items)
        if not items:
            return
        try:
            meta = items[-1].as_obj_reid()
            vector = [float(value) for value in meta.feature_vector]
            feature_size = int(meta.feature_size)
        except (AttributeError, TypeError, ValueError):
            return
        if feature_size <= 0 or len(vector) != feature_size:
            return
        record = {
            "source_id": int(frame.source_id), "local_track_id": track_id,
            "frame_number": frame_number,
            "timestamp_seconds": float(frame.buffer_pts) / 1e9,
            "feature_size": feature_size, "embedding": vector,
        }
        self.embedding_path.parent.mkdir(parents=True, exist_ok=True)
        with self.embedding_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.last_embedding_frame[key] = frame_number
        self.embedding_count += 1

    @staticmethod
    def _object_embedding(obj):
        try:
            items = list(obj.obj_reid_items)
            if not items:
                return []
            meta = items[-1].as_obj_reid()
            vector = [float(value) for value in meta.feature_vector]
            if int(meta.feature_size) != 256 or len(vector) != 256:
                return []
            norm = sum(value * value for value in vector) ** 0.5
            return [value / norm for value in vector] if norm > 0 else []
        except (AttributeError, TypeError, ValueError):
            return []

    def _matches_staff_gallery(self, obj):
        vector = self._object_embedding(obj)
        if not vector or not self.staff_gallery:
            return False
        return max(sum(a * b for a, b in zip(vector, item["embedding"]))
                   for item in self.staff_gallery) >= float(os.getenv("FOOTFALL_STAFF_THRESHOLD", "0.78"))

    @staticmethod
    def _classifier_is_staff(obj):
        """Read the TAO staff/customer result attached by secondary nvinfer."""
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

    def _write_candidates(self, candidates, now):
        if now - self.last_candidates_write < 0.5:
            return
        self.candidates_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.candidates_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"updated_at": now, "candidates": candidates}, separators=(",", ":")))
        temporary.replace(self.candidates_path)
        self.last_candidates_write = now

    def _retract_alias_events(self, key):
        """Remove earlier crossings if a customer track is later confirmed staff."""
        if key in self.retracted_staff_aliases:
            return
        events = self.alias_events.get(key)
        if events:
            self.state["entries"] = max(0, int(self.state["entries"]) - events["entries"])
            self.state["exits"] = max(0, int(self.state["exits"]) - events["exits"])
        self.retracted_staff_aliases.add(key)

    def handle_metadata(self, batch_meta):
        self._reload()
        line1 = line2 = None
        if self.lines is not None:
            line1, line2 = self.lines["line1"], self.lines["line2"]
        now = time.time()
        width = float(self.lines.get("width", 1280)) if self.lines else 1280.0
        height = float(self.lines.get("height", 720)) if self.lines else 720.0
        candidates = []
        for frame in batch_meta.frame_items:
            for obj in frame.object_items:
                if int(obj.class_id) != 0:
                    continue
                track_id = int(obj.object_id)
                self._save_embedding(frame, obj, track_id)
                embedding = self._object_embedding(obj)
                rect_for_continuity = obj.rect_params
                continuity = self.continuity.resolve(
                    int(frame.source_id),
                    track_id,
                    frame_token(frame),
                    embedding,
                    (float(rect_for_continuity.left), float(rect_for_continuity.top),
                     float(rect_for_continuity.width), float(rect_for_continuity.height)),
                    classifier_staff=(
                        self._classifier_is_staff(obj)
                        or track_id in self.staff_ids
                        or self._matches_staff_gallery(obj)
                    ),
                )
                logical_id = int(continuity["logical_track_id"])
                # Staff must remain visible to the detector/tracker long
                # enough to obtain Re-ID features, but must not reach OSD or
                # counting as a visible person.  NvDCF metadata is retained;
                # only the downstream drawing is suppressed here.
                is_staff = bool(continuity["staff"])
                if is_staff:
                    try:
                        obj.rect_params.border_width = 0
                        obj.text_params.display_text = ""
                    except AttributeError:
                        pass
                    self._retract_alias_events((int(frame.source_id), logical_id))
                elif os.getenv("FOOTFALL_LOGICAL_ID_OSD", "0") == "1":
                    try:
                        obj.text_params.display_text = f"person {logical_id}"
                    except AttributeError:
                        pass
                # Embedding collection is independent of line configuration.
                # This lets the operator review staff identities before enabling
                # counting, and avoids losing the first part of the day.
                if self.lines is None:
                    continue
                rect = obj.rect_params
                candidates.append({
                    "source_id": int(frame.source_id), "local_track_id": track_id,
                    "logical_track_id": logical_id,
                    "bbox": [float(rect.left) / width, float(rect.top) / height,
                             float(rect.width) / width, float(rect.height) / height],
                    "embedding": embedding, "staff_match": is_staff,
                })
                if is_staff:
                    continue
                point = ((float(rect.left) + float(rect.width) / 2.0) / width,
                         (float(rect.top) + float(rect.height)) / height)
                key = (int(frame.source_id), logical_id)
                previous = self.tracks.get(key)
                if previous is not None:
                    crossed = 1 if _crosses(previous["point"], point, *line1) else 2 if _crosses(previous["point"], point, *line2) else None
                    if crossed is not None:
                        last = previous.get("last_line")
                        if last == 1 and crossed == 2 and now - previous["line_time"] <= 30:
                            self.state["entries"] += 1
                            self.alias_events.setdefault(key, {"entries": 0, "exits": 0})["entries"] += 1
                            previous["last_line"] = None
                        elif last == 2 and crossed == 1 and now - previous["line_time"] <= 30:
                            self.state["exits"] += 1
                            self.alias_events.setdefault(key, {"entries": 0, "exits": 0})["exits"] += 1
                            previous["last_line"] = None
                        else:
                            previous["last_line"], previous["line_time"] = crossed, now
                else:
                    previous = {"point": point, "last_line": None, "line_time": now}
                    self.tracks[key] = previous
                previous["point"], previous["seen"] = point, now
        if self.lines is not None:
            self.tracks = {key: value for key, value in self.tracks.items()
                           if now - value.get("seen", now) <= self.track_ttl_seconds}
        self.state["embedding_count"] = self.embedding_count
        self._write_candidates(candidates, now)
        self._write_state()
