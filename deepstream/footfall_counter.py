"""Two-line footfall counter using only YOLO person boxes and NvDCF IDs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from pyservicemaker import BatchMetadataOperator


def _side(line, point):
    (x1, y1), (x2, y2) = line
    return (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)


def _crosses(old_point, new_point, start, end):
    before, after = _side((start, end), old_point), _side((start, end), new_point)
    return before == 0 or after == 0 or (before < 0) != (after < 0)


class FootfallCounter(BatchMetadataOperator):
    def __init__(self):
        super().__init__()
        root = Path(os.getenv("FOOTFALL_ROOT", "/workspace/runs/footfall"))
        self.lines_path = Path(os.getenv("FOOTFALL_LINES", str(root / "lines.json")))
        self.state_path = Path(os.getenv("FOOTFALL_STATE", str(root / "state.json")))
        self.width = int(os.getenv("FOOTFALL_WIDTH", "1280"))
        self.height = int(os.getenv("FOOTFALL_HEIGHT", "720"))
        self.track_ttl = float(os.getenv("FOOTFALL_TRACK_TTL_SECONDS", "120"))
        self.tracks = {}
        self.state = {"entries": 0, "exits": 0, "footfall": 0, "updated_at": 0}
        self.lines = None
        self.last_reload = 0.0
        self.last_write = 0.0
        self._reload()

    def _reload(self):
        now = time.time()
        if now - self.last_reload < 1.0:
            return
        self.last_reload = now
        try:
            value = json.loads(self.lines_path.read_text())
            self.lines = value if value.get("line1") and value.get("line2") else None
        except (OSError, ValueError, TypeError):
            self.lines = None
        try:
            value = json.loads(self.state_path.read_text())
            for key in ("entries", "exits", "footfall"):
                self.state[key] = int(value.get(key, 0))
        except (OSError, ValueError, TypeError):
            pass

    def _write(self, now):
        if now - self.last_write < 0.5:
            return
        self.state["footfall"] = max(0, self.state["entries"] - self.state["exits"])
        self.state["updated_at"] = now
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, separators=(",", ":")))
        temporary.replace(self.state_path)
        self.last_write = now

    def handle_metadata(self, batch_meta):
        self._reload()
        now = time.time()
        if not self.lines:
            self._write(now)
            return
        line1, line2 = self.lines["line1"], self.lines["line2"]
        for frame in batch_meta.frame_items:
            for obj in frame.object_items:
                if int(obj.class_id) != 0:
                    continue
                track_id = int(obj.object_id)
                if track_id == (1 << 64) - 1:
                    continue
                rect = obj.rect_params
                point = ((float(rect.left) + float(rect.width) / 2) / self.width,
                         (float(rect.top) + float(rect.height)) / self.height)
                key = (int(frame.source_id), track_id)
                previous = self.tracks.get(key)
                if previous is None:
                    self.tracks[key] = {"point": point, "last_line": None, "line_time": now, "seen": now}
                    continue
                crossed = 1 if _crosses(previous["point"], point, *line1) else 2 if _crosses(previous["point"], point, *line2) else None
                if crossed is not None:
                    if previous.get("last_line") == 1 and crossed == 2 and now - previous["line_time"] <= 30:
                        self.state["entries"] += 1
                        previous["last_line"] = None
                    elif previous.get("last_line") == 2 and crossed == 1 and now - previous["line_time"] <= 30:
                        self.state["exits"] += 1
                        previous["last_line"] = None
                    else:
                        previous["last_line"], previous["line_time"] = crossed, now
                previous["point"], previous["seen"] = point, now
        self.tracks = {key: value for key, value in self.tracks.items()
                       if now - value.get("seen", now) <= self.track_ttl}
        self._write(now)
