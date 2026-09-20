"""Small local-tracklet ledger for Checkpoint C; no cross-camera matching."""

import json
import os
import time
import threading
from pathlib import Path


class LocalTracklets:
    def __init__(self, camera_ids, output_dir=None, missing_frames=30):
        self.camera_ids = tuple(camera_ids)
        self.missing_frames = missing_frames
        self.tracks = {}
        self.lock = threading.RLock()
        self.pending_events = []
        self.live = {camera_id: [] for camera_id in self.camera_ids}
        self.camera_stats = {
            camera_id: {"active_tracks": 0, "created_tracks": 0, "ended_tracks": 0}
            for camera_id in self.camera_ids
        }
        self.events_path = Path(output_dir or "runs") / "tracking_events.jsonl"
        self.stats_path = Path(output_dir or "runs") / "tracking_stats.json"
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def _event(self, camera_id, local_id, event):
        record = {"time": time.time(), "camera_id": camera_id,
                  "local_track_id": local_id, "event": event}
        self.pending_events.append(record)

    @staticmethod
    def _bbox(obj):
        box = obj.rect_params
        return {"left": round(float(box.left), 2), "top": round(float(box.top), 2),
                "width": round(float(box.width), 2), "height": round(float(box.height), 2)}

    def observe(self, camera_id, frame_num, objects, now=None):
        with self.lock:
            self._observe(camera_id, frame_num, objects, now)

    def observe_values(self, camera_id, frame_num, observations, now=None):
        """Record plain Python values copied from native metadata in the probe."""
        with self.lock:
            self._observe_values(camera_id, frame_num, observations, now)

    def _observe(self, camera_id, frame_num, objects, now=None):
        observations = []
        for obj in objects:
            if int(obj.class_id) != 0 or int(obj.object_id) == 0xffffffffffffffff:
                continue
            observations.append({"local_track_id": int(obj.object_id), "bbox": self._bbox(obj)})
        self._observe_values(camera_id, frame_num, observations, now)

    def _observe_values(self, camera_id, frame_num, observations, now=None):
        now = time.time() if now is None else now
        seen = set()
        live = []
        stats = self.camera_stats[camera_id]
        for observation in observations:
            local_id = int(observation["local_track_id"])
            key = (camera_id, local_id)
            bbox = observation["bbox"]
            seen.add(key)
            track = self.tracks.get(key)
            if track is None:
                track = self.tracks[key] = {
                    "camera_id": camera_id, "local_track_id": local_id,
                    "first_seen": now, "last_seen": now, "frame_count": 0,
                    "latest_bbox": bbox, "observations": [], "last_frame": frame_num,
                }
                stats["created_tracks"] += 1
                self._event(camera_id, local_id, "START")
            track.update(last_seen=now, frame_count=track["frame_count"] + 1,
                         latest_bbox=bbox, last_frame=frame_num)
            track["observations"].append({"frame": int(frame_num), "time": now, "bbox": bbox})
            live.append({"local_track_id": local_id, "bbox": [bbox["left"], bbox["top"], bbox["width"], bbox["height"]],
                         "frame_width": 1280, "frame_height": 720, "timestamp": now})
            # Keep the ledger bounded while retaining multiple observations for later stages.
            if len(track["observations"]) > 300:
                del track["observations"][:-300]

        for key, track in list(self.tracks.items()):
            if key[0] != camera_id or key in seen:
                continue
            if int(frame_num) - int(track["last_frame"]) >= self.missing_frames:
                stats["ended_tracks"] += 1
                self._event(key[0], key[1], "END")
                del self.tracks[key]
        stats["active_tracks"] = sum(1 for key in self.tracks if key[0] == camera_id)
        self.live[camera_id] = live
        # Publication is deliberately performed by the background publisher.

    def publish(self):
        with self.lock:
            payload = {"updated_at": time.time(), "cameras": json.loads(json.dumps(self.camera_stats)),
                       "active_tracklets": len(self.tracks), "cross_camera_matching": False}
            live = json.loads(json.dumps(self.live))
            events = self.pending_events[:]
            self.pending_events.clear()
        temp = self.stats_path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, indent=2))
        os.replace(temp, self.stats_path)
        live_temp = self.stats_path.with_name("live_tracks.tmp")
        live_temp.write_text(json.dumps({"updated_at": time.time(), "cameras": live}, separators=(",", ":")))
        os.replace(live_temp, self.stats_path.with_name("live_tracks.json"))
        if events:
            with self.events_path.open("a") as stream:
                for event in events:
                    stream.write(json.dumps(event, separators=(",", ":")) + "\n")
