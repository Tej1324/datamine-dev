"""Attach asynchronous identity results to current tracked objects, never old boxes."""
import json
import time
from pathlib import Path

from pyservicemaker import BatchMetadataOperator


class SoliderDisplay(BatchMetadataOperator):
    def __init__(self):
        super().__init__()
        self.path = Path('/workspace/runs/reid_experiment/live_state.json')
        self.checked = 0.0
        self.tracks = {}
        self.updated = 0.0

    @staticmethod
    def _set_border_color(rect, red, green, blue):
        """Set NvOSD color when exposed by the Python metadata wrapper."""
        color = getattr(rect, 'border_color', None)
        if color is None:
            return
        color.red = float(red)
        color.green = float(green)
        color.blue = float(blue)
        color.alpha = 1.0
        rect.border_color = color

    def handle_metadata(self, batch_meta):
        now = time.monotonic()
        if now - self.checked >= 0.25:
            self.checked = now
            try:
                state = json.loads(self.path.read_text())
                self.updated = float(state['updated_at'])
                self.tracks = {(int(t['source_id']), int(t['local_track_id'])): t
                               for t in state.get('tracks', [])}
            except (OSError, ValueError, KeyError):
                self.tracks = {}
        for frame in batch_meta.frame_items:
            for obj in frame.object_items:
                if int(obj.class_id) != 0:
                    continue
                # This probe is the sole identity OSD for the experimental
                # SOLIDER path. Hide detector/tracker rectangles and labels
                # unless the current frame has a fresh SOLIDER result.
                rect = obj.rect_params
                rect.border_width = 0
                obj.rect_params = rect
                local = int(obj.object_id)
                track = self.tracks.get((int(frame.source_id), local))
                label = ''
                if track and time.time() - self.updated < 3:
                    pts = int(frame.buffer_pts) / 1e9
                    age = pts - float(track['timestamp'])
                    gid = track.get('experimental_global_id')
                    if gid is not None and 0 <= age <= 2:
                        label = (
                            f'SOL-E{gid}'
                            if track.get('cross_camera_confirmed')
                            else f'PENDING-E{gid}'
                        )
                        rect.border_width = 3
                        if track.get('cross_camera_confirmed'):
                            self._set_border_color(rect, 0.0, 0.65, 1.0)
                        else:
                            self._set_border_color(rect, 1.0, 0.75, 0.0)
                        obj.rect_params = rect
                params = obj.text_params
                params.display_text = label
                obj.text_params = params
