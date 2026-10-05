"""Low-overhead DeepStream buffer counters for native pipeline diagnosis."""

from __future__ import annotations

import threading
import time

from pyservicemaker import BatchMetadataOperator


class StageCounter(BatchMetadataOperator):
    """Report batch/frame counts without changing metadata or video buffers."""

    def __init__(self, stage: str, report_seconds: float = 5.0):
        super().__init__()
        self.stage = stage
        self.report_seconds = report_seconds
        self.started = time.monotonic()
        self.last_report = self.started
        self.batches = 0
        self.frames = 0
        self.last_frame_by_source = {}
        self.lock = threading.Lock()

    def handle_metadata(self, batch_meta) -> None:
        now = time.monotonic()
        with self.lock:
            self.batches += 1
            self.frames += int(batch_meta.n_frames)
            for frame_meta in batch_meta.frame_items:
                self.last_frame_by_source[str(frame_meta.source_id)] = int(frame_meta.frame_number)
            if now - self.last_report < self.report_seconds:
                return
            elapsed = max(now - self.started, 1e-6)
            batches = self.batches
            frames = self.frames
            last = dict(self.last_frame_by_source)
            self.last_report = now
        print(
            f"STAGE {self.stage}: batches={batches} frames={frames} "
            f"avg_fps={frames / elapsed:.2f} last_frame_by_source={last}",
            flush=True,
        )
