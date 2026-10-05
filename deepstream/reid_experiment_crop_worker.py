"""Bounded worker for crop packets emitted by the isolated C++ bridge."""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from src.reid_experiment.calibration import FloorPlan2DProjector, GroundProjector
from src.reid_experiment.gallery import ExperimentalGallery
from src.reid_experiment.solider_backend import SoliderBackend


HEADER = struct.Struct("<IIIIQQQffffffIII")
MAGIC = 0x52454944
VERSION = 1


class CropWorker:
    def __init__(self, socket_path: Path, output: Path, sample_dir: Path | None,
                 queue_size: int, batch_size: int, device: str, camera_model_dir: Path | None,
                 live_state: Path):
        self.socket_path = socket_path
        self.output = output
        self.sample_dir = sample_dir
        self.queue = deque(maxlen=queue_size)
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.stop = False
        self.batch_size = max(1, batch_size)
        self.sample_limit = max(0, int(os.getenv("REID_EXPERIMENT_SAMPLE_LIMIT", "100")))
        self.samples_saved = 0
        self.last_report = time.monotonic()
        self.live_state = live_state
        dataset_root = os.getenv("REID_EXPERIMENT_DATASET_DIR", "").strip()
        self.dataset_dir = Path(dataset_root).resolve() if dataset_root else None
        self.dataset_limit = max(0, int(os.getenv("REID_EXPERIMENT_DATASET_LIMIT", "20000")))
        self.dataset_saved = 0
        self.dataset_manifest = None
        if self.dataset_dir is not None:
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            self.dataset_manifest = self.dataset_dir / "manifest.jsonl"
        self.live_tracks: dict[tuple[int, int], dict] = {}
        self.latest_crops: dict[tuple[int, int], tuple[np.ndarray, dict]] = {}
        floorplan_file = os.getenv("REID_EXPERIMENT_2D_CALIBRATION", "").strip()
        vggt = os.getenv("REID_EXPERIMENT_VGGT", "0") == "1"
        self.projector_mode = (
            "floorplan_2d" if floorplan_file and Path(floorplan_file).is_file()
            else ("vggt_ground_plane" if vggt else "amc_world_3d") if camera_model_dir else "embedding_only"
        )
        self.projector = (FloorPlan2DProjector(Path(floorplan_file)) if self.projector_mode == "floorplan_2d"
                          else GroundProjector(camera_model_dir, image_size=(
                              int(os.getenv("REID_EXPERIMENT_IMAGE_WIDTH", "1280")),
                              int(os.getenv("REID_EXPERIMENT_IMAGE_HEIGHT", "720"))) if vggt else None)
                          if camera_model_dir else None)
        self.backend = SoliderBackend(device=device)
        disable_world_gate = os.getenv("REID_EXPERIMENT_DISABLE_WORLD_GATE", "0").lower() in {"1", "true", "yes"}
        print(json.dumps({"event": "CALIBRATION_CONFIG", "mode": self.projector_mode,
                          "camera_model_dir": str(camera_model_dir),
                          "world_gate_enabled": not disable_world_gate}), flush=True)
        self.gallery = ExperimentalGallery(
            output,
            similarity_threshold=float(os.getenv("REID_EXPERIMENT_SIMILARITY_THRESHOLD", "0.92")),
            minimum_margin=float(os.getenv("REID_EXPERIMENT_MIN_MARGIN", "0.08")),
            confirmations=max(1, int(os.getenv("REID_EXPERIMENT_MATCH_CONFIRMATIONS", "4"))),
            # Keep identities available for the store journey; the local
            # tracker still controls short-term continuity.
            max_gap_seconds=float(os.getenv("REID_EXPERIMENT_MAX_GAP_SECONDS", "3600.0")),
            max_world_distance=None if disable_world_gate else float(os.getenv(
                "REID_EXPERIMENT_MAX_WORLD_DISTANCE",
                "0.08" if self.projector_mode == "floorplan_2d" else "1.25")),
            ch11_only=os.getenv("REID_EXPERIMENT_CH11_ANCHORED", "0").lower() in {"1", "true", "yes"},
            anchor_source=int(os.getenv("REID_EXPERIMENT_ANCHOR_SOURCE", "1")),
            track_min_embeddings=int(os.getenv("REID_EXPERIMENT_MIN_TRACK_EMBEDDINGS", "1")),
            local_track_gap_seconds=float(os.getenv(
                "REID_EXPERIMENT_LOCAL_TRACK_GAP_SECONDS", "3.0")),
            minimum_gallery_samples=max(1, int(os.getenv(
                "REID_EXPERIMENT_MIN_GALLERY_SAMPLES", "1"))),
        )
        self.stats = {"crop_submitted": 0, "crop_processed": 0, "crop_dropped": 0,
                      "crop_invalid": 0, "crop_too_small": 0, "crop_queue_full": 0,
                      "embedding_norms": [], "latencies_ms": []}

    @staticmethod
    def camera_label(source_id: int) -> str:
        labels = [item.strip() for item in os.getenv("REID_CAMERA_LABELS", "").split(",") if item.strip()]
        return labels[int(source_id)] if int(source_id) < len(labels) else f"source_{int(source_id):02d}"

    def submit(self, record):
        with self.condition:
            if len(self.queue) == self.queue.maxlen:
                self.queue.popleft()
                self.stats["crop_dropped"] += 1
                self.stats["crop_queue_full"] += 1
            self.queue.append(record)
            self.stats["crop_submitted"] += 1
            self.condition.notify()

    def process_batch(self, records):
        if not records:
            return
        crops = [record["crop"] for record in records]
        started = time.perf_counter()
        if len(crops) == 1:
            embeddings = [self.backend.embed(crops[0])]
        else:
            embeddings, _ = self.backend.embed_batch(crops)
        latency = (time.perf_counter() - started) * 1000.0
        per_crop_latency = latency / len(records)
        for record, embedding, crop in zip(records, embeddings, crops):
            vector = np.asarray(embedding, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            self.stats["crop_processed"] += 1
            self.stats["embedding_norms"].append(norm)
            self.stats["latencies_ms"].append(per_crop_latency)
            result = {key: value for key, value in record.items() if key != "crop"}
            # Timestamps from NvDsFrameMeta are nanoseconds in this pipeline.
            timestamp = float(record["timestamp"])
            if timestamp > 1_000_000:
                timestamp /= 1_000_000_000.0
            crop_h, crop_w = crop.shape[:2]
            # Tiny/partial detections are useful for local tracking but are
            # not reliable evidence for cross-camera identity association.
            crop_quality = min(1.0, crop_h / 160.0) if crop_h else 0.0
            result.update({"embedding": embedding, "embedding_dim": len(embedding),
                           "embedding_norm": norm, "inference_latency_ms": per_crop_latency,
                           "quality": crop_quality, "timestamp_seconds": timestamp})
            world = self.projector.world_point(int(record["source_id"]), result["bbox"]) if self.projector else None
            decision = self.gallery.observe(
                source=int(record["source_id"]), local_track=int(record["local_track_id"]),
                timestamp=timestamp, embedding=embedding,
                world=world, quality=crop_quality,
                detector_confidence=float(record["detector_confidence"]),
            )
            result.update(decision)
            result["world"] = world
            identity_id = decision.get("experimental_global_id")
            identity = self.gallery.identities.get(identity_id)
            cross_camera_confirmed = bool(
                decision.get("cross_camera_confirmed") or
                (self.gallery.ch11_only and decision.get("decision") == "STICKY_CH10_MATCH") or
                (identity is not None and len(identity.source_ids) > 1)
            )
            result["cross_camera_confirmed"] = cross_camera_confirmed
            result["identity_status"] = (
                "GLOBAL_CONFIRMED" if cross_camera_confirmed else
                "GLOBAL_PROVISIONAL" if identity_id is not None else
                "PENDING"
            )
            self.gallery.write_event(result)
            if decision.get("decision") in {"CROSS_CAMERA_MATCH", "RETROACTIVE_MERGE"}:
                print(json.dumps({
                    "event": "SOLIDER_CROSS_CAMERA_MATCH", "camera": int(record["source_id"]),
                    "local_track": int(record["local_track_id"]),
                    "experimental_global_id": decision.get("experimental_global_id"),
                    "similarity": decision.get("similarity"),
                    "world_distance": decision.get("world_distance"),
                    "evidence": decision.get("track_embedding_count"),
                    "timestamp": timestamp,
                }), flush=True)
            elif decision.get("decision") == "NEW_ID":
                print(json.dumps({
                    "event": "SOLIDER_NEW_GLOBAL", "camera": int(record["source_id"]),
                    "local_track": int(record["local_track_id"]),
                    "experimental_global_id": decision.get("experimental_global_id"),
                    "timestamp": timestamp,
                }), flush=True)
            crop_path = None
            if self.sample_dir is not None and self.samples_saved < self.sample_limit:
                camera_name = self.camera_label(record["source_id"])
                crop_dir = self.sample_dir / camera_name
                crop_dir.mkdir(parents=True, exist_ok=True)
                # BGR is the bridge wire format; OpenCV writes it directly as
                # a viewable PNG without changing the crop sent to SOLIDER.
                import cv2
                path = crop_dir / (
                    f"{camera_name}_track{record['local_track_id']}"
                    f"_f{record['frame_number']}.png")
                if cv2.imwrite(str(path), crop):
                    self.samples_saved += 1
                    crop_path = str(path)
            if crop_path:
                result["crop_path"] = crop_path
            track_key = (int(record["source_id"]), int(record["local_track_id"]))
            self.latest_crops[track_key] = (crop.copy(), result)
            self.save_dataset_observation(track_key, crop, result)
            self.live_tracks[(int(record["source_id"]), int(record["local_track_id"]))] = {
                "source_id": int(record["source_id"]), "local_track_id": int(record["local_track_id"]),
                "frame_number": int(record["frame_number"]), "timestamp": timestamp,
                "bbox": result["bbox"], "experimental_global_id": decision.get("experimental_global_id"),
                # The gallery explicitly marks a cross-camera confirmation.
                # Keep the legacy sticky fallback only for CH11-anchored mode.
                "cross_camera_confirmed": cross_camera_confirmed,
                "decision": decision.get("decision"), "world": world,
                "identity_status": result["identity_status"],
                "similarity": decision.get("similarity"), "world_distance": decision.get("world_distance"),
                "crop_path": crop_path,
            }
            if decision.get("decision") in {"CROSS_CAMERA_MATCH", "RETROACTIVE_MERGE"}:
                self.save_match_crops(track_key, decision.get("experimental_global_id"), crop)
            self.write_live_state(timestamp)
        self.report()

    def save_dataset_observation(self, track_key, crop, result):
        """Persist the exact live crop and metadata used by the experiment."""
        if self.dataset_dir is None or self.dataset_manifest is None:
            return
        if self.dataset_limit and self.dataset_saved >= self.dataset_limit:
            return
        import cv2
        source, local_track = track_key
        camera = self.camera_label(source)
        name = f"{camera}_f{int(result['frame_number']):08d}_t{self.dataset_saved:08d}.jpg"
        path = self.dataset_dir / camera / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), crop):
            return
        row = {
            "index": self.dataset_saved,
            "path": str(path.relative_to(self.dataset_dir)),
            "source_id": source,
            "camera": camera,
            "local_track_id": local_track,
            "frame_number": int(result["frame_number"]),
            "timestamp": float(result["timestamp_seconds"]),
            "bbox": result["bbox"],
            "crop_width": int(result["crop_width"]),
            "crop_height": int(result["crop_height"]),
            "detector_confidence": float(result["detector_confidence"]),
            "quality": float(result["quality"]),
            "experimental_global_id": result.get("experimental_global_id"),
            "cross_camera_confirmed": bool(result.get("cross_camera_confirmed", False)),
            "identity_status": result.get("identity_status", "PENDING"),
            "decision": result.get("decision"),
            "similarity": result.get("similarity"),
            "world_distance": result.get("world_distance"),
            "world": result.get("world"),
            "embedding": result.get("embedding"),
        }
        with self.dataset_manifest.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.dataset_saved += 1

    def report(self):
        now = time.monotonic()
        if now - self.last_report < 10.0:
            return
        self.last_report = now
        latencies = sorted(self.stats["latencies_ms"])
        p50 = latencies[len(latencies) // 2] if latencies else 0.0
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0
        print(json.dumps({
            "event": "SOLIDER_EXPERIMENT",
            "crop_submitted": self.stats["crop_submitted"],
            "crop_processed": self.stats["crop_processed"],
            "crop_dropped": self.stats["crop_dropped"],
            "crop_invalid": self.stats["crop_invalid"],
            "crop_queue_full": self.stats["crop_queue_full"],
            "embeddings_generated": self.stats["crop_processed"],
            "batch_size": self.batch_size,
            "latency_ms": {"p50": round(p50, 2), "p95": round(p95, 2)},
            "samples_saved": self.samples_saved,
            "dataset_saved": self.dataset_saved,
            "experimental_global_ids": len(self.gallery.identities),
            "coordinate_system": self.projector_mode,
            "cross_camera_candidates": self.gallery.stats["cross_camera_candidates"],
            "confirmed_cross_camera_matches": self.gallery.stats["cross_camera_match"],
            "pending_matches": self.gallery.stats["ambiguous"],
            "rejected_matches": self.gallery.stats["rejected_quality"],
        }), flush=True)

    def save_match_crops(self, current_key, identity_id, current_crop):
        """Keep a small, directly reviewable crop pair for each confirmation."""
        if self.sample_dir is None or identity_id is None:
            return
        import cv2
        match_dir = self.sample_dir.parent / "matches"
        match_dir.mkdir(parents=True, exist_ok=True)
        current_source, current_track = current_key
        current_name = self.camera_label(current_source)
        cv2.imwrite(str(match_dir / f"SOL-E{identity_id}_{current_name}_track{current_track}.png"), current_crop)
        for key, state in self.live_tracks.items():
            if key[0] == current_source or state.get("experimental_global_id") != identity_id:
                continue
            prior = self.latest_crops.get(key)
            if prior is None:
                continue
            prior_crop, _ = prior
            camera = self.camera_label(key[0])
            cv2.imwrite(str(match_dir / f"SOL-E{identity_id}_{camera}_track{key[1]}.png"), prior_crop)
            break

    def write_live_state(self, timestamp):
        cutoff = timestamp - 5.0
        self.live_tracks = {key: value for key, value in self.live_tracks.items()
                            if value["timestamp"] >= cutoff}
        document = {
            "overlay_in_video": True,
            "updated_at": time.time(), "tracks": list(self.live_tracks.values()),
            "journeys": self.gallery.journey_snapshot(),
            "stats": {key: int(value) if isinstance(value, (int, np.integer)) else value
                      for key, value in self.gallery.stats.items()},
        }
        self.live_state.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.live_state.with_suffix(".tmp")
        temporary.write_text(json.dumps(document), encoding="utf-8")
        temporary.replace(self.live_state)

    def run(self):
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_path))
        # The DeepStream container connects as triton-server, not as the host
        # user that launched this worker.
        os.chmod(self.socket_path, 0o777)
        server.listen(1)
        server.settimeout(1.0)
        print(json.dumps({"event": "SOLIDER_EXPERIMENT", "socket": str(self.socket_path)}), flush=True)
        try:
            while not self.stop:
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                with connection:
                    while not self.stop:
                        raw_header = _read_exact(connection, HEADER.size)
                        if raw_header is None:
                            break
                        values = HEADER.unpack(raw_header)
                        magic, version, payload_bytes, source_id, local_id, frame, timestamp, left, top, width, height, detector, tracker, crop_width, crop_height, fmt = values
                        if magic != MAGIC or version != VERSION or fmt != 1 or payload_bytes != crop_width * crop_height * 3:
                            self.stats["crop_invalid"] += 1
                            break
                        payload = _read_exact(connection, payload_bytes)
                        if payload is None:
                            break
                        crop = np.frombuffer(payload, dtype=np.uint8).reshape((crop_height, crop_width, 3)).copy()
                        self.submit({"source_id": source_id, "local_track_id": local_id,
                                     "frame_number": frame, "timestamp": timestamp,
                                     "bbox": [left, top, width, height],
                                     "crop_width": crop_width, "crop_height": crop_height,
                                     "detector_confidence": detector, "tracker_confidence": tracker,
                                     "crop": crop})
                        with self.lock:
                            batch = []
                            while self.queue and len(batch) < self.batch_size:
                                batch.append(self.queue.popleft())
                        self.process_batch(batch)
        finally:
            server.close()
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass


def _read_exact(connection, size):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, default=Path("runs/reid_experiment/crops.sock"))
    parser.add_argument("--output", type=Path, default=Path("runs/reid_experiment/results.jsonl"))
    parser.add_argument("--sample-dir", type=Path)
    parser.add_argument("--queue-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera-model-dir", type=Path)
    parser.add_argument("--live-state", type=Path, default=Path("runs/reid_experiment/live_state.json"))
    args = parser.parse_args()
    CropWorker(args.socket, args.output, args.sample_dir, args.queue_size, args.batch_size,
               args.device, args.camera_model_dir, args.live_state).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
