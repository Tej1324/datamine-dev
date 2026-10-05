"""CPU-only bounded crop collector for the TAO staff review experiment."""

from __future__ import annotations

import argparse
import os
import socket
import struct
from pathlib import Path

import cv2
import numpy as np

HEADER = struct.Struct("<IIIIQQQffffffIII")
MAGIC = 0x52454944
VERSION = 1


def read_exact(conn, size):
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20000)
    args = parser.parse_args()
    labels = [x.strip() for x in os.getenv("REID_CAMERA_LABELS", "").split(",") if x.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        args.socket.unlink()
    except FileNotFoundError:
        pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket))
    os.chmod(args.socket, 0o777)
    server.listen(1)
    saved = 0
    try:
        print(f"crop collector listening on {args.socket}", flush=True)
        while saved < args.limit:
            conn, _ = server.accept()
            with conn:
                while saved < args.limit:
                    raw = read_exact(conn, HEADER.size)
                    if raw is None:
                        break
                    values = HEADER.unpack(raw)
                    magic, version, payload_bytes, source_id, local_id, frame, timestamp, left, top, width, height, detector, tracker, crop_width, crop_height, fmt = values
                    if magic != MAGIC or version != VERSION or fmt != 1 or payload_bytes != crop_width * crop_height * 3:
                        break
                    payload = read_exact(conn, payload_bytes)
                    if payload is None:
                        break
                    crop = np.frombuffer(payload, dtype=np.uint8).reshape((crop_height, crop_width, 3))
                    label = labels[int(source_id)] if int(source_id) < len(labels) else f"source_{int(source_id):02d}"
                    directory = args.output / label
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / f"{label}_f{int(frame):08d}_t{saved:08d}_track{int(local_id)}.jpg"
                    if cv2.imwrite(str(path), crop):
                        saved += 1
                        if saved % 100 == 0:
                            print(f"saved={saved}", flush=True)
    finally:
        server.close()
        try:
            args.socket.unlink()
        except FileNotFoundError:
            pass
        print(f"crop collector complete saved={saved}", flush=True)


if __name__ == "__main__":
    main()
