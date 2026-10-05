#!/usr/bin/env python3
"""Record synchronized CH10/CH11 calibration videos into a new project."""

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]


def read_env():
    values = {}
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({key: value for key, value in os.environ.items() if key in values})
    return values


def rtsp_url(values, channel, subtype):
    username = quote(values["CAMERA_USERNAME"], safe="")
    password = quote(values["CAMERA_PASSWORD"], safe="")
    host = values["NVR_225_HOST"]
    port = values["NVR_225_PORT"]
    return f"rtsp://{username}:{password}@{host}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path)
    parser.add_argument("--subtype", type=int, default=0,
                        help="NVR stream subtype; 0 is the 1920x1080 main stream")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="seconds; 0 records until SIGINT/SIGTERM")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    project = args.project_dir or ROOT / f"calibration/ground_floor_ch10_ch11_{stamp}"
    videos = project / "inputs/videos"
    logs = project / "capture_logs"
    videos.mkdir(parents=True, exist_ok=False)
    logs.mkdir()

    values = read_env()
    cameras = (
        ("cam_00", "CH10", 10),
        ("cam_01", "CH11", 11),
    )
    started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "project": project.name,
        "started_at_utc": started_at,
        "camera_order": [item[0] for item in cameras],
        "streams": [
            {"id": stream_id, "label": label, "channel": channel,
             "nvr": 225, "subtype": args.subtype,
             "video": str(videos / f"{stream_id}.mp4")}
            for stream_id, label, channel in cameras
        ],
        "encoding": "HEVC stream copy; no re-encoding",
        "camera_positions": "new placement for triangular-geometry calibration",
    }
    manifest_path = project / "capture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    processes = []
    stopping = False

    def stop_all(*_):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)

    try:
        for stream_id, _, channel in cameras:
            output = videos / f"{stream_id}.mp4"
            log = (logs / f"{stream_id}.log").open("w")
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "info",
                "-rtsp_transport", "tcp", "-use_wallclock_as_timestamps", "1",
                "-i", rtsp_url(values, channel, args.subtype), "-map", "0:v:0", "-an",
                "-c:v", "copy", "-movflags", "+faststart",
            ]
            if args.duration > 0:
                command += ["-t", str(args.duration)]
            command += ["-y", str(output)]
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
            print(f"started {stream_id}=CH{channel} pid={processes[-1].pid}", flush=True)

        if args.duration > 0:
            deadline = time.monotonic() + args.duration + 10
            while time.monotonic() < deadline and any(p.poll() is None for p in processes):
                time.sleep(1)
        else:
            while any(process.poll() is None for process in processes):
                time.sleep(1)
    finally:
        stop_all()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        manifest["stopped_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"capture project: {project}", flush=True)


if __name__ == "__main__":
    main()
