#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 NVIDIA
# SPDX-License-Identifier: Apache-2.0
"""Small local UI for marking matching people in synchronized camera videos.

The page samples both videos at one-second intervals.  Click the same person
in the left and right images, then save the pair.  Coordinates are always
saved; optional displayed Global IDs are copied into an evaluator CSV when
both IDs are supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_file


HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CH10 / CH11 manual matching</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#111827;color:#e5e7eb;font:16px system-ui,sans-serif}
header{padding:18px 24px;background:#0b1220;position:sticky;top:0;z-index:3}
h1{margin:0 0 6px;font-size:24px}.muted{color:#9ca3af}.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:14px 24px;background:#172033}
button,input{font:inherit;border-radius:6px;border:1px solid #4b5563;padding:9px 12px;background:#1f2937;color:#f9fafb}
button{cursor:pointer}button.primary{background:#16a34a;border-color:#22c55e}button.warn{background:#92400e;border-color:#f59e0b}
input{width:100px}.time{font-size:20px;font-weight:700;min-width:120px;text-align:center}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:14px 24px}.card{background:#1f2937;border-radius:8px;padding:10px}.card h2{font-size:18px;margin:0 0 8px}.stage{position:relative;background:#000;line-height:0;overflow:hidden;cursor:crosshair}.stage img{width:100%;height:auto;display:block}.detection{position:absolute;display:block;padding:0;background:transparent;border:3px solid #f97316;border-radius:0;cursor:pointer;line-height:normal}.detection .tag{position:absolute;top:-24px;left:-3px;padding:3px 5px;background:#ea580c;color:#fff;font:700 12px system-ui;white-space:nowrap}.detection.selected-left{border-color:#22c55e;box-shadow:0 0 0 2px #052e16}.detection.selected-left .tag{background:#16a34a}.detection.selected-right{border-color:#38bdf8;box-shadow:0 0 0 2px #082f49}.detection.selected-right .tag{background:#0284c7}.fields{display:flex;gap:8px;align-items:center;padding:12px 24px;flex-wrap:wrap}.fields label{color:#cbd5e1}.status{padding:8px 24px;color:#93c5fd;min-height:36px}.help{padding:0 24px 18px;color:#9ca3af}.count{color:#86efac}
@media(max-width:800px){.pair{grid-template-columns:1fr}.stage img{max-height:55vh;object-fit:contain}}
</style>
</head>
<body>
<header><h1>CH10 / CH11 manual matching</h1><div class="muted">Click one box on the left, click the matching box on the right, then save.</div></header>
<div class="toolbar">
  <button id="prev">◀ Previous second</button><span class="time" id="time">0.0 s</span><button id="next">Next second ▶</button>
  <label>Jump to <input id="jump" type="number" min="0" step="1" value="0"> s</label>
  <span class="muted" id="duration"></span>
</div>
<div class="pair">
  <section class="card"><h2>Left / CH10</h2><div class="stage" id="leftStage"><img id="leftImage"></div></section>
  <section class="card"><h2>Right / CH11</h2><div class="stage" id="rightStage"><img id="rightImage"></div></section>
</div>
<div class="fields">
  <button class="primary" id="same">Save SAME person</button>
  <span class="count" id="saved">Saved pairs: 0</span>
</div>
<div class="status" id="status"></div>
<div class="help">Click an orange detection box. The selected CH10 box turns green and the selected CH11 box turns blue. Clicking another box replaces that selection.</div>
<script>
let second=1, step=1, rightOffset=0, duration=0, leftSelection=null, rightSelection=null, saved=0;
const $=id=>document.getElementById(id);
function status(text){$('status').textContent=text}
function clearBoxes(stage){document.querySelectorAll('#'+stage+' .detection').forEach(x=>x.remove())}
function drawBoxes(stage,side,boxes){clearBoxes(stage);for(const box of boxes){let el=document.createElement('button');el.type='button';el.className='detection';el.dataset.id=box.id;el.style.left=(box.x*100)+'%';el.style.top=(box.y*100)+'%';el.style.width=(box.width*100)+'%';el.style.height=(box.height*100)+'%';let tag=document.createElement('span');tag.className='tag';tag.textContent='D'+box.id+' · '+Math.round(box.score*100)+'%';el.appendChild(tag);el.onclick=(event)=>{event.stopPropagation();selectBox(stage,side,box)};$(stage).appendChild(el)}}
function redrawSelection(stage,side){document.querySelectorAll('#'+stage+' .detection').forEach(el=>el.classList.remove('selected-left','selected-right'));let item=side==='left'?leftSelection:rightSelection;if(!item)return;let target=[...document.querySelectorAll('#'+stage+' .detection')].find(el=>Number(el.dataset.id)===item.id);if(target)target.classList.add(side==='left'?'selected-left':'selected-right')}
function selectBox(stage,side,box){if(side==='left')leftSelection=box;else rightSelection=box;redrawSelection(stage,side);status('Selected '+side+' D'+box.id+'. Select the matching box on the other side, then save.')}
async function loadSide(side,sourceSecond){let stage=side==='left'?'leftStage':'rightStage';let image=side==='left'?'leftImage':'rightImage';$(image).src='/frame/'+side+'/'+sourceSecond+'?t='+Date.now();let response=await fetch('/api/detections/'+side+'/'+sourceSecond);if(!response.ok)throw new Error('Could not load '+side+' detections');let data=await response.json();drawBoxes(stage,side,data.boxes)}
async function load(){ $('time').textContent=second.toFixed(1)+' s'; $('jump').value=second; leftSelection=null;rightSelection=null;status('Loading person boxes…');try{await Promise.all([loadSide('left',second),loadSide('right',second)]);status('Click one detected person box in each image, then save.')}catch(error){status(error.message)}}
async function save(){if(!leftSelection||!rightSelection){status('Click one detected person box in both images first.');return}let body={second,left:leftSelection,right:rightSelection,same_person:true};let r=await fetch('/api/annotations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});let data=await r.json();if(!r.ok){status(data.error||'Save failed');return}saved=data.count;$('saved').textContent='Saved pairs: '+saved;status('Saved D'+leftSelection.id+' ↔ D'+rightSelection.id+' at '+second.toFixed(1)+' s.');leftSelection=null;rightSelection=null;redrawSelection('leftStage','left');redrawSelection('rightStage','right')}
$('same').onclick=save;$('prev').onclick=()=>{second=Math.max(0,second-step);load()};$('next').onclick=()=>{second=Math.min(duration,second+step);load()};$('jump').onchange=()=>{second=Math.max(0,Math.min(duration,Number($('jump').value)||0));load()};
fetch('/api/info').then(r=>r.json()).then(data=>{duration=data.duration;step=data.step;rightOffset=data.right_offset||0;second=Math.min(1,duration);$('duration').textContent='duration '+duration.toFixed(1)+' s';saved=data.count;$('saved').textContent='Saved pairs: '+saved;load()}).catch(()=>status('Could not load video information.'));
</script>
</body></html>
"""


class PersonDetector:
    """Small offline YOLO26 person detector used only to draw click targets."""

    def __init__(self, model: Path, confidence: float):
        try:
            import cv2
            import numpy as np  # noqa: F401 - verifies the OpenCV dependency
        except ImportError as error:
            raise RuntimeError("OpenCV and NumPy are required for offline boxes") from error
        self.cv2 = cv2
        self.net = cv2.dnn.readNetFromONNX(str(model))
        self.confidence = confidence

    def detect(self, source: Path) -> list[dict[str, float | int]]:
        """Return stable, normalized person boxes for one extracted frame."""
        image = self.cv2.imread(str(source))
        if image is None:
            raise RuntimeError(f"unable to read extracted frame: {source}")
        height, width = image.shape[:2]
        input_width, input_height = 1280, 736
        blob = self.cv2.dnn.blobFromImage(
            image, 1.0 / 255.0, (input_width, input_height),
            swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        output = self.net.forward()
        rows = output.reshape(-1, output.shape[-1])
        boxes, scores = [], []
        scale_x = width / input_width
        scale_y = height / input_height
        for row in rows:
            if row.shape[0] < 6 or int(row[5]) != 0 or float(row[4]) < self.confidence:
                continue
            x1, y1, x2, y2 = (float(value) for value in row[:4])
            left = max(0, min(width - 1, round(x1 * scale_x)))
            top = max(0, min(height - 1, round(y1 * scale_y)))
            right = max(left + 1, min(width - 1, round(x2 * scale_x)))
            bottom = max(top + 1, min(height - 1, round(y2 * scale_y)))
            boxes.append([left, top, right - left, bottom - top])
            scores.append(float(row[4]))
        kept = self.cv2.dnn.NMSBoxes(boxes, scores, self.confidence, 0.45)
        indexes = [int(item) for item in kept] if len(kept) else []
        detections: list[dict[str, float | int]] = []
        for number, index in enumerate(indexes, 1):
            left, top, box_width, box_height = boxes[index]
            detections.append({
                "id": number,
                "x": left / width,
                "y": top / height,
                "width": box_width / width,
                "height": box_height / height,
                "score": scores[index],
            })
        return detections


class Labeler:
    def __init__(self, video_left: Path, video_right: Path, output: Path, step: float,
                 detector: PersonDetector | None, right_offset: float):
        self.video_left = video_left.resolve()
        self.video_right = video_right.resolve()
        self.output = output.resolve()
        self.step = step
        self.detector = detector
        self.right_offset = right_offset
        self.cache = self.output.parent / (self.output.stem + "_frames")
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.rows: list[dict] = []
        if self.output.exists():
            for line in self.output.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.rows.append(json.loads(line))
        self.duration = min(self._duration(self.video_left),
                            self._duration(self.video_right) - self.right_offset)
        if self.duration <= 0:
            raise ValueError("right offset leaves no overlapping video duration")

    @staticmethod
    def _duration(path: Path) -> float:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())

    def frame(self, side: str, second: float) -> Path:
        source = self.video_left if side == "left" else self.video_right
        source_second = second if side == "left" else second + self.right_offset
        safe_second = max(0.0, min(self.duration if side == "left" else self.duration + self.right_offset,
                                   source_second))
        name = f"{side}_{safe_second:010.3f}.jpg"
        target = self.cache / name
        if not target.exists():
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(safe_second),
                 "-i", str(source), "-frames:v", "1", "-q:v", "2", "-y", str(target)],
                check=True,
            )
        return target

    def detections(self, side: str, second: float) -> list[dict[str, float | int]]:
        """Detect once, cache JSON, and return the exact selectable boxes."""
        frame = self.frame(side, second)
        if self.detector is None:
            return []
        cache = frame.with_suffix(".detections.json")
        if not cache.exists():
            cache.write_text(json.dumps(self.detector.detect(frame), separators=(",", ":")),
                             encoding="utf-8")
        return json.loads(cache.read_text(encoding="utf-8"))

    def add(self, row: dict) -> int:
        with self.lock:
            row["reference_id"] = f"manual_{len(self.rows) + 1:04d}"
            self.rows.append(row)
            self.output.parent.mkdir(parents=True, exist_ok=True)
            with self.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            self._write_csv()
            return len(self.rows)

    def _write_csv(self):
        csv_path = self.output.with_suffix(".csv")
        rows = [row for row in self.rows if row.get("left_id") and row.get("right_id")]
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            fields = ["second", "reference_id", "left_id", "right_id", "same"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({"second": row["second"], "reference_id": row["reference_id"],
                                 "left_id": row["left_id"], "right_id": row["right_id"],
                                 "same": "yes" if row["same_person"] else "no"})


def create_app(labeler: Labeler) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        from flask import Response
        return Response(HTML, mimetype="text/html")

    @app.get("/api/info")
    def info():
        return jsonify(duration=labeler.duration, step=labeler.step, right_offset=labeler.right_offset,
                       count=len(labeler.rows),
                       output=str(labeler.output))

    @app.get("/frame/<side>/<second>")
    def frame(side: str, second: str):
        if side not in {"left", "right"}:
            return jsonify(error="side must be left or right"), 400
        try:
            value = float(second)
            if value < 0 or value > labeler.duration:
                raise ValueError
            return send_file(labeler.frame(side, value), mimetype="image/jpeg", max_age=0)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            return jsonify(error=f"unable to extract frame: {error}"), 400

    @app.get("/api/detections/<side>/<second>")
    def detections(side: str, second: str):
        if side not in {"left", "right"}:
            return jsonify(error="side must be left or right"), 400
        try:
            value = float(second)
            if value < 0 or value > labeler.duration:
                raise ValueError
            return jsonify(boxes=labeler.detections(side, value))
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            return jsonify(error=f"unable to detect people: {error}"), 400

    @app.post("/api/annotations")
    def annotations():
        data = request.get_json(silent=True) or {}
        try:
            second = float(data["second"])
            left = data["left"]
            right = data["right"]
            if not (0 <= second <= labeler.duration):
                raise ValueError("second is outside the video duration")
            for point in (left, right):
                if not isinstance(point.get("id"), int):
                    raise ValueError("select a detected person box")
                for field in ("x", "y", "width", "height", "score"):
                    if field not in point or not isinstance(point[field], (int, float)):
                        raise ValueError("invalid detection selection")
                if not (0 <= float(point["x"]) <= 1 and 0 <= float(point["y"]) <= 1
                        and 0 < float(point["width"]) <= 1 and 0 < float(point["height"]) <= 1):
                    raise ValueError("detection box is outside the image")
            row = {
                "second": second,
                "right_second": max(0.0, second + labeler.right_offset),
                "left": left, "right": right,
                "left_id": "", "right_id": "",
                "same_person": bool(data.get("same_person")),
            }
            return jsonify(count=labeler.add(row))
        except (KeyError, TypeError, ValueError) as error:
            return jsonify(error=str(error)), 400

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-left", type=Path, required=True, help="CH10 MP4")
    parser.add_argument("--video-right", type=Path, required=True, help="CH11 MP4")
    parser.add_argument("--output", type=Path, default=Path("runs/ch10_ch11_manual_pairs.jsonl"))
    parser.add_argument("--step", type=float, default=1.0, help="sampling interval in seconds")
    parser.add_argument("--right-offset", type=float, default=-0.26,
                        help="seconds added to CH11; this recording starts CH11 about 0.26s later")
    parser.add_argument("--detector", type=Path,
                        default=Path("models/yolo26s/yolo26s.onnx"),
                        help="YOLO ONNX model used to draw offline person boxes")
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    if args.step <= 0:
        parser.error("--step must be positive")
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be between 0 and 1")
    detector = PersonDetector(args.detector.resolve(), args.confidence)
    labeler = Labeler(args.video_left, args.video_right, args.output, args.step, detector,
                      args.right_offset)
    print(f"labeler: http://{args.host}:{args.port}/", flush=True)
    print(f"videos: {labeler.video_left} / {labeler.video_right}", flush=True)
    print(f"saved JSONL: {labeler.output}", flush=True)
    print(f"saved CSV (when IDs are entered): {labeler.output.with_suffix('.csv')}", flush=True)
    create_app(labeler).run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
