#!/usr/bin/env python3
"""Small CH7 footfall dashboard: live JPEG, two-click lines, and counts."""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request

PAGE = r'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CH7 Footfall</title>
<style>body{margin:0;background:#0f172a;color:#e5e7eb;font:16px system-ui}header{padding:18px 24px;background:#172033;border-bottom:1px solid #334155}h1{margin:0 0 4px;font-size:24px}main{max-width:1320px;margin:auto;padding:18px}.panel{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:16px}.toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:12px 0}button{padding:10px 14px;border:1px solid #64748b;border-radius:6px;background:#26364e;color:#fff;cursor:pointer}button.active,button.primary{background:#2563eb;border-color:#60a5fa}.counts,.fps{padding:10px 14px;background:#0b1220;border-radius:7px;font-size:20px;font-weight:700}.counts{margin-left:auto}.fps{color:#93c5fd}.stage{position:relative;background:#000;line-height:0;aspect-ratio:16/9;overflow:hidden;border-radius:8px}.stage img{width:100%;height:100%;object-fit:contain}.stage canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}.hint{color:#aab8cc;margin:10px 0 0}</style>
<header><h1>CH7 Footfall Analytics</h1><div class=hint>Line 1 → line 2 = entry · line 2 → line 1 = exit · every detected person is counted</div></header><main><section class=panel><div class=toolbar><button id=l1 class=primary>Draw line 1</button><button id=l2 class=primary>Draw line 2</button><button id=save class=primary>Save lines</button><button id=reset>Reset counts</button><span id=fps class=fps>RTSP FPS: --</span><span id=counts class=counts>Entries: 0 · Exits: 0 · Footfall: 0</span></div><div class=stage><img id=video alt="CH7 live video"><canvas id=canvas></canvas></div><p id=message class=hint>Choose a line, then click its two endpoints.</p></section></main>
<script>const stage=document.querySelector('.stage'),img=document.querySelector('#video'),canvas=document.querySelector('#canvas'),ctx=canvas.getContext('2d');let lines={line1:null,line2:null},mode=0,first=null,dirty=false;function resize(){canvas.width=stage.clientWidth;canvas.height=stage.clientHeight;draw()}function point(e){const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))]}function message(v){document.querySelector('#message').textContent=v}function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);for(let i=1;i<=2;i++){const q=lines['line'+i];if(!q)continue;ctx.strokeStyle=i===1?'#22c55e':'#f59e0b';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(q[0][0]*canvas.width,q[0][1]*canvas.height);ctx.lineTo(q[1][0]*canvas.width,q[1][1]*canvas.height);ctx.stroke();ctx.font='bold 18px system-ui';ctx.fillStyle=ctx.strokeStyle;ctx.fillText('LINE '+i,q[0][0]*canvas.width+6,q[0][1]*canvas.height-8)}if(first){ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(first[0]*canvas.width,first[1]*canvas.height,6,0,Math.PI*2);ctx.fill()}}document.querySelector('#l1').onclick=()=>{mode=1;first=null;message('Line 1: click the second endpoint.');draw()};document.querySelector('#l2').onclick=()=>{mode=2;first=null;message('Line 2: click the second endpoint.');draw()};canvas.onclick=e=>{if(!mode)return;const p=point(e);if(!first){first=p;message('Now click the second endpoint.');draw();return}lines['line'+mode]=[first,p];mode=0;first=null;dirty=true;message('Line created. Save both lines when ready.');draw()};document.querySelector('#save').onclick=async()=>{if(!lines.line1||!lines.line2){message('Create both lines first.');return}const r=await fetch('/api/lines',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(lines)});dirty=false;message(r.ok?'SAVED ✓ Counting is active.':'Save failed.')};document.querySelector('#reset').onclick=async()=>{if(!confirm('Reset today\'s counts?'))return;await fetch('/api/reset',{method:'POST'});message('Counts reset.')};async function poll(){try{const s=await (await fetch('/api/state?t='+Date.now(),{cache:'no-store'})).json();document.querySelector('#counts').textContent=`Entries: ${s.entries||0} · Exits: ${s.exits||0} · Footfall: ${s.footfall||0}`;const fps=Number(s.fps||0);document.querySelector('#fps').textContent=`RTSP FPS: ${fps.toFixed(1)}`;if(!dirty&&s.lines)lines=s.lines;draw()}catch(e){}setTimeout(poll,1000)}function frame(){img.src='/snapshot.jpg?t='+Date.now();setTimeout(frame,150)}img.onload=resize;addEventListener('resize',resize);fetch('/api/lines').then(r=>r.json()).then(v=>{lines=v;resize();poll()});frame();</script>'''


class LatestJpeg:
    def __init__(self, port):
        self.latest = None
        self.lock = threading.Lock()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", port))
        self.server.listen(2)
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            conn, _ = self.server.accept()
            buffer = b""
            try:
                while True:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buffer += chunk
                    while True:
                        start = buffer.find(b"\xff\xd8")
                        if start < 0:
                            buffer = buffer[-2:]
                            break
                        end = buffer.find(b"\xff\xd9", start + 2)
                        if end < 0:
                            buffer = buffer[start:]
                            break
                        with self.lock:
                            self.latest = buffer[start:end + 2]
                        buffer = buffer[end + 2:]
            finally:
                conn.close()

    def get(self):
        with self.lock:
            return self.latest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--tcp-port", type=int, default=7010)
    parser.add_argument("--root", type=Path, default=Path("runs/footfall"))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    lines = args.root / "lines.json"
    state = args.root / "state.json"
    if not lines.exists():
        lines.write_text(json.dumps({"width": 1280, "height": 720, "line1": None, "line2": None}))
    if not state.exists():
        state.write_text(json.dumps({"entries": 0, "exits": 0, "footfall": 0, "updated_at": 0}))
    relay = LatestJpeg(args.tcp_port)
    app = Flask(__name__)

    @app.get("/")
    def index():
        return Response(PAGE, mimetype="text/html")

    @app.get("/snapshot.jpg")
    def snapshot():
        frame = relay.get()
        return Response(frame or b"", status=200 if frame else 503, mimetype="image/jpeg")

    @app.get("/stream.mjpg")
    def stream():
        def frames():
            while True:
                frame = relay.get()
                if frame:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.1)
        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/lines")
    def get_lines():
        return jsonify(json.loads(lines.read_text()))

    @app.post("/api/lines")
    def set_lines():
        value = request.get_json(force=True) or {}
        value.update(width=1280, height=720)
        lines.write_text(json.dumps(value, separators=(",", ":")))
        return jsonify(value)

    @app.get("/api/state")
    def get_state():
        value = json.loads(state.read_text())
        value["lines"] = json.loads(lines.read_text())
        return jsonify(value)

    @app.post("/api/reset")
    def reset():
        state.write_text(json.dumps({"entries": 0, "exits": 0, "footfall": 0, "updated_at": time.time()}))
        return jsonify(ok=True)

    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
