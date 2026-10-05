#!/usr/bin/env python3
"""Offline CH10/CH11 synchronized frame and floor-plan point picker."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file


HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>CH10 / CH11 point calibration</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#111827;color:#e5e7eb;font:14px system-ui, sans-serif}
header{padding:14px 18px;background:#0b1220;position:sticky;top:0;z-index:4}h1{font-size:20px;margin:0 0 5px}.sub{color:#9ca3af}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:10px 18px;background:#172033}.bar button,.bar input{background:#26354c;color:#fff;border:1px solid #4b6384;border-radius:5px;padding:7px 10px}.bar button{cursor:pointer}.bar button.active{background:#0f766e;border-color:#34d399}.bar input{width:90px}
.grid{display:grid;grid-template-columns:minmax(280px,1fr) minmax(260px, .8fr) minmax(280px,1fr);gap:10px;padding:10px}.panel{background:#1f2937;border-radius:8px;padding:8px;min-width:0}.panel h2{font-size:16px;margin:0 0 7px}.stage{position:relative;background:#05080d;line-height:0;overflow:hidden}.stage img{width:100%;height:auto;display:block;cursor:crosshair}.marker{position:absolute;transform:translate(-50%,-50%);width:18px;height:18px;border:2px solid #fff;border-radius:50%;background:#ef4444;color:#fff;font:bold 10px system-ui;text-align:center;line-height:14px;pointer-events:none}.layout .stage img{cursor:crosshair}.hint{color:#9ca3af;font-size:12px;margin-top:6px}.points{padding:0 18px 20px}.pointrow{display:flex;align-items:center;gap:8px;border-bottom:1px solid #273449;padding:7px 0}.pointrow button{background:#26354c;border:1px solid #4b6384;color:#fff;border-radius:4px;padding:4px 8px}.ok{color:#34d399}.warn{color:#fbbf24}
@media(max-width:900px){.grid{grid-template-columns:1fr}.layout{order:-1}}
</style></head><body>
<header><h1>CH10 / CH11 floor-plane calibration picker</h1><div class="sub">Click the same physical floor point in CH10, the floor plan, and CH11. Save each point as P1–P12.</div></header>
<div class="bar"><button onclick="step(-1)">◀</button><button onclick="step(1)">▶</button><label>Time <input id="sec" type="number" min="0" step="0.5" value="0" onchange="loadTime()"> s</label><label>CH11 offset <input id="offset" type="number" step="0.1" value="0" onchange="loadTime()"> s</label><button onclick="loadTime()">Go</button><span id="status"></span><span style="margin-left:auto">Selected: <b id="selected">P1</b></span><button onclick="newPoint()">New point</button><button class="active" onclick="savePoint()">Save point</button></div>
<div class="grid"><section class="panel"><h2>CH10</h2><div class="stage" id="s10"><img id="im10"><div id="m10"></div></div><div class="hint">Click the footpoint (where shoes meet the floor).</div></section>
<section class="panel layout"><h2>Floor plan / layout</h2><div class="stage" id="sl"><img id="layout"><div id="ml"></div></div><div class="hint">Click the corresponding physical location. Mark fixed landmarks, not people.</div></section>
<section class="panel"><h2>CH11</h2><div class="stage" id="s11"><img id="im11"><div id="m11"></div></div><div class="hint">Use the same timestamp and click the matching floor point.</div></section></div>
<div class="points"><h2>Saved points</h2><div id="rows"></div></div>
<script>
let cur=1, picks={ch10:null,layout:null,ch11:null}, saved=[];
const $=id=>document.getElementById(id);
function setStatus(x,cls=''){ $('status').textContent=x; $('status').className=cls }
function markers(){for(const [k,id] of [['ch10','m10'],['layout','ml'],['ch11','m11']]){const m=$(id);m.innerHTML='';const p=picks[k];if(p){const e=document.createElement('div');e.className='marker';e.textContent='P'+cur;e.style.left=(p.x*100)+'%';e.style.top=(p.y*100)+'%';m.appendChild(e)}}}
function clickStage(stage,key,ev){const r=stage.getBoundingClientRect();picks[key]={x:Math.max(0,Math.min(1,(ev.clientX-r.left)/r.width)),y:Math.max(0,Math.min(1,(ev.clientY-r.top)/r.height))};markers();setStatus('Point '+('P'+cur)+' '+key+' selected')}
[['s10','ch10'],['sl','layout'],['s11','ch11']].forEach(([id,k])=>$(id).addEventListener('click',e=>{if(e.target.tagName==='IMG')clickStage($(id),k,e)}));
async function loadTime(){let s=parseFloat($('sec').value)||0,o=parseFloat($('offset').value)||0;$('layout').src='/layout';let r=await fetch(`/api/frame?second=${s}&offset=${o}`);if(!r.ok){setStatus(await r.text(),'warn');return}let j=await r.json();$('im10').src=j.ch10;$('im11').src=j.ch11;$('sec').value=j.second.toFixed(1);setStatus('Frames aligned at '+j.second.toFixed(1)+' s')}
function step(d){$('sec').value=Math.max(0,(parseFloat($('sec').value)||0)+d);loadTime()}
function newPoint(){cur=saved.length+1;picks={ch10:null,layout:null,ch11:null};$('selected').textContent='P'+cur;markers();setStatus('Select P'+cur+' in all three panels')}
async function savePoint(){if(!picks.ch10||!picks.layout||!picks.ch11){setStatus('Select all three locations before saving','warn');return}let body={point_id:'P'+cur,second:parseFloat($('sec').value)||0,offset:parseFloat($('offset').value)||0,picks};let r=await fetch('/api/points',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!r.ok){setStatus(await r.text(),'warn');return}saved=await r.json();renderRows();setStatus('Saved P'+cur,'ok');newPoint()}
async function removePoint(id){await fetch('/api/points/'+id,{method:'DELETE'});saved=await (await fetch('/api/points')).json();renderRows()}
function renderRows(){$('rows').innerHTML=saved.length?saved.map(p=>`<div class="pointrow"><b>${p.point_id}</b><span>${Number(p.second).toFixed(1)}s</span><span class="ok">CH10 ✓ layout ✓ CH11 ✓</span><button onclick="removePoint('${p.point_id}')">remove</button></div>`).join(''):'<span class="hint">No points saved yet.</span>'}
loadTime();renderRows();
</script></body></html>'''


def probe_duration(path: str) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path], text=True)
    return float(out.strip())


def frame_data_uri(path: str, second: float) -> str:
    # PNG avoids JPEG artifacts when the user clicks small floor landmarks.
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0, second):.3f}",
           "-i", path, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
    raw = subprocess.check_output(cmd)
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def build_app(args: argparse.Namespace) -> Flask:
    app = Flask(__name__)
    lock = threading.Lock()
    points_path = Path(args.output).expanduser().resolve()
    points_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        points = json.loads(points_path.read_text()) if points_path.exists() else []
    except (json.JSONDecodeError, OSError):
        points = []
    state = {"points": points}

    @app.get("/")
    def index(): return Response(HTML, mimetype="text/html")

    @app.get("/layout")
    def layout(): return send_file(args.layout)

    @app.get("/api/frame")
    def frame():
        try:
            second = float(request.args.get("second", "0"))
            offset = float(request.args.get("offset", "0"))
            if second < 0 or second > args.duration:
                return Response("time is outside the recording", status=400)
            right = min(args.duration, max(0.0, second + offset))
            return jsonify({"second": second, "ch10": frame_data_uri(args.ch10, second),
                            "ch11": frame_data_uri(args.ch11, right)})
        except Exception as exc:
            return Response(f"frame extraction failed: {exc}", status=500)

    @app.get("/api/points")
    def get_points():
        with lock: return jsonify(state["points"])

    @app.post("/api/points")
    def add_point():
        item = request.get_json(force=True)
        if not item.get("point_id") or not all(item.get("picks", {}).get(k) for k in ("ch10", "layout", "ch11")):
            return Response("point_id, ch10, layout and ch11 are required", status=400)
        item.update({"ch10_video": args.ch10, "ch11_video": args.ch11, "layout": args.layout,
                     "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        with lock:
            state["points"] = [p for p in state["points"] if p.get("point_id") != item["point_id"]]
            state["points"].append(item)
            points_path.write_text(json.dumps(state["points"], indent=2) + "\n")
            return jsonify(state["points"])

    @app.delete("/api/points/<point_id>")
    def delete_point(point_id):
        with lock:
            state["points"] = [p for p in state["points"] if p.get("point_id") != point_id]
            points_path.write_text(json.dumps(state["points"], indent=2) + "\n")
            return jsonify(state["points"])

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ch10", required=True)
    p.add_argument("--ch11", required=True)
    p.add_argument("--layout", required=True)
    p.add_argument("--output", default="runs/point_calibration/points.json")
    p.add_argument("--port", type=int, default=18082)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    for f in (args.ch10, args.ch11, args.layout):
        if not os.path.isfile(f): p.error(f"not found: {f}")
    args.ch10 = str(Path(args.ch10).expanduser().resolve())
    args.ch11 = str(Path(args.ch11).expanduser().resolve())
    args.layout = str(Path(args.layout).expanduser().resolve())
    args.duration = min(probe_duration(args.ch10), probe_duration(args.ch11))
    build_app(args).run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__": main()
