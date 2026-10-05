#!/usr/bin/env python3
"""Isolated CH7 footfall dashboard and two-line editor."""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory


PAGE = r'''<!doctype html><meta charset="utf-8"><title>CH7 Footfall</title>
<style>body{margin:0;background:#0f172a;color:#e5e7eb;font:15px system-ui}header{padding:18px 24px;background:#172033;border-bottom:1px solid #334155}header h1{margin:0;font-size:22px}main{padding:18px 24px;max-width:1320px;margin:auto}.plan{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}.step,.panel{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:14px}.step b{display:block;color:#93c5fd;margin-bottom:5px}.step span{color:#cbd5e1}.stage{position:relative;background:#000;line-height:0;border-radius:8px;overflow:hidden}.stage img{width:100%;display:block;min-height:240px;object-fit:contain}.stage canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:12px 0}button{padding:9px 12px;border-radius:6px;border:1px solid #64748b;background:#1e293b;color:#fff;cursor:pointer}button:hover{background:#334155}.primary{background:#2563eb;border-color:#3b82f6}.staff{background:#166534;border-color:#22c55e}.customer{background:#475569}.card{font-size:19px;font-weight:700;padding:9px 13px;border-radius:7px;background:#0b1220;margin-left:auto}.hint{color:#9ca3af}.panel{margin-top:16px}.panel-title{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px}.review-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}.review-item{border:3px solid transparent;padding:4px;background:#111827;color:#e5e7eb}.review-item.selected{border-color:#22c55e}.review-item img{width:100%;height:150px;object-fit:contain;background:#000}.review-item small{display:block;line-height:1.25}.pager{display:flex;gap:8px;align-items:center;margin-top:12px}@media(max-width:900px){.plan{grid-template-columns:1fr}.review-grid{grid-template-columns:repeat(3,1fr)}.card{margin-left:0}}</style>
<header><h1>CH7 Footfall Analytics</h1><div class="hint">Entrance camera · line 1 → line 2 counts entry · line 2 → line 1 counts exit</div></header><main>
<div class=plan><div class=step><b>1 · Configure entrance</b><span>Draw two floor lines and save them.</span></div><div class=step><b>2 · Label staff</b><span>Review crops and save only staff examples.</span></div><div class=step><b>3 · Monitor count</b><span>Staff matches are excluded from customer footfall.</span></div></div>
<section class=panel><div class=panel-title><b>Live entrance view</b><button id=reconnect>Reconnect live video</button></div><div class=bar><button id=l1 class=primary>Draw line 1</button><button id=l2 class=primary>Draw line 2</button><button id=save class=primary>Save lines</button><button id=uniform class=staff>Mark uniform staff</button><button id=badge class=staff>Mark ID-card staff</button><button id=both class=staff>Mark uniform + ID card</button><button id=staffsave class=staff>Save staff</button><button id=customer class=customer>Mark customer</button><button id=customersave class=customer>Dismiss customer</button><span class=card id=counts>Entries: 0 · Exits: 0 · Footfall: 0</span></div><div class=stage><img id=video src="/snapshot.jpg"><canvas id=c></canvas></div><p class=hint id=msg>Draw line 1 or line 2 using two clicks.</p></section>
<section class=panel><div class=panel-title><b>Review queue</b><span class=hint id=review-count>0 pending crops</span></div><div class=hint>Representative crops are collected automatically. Select a crop, choose a label, and save or dismiss it.</div><div class=bar><button id=reload-review>Reload current crops</button><span class=hint id=review-status></span></div><div id=review class=review-grid></div><div class=pager><button id=prev-review>Previous</button><span id=review-page>Page 1</span><button id=next-review>Next</button></div></section></main>
<script>const c=document.getElementById('c'),x=c.getContext('2d');let mode=0,lines={line1:null,line2:null},start=null,dirty=false,staffMode=false,staffCategory=null,candidates=[],selectedStaff=new Set(),reviewRows=[],selectedReview=new Map(),lastReviewSignature='',reviewPage=0;
function size(){const r=document.querySelector('.stage').getBoundingClientRect();if(r.width>0&&r.height>0){c.width=Math.round(r.width);c.height=Math.round(r.height);draw()}}addEventListener('resize',size);document.getElementById('video').onload=size;new ResizeObserver(size).observe(document.querySelector('.stage'));document.getElementById('l1').onclick=()=>{staffMode=false;mode=1;start=null;msg('Line 1: click the first endpoint, then click the second endpoint.')};document.getElementById('l2').onclick=()=>{staffMode=false;mode=2;start=null;msg('Line 2: click the first endpoint, then click the second endpoint.')};
function msg(v){document.getElementById('msg').textContent=v}function p(e){const r=c.getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))]}
function draw(){x.clearRect(0,0,c.width,c.height);for(let i=1;i<=2;i++){const q=lines['line'+i];if(!q)continue;x.strokeStyle=i===1?'#22c55e':'#f59e0b';x.lineWidth=4;x.beginPath();x.moveTo(q[0][0]*c.width,q[0][1]*c.height);x.lineTo(q[1][0]*c.width,q[1][1]*c.height);x.stroke();x.fillStyle=x.strokeStyle;x.font='bold 18px sans-serif';x.fillText('LINE '+i,q[0][0]*c.width+6,q[0][1]*c.height-8)}if(staffMode){candidates.forEach((v,i)=>{const b=v.bbox,s=selectedStaff.has(i);x.strokeStyle=s?'#00ff70':v.staff_match?'#a855f7':'#38bdf8';x.lineWidth=s?6:3;x.strokeRect(b[0]*c.width,b[1]*c.height,b[2]*c.width,b[3]*c.height);x.fillStyle=x.strokeStyle;x.font='bold 16px sans-serif';x.fillText((s?'SELECTED ':'')+'T'+v.local_track_id,b[0]*c.width+3,Math.max(18,b[1]*c.height+18))})}if(start){x.fillStyle='#fff';x.beginPath();x.arc(start[0]*c.width,start[1]*c.height,7,0,Math.PI*2);x.fill()}}
c.onclick=e=>{const point=p(e);if(staffMode){let hit=-1;for(let i=candidates.length-1;i>=0;i--){const b=candidates[i].bbox;if(point[0]>=b[0]&&point[0]<=b[0]+b[2]&&point[1]>=b[1]&&point[1]<=b[1]+b[3]){hit=i;break}}if(hit>=0){selectedStaff.has(hit)?selectedStaff.delete(hit):selectedStaff.add(hit);msg(selectedStaff.size+' staff box(es) selected. Click Save staff.');draw()}return}if(!mode)return;if(!start){start=point;msg('Now click the second endpoint.');draw();return}lines['line'+mode]=[start,point];dirty=true;mode=0;start=null;msg('Line created. Draw the other line or click Save lines.');draw()};
document.getElementById('save').onclick=async()=>{if(!lines.line1||!lines.line2){msg('Please create both lines before saving.');return}const r=await fetch('/api/lines',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(lines)});if(r.ok){dirty=false;msg('SAVED ✓ Lines saved successfully. Counting started.')}else msg('SAVE FAILED ✗')};
function renderReview(){const root=document.getElementById('review');root.innerHTML='';const ordered=reviewRows.slice().reverse(),pageSize=12;const pages=Math.max(1,Math.ceil(ordered.length/pageSize));reviewPage=Math.min(reviewPage,pages-1);const cacheKey=Date.now();ordered.slice(reviewPage*pageSize,(reviewPage+1)*pageSize).forEach((v)=>{const b=document.createElement('button');b.className='review-item'+(selectedReview.has(v.id)?' selected':'');const im=document.createElement('img');im.loading='eager';im.alt='crop '+v.id;im.src=v.crop_url+'?t='+v.id+'-'+cacheKey;im.onerror=()=>{im.replaceWith(Object.assign(document.createElement('span'),{textContent:'Image unavailable',className:'hint'}))};const text=document.createElement('small');text.textContent='crop '+v.id+' · track '+v.local_track_id+(v.staff_match?' · current staff match':'');b.append(im,text);b.onclick=()=>{if(selectedReview.has(v.id))selectedReview.delete(v.id);else selectedReview.set(v.id,v);b.classList.toggle('selected');msg(selectedReview.size+' review crop(s) selected. Choose staff or customer, then save.')};root.append(b)});document.getElementById('review-count').textContent=ordered.length+' pending crops';document.getElementById('review-page').textContent='Page '+(reviewPage+1)+' / '+pages;document.getElementById('prev-review').disabled=reviewPage===0;document.getElementById('next-review').disabled=reviewPage>=pages-1}
function chooseStaff(category){mode=0;start=null;staffMode=true;staffCategory=category;selectedStaff=new Set();msg('Click staff boxes or saved review crops wearing '+category+'. Then click Save staff.');draw()}function chooseCustomer(){mode=0;start=null;staffMode=true;staffCategory='customer';selectedStaff=new Set();msg('Click customer boxes or saved review crops. Then click Dismiss customer.');draw()}document.getElementById('uniform').onclick=()=>chooseStaff('uniform');document.getElementById('badge').onclick=()=>chooseStaff('an ID card');document.getElementById('both').onclick=()=>chooseStaff('both');document.getElementById('customer').onclick=chooseCustomer;document.getElementById('staffsave').onclick=async()=>{if(!staffMode||staffCategory==='customer'||(!selectedStaff.size&&!selectedReview.size)){msg('Choose a staff category and select at least one box or review crop.');return}const r=await fetch('/api/staff',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({category:staffCategory,indices:[...selectedStaff],records:[...selectedReview.values()]})});const b=await r.json();if(r.ok){staffMode=false;selectedStaff=new Set();selectedReview=new Map();msg('SAVED ✓ '+b.saved+' staff embedding(s) written to the gallery. Future matching staff will be excluded.');draw();renderReview()}else msg('SAVE FAILED ✗ '+(b.error||'Staff save failed'))};document.getElementById('customersave').onclick=async()=>{if(!staffMode||staffCategory!=='customer'||(!selectedStaff.size&&!selectedReview.size)){msg('Choose Mark customer and select at least one box or review crop.');return}const r=await fetch('/api/customer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({records:[...selectedReview.keys()].map(id=>({id}))})});const b=await r.json();if(r.ok){staffMode=false;selectedStaff=new Set();reviewRows=reviewRows.filter(v=>!selectedReview.has(v.id));selectedReview=new Map();msg('DISMISSED ✓ '+b.dismissed+' crops. No customer training embeddings saved.');draw();renderReview()}else msg('SAVE FAILED ✗ '+(b.error||'Customer save failed'))};
document.getElementById('prev-review').onclick=()=>{reviewPage=Math.max(0,reviewPage-1);renderReview()};document.getElementById('next-review').onclick=()=>{reviewPage++;renderReview()};document.getElementById('reload-review').onclick=()=>{lastReviewSignature='';reviewPage=0;poll().catch(()=>{});document.getElementById('review-status').textContent='Reloading current crops…'};let videoRetry=0,videoTimer=null;function scheduleVideo(delay=500){clearTimeout(videoTimer);videoTimer=setTimeout(()=>{document.getElementById('video').src='/snapshot.jpg?refresh='+Date.now()},delay)}function reconnectVideo(){videoRetry++;msg('Connecting live video… retry '+videoRetry);scheduleVideo(videoRetry>3?1500:500)}document.getElementById('reconnect').onclick=()=>{videoRetry=0;reconnectVideo()};document.getElementById('video').onload=()=>{videoRetry=0;size();scheduleVideo()};document.getElementById('video').onerror=()=>{msg('Live video disconnected; retrying…');scheduleVideo(1500)};
async function poll(){const s=await (await fetch('/api/state?refresh='+Date.now(),{cache:'no-store'})).json();document.getElementById('counts').textContent=`Entries: ${s.entries||0} · Exits: ${s.exits||0} · Footfall: ${s.footfall||0}`;if(s.lines&&!dirty&&!start){lines=s.lines}const r=await fetch('/api/staff/candidates?refresh='+Date.now(),{cache:'no-store'});const b=await r.json();candidates=b.candidates||[];const q=await fetch('/api/staff/review?refresh='+Date.now(),{cache:'no-store'});const nextRows=(await q.json()).rows||[];const signature=nextRows.map(v=>v.id).join(',');if(signature!==lastReviewSignature){reviewRows=nextRows;lastReviewSignature=signature;renderReview();document.getElementById('review-status').textContent='Showing latest crops';}draw()}setInterval(()=>poll().catch(()=>{}),3000);fetch('/api/lines').then(r=>r.json()).then(v=>{lines=v;size();return poll()}).catch(()=>{msg('Dashboard connection is retrying…')});</script>'''


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
                        end = buffer.find(b"\xff\xd9", start + 2)
                        if start < 0 or end < 0:
                            break
                        frame = buffer[start:end + 2]
                        buffer = buffer[end + 2:]
                        with self.lock:
                            self.latest = frame
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
    candidates = args.root / "staff_candidates.json"
    gallery = args.root / "staff_gallery.jsonl"
    review = args.root / "staff_review.jsonl"
    review_labels = args.root / "staff_review_labels.json"
    crop_dir = args.root / "staff_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    review_seen = set()
    review_last = {}
    review_sequence = 0
    review_limit = int(os.getenv("FOOTFALL_REVIEW_LIMIT", "5000"))
    if not lines.exists():
        lines.write_text(json.dumps({"width": 1280, "height": 720, "line1": None, "line2": None}))
    if not state.exists():
        state.write_text(json.dumps({"entries": 0, "exits": 0, "footfall": 0}))
    if not candidates.exists():
        candidates.write_text(json.dumps({"updated_at": 0, "candidates": []}))
    if not gallery.exists():
        gallery.touch()
    if not review.exists():
        review.touch()
    if not review_labels.exists():
        review_labels.write_text("{}")
    try:
        review_sequence = sum(1 for line in review.read_text().splitlines() if line.strip())
    except OSError:
        review_sequence = 0
    relay = LatestJpeg(args.tcp_port)
    app = Flask(__name__)

    @app.get("/")
    def index():
        return Response(PAGE, mimetype="text/html")

    @app.get("/stream.mjpg")
    def stream():
        def frames():
            while True:
                frame = relay.get()
                if frame:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                # Keep the browser relay at 10 FPS so a slow SSH tunnel does
                # not build a large socket backlog while the detector runs at
                # its own cadence.
                time.sleep(0.1)
        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/snapshot.jpg")
    def snapshot():
        frame = relay.get()
        if not frame:
            return Response(status=503)
        return Response(frame, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/lines")
    def get_lines():
        return jsonify(json.loads(lines.read_text()))

    @app.post("/api/lines")
    def set_lines():
        value = request.get_json(force=True)
        value.update(width=1280, height=720)
        lines.write_text(json.dumps(value))
        return jsonify(value)

    @app.get("/api/state")
    def get_state():
        value = json.loads(state.read_text())
        value["lines"] = json.loads(lines.read_text())
        return jsonify(value)

    @app.get("/api/staff/candidates")
    def staff_candidates():
        try:
            value = json.loads(candidates.read_text())
            frame = relay.get()
            image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR) if frame else None
            if image is not None:
                height, width = image.shape[:2]
                nonlocal review_sequence
                now = time.time()
                for item in value.get("candidates", []):
                    vector = item.get("embedding", [])
                    track_key = (item.get("source_id"), item.get("local_track_id"))
                    if len(vector) != 256 or now - review_last.get(track_key, 0) < 3:
                        continue
                    left, top, box_width, box_height = item["bbox"]
                    x1, y1 = max(0, int(left * width)), max(0, int(top * height))
                    x2, y2 = min(width, int((left + box_width) * width)), min(height, int((top + box_height) * height))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    if review_sequence >= review_limit:
                        continue
                    review_sequence += 1
                    filename = f"staff_{review_sequence:06d}.jpg"
                    if cv2.imwrite(str(crop_dir / filename), image[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 90]):
                        record = {"id": review_sequence, "crop": filename, "source_id": item.get("source_id"),
                                  "local_track_id": item.get("local_track_id"), "embedding": vector,
                                  "staff_match": item.get("staff_match", False), "created_at": now}
                        with review.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                        review_last[track_key] = now
            return jsonify(value)
        except (OSError, ValueError):
            return jsonify(updated_at=0, candidates=[])

    @app.get("/api/staff/review")
    def staff_review():
        rows = []
        try:
            labels = json.loads(review_labels.read_text())
            # Keep the browser payload small; the full review archive remains
            # on disk and can be paginated later if needed.
            for line in review.read_text().splitlines()[-60:]:
                item = json.loads(line)
                if str(item.get("id")) in labels:
                    continue
                item["crop_url"] = "/staff-crop/" + item["crop"]
                rows.append(item)
        except (OSError, ValueError, TypeError):
            pass
        return jsonify(rows=rows)

    @app.get("/staff-crop/<path:name>")
    def staff_crop(name):
        response = send_from_directory(crop_dir, name, max_age=0)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

    @app.post("/api/staff")
    def save_staff():
        value = request.get_json(force=True) or {}
        category = str(value.get("category", "")).strip().lower()
        if category not in {"uniform", "an id card", "both"}:
            return jsonify(error="invalid staff category"), 400
        try:
            rows = json.loads(candidates.read_text()).get("candidates", [])
        except (OSError, ValueError):
            rows = []
        saved = 0
        labels = json.loads(review_labels.read_text())
        with gallery.open("a", encoding="utf-8") as stream:
            for record in value.get("records", []):
                vector = record.get("embedding", []) if isinstance(record, dict) else []
                if len(vector) == 256:
                    stream.write(json.dumps({"category": category, "source_id": record.get("source_id"),
                                             "local_track_id": record.get("local_track_id"),
                                             "embedding": vector}, separators=(",", ":")) + "\n")
                    saved += 1
                    if record.get("id") is not None:
                        labels[str(record["id"])] = category
            for raw_index in value.get("indices", []):
                try:
                    item = rows[int(raw_index)]
                    vector = item.get("embedding", [])
                    if len(vector) != 256:
                        continue
                    stream.write(json.dumps({
                        "category": category, "source_id": item.get("source_id"),
                        "local_track_id": item.get("local_track_id"),
                        "embedding": vector,
                    }, separators=(",", ":")) + "\n")
                    saved += 1
                except (IndexError, TypeError, ValueError):
                    continue
        if not saved:
            return jsonify(error="selected box has no usable NVIDIA Re-ID embedding yet"), 400
        review_labels.write_text(json.dumps(labels, separators=(",", ":")))
        return jsonify(saved=saved, category=category)

    @app.post("/api/customer")
    def save_customer():
        value = request.get_json(force=True) or {}
        try:
            rows = json.loads(candidates.read_text()).get("candidates", [])
        except (OSError, ValueError):
            rows = []
        try:
            labels = json.loads(review_labels.read_text())
        except (OSError, ValueError):
            labels = {}
        # Customer means dismiss from review, never enroll in a model gallery.
        selected = {str(record.get("id")) for record in value.get("records", [])
                    if isinstance(record, dict) and record.get("id") is not None}
        if not selected:
            return jsonify(error="Select saved review crops to dismiss customers."), 400
        for identifier in selected:
            labels[identifier] = "dismissed"
        temporary = review_labels.with_suffix(".tmp")
        temporary.write_text(json.dumps(labels, separators=(",", ":")))
        temporary.replace(review_labels)
        return jsonify(dismissed=len(selected))

    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
