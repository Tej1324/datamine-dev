#!/usr/bin/env python3
"""Review exact live CH10/CH11 crops and measure Re-ID decisions.

The UI presents one-second buckets from a dataset recorded by the isolated
SOLIDER worker. Select one crop from each camera and label SAME or DIFFERENT.
Annotations are written beside the dataset and can be scored with
``tools/evaluate_live_reid.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file


HTML = r'''<!doctype html><html><head><meta charset="utf-8"><title>Live Re-ID evaluator</title>
<style>body{margin:0;background:#111827;color:#e5e7eb;font:15px system-ui}header{padding:14px 22px;background:#0b1220;position:sticky;top:0;z-index:2}.toolbar{padding:12px 22px;background:#172033;display:flex;gap:10px;align-items:center;flex-wrap:wrap}button,input{font:inherit;padding:8px 11px;border-radius:5px;border:1px solid #4b5563;background:#1f2937;color:#fff}button{cursor:pointer}.same{background:#15803d}.different{background:#9f1239}.time{font-weight:700;min-width:120px;text-align:center}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:14px}.panel{background:#1f2937;border-radius:8px;padding:10px}.panel h2{margin:0 0 8px;font-size:18px}.crops{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:9px}.crop{border:3px solid transparent;border-radius:6px;padding:5px;background:#111827;cursor:pointer;min-height:150px}.crop.selected-left{border-color:#22c55e}.crop.selected-right{border-color:#38bdf8}.crop img{width:100%;height:180px;object-fit:contain;background:#000}.matches{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px}.match{background:#111827;border-radius:7px;padding:8px}.match-images{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:7px 0}.match-images img{width:100%;height:220px;object-fit:contain;background:#000}.meta{font-size:11px;line-height:1.35;color:#cbd5e1}.status{padding:8px 22px;color:#93c5fd;min-height:22px}@media(max-width:800px){.grid{grid-template-columns:1fr}.matches{grid-template-columns:1fr}}</style></head>
<body><header><b>CH10 / CH11 live-crop Re-ID evaluator</b><div>Pick one crop per camera, then label whether they are the same person.</div></header>
<div class="toolbar"><button id="prev">◀</button><span class="time" id="time"></span><button id="next">▶</button><label>Bucket <input id="bucket" type="number" min="0" step="1" value="0"></label><button id="mode">MATCHED SETS</button><button class="same" id="same">SAME PERSON</button><button class="different" id="different">DIFFERENT</button><span id="count"></span></div>
<div class="grid"><section class="panel"><h2>CH10</h2><div id="left" class="crops"></div></section><section class="panel"><h2>CH11</h2><div id="right" class="crops"></div></section></div><div class="panel" id="matches-panel" style="display:none;margin:14px"><h2>System-suggested matched sets</h2><div id="matches" class="matches"></div></div><div class="status" id="status"></div>
<script>
let bucket=0,step=1,left=null,right=null,info=null,mode='crops';const $=x=>document.getElementById(x);
function msg(x){$('status').textContent=x} function clear(){left=null;right=null}
function draw(side,rows){let root=$(side);root.innerHTML='';for(const r of rows){let b=document.createElement('button');b.className='crop';b.dataset.index=r.index;let im=document.createElement('img');im.src='/crop/'+r.index+'?t='+Date.now();let m=document.createElement('div');m.className='meta';let sim=r.similarity==null?'—':Number(r.similarity).toFixed(3);let world=Array.isArray(r.world)?`XY ${r.world.map(x=>Number(x).toFixed(2)).join(',')}`:'XY —';m.textContent=`#${r.index} · f${r.frame_number} · t${Number(r.timestamp).toFixed(2)}s · L${r.local_track_id} · ${r.decision||'—'} · ${r.experimental_global_id?'SOL-E'+r.experimental_global_id:'no E-ID'} · sim ${sim} · conf ${Number(r.detector_confidence||0).toFixed(2)} · q ${Number(r.quality||0).toFixed(2)} · ${world}`;b.append(im,m);b.onclick=()=>{if(side==='left')left=r;else right=r;document.querySelectorAll('#'+side+' .crop').forEach(x=>x.classList.remove(side==='left'?'selected-left':'selected-right'));b.classList.add(side==='left'?'selected-left':'selected-right');msg(`Selected ${side} crop #${r.index}`)};root.append(b)}}
async function load(){bucket=Math.max(0,Number($('bucket').value)||0);clear();$('time').textContent=`${bucket}s`;let r=await fetch('/api/bucket/'+bucket);let d=await r.json();draw('left',d.left);draw('right',d.right);$('count').textContent=`Saved: ${d.annotation_count}`;msg(`Showing ${d.left.length} CH10 and ${d.right.length} CH11 crops`)}
async function save(same){if(!left||!right){msg('Select one crop from each camera first');return}return savePair(left,right,same)}
async function savePair(a,b,same){let r=await fetch('/api/annotations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({bucket,left_index:a.index,right_index:b.index,same_person:same})});let d=await r.json();if(!r.ok){msg(d.error||'save failed');return}$('count').textContent=`Saved: ${d.count}`;msg(`Saved ${same?'SAME':'DIFFERENT'}: CH10 #${a.index} ↔ CH11 #${b.index}`);if(mode==='matches')loadMatches();else{clear();load()}}
function drawMatches(rows){let root=$('matches');root.innerHTML='';if(!rows.length){root.textContent='No system-matched sets in this time bucket';return}for(const p of rows){let card=document.createElement('div');card.className='match';let title=document.createElement('div');title.className='meta';title.textContent=`SOL-E${p.global_id} · CH10 #${p.left.index} ↔ CH11 #${p.right.index} · similarity ${p.similarity==null?'—':Number(p.similarity).toFixed(3)}`;let imgs=document.createElement('div');imgs.className='match-images';for(const r of [p.left,p.right]){let im=document.createElement('img');im.src='/crop/'+r.index+'?t='+Date.now();imgs.append(im)}let actions=document.createElement('div');actions.innerHTML='<button class="same">SAME PERSON</button> <button class="different">DIFFERENT</button>';actions.children[0].onclick=()=>savePair(p.left,p.right,true);actions.children[1].onclick=()=>savePair(p.left,p.right,false);card.append(title,imgs,actions);root.append(card)}}
async function loadMatches(){bucket=Math.max(0,Number($('bucket').value)||0);$('time').textContent=`${bucket}s`;let r=await fetch('/api/matches/'+bucket);let d=await r.json();drawMatches(d.matches);$('count').textContent=`Saved: ${d.annotation_count}`;msg(`Showing ${d.matches.length} system-suggested matched sets`)}
function setMode(next){mode=next;$('matches-panel').style.display=mode==='matches'?'block':'none';document.querySelector('.grid').style.display=mode==='matches'?'none':'grid';$('mode').textContent=mode==='matches'?'ALL CROPS':'MATCHED SETS';if(mode==='matches')loadMatches();else load()}
 $('prev').onclick=()=>{ $('bucket').value=Math.max(0,bucket-step);mode==='matches'?loadMatches():load()};$('next').onclick=()=>{ $('bucket').value=bucket+step;mode==='matches'?loadMatches():load()};$('bucket').onchange=()=>mode==='matches'?loadMatches():load();$('mode').onclick=()=>setMode(mode==='matches'?'crops':'matches');$('same').onclick=()=>save(true);$('different').onclick=()=>save(false);fetch('/api/info').then(r=>r.json()).then(d=>{info=d;step=d.step;bucket=Math.floor(d.start_time/d.step);$('bucket').value=bucket;load()});
</script></body></html>'''


def load_manifest(dataset: Path, include_embeddings: bool = False) -> list[dict]:
    path = dataset / "manifest.jsonl"
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                try:
                    row = json.loads(line)
                    if not include_embeddings:
                        row.pop("embedding", None)
                    rows.append(row)
                except json.JSONDecodeError:
                    # The worker may be in the middle of appending the last
                    # line while the browser refreshes.
                    continue
    # Camera PTS values may have a fixed startup offset.  Keep the raw
    # timestamp, but use each camera's first observed PTS as the review clock
    # so a one-second bucket contains the corresponding live observations.
    starts = {}
    for row in rows:
        source = int(row["source_id"])
        # DeepStream's original timestamp is nanoseconds. The worker also
        # stores timestamp_seconds, which is the review-friendly normalized
        # value; keep the raw field in the manifest for diagnostics.
        review_timestamp = row.get("timestamp_seconds", row["timestamp"])
        review_timestamp = float(review_timestamp)
        # Some file-source timestamps are sub-second nanoseconds and are
        # therefore below 1e9; treat values above one million as ns.
        if review_timestamp > 1_000_000:
            review_timestamp /= 1_000_000_000.0
        row["review_timestamp"] = review_timestamp
        starts[source] = min(review_timestamp, starts.get(source, float("inf")))
    for row in rows:
        row["aligned_timestamp"] = row["review_timestamp"] - starts[int(row["source_id"])]
    return rows


def create_app(dataset: Path, step: float) -> Flask:
    rows = load_manifest(dataset, include_embeddings=False)
    by_index = {int(row["index"]): row for row in rows}
    annotations_path = dataset / "annotations.jsonl"
    app = Flask(__name__)

    def public(row):
        return {key: value for key, value in row.items() if key != "embedding"}

    @app.get("/")
    def index():
        return Response(HTML, mimetype="text/html")

    @app.get("/api/info")
    def info():
        maximum = max((float(r["aligned_timestamp"]) for r in rows), default=0.0)
        count = sum(1 for line in annotations_path.read_text(encoding="utf-8").splitlines() if line.strip()) if annotations_path.exists() else 0
        minimum = min((float(r["aligned_timestamp"]) for r in rows), default=0.0)
        return jsonify(duration=maximum, start_time=minimum, step=step, count=count, crops=len(rows))

    @app.get("/api/bucket/<int:bucket>")
    def bucket(bucket: int):
        start, end = bucket * step, (bucket + 1) * step
        selected = [r for r in rows if start <= float(r["aligned_timestamp"]) < end]
        left = [public(r) for r in selected if int(r["source_id"]) == 0]
        right = [public(r) for r in selected if int(r["source_id"]) == 1]
        count = sum(1 for line in annotations_path.read_text(encoding="utf-8").splitlines() if line.strip()) if annotations_path.exists() else 0
        return jsonify(left=left, right=right, annotation_count=count)

    @app.get("/api/matches/<int:bucket>")
    def matches(bucket: int):
        """Return one CH10/CH11 pair for every system Global ID in a bucket."""
        start, end = bucket * step, (bucket + 1) * step
        groups = {}
        for row in rows:
            identity = row.get("experimental_global_id")
            if identity is None or not (start <= float(row["aligned_timestamp"]) < end):
                continue
            group = groups.setdefault(str(identity), {0: [], 1: []})
            source = int(row["source_id"])
            if source in group:
                group[source].append(row)
        output = []
        for identity, group in groups.items():
            if not group[0] or not group[1]:
                continue
            # Prefer the clearest crop from each camera while keeping the
            # system's assigned identity and metadata visible for review.
            choose = lambda values: max(values, key=lambda item: (
                float(item.get("quality", 0.0)), int(item.get("crop_height", 0))))
            left = choose(group[0])
            right = choose(group[1])
            output.append({"global_id": int(identity), "left": public(left),
                           "right": public(right),
                           "similarity": left.get("similarity") or right.get("similarity")})
        output.sort(key=lambda item: item["global_id"])
        count = sum(1 for line in annotations_path.read_text(encoding="utf-8").splitlines() if line.strip()) if annotations_path.exists() else 0
        return jsonify(matches=output, annotation_count=count)

    @app.get("/crop/<int:index>")
    def crop(index: int):
        row = by_index.get(index)
        if row is None:
            return jsonify(error="unknown crop"), 404
        path = (dataset / row["path"]).resolve()
        if dataset not in path.parents or not path.is_file():
            return jsonify(error="crop is outside dataset"), 404
        return send_file(path, mimetype="image/jpeg", max_age=0)

    @app.post("/api/annotations")
    def annotate():
        data = request.get_json(silent=True) or {}
        try:
            left = by_index[int(data["left_index"])]
            right = by_index[int(data["right_index"])]
            if int(left["source_id"]) != 0 or int(right["source_id"]) != 1:
                raise ValueError("select CH10 on the left and CH11 on the right")
            row = {"reference_id": f"manual_{sum(1 for _ in annotations_path.open()) + 1:05d}" if annotations_path.exists() else "manual_00001",
                   "bucket": int(data["bucket"]), "left_index": left["index"], "right_index": right["index"],
                   "same_person": bool(data["same_person"])}
            with annotations_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            count = sum(1 for line in annotations_path.open(encoding="utf-8") if line.strip())
            return jsonify(count=count)
        except (KeyError, TypeError, ValueError) as error:
            return jsonify(error=str(error)), 400

    return app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--step", type=float, default=1.0)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    if not (dataset / "manifest.jsonl").is_file():
        parser.error(f"missing {dataset / 'manifest.jsonl'}")
    create_app(dataset, args.step).run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
