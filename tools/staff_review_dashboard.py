"""Offline TAO staff/customer review dashboard for a bounded evaluation run."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template_string, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("STAFF_REVIEW_RUN", str(ROOT / "runs/staff_review_eval")))
CROPS = RUN / "crops"
MANIFEST = RUN / "review_manifest.jsonl"
CORRECTIONS = RUN / "corrections.jsonl"
MODEL = Path(os.environ.get(
    "STAFF_REVIEW_MODEL",
    str(ROOT / "runs/tao_staff_classifier_v1/staff_customer_fan_tiny_dynamic.onnx"),
))
LOCK = threading.Lock()


def classify_image(session, path: Path) -> tuple[str, float]:
    image = cv2.imread(str(path))
    if image is None:
        return "customer", 0.0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    image = (image - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
    image = np.transpose(image, (2, 0, 1))[None, ...]
    # TAO's binary export emits a signed margin: staff is the negative side,
    # customer is the positive side.
    margin = float(session.run(None, {session.get_inputs()[0].name: image})[0].reshape(-1)[0])
    staff_probability = 1.0 / (1.0 + float(np.exp(np.clip(margin, -60.0, 60.0))))
    return ("staff" if margin <= 0.0 else "customer"), staff_probability


def classify_batch(session, paths: list[Path]) -> list[tuple[str, float]]:
    images = []
    valid = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        image = (image - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
        images.append(np.transpose(image, (2, 0, 1)))
        valid.append(path)
    if not images:
        return []
    margins = session.run(None, {session.get_inputs()[0].name: np.stack(images)})[0].reshape(-1)
    result = []
    for margin in margins:
        margin = float(margin)
        staff_probability = 1.0 / (1.0 + float(np.exp(np.clip(margin, -60.0, 60.0))))
        result.append(("staff" if margin <= 0.0 else "customer", staff_probability))
    return result


def build_manifest() -> None:
    if MANIFEST.exists():
        return
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    rows = []
    paths = sorted(CROPS.rglob("*.jpg")) + sorted(CROPS.rglob("*.png"))
    for offset in range(0, len(paths), 64):
        batch_paths = paths[offset:offset + 64]
        predictions = classify_batch(session, batch_paths)
        for path, (label, staff_probability) in zip(batch_paths, predictions):
            rows.append({
                "id": len(rows), "path": str(path.relative_to(RUN)),
                "camera": path.parent.name, "label": label,
                "model_label": label, "staff_probability": round(staff_probability, 6),
            })
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def load_rows() -> list[dict]:
    build_manifest()
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    for row in rows:
        match = re.search(r"^crops/([^/]+)/.*_track(\d+)\.", row.get("path", ""))
        row["group"] = f"{match.group(1)}:track{match.group(2)}" if match else f"crop:{row['id']}"
    return rows


HTML = r'''<!doctype html><meta charset="utf-8"><title>TAO Staff Review</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#101827;color:#edf3fb;font:14px system-ui,sans-serif}
.wrap{max-width:1500px;margin:auto;padding:20px}.top{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:14px}
h1{font-size:22px;margin:0 0 4px}.muted{color:#9eafc4}.stats{background:#1d2a3e;border-radius:8px;padding:10px 14px;font-weight:700}
.note{background:#18263a;padding:10px 14px;border-radius:8px;margin-bottom:14px;color:#c5d2e2}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{background:#1b293d;border-radius:10px;padding:12px;min-height:300px}.panel h2{margin:0 0 10px;font-size:17px}
.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}.card{background:#111b2b;border:2px solid transparent;border-radius:7px;padding:5px;cursor:pointer}.card.selected{border-color:#ffd166}.card img{display:block;width:100%;height:190px;object-fit:contain;background:#000}.meta{font-size:11px;color:#c3d2e5;margin-top:4px;line-height:1.3}.empty{color:#9eafc4;padding:30px;text-align:center}
.toolbar{position:sticky;top:0;z-index:5;background:#101827;padding:10px 0;display:flex;gap:8px;align-items:center}.toolbar button{background:#596579}.toolbar button.danger{background:#a72b4b}
button{border:0;border-radius:6px;padding:9px 13px;font-weight:700;color:#fff;background:#315b91;cursor:pointer}button:hover{filter:brightness(1.15)}button:disabled{opacity:.45}.staff{background:#a72b4b}.customer{background:#21865a}
@media(max-width:900px){.columns{grid-template-columns:1fr}.grid{grid-template-columns:repeat(3,minmax(0,1fr))}.card img{height:150px}}
</style><div class="wrap"><div class="top"><div><h1>TAO staff/customer evaluation</h1><div class="muted">Review the model’s five-camera result. Click a crop, then move it between columns if it is wrong.</div></div><div class="stats" id="stats">Loading…</div></div>
<div class="note">Corrections are saved separately from the trained model. After review, corrected staff crops can be added to the next TAO training dataset.</div><div class="toolbar"><button class="danger" onclick="resetSelection()">Reset selection</button><button id="undoTop" onclick="undo()" disabled>Undo</button><span class="muted">These controls apply to the current selection.</span></div>
<div class="columns"><section class="panel"><h2>Customers <button class="customer" onclick="move('customer')">Move selected here</button></h2><div id="customers" class="grid"></div></section><section class="panel"><h2>Staff / filtered <button class="staff" onclick="move('staff')">Move selected here</button></h2><div id="staff" class="grid"></div></section></div><div id="pager" class="note"></div></div>
<script>
let rows=[],selected=new Set(),history=[],busy=false;
async function load(){document.getElementById('stats').textContent='Loading all crops…';let r=await fetch('/api/rows',{cache:'no-store'});rows=await r.json();render()}
function render(){for(const [elementId,label] of [['customers','customer'],['staff','staff']]){let el=document.getElementById(elementId),list=rows.filter(x=>x.label===label);el.innerHTML=list.length?list.map(x=>`<div class="card ${selected.has(x.id)?'selected':''}" onclick="pick(${x.id})"><img loading="lazy" src="/files/${x.path}"><div class="meta">${x.camera} · ${x.label.toUpperCase()} · ${x.group}<br>staff score: ${(x.staff_probability*100).toFixed(1)}%</div></div>`).join(''):'<div class="empty">No crops</div>'}let c=rows.filter(x=>x.label==='customer').length,s=rows.filter(x=>x.label==='staff').length;document.getElementById('stats').textContent=`${rows.length} crops · ${c} customers · ${s} staff · ${selected.size} selected`;document.getElementById('undoTop').disabled=!history.length;document.getElementById('pager').textContent='Click a crop to select similar Re-ID embeddings (cosine ≥ 0.86). Click a selected crop again to deselect only that crop.'}
function checkpoint(){history.push(new Set(selected));if(history.length>20)history.shift()}
function cosine(a,b){if(!a||!b||a.length!==b.length)return -1;let dot=0,na=0,nb=0;for(let i=0;i<a.length;i++){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i]}return na&&nb?dot/Math.sqrt(na*nb):-1}
function pick(id){let item=rows.find(x=>x.id===id);if(!item)return;checkpoint();if(selected.has(id))selected.delete(id);else{let similar=rows.filter(x=>cosine(item.embedding,x.embedding)>=0.86);if(similar.length>1)similar.forEach(x=>selected.add(x.id));else rows.filter(x=>x.group===item.group).forEach(x=>selected.add(x.id))}render()}
function resetSelection(){if(!selected.size)return;checkpoint();selected.clear();render()}
function undo(){if(!history.length)return;selected=history.pop();render()}
async function move(label){if(busy||!selected.size)return;busy=true;let r=await fetch('/api/move',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids:[...selected],label})});if(!r.ok){alert(await r.text());busy=false;return}selected.clear();history=[];busy=false;await load()}
load();
</script>'''

app = Flask(__name__)


@app.get("/")
def index():
    return render_template_string(HTML)


@app.get("/files/<path:name>")
def files(name):
    return send_from_directory(RUN, name)


@app.get("/api/rows")
def rows():
    return jsonify(load_rows())


@app.get("/api/row")
def row():
    values = load_rows()
    try:
        index = max(0, int(request.args.get("index", "0")))
    except ValueError:
        index = 0
    if index >= len(values):
        return jsonify(total=len(values), row=None), 404
    return jsonify(total=len(values), row=values[index])


@app.post("/api/move")
def move():
    data = request.get_json(force=True)
    ids = {int(value) for value in data.get("ids", [])}
    label = str(data.get("label", ""))
    if label not in {"staff", "customer"}:
        return jsonify(error="label must be staff or customer"), 400
    with LOCK:
        values = load_rows()
        changed = []
        for row in values:
            if int(row["id"]) in ids and row["label"] != label:
                row["label"] = label
                changed.append({"id": row["id"], "path": row["path"], "label": label})
        MANIFEST.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in values), encoding="utf-8")
        with CORRECTIONS.open("a", encoding="utf-8") as output:
            for row in changed:
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
    return jsonify(changed=len(changed))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("STAFF_REVIEW_PORT", "18083")), threaded=True)
