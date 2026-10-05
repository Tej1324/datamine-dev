"""Six-camera staff enrollment using live NvDCF/NVIDIA Re-ID metadata.

The operator labels only positive staff boxes.  Local tracker IDs are used
internally to pair the click with the embedding but are never part of the
operator workflow.  Customers are not a training class.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path

import cv2
import yaml
from flask import Flask, jsonify, render_template_string, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/staff_filter"
FRAMES = DATA / "frames"
CANDIDATES = DATA / "reid_gallery_candidates.jsonl"
PENDING = DATA / "reid_gallery_pending.jsonl"
GALLERY = DATA / "reid_gallery.tsv"
ENROLLMENT_FILE = ROOT / "runs/staff_enrollment_live.json"
CAMERAS = ROOT / "config/cameras.yaml"
ENV = ROOT / ".env"
SYNC_MAX_DELTA_SECONDS = float(os.environ.get("STAFF_ENROLLMENT_SYNC_MAX_DELTA", "2"))
MIN_CONFIDENCE = float(os.environ.get("STAFF_ENROLLMENT_MIN_CONFIDENCE", "0.20"))
MIN_HEIGHT = float(os.environ.get("STAFF_ENROLLMENT_MIN_HEIGHT", "24"))
PENDING_MAX_FRAME_GAP = int(os.environ.get("STAFF_ENROLLMENT_PENDING_MAX_FRAME_GAP", "300"))
for directory in (FRAMES, DATA):
    directory.mkdir(parents=True, exist_ok=True)
PENDING_LOCK = threading.Lock()


def env_values():
    values = {}
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def cameras():
    return (yaml.safe_load(CAMERAS.read_text()) or {}).get("cameras", [])


def source_url(camera):
    values = env_values()
    user = values["CAMERA_USERNAME"].replace("@", "%40")
    password = values["CAMERA_PASSWORD"].replace("@", "%40")
    return (f"rtsp://{user}:{password}@{values['NVR_225_HOST']}:{values['NVR_225_PORT']}"
            f"/cam/realmonitor?channel={int(camera['channel'])}&subtype={int(camera['subtype'])}")


def load_live_snapshot():
    try:
        value = json.loads(ENROLLMENT_FILE.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("observations"), list):
        return None
    return value


def source_observations(snapshot, source_id):
    if not snapshot:
        return []
    result = []
    for item in snapshot.get("observations", []):
        if int(item.get("source_id", -1)) != source_id:
            continue
        bbox = item.get("bbox")
        embedding = item.get("embedding")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            values = [float(value) for value in bbox]
            vector = [float(value) for value in embedding] if isinstance(embedding, list) else []
            confidence = float(item.get("detector_confidence", 0.0))
            tracker_confidence = float(item.get("tracker_confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if values[2] <= 0 or values[3] <= 0:
            continue
        embedding_available = len(vector) == 256 and all(math.isfinite(value) for value in vector)
        norm = math.sqrt(sum(value * value for value in vector)) if embedding_available else 0.0
        embedding_available = embedding_available and norm > 0.0
        scale = min(values[2] / 48.0, values[3] / 128.0)
        quality = max(0.0, min(1.0, 0.4 * confidence + 0.4 * tracker_confidence +
                                0.1 + 0.1 * min(1.0, scale)))
        result.append({
            "bbox": values,
            "confidence": confidence,
            "tracker_confidence": tracker_confidence,
            "quality": quality,
            "view_bucket": int(source_id) * 10 + (0 if values[3] < 110 else 1 if values[3] < 220 else 2),
            "local_object_id": int(item.get("local_object_id", 0)),
            "frame_number": int(item.get("frame_number", 0)),
            "timestamp": int(item.get("timestamp", 0)),
            "embedding_available": embedding_available,
            "embedding": [value / norm for value in vector] if embedding_available else [],
        })
    return result


def capture_one(camera, stamp):
    camera_id = camera["id"]
    path = FRAMES / f"{camera_id}_{stamp}.jpg"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp",
               "-i", source_url(camera), "-frames:v", "1", "-q:v", "2", "-y", str(path)]
    try:
        subprocess.run(command, check=True, timeout=20, capture_output=True)
    except (OSError, subprocess.SubprocessError):
        return None
    image = cv2.imread(str(path))
    snapshot = load_live_snapshot()
    if image is None or snapshot is None:
        return None
    source_id = cameras().index(camera)
    sidecar = {
        "updated_at": float(snapshot.get("updated_at", 0.0)),
        "source_id": source_id,
        "observations": source_observations(snapshot, source_id),
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    return {"camera": camera_id, "path": str(path.relative_to(ROOT)), "width": int(image.shape[1]),
            "height": int(image.shape[0]), "updated_at": sidecar["updated_at"]}


def load_frame(path):
    try:
        return json.loads((ROOT / path).with_suffix(".json").read_text())
    except (OSError, ValueError):
        return None


def public_observations(snapshot):
    result = []
    for index, item in enumerate((snapshot or {}).get("observations", [])):
        result.append({"detector_index": index, "bbox": item["bbox"],
                       "confidence": item["confidence"], "tracker_confidence": item["tracker_confidence"],
                       "quality": item["quality"],
                       "embedding_available": bool(item.get("embedding_available"))})
    return result


def append_candidate(output, camera, source_id, item):
    output.write(json.dumps({"camera": camera, "source_id": source_id,
                             "local_object_id": item["local_object_id"],
                             "frame_number": item["frame_number"], "quality": item["quality"],
                             "view_bucket": item["view_bucket"], "embedding": item["embedding"]},
                        separators=(",", ":")) + "\n")


def resolve_pending(camera, sidecar):
    """Attach a later embedding to a box previously labelled as staff."""
    if not PENDING.exists():
        return 0
    with PENDING_LOCK:
        pending = []
        for line in PENDING.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                continue
            pending.append(item)
        observations = sidecar.get("observations", [])
        current_frame = max((int(observation.get("frame_number", 0)) for observation in observations), default=0)
        resolved = 0
        remaining = []
        with CANDIDATES.open("a", encoding="utf-8") as candidates:
            for item in pending:
                pending_frame = int(item.get("frame_number", 0))
                frame_is_current = (current_frame >= pending_frame and
                                    current_frame - pending_frame <= PENDING_MAX_FRAME_GAP)
                match = next((observation for observation in observations
                              if int(observation.get("local_object_id", -1)) == int(item.get("local_object_id", -2))
                              and int(observation.get("local_object_id", -1)) != 18446744073709551615
                              and observation.get("embedding_available")), None)
                if item.get("camera") == camera and frame_is_current and match is not None:
                    append_candidate(candidates, camera, sidecar["source_id"], match)
                    resolved += 1
                else:
                    remaining.append(item)
        temporary = PENDING.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as output:
            for item in remaining:
                output.write(json.dumps(item, separators=(",", ":")) + "\n")
        os.replace(temporary, PENDING)
        if resolved:
            rebuild_gallery()
        return resolved


def latest_frames():
    result = {}
    now = time.time()
    for camera in cameras():
        paths = sorted(FRAMES.glob(f"{camera['id']}_*.jpg"))
        if not paths:
            continue
        path = paths[-1]
        image = cv2.imread(str(path))
        sidecar = load_frame(str(path.relative_to(ROOT)))
        if image is None or sidecar is None:
            continue
        frame_age = max(0.0, now - path.stat().st_mtime)
        updated_at = float(sidecar.get("updated_at", 0.0))
        delta = abs(path.stat().st_mtime - updated_at) if updated_at else None
        result[camera["id"]] = {
            "camera": camera["id"], "path": str(path.relative_to(ROOT)),
            "url": "/files/" + path.name, "width": int(image.shape[1]), "height": int(image.shape[0]),
            "frame_age_seconds": frame_age, "synchronized": bool(delta is not None and delta <= SYNC_MAX_DELTA_SECONDS),
            "synchronization_delta_seconds": delta, "live_snapshot": "observations" in sidecar,
            "observations": public_observations(sidecar),
        }
    return result


def cosine(left, right):
    return sum(a * b for a, b in zip(left, right))


def load_candidates():
    if not CANDIDATES.exists():
        return []
    values = []
    for line in CANDIDATES.read_text().splitlines():
        try:
            item = json.loads(line)
            if len(item.get("embedding", [])) == 256:
                values.append(item)
        except (ValueError, TypeError):
            continue
    return values


def rebuild_gallery():
    candidates = load_candidates()
    candidates.sort(key=lambda item: (float(item.get("quality", 0.0)), int(item.get("frame_number", 0))), reverse=True)
    chosen = []
    buckets = set()
    for item in candidates:
        bucket = (item.get("camera"), item.get("view_bucket"))
        if bucket in buckets and len(chosen) < 96 and len(chosen) < 24:
            continue
        if any(cosine(item["embedding"], previous["embedding"]) >= 0.995 for previous in chosen):
            continue
        chosen.append(item)
        buckets.add(bucket)
        if len(chosen) >= 96:
            break
    if not chosen:
        return 0
    temporary = GALLERY.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write("# pooled positive staff gallery: label source_id view_bucket quality vector[256]\n")
        for index, item in enumerate(chosen, 1):
            vector = " ".join(f"{float(value):.8g}" for value in item["embedding"])
            output.write(f"staff_{index:04d} {int(item['source_id'])} {int(item['view_bucket'])} "
                         f"{float(item['quality']):.6f} {vector}\n")
    os.replace(temporary, GALLERY)
    return len(chosen)


HTML = r'''<!doctype html><meta charset="utf-8"><title>Staff enrollment</title>
<style>
body{font:16px system-ui;background:#0f1722;color:#eef2f7;margin:0}.wrap{max-width:1100px;margin:auto;padding:24px}
header{display:flex;justify-content:space-between;align-items:center;gap:16px}.muted{color:#9eb1c9;font-size:13px}
.progress{height:8px;background:#263449;border-radius:8px;overflow:hidden;margin:18px 0}.bar{height:100%;background:#37c878;width:0;transition:width .2s}
.card{background:#192638;border-radius:12px;padding:18px;box-shadow:0 8px 30px #0004}.stage{position:relative;max-width:100%;background:#000}
.stage img{width:100%;display:block}.stage canvas{position:absolute;inset:0;width:100%;height:100%;cursor:pointer}
.actions{display:flex;align-items:center;gap:10px;margin-top:16px;flex-wrap:wrap}button{border:0;border-radius:7px;padding:11px 18px;font-weight:700;color:#fff;cursor:pointer}
button.primary{background:#198754}button.secondary{background:#315b91}button.skip{background:#596579}button:disabled{opacity:.45;cursor:not-allowed}
#status{white-space:pre-wrap;background:#111b29;border-radius:7px;padding:12px;margin-top:16px}.empty{text-align:center;padding:60px;color:#9eb1c9}
</style><div class="wrap"><header><div><h1>Staff enrollment</h1><div class="muted">Click every staff box. Customers are ignored. The system collects NVIDIA Re-ID embeddings automatically.</div></div><div id="step" class="muted"></div></header>
<div class="progress"><div id="bar" class="bar"></div></div><main id="main"></main><div id="status">Preparing synchronized camera frames…</div></div>
<script>
let frames={},order=[],current=0,selected=new Set(),saving=false,capturing=false;
function status(v){document.querySelector('#status').textContent=v}
function name(p){return p.split('/').pop()}
function valid(f){return(f.observations||[]).filter(d=>d.confidence>=.20&&d.bbox[3]>=24)}
async function load(){let r=await fetch('/api/frames',{cache:'no-store'});if(!r.ok)throw new Error('Unable to load current frames');frames=await r.json();order=Object.keys(frames).sort();current=Math.min(current,Math.max(0,order.length-1));render()}
function render(){let main=document.querySelector('#main');if(!order.length){main.innerHTML='<div class="empty">No captured camera frames. Press CAPTURE SIX CAMERAS.</div>';return}let id=order[current],f=frames[id],ds=valid(f),usable=ds.filter(d=>d.embedding_available).length,fresh=!!f.synchronized;selected=new Set();document.querySelector('#step').textContent='Camera '+(current+1)+' / '+order.length+' · '+id;document.querySelector('#bar').style.width=((current+1)/order.length*100)+'%';main.innerHTML='<section class="card"><div class="muted">'+f.width+'×'+f.height+' · synchronization delta '+(f.synchronization_delta_seconds==null?'unknown':f.synchronization_delta_seconds.toFixed(2)+'s')+' · '+ds.length+' tracked boxes · '+usable+' usable Re-ID embeddings</div><div class="stage"><img id="image" src="/files/'+name(f.path)+'?t='+Date.now()+'"><canvas id="canvas"></canvas></div><div class="actions"><button class="primary" id="save" disabled>SAVE STAFF → NEXT</button><button class="secondary" id="nostaff">NO STAFF → NEXT</button><button class="skip" id="skip">SKIP</button><button class="secondary" id="capture">CAPTURE SIX CAMERAS</button></div></section>';let image=document.querySelector('#image'),canvas=document.querySelector('#canvas');image.onload=()=>{canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;draw(ds)};canvas.onclick=e=>{let r=canvas.getBoundingClientRect(),p={x:(e.clientX-r.left)*canvas.width/r.width,y:(e.clientY-r.top)*canvas.height/r.height},hit=null;for(let i=ds.length-1;i>=0;i--){let b=ds[i].bbox;if(p.x>=b[0]&&p.x<=b[0]+b[2]&&p.y>=b[1]&&p.y<=b[1]+b[3]){hit=i;break}}if(hit!==null){selected.has(hit)?selected.delete(hit):selected.add(hit);draw(ds);document.querySelector('#save').disabled=!fresh||!selected.size;let pending=[...selected].filter(index=>!ds[index].embedding_available).length;status(pending?pending+' selected staff box(es) are pending an embedding; they will be enrolled when this track produces one.':selected.size+' staff selected. Customers remain unselected.')}};document.querySelector('#save').onclick=()=>save(false);document.querySelector('#nostaff').onclick=()=>save(true);document.querySelector('#skip').onclick=()=>next();document.querySelector('#capture').onclick=capture;if(!fresh)status('Frames are not synchronized with the live tracker snapshot. Capture a new six-camera set.');else status('Click any tracked box to mark it as staff. Missing embeddings are enrolled later when available.');draw(ds)}
function draw(ds){let c=document.querySelector('#canvas');if(!c)return;let x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);ds.forEach((d,i)=>{let b=d.bbox,s=selected.has(i);x.strokeStyle=s?'#00ff70':d.embedding_available?'#ffd166':'#94a3b8';x.lineWidth=s?6:3;x.strokeRect(b[0],b[1],b[2],b[3]);x.fillStyle=x.strokeStyle;x.font='20px sans-serif';let label=s?'STAFF ':'#'+(i+1)+' ';label+=d.embedding_available?'RE-ID ':'WAIT ';x.fillText(label+d.quality.toFixed(2),b[0]+3,Math.max(20,b[1]+20))})}
async function save(noStaff){if(saving)return;let id=order[current],f=frames[id];saving=true;let r=await fetch('/api/enroll',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({camera:id,frame:f.path,indices:[...selected],no_staff:noStaff})}),b=await r.json();saving=false;if(!r.ok){status(JSON.stringify(b,null,2));return}status(noStaff?'No staff recorded. Opening next camera…':'Saved '+b.saved+' embeddings and queued '+b.pending+' pending staff boxes. Gallery now has '+b.gallery_entries+' views. Opening next camera…');setTimeout(next,250)}
function next(){if(current+1<order.length){current++;render()}else{status('Six-camera set complete. Capturing the next set…');capture()}}async function capture(){if(capturing)return;capturing=true;status('Capturing synchronized frames from all six cameras…');let r=await fetch('/api/capture',{method:'POST'}),b=await r.json();capturing=false;if(!r.ok){status(JSON.stringify(b,null,2));return}current=0;await load();status('New six-camera set ready. Select staff boxes marked RE-ID.')}async function init(){try{await load();if(!order.length||!Object.values(frames).some(f=>f.live_snapshot))await capture()}catch(e){status(e.message)}}init();
</script>'''

app = Flask(__name__)


@app.get("/")
def index():
    return render_template_string(HTML)


@app.get("/files/<path:name>")
def files(name):
    return send_from_directory(FRAMES, name)


@app.get("/api/frames")
def frames():
    return jsonify(latest_frames())


@app.post("/api/capture")
def capture():
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    camera_list = cameras()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        values = list(pool.map(lambda camera: capture_one(camera, stamp), camera_list))
    resolved_pending = 0
    for item in values:
        if item:
            sidecar = load_frame(item["path"])
            if sidecar:
                resolved_pending += resolve_pending(item["camera"], sidecar)
    return jsonify({"captured": [item for item in values if item],
                    "failed": [camera["id"] for camera, item in zip(camera_list, values) if item is None],
                    "resolved_pending": resolved_pending})


@app.route("/api/enroll", methods=["POST", "OPTIONS"])
def enroll():
    if request.method == "OPTIONS":
        return ("", 204, {"Access-Control-Allow-Origin": "*",
                           "Access-Control-Allow-Methods": "POST, OPTIONS",
                           "Access-Control-Allow-Headers": "Content-Type"})
    data = request.get_json(force=True)
    frame = str(data.get("frame", ""))
    camera = str(data.get("camera", ""))
    selected = {int(value) for value in data.get("indices", [])}
    sidecar = load_frame(frame)
    current = latest_frames().get(camera)
    if sidecar is None or current is None or current["path"] != frame:
        return jsonify(error="capture is no longer current; capture a new six-camera set"), 409
    if not current["synchronized"]:
        return jsonify(error="image and tracker snapshot are not synchronized",
                       synchronization_delta_seconds=current["synchronization_delta_seconds"]), 409
    observations = sidecar.get("observations", [])
    chosen = [item for index, item in enumerate(observations) if index in selected]
    if not chosen and not data.get("no_staff"):
        return jsonify(error="select one or more staff boxes, or choose NO STAFF"), 400
    if any(float(item.get("confidence", 0.0)) < MIN_CONFIDENCE or
           float(item.get("bbox", [0, 0, 0, 0])[3]) < MIN_HEIGHT for item in chosen):
        return jsonify(error="selected staff box is too small or low-confidence"), 400
    immediate = [item for item in chosen if item.get("embedding_available")]
    waiting = [item for item in chosen if not item.get("embedding_available")]
    with PENDING_LOCK:
        with CANDIDATES.open("a", encoding="utf-8") as output:
            for item in immediate:
                append_candidate(output, camera, sidecar["source_id"], item)
        with PENDING.open("a", encoding="utf-8") as output:
            for item in waiting:
                output.write(json.dumps({"camera": camera, "source_id": sidecar["source_id"],
                                         "local_object_id": item["local_object_id"],
                                         "frame_number": item["frame_number"], "bbox": item["bbox"]},
                                        separators=(",", ":")) + "\n")
        gallery_entries = rebuild_gallery()
    return jsonify(saved=len(immediate), pending=len(waiting), gallery_entries=gallery_entries)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("STAFF_LABELER_PORT", "8780")), threaded=True)
