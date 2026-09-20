"""Small manual staff-uniform labeling UI for production-geometry frames."""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from flask import Flask, jsonify, render_template_string, request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/staff_filter"
FRAMES = DATA / "frames"
CROP_DIRS = {"staff": DATA / "staff", "customer": DATA / "customer"}
LABELS = DATA / "labels.yaml"
PROFILE = DATA / "profile.yml"
CAMERAS = ROOT / "config/cameras.yaml"
ENV = ROOT / ".env"
for directory in (FRAMES, *CROP_DIRS.values()):
    directory.mkdir(parents=True, exist_ok=True)


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


def capture_one(camera, stamp):
    camera_id = camera["id"]
    path = FRAMES / f"{camera_id}_{stamp}.jpg"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp",
        "-i", source_url(camera), "-frames:v", "1", "-q:v", "2", "-y", str(path),
    ]
    try:
        subprocess.run(command, check=True, timeout=20, capture_output=True)
    except (OSError, subprocess.SubprocessError):
        return None
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return {"camera": camera_id, "path": str(path.relative_to(ROOT)), "width": width, "height": height, "captured_at": stamp}


def latest_frames():
    result = {}
    for camera in cameras():
        paths = sorted(FRAMES.glob(f"{camera['id']}_*.jpg"))
        if paths:
            path = paths[-1]
            image = cv2.imread(str(path))
            if image is not None:
                available = []
                for candidate in paths:
                    available.append({
                        "path": str(candidate.relative_to(ROOT)),
                        "name": candidate.stem,
                    })
                result[camera["id"]] = {
                    "camera": camera["id"],
                    "path": str(path.relative_to(ROOT)),
                    "url": "/files/" + str(path.relative_to(DATA)).replace(os.sep, "/"),
                    "width": int(image.shape[1]), "height": int(image.shape[0]),
                    "available": available,
                }
    return result


def load_labels():
    if not LABELS.exists():
        return []
    return yaml.safe_load(LABELS.read_text()) or []


def save_labels(items):
    LABELS.write_text(yaml.safe_dump(items, sort_keys=False), encoding="utf-8")


def rgb_features(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    delta = mx - mn
    h = np.zeros_like(mx)
    mask = delta > 1e-6
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h[mask & (mx == r)] = (60 * ((g - b) / np.maximum(delta, 1e-6)) % 360)[mask & (mx == r)]
    h[mask & (mx == g)] = (60 * ((b - r) / np.maximum(delta, 1e-6)) + 120)[mask & (mx == g)]
    h[mask & (mx == b)] = (60 * ((r - g) / np.maximum(delta, 1e-6)) + 240)[mask & (mx == b)]
    hsv = np.stack((h / 2, np.where(mx > 1e-6, delta / np.maximum(mx, 1e-6) * 255, 0), mx * 255), axis=2)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    X = (linear[..., 0] * .4124564 + linear[..., 1] * .3575761 + linear[..., 2] * .1804375) / .95047
    Y = linear[..., 0] * .2126729 + linear[..., 1] * .7151522 + linear[..., 2] * .0721750
    Z = (linear[..., 0] * .0193339 + linear[..., 1] * .1191920 + linear[..., 2] * .9503041) / 1.08883
    def lab_f(value):
        return np.where(value > .008856, np.cbrt(value), 7.787 * value + 16 / 116)
    fx, fy, fz = lab_f(X), lab_f(Y), lab_f(Z)
    lab = np.stack((np.clip((116 * fy - 16) * 2.55, 0, 255),
                    np.clip(500 * (fx - fy) + 128, 0, 255),
                    np.clip(200 * (fy - fz) + 128, 0, 255)), axis=2)
    return hsv, lab


def derive_profile():
    labels = load_labels()
    hsv_values, lab_values = [], []
    for item in labels:
        if item.get("class", "staff") != "staff":
            continue
        image = cv2.imread(str(ROOT / item["crop"]))
        if image is None or image.size == 0:
            continue
        h, w = image.shape[:2]
        upper = image[int(.10 * h):max(int(.11 * h), int(.65 * h)), int(.15 * w):max(int(.16 * w), int(.85 * w))]
        hsv, lab = rgb_features(upper[::2, ::2])
        valid = (hsv[..., 1] > 28) & (hsv[..., 2] > 25)
        if valid.any():
            hsv_values.append(hsv[valid])
            lab_values.append(lab[valid])
    if not hsv_values:
        return None
    hsv_all = np.concatenate(hsv_values)
    lab_all = np.concatenate(lab_values)
    lower_hsv = np.percentile(hsv_all, 2, axis=0).round(3).tolist()
    upper_hsv = np.percentile(hsv_all, 98, axis=0).round(3).tolist()
    lower_lab = np.percentile(lab_all, 2, axis=0).round(3).tolist()
    upper_lab = np.percentile(lab_all, 98, axis=0).round(3).tolist()
    profile = {
        "enabled": True,
        "source_crop_count": len(labels),
        "sample_count": int(len(hsv_all)),
        "hsv": {"lower": lower_hsv, "upper": upper_hsv},
        "lab": {"lower": lower_lab, "upper": upper_lab},
        "generated_at": time.time(),
    }
    PROFILE.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return profile


HTML = r'''<!doctype html><meta charset="utf-8"><title>Staff classifier labeling</title>
<style>body{font:15px system-ui;background:#101722;color:#eee;margin:20px}button{padding:8px 14px;margin:4px;background:#1d6fbd;color:white;border:0;border-radius:4px}.grid{display:grid;grid-template-columns:repeat(2,minmax(420px,1fr));gap:14px}.card{background:#1b2738;padding:10px;border-radius:6px}.stage{position:relative}.stage img{width:100%;display:block}.stage canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}.small{color:#9eb1c9;font-size:12px}#status{white-space:pre-wrap;background:#192536;padding:10px}</style>
<h1>Staff/customer classifier labeling</h1><p>Capture current production-geometry frames, drag every full person bbox, then choose STAFF or CUSTOMER when saving. Include difficult look-alikes as customer examples.</p>
<button onclick="capture()">CAPTURE 6 CURRENT FRAMES</button><button onclick="profile()">DERIVE LEGACY COLOR PROFILE</button><span id="status">Ready.</span><div id="grid" class="grid"></div>
<script>
let frames={}, boxes={}, classes={};
async function load(){let r=await fetch('/api/frames');frames=await r.json();let g=document.querySelector('#grid');g.innerHTML='';for(const id of Object.keys(frames)){let f=frames[id];boxes[id]=[];let options=f.available.map(v=>'<option value="'+v.path+'">'+v.name+'</option>').join('');let c=document.createElement('div');c.className='card';c.innerHTML='<h2>'+id+'</h2><div class="small">'+f.width+'×'+f.height+' · choose a frame, then drag every staff bbox</div><select id="sel-'+id+'" onchange="selectFrame(\''+id+'\',this.value)">'+options+'</select><div class="stage"><img id="im-'+id+'" src="/files/'+f.path.split('/').slice(-1)[0]+'?t='+Date.now()+'"><canvas id="cv-'+id+'"></canvas></div><button onclick="saveBox(\''+id+'\')">SAVE BOX</button><button onclick="boxes[\''+id+'\']=[];draw(\''+id+'\')">CLEAR BOXES</button>';g.appendChild(c);let im=document.querySelector('#im-'+id),cv=document.querySelector('#cv-'+id);im.onload=()=>{cv.width=im.naturalWidth;cv.height=im.naturalHeight;draw(id)};let down=null;cv.onmousedown=e=>{let p=point(e,cv);down=p};cv.onmousemove=e=>{if(down){let p=point(e,cv);draw(id);let x=Math.min(down.x,p.x),y=Math.min(down.y,p.y),w=Math.abs(p.x-down.x),h=Math.abs(p.y-down.y),xctx=cv.getContext('2d');xctx.strokeStyle='#ff3030';xctx.lineWidth=3;xctx.strokeRect(x,y,w,h)}};cv.onmouseup=e=>{if(down){let p=point(e,cv);let b={x:Math.min(down.x,p.x),y:Math.min(down.y,p.y),w:Math.abs(p.x-down.x),h:Math.abs(p.y-down.y)};if(b.w>8&&b.h>16)boxes[id].push(b);down=null;draw(id)}}}}
function selectFrame(id,path){let im=document.querySelector('#im-'+id);im.src='/files/'+path.split('/').slice(-1)[0]+'?t='+Date.now();boxes[id]=[];}
function point(e,c){let r=c.getBoundingClientRect();return{x:(e.clientX-r.left)*c.width/r.width,y:(e.clientY-r.top)*c.height/r.height}}
function draw(id){let cv=document.querySelector('#cv-'+id);if(!cv)return;let x=cv.getContext('2d');x.clearRect(0,0,cv.width,cv.height);x.strokeStyle='#00ff70';x.lineWidth=3;boxes[id].forEach((b,i)=>{x.strokeRect(b.x,b.y,b.w,b.h);x.fillStyle='#00ff70';x.font='24px sans-serif';x.fillText('staff '+(i+1),b.x+4,b.y+25)})}
async function capture(){document.querySelector('#status').textContent='Capturing six frames…';let r=await fetch('/api/capture',{method:'POST'});document.querySelector('#status').textContent=JSON.stringify(await r.json(),null,2);load()}
async function saveBox(id){let f=frames[id];let class_name=(window.prompt('Enter class: staff or customer','staff')||'staff').toLowerCase();let r=await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({camera:id,frame:f.path,class_name:class_name,bboxes:boxes[id]})});document.querySelector('#status').textContent=JSON.stringify(await r.json(),null,2);boxes[id]=[];draw(id)}
async function profile(){let r=await fetch('/api/profile',{method:'POST'});document.querySelector('#status').textContent=JSON.stringify(await r.json(),null,2)}
load();
</script>'''

app = Flask(__name__)


@app.get("/")
def index():
    return render_template_string(HTML)


@app.get("/files/<path:name>")
def files(name):
    from flask import send_from_directory
    return send_from_directory(FRAMES, name)


@app.get("/api/frames")
def frames():
    return jsonify(latest_frames())


@app.post("/api/capture")
def capture():
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        values = list(pool.map(lambda camera: capture_one(camera, stamp), cameras()))
    return jsonify({"captured": [item for item in values if item], "failed": [camera["id"] for camera, item in zip(cameras(), values) if item is None]})


@app.post("/api/label")
def label():
    data = request.get_json(force=True)
    frame_rel = Path(data["frame"])
    frame_path = ROOT / frame_rel
    image = cv2.imread(str(frame_path))
    if image is None:
        return jsonify(error="frame not found"), 400
    labels = load_labels()
    class_name = str(data.get("class_name", "staff")).lower()
    if class_name not in CROP_DIRS:
        return jsonify(error="class_name must be staff or customer"), 400
    saved = []
    for bbox in data.get("bboxes", []):
        x, y, w, h = [float(bbox[key]) for key in ("x", "y", "w", "h")]
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(image.shape[1], int(x + w)), min(image.shape[0], int(y + h))
        if x1 <= x0 or y1 <= y0:
            continue
        class_count = sum(1 for item in labels if item.get("class", "staff") == class_name)
        label_id = f"{class_name}_{class_count + 1:05d}"
        while any(item.get("id") == label_id for item in labels):
            class_count += 1
            label_id = f"{class_name}_{class_count + 1:05d}"
        crop_rel = Path("data/staff_filter") / class_name / f"{label_id}.jpg"
        cv2.imwrite(str(ROOT / crop_rel), image[y0:y1, x0:x1])
        item = {"id": label_id, "class": class_name, "camera": data["camera"], "frame": str(frame_rel), "captured_at": frame_path.stat().st_mtime, "bbox": [x0, y0, x1 - x0, y1 - y0], "crop": str(crop_rel), "source_width": int(image.shape[1]), "source_height": int(image.shape[0])}
        labels.append(item); saved.append(item)
    save_labels(labels)
    return jsonify(saved=saved, total=len(labels))


@app.post("/api/profile")
def profile():
    value = derive_profile()
    return jsonify(value or {"enabled": False, "reason": "label staff crops first"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("STAFF_LABELER_PORT", "8780")), threaded=True)
