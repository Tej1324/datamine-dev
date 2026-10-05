#!/usr/bin/env python3
"""Run optional AMC VGGT refinement for the existing CH10/CH11 project."""
import json, os, time
from pathlib import Path
from urllib.request import Request, urlopen

BASE_URL = os.environ.get("AMC_BASE_URL", "http://127.0.0.1:8100/v1").rstrip("/")
PROJECT_ID = os.environ.get("AMC_PROJECT_ID", "20260918_011215_6295")
MODEL = Path(__file__).resolve().parents[1] / "calibration/auto-magic-calib/tools/auto-magic-calib/models/vggt/vggt_1B_commercial.pt"

def api(path, method="GET"):
    with urlopen(Request(f"{BASE_URL}{path}", method=method), timeout=30) as response:
        return json.loads(response.read())

def main():
    if not MODEL.is_file():
        raise SystemExit(f"VGGT model missing: {MODEL}; run scripts/download_vggt_model.sh")
    info = api(f"/get_project_info/{PROJECT_ID}")
    project = info.get("project_info", info)
    state = project.get("vggt_state", "INIT")
    print(json.dumps({"project_id": PROJECT_ID, "project_state": project.get("project_state"), "vggt_state": state}, indent=2))
    if state != "READY":
        raise SystemExit(f"VGGT is not ready; current vggt_state={state}")
    api(f"/vggt/calibrate/{PROJECT_ID}", method="POST")
    while True:
        time.sleep(10)
        project = api(f"/get_project_info/{PROJECT_ID}").get("project_info", {})
        state = project.get("vggt_state", "UNKNOWN")
        print(f"vggt_state={state}", flush=True)
        if state in {"COMPLETED", "ERROR"}: break
    if state != "COMPLETED": raise SystemExit("VGGT failed; inspect AMC logs")
    print(json.dumps(api(f"/vggt_results/{PROJECT_ID}/evaluation_statistics"), indent=2))

if __name__ == "__main__": main()
