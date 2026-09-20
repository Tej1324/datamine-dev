# CH18 ↔ CH20 planar calibration

This project is an OpenCV floor-plane homography workflow for the observed
common central display/rack area. It is independent of AutoMagicCalib and does
not produce an MV3DT `projectionMatrix_3x4`.

The source videos are referenced in `inputs/capture_manifest.yaml`; they are
not copied into this project. Representative 1920×1080 frames and
aspect-preserving 1280×720 working frames are under `frames/`. The existing
ground-floor layout is copied under `layout/` for reference only.

Run:

```bash
python3 calibration/ground_floor_ch18_ch20/calibrate.py
```

For manual correspondence selection, start the loopback-only picker:

```bash
python3 calibration/ground_floor_ch18_ch20/pick_landmarks.py
```

Open `http://127.0.0.1:8765/`. Click each permanent floor-plane point in
CH18 first and the same point in CH20 second. Use at least eight well-spread
pairs, then click **Save landmarks.yaml** and **Run RANSAC validation**. For a
remote browser, forward the picker port with `ssh -N -L
8765:127.0.0.1:8765 <ssh-target>`.

The script uses `cv2.findHomography(..., cv2.RANSAC, ...)`, emits candidate
`H18_to_world`, `H20_to_world`, `H18_to_20`, and `H20_to_18` matrices, creates
point/warp/overlay artifacts, and rejects a narrow or unstable landmark set.
When the picker has not been given floor coordinates, validation runs in
pixel-only mode and emits only CH18↔CH20 planar homographies; world mappings
remain null.

The current point set is intentionally marked as candidate-only. Its floor
coordinate units are arbitrary and its metric scale is unverified. A successful
run of this tool does not authorize MV3DT integration.
