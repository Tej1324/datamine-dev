# CH16/CH18 calibration inputs

This directory is reserved for the NVIDIA AutoMagicCalib input and output
artifacts. It does not contain credentials, RTSP URLs, or fabricated
calibration values.

## Required inputs

Place time-synchronized main-stream recordings here:

```text
inputs/videos/ch16.mp4
inputs/videos/ch18.mp4
inputs/layout.png
```

The recordings must be from the NVR main streams at 1920x1080. Capture at
least five minutes with approximately ten or more different people moving
through the shared CH16/CH18 area and visible common landmarks.

The layout image must show the CH16/CH18 floor area and the fixed landmarks
used for manual alignment. Do not add guessed camera poses or world points.

## Output locations

AutoMagicCalib results belong here:

```text
outputs/automagiccalib/
outputs/mv3dt/
```

The production pipeline must continue using the 1280x720 substreams. Any
exported camera matrices must be converted to that image coordinate system
before a future MV3DT integration. MV3DT is not enabled by this setup file.
