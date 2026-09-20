# Minimum physical inputs for CH16/CH18 AutoMagicCalib

This is the smallest measurement set needed to turn the existing preliminary
layout and candidate landmarks into calibration inputs. Do not measure people,
temporary displays, or travel times.

## Required

1. Measure one straight-line distance in metres between two permanent floor-level
   landmarks that are visible in both CH16 and CH18 (for example, two corners of
   the same fixed display island). Record the two landmark IDs and the distance.
   This establishes the layout map's pixels-per-metre scale.
2. On the layout image, mark at least four distinct permanent floor-level points
   that are visible in both cameras. For each point, record its pixel coordinate
   on the CH16 frame, CH18 frame, and `layout.png`. Use points spread across near,
   far, left, and right parts of the shared floor area.
3. Confirm the exact CH16↔CH18 overlap extent and identify which marked points are
   on the walking surface rather than on elevated shelving or walls.

## Optional only if available

- Measured camera mounting position, height, and orientation.
- Lens focal length or camera calibration data.

These are not required for the documented manual-alignment path; do not collect
them solely to run the first AMC attempt.

## Not required for this phase

- Camera-to-camera travel time.
- Homography or projection matrices.
- A full architectural floor-plan survey beyond the shared CH16/CH18 area.
- Any production-stream changes.

The existing `landmarks.yaml` deliberately leaves `floor_coordinate` as
`UNKNOWN` and marks the pixel candidates as non-usable until the physical and
layout correspondences above are confirmed.
