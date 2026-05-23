# rage-cage-arm-hackstorm
Rage Cage playing robotic arm

## ArUco Table Plane Detection

Install dependencies:

```bash
pip install -r requirements.txt
```

Or use `uv`:

```bash
uv sync
```

Run on a camera:

```bash
uv run python scripts/detect_aruco_table.py \
  --camera 0 \
  --marker-length-m 0.04 \
  --camera-matrix FX FY CX CY \
  --output-json aruco_table_plane.json
```

Run on an image:

```bash
uv run python scripts/detect_aruco_table.py \
  --image frame.jpg \
  --marker-length-m 0.04 \
  --calibration camera_calibration.npz
```

For a 4x4 AprilTag with 8 mm cells, use the measured outer tag side length. If
the full detected tag is 4 cells across, that is:

```bash
uv run python scripts/detect_aruco_table.py \
  --image frame.jpg \
  --dict DICT_APRILTAG_16h5 \
  --marker-length-m 0.032 \
  --no-show
```

Without camera intrinsics the script can detect marker pixel corners, but it
cannot estimate the metric table plane. The table plane output is expressed in
the camera frame as:

```text
normal_x*x + normal_y*y + normal_z*z + d = 0
```

When `--reference-id` is provided, the same fitted table plane is also printed
in that marker's coordinate frame as `table_plane_in_idN`. Each marker line also
prints the marker's pose normal in the camera frame and its angle from the fitted
table normal.
