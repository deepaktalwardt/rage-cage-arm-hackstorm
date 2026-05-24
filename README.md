# rage-cage-arm-hackstorm
Rage Cage playing robotic arm

## ArUco Table Plane Detection

Install dependencies:

```bash
pip install -r requirements.txt
```

## ROS 2 ArUco Detection

Run the direct ROS 2 subscriber from a ROS-sourced shell or container:

```bash
python3 scripts/aruco_ros2_node.py \
  --ros-args \
  -p image_topic:=/camera/d435i/color/image_raw \
  -p camera_info_topic:=/camera/d435i/color/camera_info \
  -p detections_topic:=/aruco/detections \
  -p annotated_topic:=/aruco/annotated_image \
  -p pose_topic:=/aruco/pose \
  -p marker_length_m:=0.04
```

The detections topic publishes JSON in `std_msgs/String` with marker IDs and
pixel corners. When `marker_length_m` is greater than zero and camera intrinsics
arrive on `camera_info_topic`, each JSON marker includes a `pose_stamped` object
with the same `header` and `pose` shape as `geometry_msgs/PoseStamped`.

The node also publishes:

```text
/aruco/annotated_image  sensor_msgs/Image
/aruco/pose             geometry_msgs/PoseStamped
```

In Foxglove, add an Image panel for `/aruco/annotated_image` to see marker
outlines, and add a 3D panel display for `/aruco/pose` to inspect the selected
marker pose. In the 3D panel, set the fixed frame to the pose message frame,
usually `d435i_color_optical_frame`, then add `/aruco/pose` as a Pose display.

Useful parameters:

```text
dictionary:=DICT_4X4_50
marker_length_m:=0.04
axis_length_m:=0.0
target_marker_id:=-1
reference_marker_id:=-1
process_every_n:=3
publish_annotated:=true
pose_smoothing_alpha:=1.0
```

`axis_length_m:=0.0` uses half of `marker_length_m` for the drawn axes.
`pose_smoothing_alpha:=1.0` disables smoothing. Lower values smooth `/aruco/pose`
more strongly but add lag; try `0.2` for a stable debug view.

Set `reference_marker_id:=4` to publish `/aruco/pose` relative to marker `id4`
instead of the camera frame. For example, to publish marker `id3` in marker
`id4`'s frame:

```bash
python3 scripts/aruco_ros2_node.py --ros-args \
  -p target_marker_id:=3 \
  -p reference_marker_id:=4
```

Print the 3D distance between two detected marker centers:

```bash
python3 scripts/aruco_distance.py --origin-id 3 --target-id 4
```

The script prints `delta_m` in the camera optical frame and
`delta_in_id3_frame_m` in the origin marker frame. For two markers on the same
flat surface, the origin-frame Z value should be close to zero.

Add `--once` to exit after the first frame where both marker poses are visible.

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
