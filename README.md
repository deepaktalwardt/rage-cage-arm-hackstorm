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
```

`axis_length_m:=0.0` uses half of `marker_length_m` for the drawn axes.

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

## Ping-Pong 3D Perception

This branch contains one ROS 2 node, `perception_node`, that consumes the camera
color image and camera calibration once, then:

- detects the ping-pong ball in 2D with `model/yolo11n.onnx`
- detects the table ArUco marker with `table_marker_id:=4`
- computes the camera-frame ray through the ball centroid
- intersects that ray with a plane `ball_radius_m` above the ArUco marker plane
- publishes marker and ball poses in camera frame and marker-4 frame

Published topics:

```text
/perception/debug/ball_detection                 vision_msgs/Detection2DArray
/perception/debug/ball_pose                      geometry_msgs/PoseStamped
/perception/debug/marker_4_pose                  geometry_msgs/PoseStamped
/perception/debug/marker_3_pose                  geometry_msgs/PoseStamped
/perception/debug/annotated_image                sensor_msgs/Image
/perception/debug/ball_marker                    visualization_msgs/Marker
/perception/debug/ball_ray                       visualization_msgs/Marker
/perception/debug/aruco_detections              std_msgs/String
/perception/output/ball_pose_marker_4_frame      geometry_msgs/PoseStamped
/perception/output/cup_pose                      geometry_msgs/PoseStamped
```

Run it as a mounted overlay on the existing Piper image without editing the Piper
repo:

```bash
cd /home/orin/hackathon/xinyi/rage-cage-arm-hackstorm
ROS_DOMAIN_ID=12 docker compose -f compose.perception.yaml up
```

Useful parameters:

```text
image_topic:=/camera/d435i/color/image_raw
camera_info_topic:=/camera/d435i/color/camera_info
table_marker_id:=4
marker_length_m:=0.04
ball_radius_m:=0.025
cup_offset_marker_3_x_m:=0.0
cup_offset_marker_3_y_m:=0.029
cup_offset_marker_3_z_m:=-0.0555
robot_arm_base_frame:=base_link
marker_4_in_robot_base_x_m:=0.0
marker_4_in_robot_base_y_m:=0.0
marker_4_in_robot_base_z_m:=0.0
marker_4_in_robot_base_qx:=0.0
marker_4_in_robot_base_qy:=0.0
marker_4_in_robot_base_qz:=0.0
marker_4_in_robot_base_qw:=1.0
process_every_n:=3
```

In Foxglove, set the 3D panel fixed frame to `d435i_color_optical_frame`, then
add `/perception/debug/ball_marker` and `/perception/debug/ball_ray` as Marker
topics. To show the annotated camera image inside the 3D panel, add
`/perception/debug/annotated_image` as an Image/Camera layer and pair it with
`/camera/d435i/color/camera_info`. `/perception/debug/ball_pose` is also
available as a PoseStamped topic in the camera frame, and
`/perception/output/ball_pose_marker_4_frame` publishes the same ball point in
`aruco_marker_4` coordinates. With the default `ball_radius_m:=0.025`, the
marker-frame `z` value should be near `+0.025 m` because the pose is the ball
center, not the contact point on the table.

`/perception/output/cup_pose` publishes the center of the cup opening in the
robot arm base frame, default `base_link`. It is computed from marker 3 by
applying the marker-3-frame offset
`[cup_offset_marker_3_x_m, cup_offset_marker_3_y_m, cup_offset_marker_3_z_m]`.
The defaults assume the opening center is 29 mm in marker-3 +Y and 55.5 mm in
marker-3 -Z from marker 3's origin. The resulting marker-4-frame point is then
transformed by the hard-coded marker 4 pose in robot base,
`marker_4_in_robot_base_*`. Those parameters default to identity, so the current
numeric cup position is unchanged from marker 4 coordinates while the message
frame is `base_link`.
