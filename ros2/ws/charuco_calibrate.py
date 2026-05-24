#!/usr/bin/env python3
"""ChArUco intrinsic calibration node.

Subscribes to a ROS Image topic, lets you select frames via stdin keypresses
while watching annotated detections in Foxglove, then runs OpenCV's
calibrateCameraCharuco and writes a ROS camera_info YAML.

Targets the legacy OpenCV 4.6 aruco API (the version shipped in the piper:jazzy
container): detectMarkers + interpolateCornersCharuco + calibrateCameraCharuco.

Keys (in the terminal running this script):
  SPACE  capture current frame (only if charuco corners were detected)
  c      run intrinsic calibration on the captured frames
  u      undo last capture
  r      reset all captures
  s      save last calibration to YAML (also auto-saved after 'c')
  q      quit

Example:
  python3 charuco_calibrate.py \
    --topic /camera/d435i/color/image_raw \
    --squares-x 5 --squares-y 7 \
    --square-length 0.035 --marker-length 0.026 \
    --dictionary DICT_4X4_50 \
    --output ~/d435i_color_intrinsics.yaml
"""

import argparse
import os
import select
import signal
import sys
import termios
import threading
import tty
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


ARUCO_DICTS = {
    name: getattr(cv2.aruco, name)
    for name in [
        "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
        "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
        "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
        "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
        "DICT_ARUCO_ORIGINAL",
    ]
    if hasattr(cv2.aruco, name)
}


@dataclass
class Capture:
    charuco_corners: np.ndarray
    charuco_ids: np.ndarray
    image_size: Tuple[int, int]  # (w, h)


class CharucoCalibNode(Node):
    def __init__(self, args):
        super().__init__("charuco_calibrate")
        self.bridge = CvBridge()

        self.dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[args.dictionary])
        self.board = cv2.aruco.CharucoBoard_create(
            args.squares_x,
            args.squares_y,
            args.square_length,
            args.marker_length,
            self.dictionary,
        )

        self.output_path = os.path.expanduser(args.output)
        self.camera_name = args.camera_name
        self.min_corners = args.min_corners
        self.min_captures = args.min_captures

        self.lock = threading.Lock()
        self.captures: List[Capture] = []
        self.last_detection: Optional[Capture] = None
        self.last_status = "waiting for images"
        self.last_rms: Optional[float] = None
        self.last_K: Optional[np.ndarray] = None
        self.last_D: Optional[np.ndarray] = None
        self.last_image_size: Optional[Tuple[int, int]] = None

        self.sub = self.create_subscription(Image, args.topic, self._image_cb, 10)
        self.pub = self.create_publisher(Image, args.feedback_topic, 10)
        self.get_logger().info(
            f"sub={args.topic}  feedback={args.feedback_topic}  "
            f"board={args.squares_x}x{args.squares_y} "
            f"square={args.square_length}m marker={args.marker_length}m "
            f"dict={args.dictionary}"
        )
        self.get_logger().info(
            "keys: SPACE=capture  c=calibrate  u=undo  r=reset  s=save  q=quit"
        )

    def _image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        m_corners, m_ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
        c_corners = None
        c_ids = None
        if m_ids is not None and len(m_ids) > 0:
            _, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(
                m_corners, m_ids, gray, self.board
            )

        annotated = frame.copy()
        if m_ids is not None and len(m_ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated, m_corners, m_ids)
        n_corners = 0
        if c_ids is not None and len(c_ids) > 0:
            n_corners = len(c_ids)
            cv2.aruco.drawDetectedCornersCharuco(
                annotated, c_corners, c_ids, (0, 255, 0)
            )

        h, w = frame.shape[:2]
        with self.lock:
            if n_corners >= self.min_corners:
                self.last_detection = Capture(
                    charuco_corners=c_corners.copy(),
                    charuco_ids=c_ids.copy(),
                    image_size=(w, h),
                )
            else:
                self.last_detection = None
            self._draw_overlay(annotated, n_corners)

        out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out.header = msg.header
        self.pub.publish(out)

    def _draw_overlay(self, img: np.ndarray, n_corners: int):
        ready = "READY" if self.last_detection is not None else "no detection"
        lines = [
            f"captures: {len(self.captures)}   corners: {n_corners}   {ready}",
            f"status: {self.last_status}",
        ]
        if self.last_rms is not None:
            lines.append(f"last RMS: {self.last_rms:.4f}")
        y = 24
        for line in lines:
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += 24

    # ---- keyboard actions (called from stdin thread) ----

    def capture(self):
        with self.lock:
            if self.last_detection is None:
                self.last_status = "capture rejected: no charuco detection"
                self.get_logger().warn(self.last_status)
                return
            self.captures.append(self.last_detection)
            n = len(self.last_detection.charuco_ids)
            self.last_status = f"captured frame {len(self.captures)} (corners={n})"
            self.get_logger().info(self.last_status)

    def undo(self):
        with self.lock:
            if not self.captures:
                self.last_status = "nothing to undo"
                return
            self.captures.pop()
            self.last_status = f"undo; captures={len(self.captures)}"
            self.get_logger().info(self.last_status)

    def reset(self):
        with self.lock:
            self.captures.clear()
            self.last_rms = None
            self.last_K = None
            self.last_D = None
            self.last_status = "reset"
            self.get_logger().info(self.last_status)

    def calibrate(self):
        with self.lock:
            if len(self.captures) < self.min_captures:
                self.last_status = (
                    f"need >= {self.min_captures} captures, have {len(self.captures)}"
                )
                self.get_logger().warn(self.last_status)
                return
            captures = list(self.captures)
            image_size = captures[-1].image_size  # (w, h)

        self.get_logger().info(
            f"calibrateCameraCharuco on {len(captures)} views @ {image_size}..."
        )
        rms, K, D, _, _ = cv2.aruco.calibrateCameraCharuco(
            [c.charuco_corners for c in captures],
            [c.charuco_ids for c in captures],
            self.board,
            image_size,
            None,
            None,
        )
        with self.lock:
            self.last_rms = float(rms)
            self.last_K = K
            self.last_D = D
            self.last_image_size = image_size
            self.last_status = f"calibrated RMS={rms:.4f}"
        self.get_logger().info(
            f"RMS={rms:.4f}\nK=\n{K}\nD={np.array(D).ravel()}"
        )
        self.save()

    def save(self):
        with self.lock:
            if self.last_K is None:
                self.last_status = "nothing to save; run calibrate first"
                self.get_logger().warn(self.last_status)
                return
            K = self.last_K
            D = np.array(self.last_D).ravel()
            w, h = self.last_image_size

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        data = {
            "image_width": int(w),
            "image_height": int(h),
            "camera_name": self.camera_name,
            "camera_matrix": {
                "rows": 3, "cols": 3,
                "data": [float(v) for v in K.ravel()],
            },
            "distortion_model": "plumb_bob",
            "distortion_coefficients": {
                "rows": 1, "cols": int(D.size),
                "data": [float(v) for v in D],
            },
            "rectification_matrix": {
                "rows": 3, "cols": 3,
                "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            },
            "projection_matrix": {
                "rows": 3, "cols": 4,
                "data": [fx, 0.0, cx, 0.0,
                         0.0, fy, cy, 0.0,
                         0.0, 0.0, 1.0, 0.0],
            },
        }
        out = os.path.abspath(self.output_path)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        self.get_logger().info(f"wrote {out}")


def stdin_loop(node: CharucoCalibNode, stop: threading.Event):
    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        node.get_logger().warn(
            "stdin is not a TTY; keyboard disabled. Run from a terminal."
        )
        return
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not stop.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not r:
                continue
            ch = sys.stdin.read(1)
            if ch == " ":
                node.capture()
            elif ch == "c":
                node.calibrate()
            elif ch == "u":
                node.undo()
            elif ch == "r":
                node.reset()
            elif ch == "s":
                node.save()
            elif ch in ("q", "\x03", "\x04"):
                stop.set()
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def parse_args():
    p = argparse.ArgumentParser(description="ChArUco intrinsic calibration node")
    p.add_argument("--topic", default="/camera/color/image_raw",
                   help="sensor_msgs/Image topic to subscribe to")
    p.add_argument("--feedback-topic", default="/charuco_calib/feedback",
                   help="annotated Image topic for Foxglove")
    p.add_argument("--squares-x", type=int, required=True,
                   help="number of checker squares along X")
    p.add_argument("--squares-y", type=int, required=True,
                   help="number of checker squares along Y")
    p.add_argument("--square-length", type=float, required=True,
                   help="checker square side length [m]")
    p.add_argument("--marker-length", type=float, required=True,
                   help="ArUco marker side length [m]")
    p.add_argument("--dictionary", default="DICT_4X4_50",
                   choices=sorted(ARUCO_DICTS))
    p.add_argument("--output", default="~/camera_intrinsics.yaml",
                   help="output camera_info YAML path")
    p.add_argument("--camera-name", default="camera",
                   help="camera_name field written into the YAML")
    p.add_argument("--min-corners", type=int, default=6,
                   help="min charuco corners required to accept a capture")
    p.add_argument("--min-captures", type=int, default=8,
                   help="min captures required before calibrating")
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = CharucoCalibNode(args)
    stop = threading.Event()

    key_thread = threading.Thread(target=stdin_loop, args=(node, stop), daemon=True)
    key_thread.start()

    def shutdown(*_):
        stop.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while not stop.is_set() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        stop.set()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
