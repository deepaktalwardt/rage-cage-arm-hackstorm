#!/usr/bin/env python3
"""ROS 2 node that detects ArUco markers from a sensor_msgs/Image topic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


if not hasattr(cv2, "aruco"):
    raise RuntimeError("OpenCV was installed without aruco support. Install opencv-contrib-python.")

ARUCO_DICTS = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco)
    if name.startswith("DICT_")
}


@dataclass
class MarkerPose:
    marker_id: int
    pose: PoseStamped
    rvec: np.ndarray
    tvec: np.ndarray


class ArucoDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("aruco_detector")
        self.declare_parameter("image_topic", "/camera/d435i/color/image_raw")
        self.declare_parameter("detections_topic", "/aruco/detections")
        self.declare_parameter("annotated_topic", "/aruco/annotated_image")
        self.declare_parameter("camera_info_topic", "/camera/d435i/color/camera_info")
        self.declare_parameter("pose_topic", "/aruco/pose")
        self.declare_parameter("dictionary", "DICT_4X4_50")
        self.declare_parameter("marker_length_m", 0.0)
        self.declare_parameter("target_marker_id", -1)
        self.declare_parameter("axis_length_m", 0.0)
        self.declare_parameter("process_every_n", 3)
        self.declare_parameter("publish_annotated", True)

        self.image_topic = self.get_parameter("image_topic").value
        detections_topic = self.get_parameter("detections_topic").value
        annotated_topic = self.get_parameter("annotated_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        pose_topic = self.get_parameter("pose_topic").value
        dictionary_name = self.get_parameter("dictionary").value
        self.marker_length_m = float(self.get_parameter("marker_length_m").value)
        self.target_marker_id = int(self.get_parameter("target_marker_id").value)
        self.axis_length_m = float(self.get_parameter("axis_length_m").value)
        self.process_every_n = max(1, int(self.get_parameter("process_every_n").value))
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)

        if dictionary_name not in ARUCO_DICTS:
            raise ValueError(f"unknown ArUco dictionary {dictionary_name}; choices: {sorted(ARUCO_DICTS)}")

        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary_name])
        if hasattr(cv2.aruco, "DetectorParameters"):
            parameters = cv2.aruco.DetectorParameters()
        else:
            parameters = cv2.aruco.DetectorParameters_create()
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters) if hasattr(cv2.aruco, "ArucoDetector") else None
        self.dictionary = dictionary

        self.detections_pub = self.create_publisher(String, detections_topic, 10)
        self.annotated_pub = self.create_publisher(Image, annotated_topic, 10) if self.publish_annotated else None
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.subscription = self.create_subscription(Image, self.image_topic, self._on_image, qos_profile_sensor_data)
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._on_camera_info,
            qos_profile_sensor_data,
        )
        self.frame_count = 0
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None

        self.get_logger().info(
            f"subscribing to {self.image_topic}; publishing detections on {self.detections_pub.topic_name}"
        )
        if self.marker_length_m > 0.0:
            self.get_logger().info(
                f"subscribing to {camera_info_topic}; publishing target marker pose on {self.pose_pub.topic_name}"
            )
        else:
            self.get_logger().info("marker_length_m is 0; pose output is disabled")

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(msg.d, dtype=np.float64)

    def _on_image(self, msg: Image) -> None:
        self.frame_count += 1
        if self.frame_count % self.process_every_n != 0:
            return

        try:
            bgr = image_to_bgr(msg)
            corners, ids = self._detect(bgr)
            poses = self._estimate_poses(corners, ids)
            target_pose = self._target_pose(ids, poses)
            self.detections_pub.publish(String(data=json.dumps(detection_payload(msg, corners, ids, poses))))
            if target_pose is not None:
                self.pose_pub.publish(pose_stamped_msg(msg, target_pose.pose))
            if self.annotated_pub is not None:
                self.annotated_pub.publish(
                    annotated_image_msg(
                        msg,
                        bgr,
                        corners,
                        ids,
                        poses,
                        self.camera_matrix,
                        self.dist_coeffs,
                        self.axis_length_m if self.axis_length_m > 0.0 else self.marker_length_m * 0.5,
                    )
                )
        except Exception as exc:
            self.get_logger().error(f"failed to process image: {exc}")

    def _detect(self, bgr: np.ndarray) -> tuple[list[np.ndarray], np.ndarray | None]:
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(bgr)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(bgr, self.dictionary)
        return corners, ids

    def _estimate_poses(self, corners: list[np.ndarray], ids: np.ndarray | None) -> list[MarkerPose] | None:
        if ids is None or len(ids) == 0:
            return None
        if self.marker_length_m <= 0.0 or self.camera_matrix is None:
            return None

        dist_coeffs = self.dist_coeffs if self.dist_coeffs is not None else np.zeros((5,), dtype=np.float64)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            self.marker_length_m,
            self.camera_matrix,
            dist_coeffs,
        )
        return [
            MarkerPose(int(marker_id), pose_from_rvec_tvec(rvec, tvec), rvec, tvec)
            for marker_id, rvec, tvec in zip(ids.flatten(), rvecs, tvecs)
        ]

    def _target_pose(self, ids: np.ndarray | None, poses: list[MarkerPose] | None) -> MarkerPose | None:
        if ids is None or poses is None:
            return None
        for pose in poses:
            if self.target_marker_id < 0 or pose.marker_id == self.target_marker_id:
                return pose
        return None


def image_to_bgr(msg: Image) -> np.ndarray:
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
        "8UC1": 1,
        "8UC3": 3,
    }
    if msg.encoding not in channels_by_encoding:
        raise ValueError(f"unsupported image encoding: {msg.encoding}")

    channels = channels_by_encoding[msg.encoding]
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    expected = int(msg.height * msg.step)
    if raw.size < expected:
        raise ValueError(f"image data is shorter than expected: {raw.size} < {expected}")

    rows = raw[:expected].reshape((msg.height, msg.step))
    pixels = rows[:, : msg.width * channels].reshape((msg.height, msg.width, channels))

    if msg.encoding in ("rgb8", "8UC3"):
        return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    if msg.encoding == "bgr8":
        return pixels.copy()
    if msg.encoding == "rgba8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
    if msg.encoding == "bgra8":
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)


def detection_payload(
    msg: Image,
    corners: list[np.ndarray],
    ids: np.ndarray | None,
    poses: list[MarkerPose] | None,
) -> dict[str, Any]:
    marker_ids = [] if ids is None else [int(marker_id) for marker_id in ids.flatten()]
    marker_corners = [] if ids is None else [np.squeeze(corner).tolist() for corner in corners]
    marker_poses = poses or [None] * len(marker_ids)
    return {
        "stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec},
        "frame_id": msg.header.frame_id,
        "image": {"width": msg.width, "height": msg.height, "encoding": msg.encoding},
        "detected": len(marker_ids) > 0,
        "marker_count": len(marker_ids),
        "markers": [
            {
                "id": marker_id,
                "corners_px": corners_px,
                "pose_stamped": pose_stamped_payload(msg, marker_pose.pose) if marker_pose is not None else None,
            }
            for marker_id, corners_px, marker_pose in zip(marker_ids, marker_corners, marker_poses)
        ],
    }


def annotated_image_msg(
    source: Image,
    bgr: np.ndarray,
    corners: list[np.ndarray],
    ids: np.ndarray | None,
    poses: list[MarkerPose] | None,
    camera_matrix: np.ndarray | None,
    dist_coeffs: np.ndarray | None,
    axis_length_m: float,
) -> Image:
    annotated = bgr.copy()
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        if poses is not None and camera_matrix is not None and axis_length_m > 0.0:
            coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((5,), dtype=np.float64)
            for marker_pose in poses:
                cv2.drawFrameAxes(
                    annotated,
                    camera_matrix,
                    coeffs,
                    marker_pose.rvec,
                    marker_pose.tvec,
                    axis_length_m,
                )

    msg = Image()
    msg.header = source.header
    msg.height = annotated.shape[0]
    msg.width = annotated.shape[1]
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = int(annotated_rgb.shape[1] * 3)
    msg.data = annotated_rgb.tobytes()
    return msg


def pose_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> PoseStamped:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    quaternion = quaternion_from_rotation(rotation)
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)

    pose = PoseStamped()
    pose.pose.position.x = float(translation[0])
    pose.pose.position.y = float(translation[1])
    pose.pose.position.z = float(translation[2])
    pose.pose.orientation.x = quaternion[0]
    pose.pose.orientation.y = quaternion[1]
    pose.pose.orientation.z = quaternion[2]
    pose.pose.orientation.w = quaternion[3]
    return pose


def pose_stamped_msg(source: Image, pose: PoseStamped) -> PoseStamped:
    msg = PoseStamped()
    msg.header = source.header
    msg.pose = pose.pose
    return msg


def pose_stamped_payload(source: Image, pose: PoseStamped) -> dict[str, Any]:
    stamped = pose_stamped_msg(source, pose)
    return {
        "header": {
            "stamp": {"sec": stamped.header.stamp.sec, "nanosec": stamped.header.stamp.nanosec},
            "frame_id": stamped.header.frame_id,
        },
        "pose": {
            "position": {
                "x": stamped.pose.position.x,
                "y": stamped.pose.position.y,
                "z": stamped.pose.position.z,
            },
            "orientation": {
                "x": stamped.pose.orientation.x,
                "y": stamped.pose.orientation.y,
                "z": stamped.pose.orientation.z,
                "w": stamped.pose.orientation.w,
            },
        },
    }


def quaternion_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif index == 1:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s

    norm = float(np.linalg.norm([x, y, z, w]))
    if norm < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    return float(x / norm), float(y / norm), float(z / norm), float(w / norm)



def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
