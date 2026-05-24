#!/usr/bin/env python3
"""ROS 2 perception node for table ArUco pose plus ping-pong ball 3D position."""

from __future__ import annotations

import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VENV_SITE_PACKAGES = Path(f"/opt/venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
if VENV_SITE_PACKAGES.exists():
    sys.path = [path for path in sys.path if path != str(VENV_SITE_PACKAGES)]
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

import cv2
import numpy as np
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker


if not hasattr(cv2, "aruco"):
    raise RuntimeError("OpenCV was installed without aruco support. Install opencv-contrib-python-headless.")


SPORTS_BALL_CLASS_ID = 32
COCO_CLASS_NAMES = {SPORTS_BALL_CLASS_ID: "sports ball"}
def default_model_path() -> Path:
    source_model = Path(__file__).resolve().parents[1] / "model" / "yolo11n.onnx"
    if source_model.exists():
        return source_model
    try:
        return Path(get_package_share_directory("rage_cage_perception")) / "model" / "yolo11n.onnx"
    except PackageNotFoundError:
        return source_model


DEFAULT_MODEL_PATH = default_model_path()
ARUCO_DICTS = {name: getattr(cv2.aruco, name) for name in dir(cv2.aruco) if name.startswith("DICT_")}


@dataclass(frozen=True)
class BallDetection:
    centroid: tuple[float, float]
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    color_score: float


@dataclass(frozen=True)
class MarkerPose:
    marker_id: int
    rvec: np.ndarray
    tvec: np.ndarray
    pose: PoseStamped


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("perception_node")
        cv2.setNumThreads(1)

        self.declare_parameter("image_topic", "/camera/d435i/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/d435i/color/camera_info")
        self.declare_parameter("ball_detections_topic", "/perception/debug/ball_detection")
        self.declare_parameter("ball_pose_topic", "/perception/debug/ball_pose")
        self.declare_parameter("ball_pose_marker_frame_topic", "/perception/output/ball_pose_marker_4_frame")
        self.declare_parameter("ball_ray_topic", "/perception/debug/ball_ray")
        self.declare_parameter("ball_marker_topic", "/perception/debug/ball_marker")
        self.declare_parameter("marker_3_pose_topic", "/perception/debug/marker_3_pose")
        self.declare_parameter("marker_4_pose_topic", "/perception/debug/marker_4_pose")
        self.declare_parameter("cup_pose_topic", "/perception/output/cup_pose")
        self.declare_parameter("aruco_detections_topic", "/perception/debug/aruco_detections")
        self.declare_parameter("annotated_topic", "/perception/debug/annotated_image")
        self.declare_parameter("cup_offset_marker_3_x_m", 0.0)
        self.declare_parameter("cup_offset_marker_3_y_m", 0.029)
        self.declare_parameter("cup_offset_marker_3_z_m", -0.0555)
        self.declare_parameter("robot_arm_base_frame", "base_link")
        self.declare_parameter("marker_4_in_robot_base_x_m", 0.0)
        self.declare_parameter("marker_4_in_robot_base_y_m", 0.0)
        self.declare_parameter("marker_4_in_robot_base_z_m", 0.0)
        self.declare_parameter("marker_4_in_robot_base_qx", 0.0)
        self.declare_parameter("marker_4_in_robot_base_qy", 0.0)
        self.declare_parameter("marker_4_in_robot_base_qz", 0.0)
        self.declare_parameter("marker_4_in_robot_base_qw", 1.0)
        self.declare_parameter("model_path", str(DEFAULT_MODEL_PATH))
        self.declare_parameter("conf", 0.03)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("fallback_conf", 0.60)
        self.declare_parameter("use_color_tiebreak", True)
        self.declare_parameter("enable_white_ball_fallback", True)
        self.declare_parameter("dictionary", "DICT_4X4_50")
        self.declare_parameter("marker_length_m", 0.04)
        self.declare_parameter("table_marker_id", 4)
        self.declare_parameter("axis_length_m", 0.02)
        self.declare_parameter("ray_length_m", 1.5)
        self.declare_parameter("ball_radius_m", 0.025)
        self.declare_parameter("process_every_n", 3)
        self.declare_parameter("publish_annotated", True)

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        model_path = Path(str(self.get_parameter("model_path").value))
        self.conf = float(self.get_parameter("conf").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.iou = float(self.get_parameter("iou").value)
        self.fallback_conf = float(self.get_parameter("fallback_conf").value)
        self.use_color_tiebreak = bool(self.get_parameter("use_color_tiebreak").value)
        self.enable_white_ball_fallback = bool(self.get_parameter("enable_white_ball_fallback").value)
        self.marker_length_m = float(self.get_parameter("marker_length_m").value)
        self.table_marker_id = int(self.get_parameter("table_marker_id").value)
        self.marker_3_id = 3
        self.axis_length_m = float(self.get_parameter("axis_length_m").value)
        self.ray_length_m = float(self.get_parameter("ray_length_m").value)
        self.ball_radius_m = float(self.get_parameter("ball_radius_m").value)
        self.cup_offset_marker_3_m = np.asarray(
            [
                float(self.get_parameter("cup_offset_marker_3_x_m").value),
                float(self.get_parameter("cup_offset_marker_3_y_m").value),
                float(self.get_parameter("cup_offset_marker_3_z_m").value),
            ],
            dtype=np.float64,
        )
        self.robot_arm_base_frame = str(self.get_parameter("robot_arm_base_frame").value)
        self.marker_4_in_robot_base_t = np.asarray(
            [
                float(self.get_parameter("marker_4_in_robot_base_x_m").value),
                float(self.get_parameter("marker_4_in_robot_base_y_m").value),
                float(self.get_parameter("marker_4_in_robot_base_z_m").value),
            ],
            dtype=np.float64,
        )
        self.marker_4_in_robot_base_r = rotation_from_quaternion(
            (
                float(self.get_parameter("marker_4_in_robot_base_qx").value),
                float(self.get_parameter("marker_4_in_robot_base_qy").value),
                float(self.get_parameter("marker_4_in_robot_base_qz").value),
                float(self.get_parameter("marker_4_in_robot_base_qw").value),
            )
        )
        self.process_every_n = max(1, int(self.get_parameter("process_every_n").value))
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)

        dictionary_name = str(self.get_parameter("dictionary").value)
        if dictionary_name not in ARUCO_DICTS:
            raise ValueError(f"unknown ArUco dictionary {dictionary_name}; choices: {sorted(ARUCO_DICTS)}")
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary_name])
        if hasattr(cv2.aruco, "DetectorParameters"):
            parameters = cv2.aruco.DetectorParameters()
        else:
            parameters = cv2.aruco.DetectorParameters_create()
        self.aruco_detector = cv2.aruco.ArucoDetector(dictionary, parameters) if hasattr(cv2.aruco, "ArucoDetector") else None
        self.aruco_parameters = parameters
        self.aruco_dictionary = dictionary

        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")
        self.net = cv2.dnn.readNetFromONNX(str(model_path))

        self.ball_detections_pub = self.create_publisher(
            Detection2DArray, str(self.get_parameter("ball_detections_topic").value), 10
        )
        self.ball_pose_pub = self.create_publisher(PoseStamped, str(self.get_parameter("ball_pose_topic").value), 10)
        self.ball_pose_marker_frame_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("ball_pose_marker_frame_topic").value), 10
        )
        self.ball_ray_pub = self.create_publisher(Marker, str(self.get_parameter("ball_ray_topic").value), 10)
        self.ball_marker_pub = self.create_publisher(Marker, str(self.get_parameter("ball_marker_topic").value), 10)
        self.marker_3_pose_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("marker_3_pose_topic").value), 10
        )
        self.marker_4_pose_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("marker_4_pose_topic").value), 10
        )
        self.cup_pose_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("cup_pose_topic").value), 10
        )
        self.aruco_detections_pub = self.create_publisher(
            String, str(self.get_parameter("aruco_detections_topic").value), 10
        )
        self.annotated_pub = (
            self.create_publisher(Image, str(self.get_parameter("annotated_topic").value), 10)
            if self.publish_annotated
            else None
        )

        self.image_sub = self.create_subscription(Image, self.image_topic, self._on_image, qos_profile_sensor_data)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._on_camera_info, qos_profile_sensor_data
        )
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None
        self.frame_count = 0

        self.get_logger().info(
            f"subscribing to {self.image_topic} and {self.camera_info_topic}; "
            f"using table marker id {self.table_marker_id}"
        )

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(msg.d, dtype=np.float64)

    def _on_image(self, msg: Image) -> None:
        self.frame_count += 1
        if self.frame_count % self.process_every_n != 0:
            return

        try:
            bgr = image_to_bgr(msg)
            ball_detections = self._detect_balls(bgr)
            best_ball = ball_detections[0] if ball_detections else None
            corners, ids = self._detect_markers(bgr)
            marker_poses = self._estimate_marker_poses(msg, corners, ids)
            table_marker = next((pose for pose in marker_poses if pose.marker_id == self.table_marker_id), None)
            marker_3 = next((pose for pose in marker_poses if pose.marker_id == self.marker_3_id), None)
            cup_pose = self._cup_pose_in_robot_base_frame(msg, marker_3, table_marker)

            ray = self._ball_ray(best_ball)
            ball_pose = self._ball_pose_from_table_ray(msg, ray, table_marker)
            ball_pose_marker_frame = self._ball_pose_in_marker_frame(msg, ball_pose, table_marker)

            self.ball_detections_pub.publish(detections_msg(msg, ball_detections))
            self.aruco_detections_pub.publish(String(data=aruco_payload(msg, corners, ids, marker_poses)))
            if table_marker is not None:
                self.marker_4_pose_pub.publish(pose_stamped_msg(msg, table_marker.pose))
            if marker_3 is not None:
                self.marker_3_pose_pub.publish(pose_stamped_msg(msg, marker_3.pose))
            if cup_pose is not None:
                self.cup_pose_pub.publish(cup_pose)
            if ray is not None:
                self.ball_ray_pub.publish(ray_marker_msg(msg, ray, self.ray_length_m))
            if ball_pose is not None:
                self.ball_pose_pub.publish(ball_pose)
                self.ball_marker_pub.publish(ball_marker_msg(ball_pose))
            if ball_pose_marker_frame is not None:
                self.ball_pose_marker_frame_pub.publish(ball_pose_marker_frame)
            if self.annotated_pub is not None:
                self.annotated_pub.publish(
                    annotated_image_msg(
                        msg,
                        bgr,
                        ball_detections,
                        corners,
                        ids,
                        marker_poses,
                        self.camera_matrix,
                        self.dist_coeffs,
                        self.axis_length_m,
                        ray,
                        ball_pose,
                    )
                )
        except Exception as exc:
            self.get_logger().error(f"failed to process image: {exc}")

    def _detect_markers(self, bgr: np.ndarray) -> tuple[list[np.ndarray], np.ndarray | None]:
        if hasattr(cv2.aruco, "ArucoDetector"):
            corners, ids, _ = self.aruco_detector.detectMarkers(bgr)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(bgr, self.aruco_dictionary, parameters=self.aruco_parameters)
        return corners, ids

    def _estimate_marker_poses(
        self, source: Image, corners: list[np.ndarray], ids: np.ndarray | None
    ) -> list[MarkerPose]:
        if ids is None or len(ids) == 0 or self.camera_matrix is None or self.marker_length_m <= 0.0:
            return []
        coeffs = self.dist_coeffs if self.dist_coeffs is not None else np.zeros((5,), dtype=np.float64)
        if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length_m, self.camera_matrix, coeffs
            )
            return [
                MarkerPose(
                    int(marker_id),
                    np.asarray(rvec, dtype=np.float64).reshape(3),
                    np.asarray(tvec, dtype=np.float64).reshape(3),
                    pose_from_rvec_tvec(source, rvec, tvec),
                )
                for marker_id, rvec, tvec in zip(ids.flatten(), rvecs, tvecs)
            ]

        half = self.marker_length_m / 2.0
        object_points = np.asarray(
            [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
            dtype=np.float64,
        )
        flag = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE
        poses: list[MarkerPose] = []
        for marker_id, corner in zip(ids.flatten(), corners):
            image_points = np.asarray(corner, dtype=np.float64).reshape(4, 2)
            ok, rvec, tvec = cv2.solvePnP(object_points, image_points, self.camera_matrix, coeffs, flags=flag)
            if not ok:
                continue
            poses.append(
                MarkerPose(
                    int(marker_id),
                    np.asarray(rvec, dtype=np.float64).reshape(3),
                    np.asarray(tvec, dtype=np.float64).reshape(3),
                    pose_from_rvec_tvec(source, rvec, tvec),
                )
            )
        return poses

    def _detect_balls(self, bgr: np.ndarray) -> list[BallDetection]:
        detections = self._detect_yolo11_onnx(bgr)
        if self.enable_white_ball_fallback:
            detections.extend(detect_white_ball_candidates(bgr, detections))
        return sorted(detections, key=lambda detection: detection.confidence, reverse=True)

    def _detect_yolo11_onnx(self, bgr: np.ndarray) -> list[BallDetection]:
        original_h, original_w = bgr.shape[:2]
        blob, scale, pad_left, pad_top = letterbox_blob(bgr, self.imgsz)
        self.net.setInput(blob)
        predictions = normalize_yolo_output(self.net.forward())

        boxes_xywh: list[list[int]] = []
        confidences: list[float] = []
        boxes_xyxy: list[tuple[float, float, float, float]] = []
        for prediction in predictions:
            if prediction.shape[0] < 5 + SPORTS_BALL_CLASS_ID:
                continue
            class_scores = prediction[4:]
            class_id = int(np.argmax(class_scores))
            if class_id != SPORTS_BALL_CLASS_ID:
                continue
            confidence = float(class_scores[class_id])
            if confidence < self.conf:
                continue
            cx, cy, width, height = (float(v) for v in prediction[:4])
            x1 = np.clip((cx - width / 2.0 - pad_left) / scale, 0.0, float(original_w - 1))
            y1 = np.clip((cy - height / 2.0 - pad_top) / scale, 0.0, float(original_h - 1))
            x2 = np.clip((cx + width / 2.0 - pad_left) / scale, 0.0, float(original_w - 1))
            y2 = np.clip((cy + height / 2.0 - pad_top) / scale, 0.0, float(original_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_xyxy.append((float(x1), float(y1), float(x2), float(y2)))
            boxes_xywh.append([int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))])
            confidences.append(confidence)

        if not boxes_xywh:
            return []
        keep = cv2.dnn.NMSBoxes(boxes_xywh, confidences, self.conf, self.iou)
        detections: list[BallDetection] = []
        for index in np.array(keep).reshape(-1).tolist() if len(keep) else []:
            xyxy = boxes_xyxy[index]
            x1, y1, x2, y2 = xyxy
            color_score = yellow_or_white_score(bgr, xyxy) if self.use_color_tiebreak else 0.0
            detections.append(
                BallDetection(
                    centroid=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    bbox_xyxy=xyxy,
                    confidence=confidences[index],
                    class_id=SPORTS_BALL_CLASS_ID,
                    class_name=COCO_CLASS_NAMES[SPORTS_BALL_CLASS_ID],
                    color_score=color_score,
                )
            )
        return detections

    def _ball_ray(self, detection: BallDetection | None) -> np.ndarray | None:
        if detection is None or self.camera_matrix is None:
            return None
        pixel = np.asarray([[[detection.centroid[0], detection.centroid[1]]]], dtype=np.float64)
        coeffs = self.dist_coeffs if self.dist_coeffs is not None else np.zeros((5,), dtype=np.float64)
        normalized = cv2.undistortPoints(pixel, self.camera_matrix, coeffs).reshape(2)
        ray = np.asarray([normalized[0], normalized[1], 1.0], dtype=np.float64)
        norm = float(np.linalg.norm(ray))
        return ray / norm if norm > 1e-12 else None

    def _ball_pose_from_table_ray(
        self, source: Image, ray: np.ndarray | None, table_marker: MarkerPose | None
    ) -> PoseStamped | None:
        if ray is None or table_marker is None:
            return None
        rotation, _ = cv2.Rodrigues(table_marker.rvec)
        plane_normal = rotation[:, 2]
        plane_point = table_marker.tvec + plane_normal * self.ball_radius_m
        denominator = float(np.dot(plane_normal, ray))
        if abs(denominator) < 1e-9:
            return None
        distance = float(np.dot(plane_normal, plane_point) / denominator)
        if distance <= 0.0:
            return None
        point = ray * distance
        pose = PoseStamped()
        pose.header = source.header
        pose.pose.position.x = float(point[0])
        pose.pose.position.y = float(point[1])
        pose.pose.position.z = float(point[2])
        pose.pose.orientation.w = 1.0
        return pose

    def _cup_pose_in_robot_base_frame(
        self, source: Image, target: MarkerPose | None, reference: MarkerPose | None
    ) -> PoseStamped | None:
        if target is None or reference is None:
            return None
        rotation_reference_to_camera, _ = cv2.Rodrigues(reference.rvec)
        rotation_target_to_camera, _ = cv2.Rodrigues(target.rvec)
        cup_center_camera = target.tvec + rotation_target_to_camera @ self.cup_offset_marker_3_m
        cup_center_in_marker_4 = rotation_reference_to_camera.T @ (cup_center_camera - reference.tvec)
        rotation_target_to_reference = rotation_reference_to_camera.T @ rotation_target_to_camera
        cup_center_in_base = self.marker_4_in_robot_base_r @ cup_center_in_marker_4 + self.marker_4_in_robot_base_t
        rotation_target_to_base = self.marker_4_in_robot_base_r @ rotation_target_to_reference
        quaternion = quaternion_from_rotation(rotation_target_to_base)

        pose = PoseStamped()
        pose.header.stamp = source.header.stamp
        pose.header.frame_id = self.robot_arm_base_frame
        pose.pose.position.x = float(cup_center_in_base[0])
        pose.pose.position.y = float(cup_center_in_base[1])
        pose.pose.position.z = float(cup_center_in_base[2])
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        return pose

    def _ball_pose_in_marker_frame(
        self, source: Image, ball_pose: PoseStamped | None, table_marker: MarkerPose | None
    ) -> PoseStamped | None:
        if ball_pose is None or table_marker is None:
            return None
        ball_camera = np.asarray(
            [ball_pose.pose.position.x, ball_pose.pose.position.y, ball_pose.pose.position.z],
            dtype=np.float64,
        )
        rotation_marker_to_camera, _ = cv2.Rodrigues(table_marker.rvec)
        ball_marker = rotation_marker_to_camera.T @ (ball_camera - table_marker.tvec)

        pose = PoseStamped()
        pose.header.stamp = source.header.stamp
        pose.header.frame_id = f"aruco_marker_{table_marker.marker_id}"
        pose.pose.position.x = float(ball_marker[0])
        pose.pose.position.y = float(ball_marker[1])
        pose.pose.position.z = float(ball_marker[2])
        pose.pose.orientation.w = 1.0
        return pose


def image_to_bgr(msg: Image) -> np.ndarray:
    channels_by_encoding = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "8UC1": 1, "8UC3": 3}
    if msg.encoding not in channels_by_encoding:
        raise ValueError(f"unsupported image encoding: {msg.encoding}")
    channels = channels_by_encoding[msg.encoding]
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    expected = int(msg.height * msg.step)
    if raw.size < expected:
        raise ValueError(f"image data is shorter than expected: {raw.size} < {expected}")
    pixels = raw[:expected].reshape((msg.height, msg.step))[:, : msg.width * channels]
    pixels = pixels.reshape((msg.height, msg.width, channels))
    if msg.encoding in ("rgb8", "8UC3"):
        return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    if msg.encoding == "bgr8":
        return pixels.copy()
    if msg.encoding == "rgba8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
    if msg.encoding == "bgra8":
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)


def letterbox_blob(bgr: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, float, float]:
    height, width = bgr.shape[:2]
    scale = min(imgsz / width, imgsz / height)
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))
    resized = cv2.resize(bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_left = (imgsz - resized_w) / 2.0
    pad_top = (imgsz - resized_h) / 2.0
    left = int(round(pad_left - 0.1))
    top = int(round(pad_top - 0.1))
    canvas[top : top + resized_h, left : left + resized_w] = resized
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1.0 / 255.0, size=(imgsz, imgsz), swapRB=True, crop=False)
    return blob, scale, float(left), float(top)


def normalize_yolo_output(output: np.ndarray) -> np.ndarray:
    output = np.squeeze(output)
    if output.ndim != 2:
        raise ValueError(f"unexpected YOLO output shape: {output.shape}")
    return output.T if output.shape[0] < output.shape[1] else output


def yellow_or_white_score(bgr: np.ndarray, bbox_xyxy: tuple[float, float, float, float]) -> float:
    height, width = bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(left + 1, min(width, int(round(x2))))
    bottom = max(top + 1, min(height, int(round(y2))))
    crop = bgr[top:bottom, left:right].astype(np.float32) / 255.0
    if crop.size == 0:
        return 0.0
    blue, green, red = crop[..., 0], crop[..., 1], crop[..., 2]
    max_channel = crop.max(axis=-1)
    min_channel = crop.min(axis=-1)
    saturation = max_channel - min_channel
    white = (max_channel > 0.72) & (saturation < 0.22)
    yellow = (red > 0.55) & (green > 0.45) & (blue < 0.45) & ((red - blue) > 0.18)
    return float(np.mean(white | yellow))


def detect_white_ball_candidates(bgr: np.ndarray, existing_detections: list[BallDetection]) -> list[BallDetection]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 165]), np.array([180, 70, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    detections: list[BallDetection] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 180.0 or area > 950.0:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width < 14 or height < 14 or width > 45 or height > 45:
            continue
        aspect = min(width, height) / max(width, height)
        perimeter = float(cv2.arcLength(contour, True))
        circularity = 0.0 if perimeter <= 0.0 else 4.0 * np.pi * area / (perimeter * perimeter)
        if aspect < 0.70 or circularity < 0.65:
            continue
        bbox = (float(x), float(y), float(x + width), float(y + height))
        if any(iou(bbox, detection.bbox_xyxy) > 0.20 for detection in existing_detections):
            continue
        color_score = yellow_or_white_score(bgr, bbox)
        if color_score < 0.45:
            continue
        detections.append(
            BallDetection(
                centroid=(float(x + width / 2.0), float(y + height / 2.0)),
                bbox_xyxy=bbox,
                confidence=float(min(0.99, 0.35 + 0.30 * circularity + 0.30 * color_score)),
                class_id=SPORTS_BALL_CLASS_ID,
                class_name=COCO_CLASS_NAMES[SPORTS_BALL_CLASS_ID],
                color_score=color_score,
            )
        )
    return detections


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    return float(intersection / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection))


def detections_msg(source: Image, detections: list[BallDetection]) -> Detection2DArray:
    msg = Detection2DArray()
    msg.header = source.header
    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection.bbox_xyxy
        detection_msg = Detection2D()
        detection_msg.header = source.header
        detection_msg.bbox.center.position.x = float((x1 + x2) / 2.0)
        detection_msg.bbox.center.position.y = float((y1 + y2) / 2.0)
        detection_msg.bbox.size_x = float(x2 - x1)
        detection_msg.bbox.size_y = float(y2 - y1)
        detection_msg.id = f"sports_ball_{index}"
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = str(detection.class_id)
        hypothesis.hypothesis.score = float(detection.confidence)
        hypothesis.pose.pose.position.x = float(detection.centroid[0])
        hypothesis.pose.pose.position.y = float(detection.centroid[1])
        hypothesis.pose.pose.orientation.w = 1.0
        detection_msg.results.append(hypothesis)
        msg.detections.append(detection_msg)
    return msg


def ray_marker_msg(source: Image, ray: np.ndarray, ray_length_m: float) -> Marker:
    msg = Marker()
    msg.header = source.header
    msg.ns = "ball_ray"
    msg.id = 0
    msg.type = Marker.LINE_STRIP
    msg.action = Marker.ADD
    msg.pose.orientation.w = 1.0
    msg.scale.x = 0.006
    msg.color.r = 0.0
    msg.color.g = 0.8
    msg.color.b = 1.0
    msg.color.a = 1.0
    msg.points = [Point(x=0.0, y=0.0, z=0.0), Point(x=float(ray[0] * ray_length_m), y=float(ray[1] * ray_length_m), z=float(ray[2] * ray_length_m))]
    return msg


def ball_marker_msg(ball_pose: PoseStamped) -> Marker:
    msg = Marker()
    msg.header = ball_pose.header
    msg.ns = "ball_pose"
    msg.id = 0
    msg.type = Marker.SPHERE
    msg.action = Marker.ADD
    msg.pose = ball_pose.pose
    msg.pose.orientation.w = 1.0
    msg.scale.x = 0.04
    msg.scale.y = 0.04
    msg.scale.z = 0.04
    msg.color.r = 1.0
    msg.color.g = 0.1
    msg.color.b = 0.8
    msg.color.a = 1.0
    return msg


def pose_from_rvec_tvec(source: Image, rvec: np.ndarray, tvec: np.ndarray) -> PoseStamped:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    quaternion = quaternion_from_rotation(rotation)
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    pose = PoseStamped()
    pose.header = source.header
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
    return (0.0, 0.0, 0.0, 1.0) if norm < 1e-12 else (float(x / norm), float(y / norm), float(z / norm), float(w / norm))


def rotation_from_quaternion(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = quaternion
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def aruco_payload(
    source: Image, corners: list[np.ndarray], ids: np.ndarray | None, poses: list[MarkerPose]
) -> str:
    marker_ids = [] if ids is None else [int(marker_id) for marker_id in ids.flatten()]
    corners_px = [] if ids is None else [np.squeeze(corner).tolist() for corner in corners]
    pose_by_id = {pose.marker_id: pose.pose for pose in poses}
    return json.dumps(
        {
            "stamp": {"sec": source.header.stamp.sec, "nanosec": source.header.stamp.nanosec},
            "frame_id": source.header.frame_id,
            "detected": len(marker_ids) > 0,
            "marker_count": len(marker_ids),
            "markers": [
                {"id": marker_id, "corners_px": corner, "pose": pose_payload(pose_by_id.get(marker_id))}
                for marker_id, corner in zip(marker_ids, corners_px)
            ],
        }
    )


def pose_payload(pose: PoseStamped | None) -> dict[str, Any] | None:
    if pose is None:
        return None
    return {
        "position": {
            "x": pose.pose.position.x,
            "y": pose.pose.position.y,
            "z": pose.pose.position.z,
        },
        "orientation": {
            "x": pose.pose.orientation.x,
            "y": pose.pose.orientation.y,
            "z": pose.pose.orientation.z,
            "w": pose.pose.orientation.w,
        },
    }


def annotated_image_msg(
    source: Image,
    bgr: np.ndarray,
    detections: list[BallDetection],
    corners: list[np.ndarray],
    ids: np.ndarray | None,
    poses: list[MarkerPose],
    camera_matrix: np.ndarray | None,
    dist_coeffs: np.ndarray | None,
    axis_length_m: float,
    ray: np.ndarray | None,
    ball_pose: PoseStamped | None,
) -> Image:
    annotated = bgr.copy()
    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox_xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(annotated, (int(round(detection.centroid[0])), int(round(detection.centroid[1]))), 4, (0, 0, 255), -1)
        cv2.putText(annotated, f"ball {index + 1} {detection.confidence:.2f}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        if camera_matrix is not None and axis_length_m > 0.0:
            coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((5,), dtype=np.float64)
            for marker_pose in poses:
                cv2.drawFrameAxes(annotated, camera_matrix, coeffs, marker_pose.rvec, marker_pose.tvec, axis_length_m)
    if ray is not None and camera_matrix is not None:
        draw_projected_ray(annotated, ray, camera_matrix, dist_coeffs)
    if ball_pose is not None and camera_matrix is not None:
        point = np.asarray([[ball_pose.pose.position.x, ball_pose.pose.position.y, ball_pose.pose.position.z]], dtype=np.float64)
        image_points, _ = cv2.projectPoints(point, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs if dist_coeffs is not None else np.zeros((5,), dtype=np.float64))
        x, y = (int(round(v)) for v in image_points.reshape(2))
        cv2.drawMarker(annotated, (x, y), (255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

    msg = Image()
    msg.header = source.header
    msg.height = int(annotated.shape[0])
    msg.width = int(annotated.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = int(annotated.shape[1] * 3)
    msg.data = annotated.tobytes()
    return msg


def draw_projected_ray(
    annotated: np.ndarray, ray: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray | None
) -> None:
    coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((5,), dtype=np.float64)
    points = np.asarray([[ray * 0.15], [ray * 1.5]], dtype=np.float64).reshape(-1, 3)
    image_points, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), camera_matrix, coeffs)
    p1, p2 = image_points.reshape(-1, 2).astype(int)
    cv2.line(annotated, tuple(p1), tuple(p2), (255, 180, 0), 2)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
