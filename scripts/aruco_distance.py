#!/usr/bin/env python3
"""Print distance between two detected ArUco marker poses."""

from __future__ import annotations

import argparse
import json
from typing import Any, NamedTuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MarkerPose(NamedTuple):
    position: np.ndarray
    orientation: np.ndarray
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    transform: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/aruco/detections", help="Detection JSON topic.")
    parser.add_argument("--origin-id", type=int, default=3, help="Origin marker ID.")
    parser.add_argument("--target-id", type=int, default=4, help="Target marker ID.")
    parser.add_argument("--once", action=argparse.BooleanOptionalAction, default=False, help="Exit after first match.")
    parser.add_argument("--samples", type=int, default=0, help="Average this many valid detections, then exit.")
    return parser.parse_args()


class ArucoDistanceNode(Node):
    def __init__(self, topic: str, id_a: int, id_b: int, once: bool, samples: int) -> None:
        super().__init__("aruco_distance")
        self.origin_id = id_a
        self.target_id = id_b
        self.once = once
        self.samples = samples
        self.origin_frame_deltas: list[np.ndarray] = []
        self.normal_angles_deg: list[float] = []
        self.subscription = self.create_subscription(String, topic, self._on_detection, 10)
        self.get_logger().info(f"waiting for origin ID {id_a} and target ID {id_b} on {topic}")

    def _on_detection(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            poses = marker_poses(data)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warn(f"invalid detection payload: {exc}")
            return

        missing = [marker_id for marker_id in (self.origin_id, self.target_id) if marker_id not in poses]
        if missing:
            print(f"missing ids={missing}; visible ids={sorted(poses)}")
            return

        origin = poses[self.origin_id].position
        target = poses[self.target_id].position
        delta = target - origin
        origin_rotation = rotation_matrix_from_quaternion(poses[self.origin_id].orientation)
        target_rotation = rotation_matrix_from_quaternion(poses[self.target_id].orientation)
        origin_frame_delta = origin_rotation.T @ delta
        normal_angle_deg = marker_normal_angle_deg(origin_rotation, target_rotation)
        origin_transform = pose_transform(poses[self.origin_id])
        target_transform = pose_transform(poses[self.target_id])
        relative_transform = relative_marker_transform(origin_transform, target_transform)
        distance_m = float(np.linalg.norm(delta))
        frame_id = data.get("frame_id", "")
        print(
            f"origin id{self.origin_id} -> target id{self.target_id} "
            f"distance_m={distance_m:.4f} frame={frame_id} "
            f"delta_m={round_point(delta)} "
            f"delta_in_id{self.origin_id}_frame_m={round_point(origin_frame_delta)} "
            f"marker_normal_angle_deg={normal_angle_deg:.2f} "
            f"origin={round_point(origin)} target={round_point(target)} "
            f"origin_rvec={round_optional_point(poses[self.origin_id].rvec)} "
            f"origin_tvec_m={round_optional_point(poses[self.origin_id].tvec)} "
            f"target_rvec={round_optional_point(poses[self.target_id].rvec)} "
            f"target_tvec_m={round_optional_point(poses[self.target_id].tvec)}"
        )
        print(f"origin_transform_4x4 id{self.origin_id} camera_from_marker:")
        print(format_optional_matrix(origin_transform))
        print(f"target_transform_4x4 id{self.target_id} camera_from_marker:")
        print(format_optional_matrix(target_transform))
        print(f"relative_transform_4x4 from id{self.origin_id} to id{self.target_id}:")
        print(format_optional_matrix(relative_transform))

        if self.samples > 0:
            self.origin_frame_deltas.append(origin_frame_delta)
            self.normal_angles_deg.append(normal_angle_deg)
            if len(self.origin_frame_deltas) >= self.samples:
                print_sample_summary(self.origin_id, self.samples, self.origin_frame_deltas, self.normal_angles_deg)
                rclpy.shutdown()
        elif self.once:
            rclpy.shutdown()


def marker_poses(data: dict[str, Any]) -> dict[int, MarkerPose]:
    poses = {}
    for marker in data.get("markers", []):
        marker_id = int(marker["id"])
        pose_stamped = marker.get("pose_stamped")
        if pose_stamped is None:
            continue
        position = pose_stamped["pose"]["position"]
        orientation = pose_stamped["pose"]["orientation"]
        poses[marker_id] = MarkerPose(
            position=np.array([position["x"], position["y"], position["z"]], dtype=np.float64),
            orientation=np.array(
                [orientation["x"], orientation["y"], orientation["z"], orientation["w"]],
                dtype=np.float64,
            ),
            rvec=optional_array(marker.get("rvec")),
            tvec=optional_array(marker.get("tvec_m")),
            transform=optional_matrix(marker.get("transform_4x4")),
        )
    return poses


def round_point(point: np.ndarray) -> tuple[float, ...]:
    return tuple(round(float(value), 4) for value in point)


def round_optional_point(point: np.ndarray | None) -> tuple[float, ...] | None:
    if point is None:
        return None
    return round_point(point)


def optional_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def format_optional_matrix(matrix: np.ndarray | None) -> str:
    if matrix is None:
        return "  None"
    return "\n".join(
        "  [" + " ".join(f"{value: .4f}" for value in row) + "]"
        for row in matrix
    )


def pose_transform(pose: MarkerPose) -> np.ndarray:
    if pose.transform is not None:
        return pose.transform
    if pose.rvec is None or pose.tvec is None:
        rotation = rotation_matrix_from_quaternion(pose.orientation)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = pose.position
        return transform

    rotation, _ = cv2.Rodrigues(pose.rvec.reshape(3))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = pose.tvec.reshape(3)
    return transform


def relative_marker_transform(
    origin_transform: np.ndarray | None,
    target_transform: np.ndarray | None,
) -> np.ndarray | None:
    if origin_transform is None or target_transform is None:
        return None
    return np.linalg.inv(origin_transform) @ target_transform


def optional_matrix(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def marker_normal_angle_deg(
    origin_rotation: np.ndarray,
    target_rotation: np.ndarray,
) -> float:
    origin_normal = origin_rotation[:, 2]
    target_normal = target_rotation[:, 2]
    dot = float(np.dot(origin_normal, target_normal))
    dot = float(np.clip(abs(dot), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def rotation_matrix_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)

    x, y, z, w = quaternion / norm

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def print_sample_summary(
    origin_id: int,
    sample_count: int,
    origin_frame_deltas: list[np.ndarray],
    normal_angles_deg: list[float],
) -> None:
    deltas = np.vstack(origin_frame_deltas)
    normal_angles = np.asarray(normal_angles_deg, dtype=np.float64)
    print(
        f"summary samples={sample_count} "
        f"mean_delta_in_id{origin_id}_frame_m={round_point(deltas.mean(axis=0))} "
        f"std_delta_in_id{origin_id}_frame_m={round_point(deltas.std(axis=0))} "
        f"mean_marker_normal_angle_deg={normal_angles.mean():.2f} "
        f"std_marker_normal_angle_deg={normal_angles.std():.2f}"
    )


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = ArucoDistanceNode(args.topic, args.origin_id, args.target_id, args.once, max(0, args.samples))
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
