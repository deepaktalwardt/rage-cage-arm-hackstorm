#!/usr/bin/env python3
"""Detect ArUco markers on a table and estimate the table plane.

The script can run on a still image or a live camera. With calibrated camera
intrinsics and a known marker side length, it estimates each marker pose and
fits a plane through all detected marker corners in the camera frame.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

if not hasattr(cv2, "aruco"):
    raise RuntimeError("OpenCV was installed without aruco support. Install opencv-contrib-python.")

ARUCO_DICTS = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco)
    if name.startswith("DICT_")
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Image file to process.")
    source.add_argument("--camera", type=int, help="Camera index to stream from, e.g. 0.")
    source.add_argument("--rosbridge-url", help="rosbridge WebSocket URL, e.g. ws://orin-desktop.local:8765.")
    parser.add_argument(
        "--ros-topic",
        default="/camera/d435i/color/image_raw",
        help="ROS sensor_msgs/Image topic to subscribe to when using --rosbridge-url.",
    )
    parser.add_argument(
        "--ros-image-type",
        default="sensor_msgs/msg/Image",
        help="ROS image message type for rosbridge subscription.",
    )
    parser.add_argument(
        "--ros-throttle-ms",
        type=int,
        default=100,
        help="rosbridge subscription throttle in milliseconds.",
    )
    parser.add_argument("--marker-length-m", type=float, required=True, help="Printed marker side length in meters.")
    parser.add_argument("--dict", choices=sorted(ARUCO_DICTS), default="DICT_4X4_50", help="ArUco dictionary.")
    parser.add_argument(
        "--calibration",
        type=Path,
        help=(
            "Camera calibration file. Supports .npz with camera_matrix/dist_coeffs, "
            ".json with the same keys, or OpenCV .yml/.yaml."
        ),
    )
    parser.add_argument(
        "--camera-matrix",
        type=float,
        nargs=4,
        metavar=("FX", "FY", "CX", "CY"),
        help="Inline pinhole intrinsics. Example: --camera-matrix 615 615 320 240",
    )
    parser.add_argument(
        "--dist-coeffs",
        type=float,
        nargs="*",
        default=None,
        help="Optional distortion coefficients used with --camera-matrix.",
    )
    parser.add_argument("--output-json", type=Path, help="Write latest detection result as JSON.")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True, help="Show annotated image.")
    parser.add_argument("--save-annotated", type=Path, help="Optional path to save the annotated image.")
    parser.add_argument("--axis-scale", type=float, default=1.5, help="Axis length as a fraction of marker side.")
    parser.add_argument("--reference-id", type=int, help="Marker ID to use as the reference frame.")
    parser.add_argument("--target-id", type=int, help="Marker ID to express in the reference marker frame.")
    parser.add_argument("--print-every", type=int, default=10, help="For camera mode, print every N frames.")
    return parser.parse_args()


def load_calibration(args: argparse.Namespace) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if args.camera_matrix is not None:
        fx, fy, cx, cy = args.camera_matrix
        camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist_coeffs = np.asarray(args.dist_coeffs or [0, 0, 0, 0, 0], dtype=np.float64)
        return camera_matrix, dist_coeffs

    if args.calibration is None:
        return None, None

    path = args.calibration
    if path.suffix == ".npz":
        data = np.load(path)
        return np.asarray(data["camera_matrix"], dtype=np.float64), np.asarray(data["dist_coeffs"], dtype=np.float64)

    if path.suffix == ".json":
        with path.open() as f:
            data: dict[str, Any] = json.load(f)
        return (
            np.asarray(data["camera_matrix"], dtype=np.float64),
            np.asarray(data.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64),
        )

    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f"could not open calibration file: {path}")
    camera_matrix = storage.getNode("camera_matrix").mat()
    dist_coeffs = storage.getNode("dist_coeffs").mat()
    storage.release()
    if camera_matrix is None:
        raise ValueError(f"missing camera_matrix in calibration file: {path}")
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    return np.asarray(camera_matrix, dtype=np.float64), np.asarray(dist_coeffs, dtype=np.float64)


def make_detector(dictionary_name: str) -> Tuple[Any, Any]:
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary_name])
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters) if hasattr(cv2.aruco, "ArucoDetector") else None
    return detector, dictionary


def detect_markers(frame: np.ndarray, detector: Any, dictionary: Any) -> Tuple[list[np.ndarray], Optional[np.ndarray]]:
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(frame)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary)
    return corners, ids


def decode_ros_image(msg: dict[str, Any]) -> np.ndarray:
    height = int(msg["height"])
    width = int(msg["width"])
    step = int(msg["step"])
    encoding = str(msg["encoding"]).lower()
    data = msg["data"]
    if isinstance(data, str):
        raw = base64.b64decode(data)
    else:
        raw = bytes(data)

    if encoding in ("bgr8", "rgb8", "8uc3"):
        channels = 3
        dtype = np.uint8
    elif encoding in ("bgra8", "rgba8", "8uc4"):
        channels = 4
        dtype = np.uint8
    elif encoding in ("mono8", "8uc1"):
        channels = 1
        dtype = np.uint8
    elif encoding in ("mono16", "16uc1"):
        channels = 1
        dtype = np.uint16
    else:
        raise ValueError(f"unsupported ROS image encoding: {msg['encoding']}")

    row = np.frombuffer(raw, dtype=np.uint8).reshape(height, step)
    pixel_bytes = np.dtype(dtype).itemsize * channels
    image = row[:, : width * pixel_bytes].reshape(height, width, channels, np.dtype(dtype).itemsize)
    image = image.view(dtype).reshape(height, width, channels)

    if encoding in ("rgb8", "8uc3"):
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding in ("mono8", "8uc1"):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if encoding in ("mono16", "16uc1"):
        mono8 = cv2.convertScaleAbs(image, alpha=255.0 / max(float(image.max()), 1.0))
        return cv2.cvtColor(mono8, cv2.COLOR_GRAY2BGR)
    return image


def rosbridge_image_frames(url: str, topic: str, image_type: str, throttle_ms: int):
    try:
        import websocket
    except ImportError as exc:
        raise RuntimeError("Install websocket-client to use --rosbridge-url.") from exc

    ws = websocket.create_connection(url)
    subscribe = {
        "op": "subscribe",
        "topic": topic,
        "type": image_type,
        "queue_length": 1,
        "throttle_rate": max(throttle_ms, 0),
    }
    ws.send(json.dumps(subscribe))
    try:
        while True:
            payload = json.loads(ws.recv())
            if payload.get("op") != "publish" or payload.get("topic") != topic:
                continue
            yield decode_ros_image(payload["msg"])
    finally:
        try:
            ws.send(json.dumps({"op": "unsubscribe", "topic": topic}))
        finally:
            ws.close()


def marker_object_corners(marker_length_m: float) -> np.ndarray:
    half = marker_length_m / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def fit_plane(points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)

    # Orient the normal toward the camera origin.
    if float(np.dot(normal, centroid)) > 0.0:
        normal = -normal
    d = -float(np.dot(normal, centroid))
    residuals = np.abs(points @ normal + d)
    return normal, d, float(residuals.mean())


def orient_normal_toward_camera(normal: np.ndarray, point_on_plane: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    point_on_plane = np.asarray(point_on_plane, dtype=np.float64).reshape(3)
    if float(np.dot(normal, point_on_plane)) > 0.0:
        return -normal
    return normal


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return float("nan")
    cosine = float(np.dot(a, b) / denom)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def rpy_from_rotation(rotation: np.ndarray) -> list[float]:
    sy = float(np.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0]))
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = 0.0
    return [float(roll), float(pitch), float(yaw)]


def draw_marker_overlay(
    image: np.ndarray,
    corner: np.ndarray,
    marker_id: int,
    axis_scale: float,
) -> None:
    points = np.squeeze(corner).astype(np.int32)
    center = points.mean(axis=0).astype(np.int32)

    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 255), thickness=5, lineType=cv2.LINE_AA)
    for idx, point in enumerate(points):
        cv2.circle(image, tuple(point), 10, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(image, tuple(point), 10, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            image,
            str(idx),
            tuple(point + np.array([12, -12])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(idx),
            tuple(point + np.array([12, -12])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    x_dir = ((points[1] + points[2]) * 0.5 - center).astype(np.float64)
    y_dir = ((points[0] + points[1]) * 0.5 - center).astype(np.float64)
    marker_side_px = float(
        np.mean(
            [
                np.linalg.norm(points[1] - points[0]),
                np.linalg.norm(points[2] - points[1]),
                np.linalg.norm(points[3] - points[2]),
                np.linalg.norm(points[0] - points[3]),
            ]
        )
    )

    for label, direction, color in (("x", x_dir, (0, 0, 255)), ("y", y_dir, (0, 255, 0))):
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        end = center + (direction / norm * marker_side_px * axis_scale).astype(np.int32)
        cv2.arrowedLine(image, tuple(center), tuple(end), color, thickness=6, tipLength=0.2, line_type=cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            tuple(end + np.array([8, 8])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            4,
            cv2.LINE_AA,
        )

    label_origin = tuple(center + np.array([16, -16]))
    cv2.putText(image, f"id {marker_id}", label_origin, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 7, cv2.LINE_AA)
    cv2.putText(image, f"id {marker_id}", label_origin, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)


def estimate(
    frame: np.ndarray,
    detector: Any,
    dictionary: Any,
    marker_length_m: float,
    camera_matrix: Optional[np.ndarray],
    dist_coeffs: Optional[np.ndarray],
    axis_scale: float,
    reference_id: Optional[int],
    target_id: Optional[int],
) -> Tuple[np.ndarray, dict[str, Any]]:
    corners, ids = detect_markers(frame, detector, dictionary)
    result: dict[str, Any] = {
        "marker_count": 0 if ids is None else int(len(ids)),
        "markers": [],
        "table_plane_camera": None,
        "table_plane_reference": None,
        "relative_pose": None,
    }
    annotated = frame.copy()

    if ids is None or len(ids) == 0:
        return annotated, result

    result["markers"] = [{"id": int(marker_id), "corners_px": np.squeeze(corner).tolist()} for marker_id, corner in zip(ids.flatten(), corners)]
    for marker_id, corner in zip(ids.flatten(), corners):
        draw_marker_overlay(annotated, corner, int(marker_id), axis_scale)

    if camera_matrix is None:
        return annotated, result

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length_m, camera_matrix, dist_coeffs)
    object_corners = marker_object_corners(marker_length_m)
    all_corner_points = []
    marker_poses: dict[int, np.ndarray] = {}
    for marker_result, rvec, tvec in zip(result["markers"], rvecs, tvecs):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        marker_pose = transform_from_rvec_tvec(rvec, tvec)
        marker_poses[int(marker_result["id"])] = marker_pose
        rotation = marker_pose[:3, :3]
        corner_points = (rotation @ object_corners.T).T + tvec
        all_corner_points.append(corner_points)
        marker_normal = orient_normal_toward_camera(rotation[:, 2], tvec)

        cv2.drawFrameAxes(annotated, camera_matrix, dist_coeffs, rvec, tvec, marker_length_m * axis_scale)
        marker_result["rvec"] = rvec.tolist()
        marker_result["tvec_m"] = tvec.tolist()
        marker_result["normal_camera"] = marker_normal.tolist()

    points = np.concatenate(all_corner_points, axis=0)
    normal, d, mean_abs_error = fit_plane(points)
    result["table_plane_camera"] = {
        "equation": "normal_x*x + normal_y*y + normal_z*z + d = 0",
        "normal": normal.tolist(),
        "d_m": d,
        "mean_abs_error_m": mean_abs_error,
        "points_used": int(len(points)),
    }
    for marker_result in result["markers"]:
        if "normal_camera" in marker_result:
            marker_result["angle_to_table_normal_deg"] = angle_between_deg(marker_result["normal_camera"], normal)

    if reference_id is not None and reference_id in marker_poses:
        camera_from_reference = marker_poses[reference_id]
        reference_rotation_from_camera = camera_from_reference[:3, :3].T
        normal_reference = reference_rotation_from_camera @ normal
        d_reference = float(np.dot(normal, camera_from_reference[:3, 3]) + d)
        result["table_plane_reference"] = {
            "reference_id": reference_id,
            "equation": "normal_x*x + normal_y*y + normal_z*z + d = 0",
            "meaning": "table plane expressed in the reference marker frame",
            "normal": normal_reference.tolist(),
            "d_m": d_reference,
        }

    if reference_id is not None or target_id is not None:
        if reference_id is None or target_id is None:
            raise ValueError("Use --reference-id and --target-id together.")
        if reference_id in marker_poses and target_id in marker_poses:
            reference_from_camera = np.linalg.inv(marker_poses[reference_id])
            reference_from_target = reference_from_camera @ marker_poses[target_id]
            translation = reference_from_target[:3, 3]
            result["relative_pose"] = {
                "reference_id": reference_id,
                "target_id": target_id,
                "meaning": "target marker pose expressed in reference marker frame",
                "translation_m": translation.tolist(),
                "distance_m": float(np.linalg.norm(translation)),
                "rpy_rad": rpy_from_rotation(reference_from_target[:3, :3]),
                "transform_4x4": reference_from_target.tolist(),
            }
            text = f"id{target_id} in id{reference_id}: {translation[0]:+.3f}, {translation[1]:+.3f}, {translation[2]:+.3f} m"
            cv2.putText(annotated, text, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.putText(annotated, text, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
        else:
            result["relative_pose"] = {
                "reference_id": reference_id,
                "target_id": target_id,
                "error": "one or both marker IDs were not detected",
                "detected_ids": sorted(marker_poses),
            }
    return annotated, result


def print_result(result: dict[str, Any]) -> None:
    print(f"markers={result['marker_count']}")
    for marker in result["markers"]:
        if "tvec_m" in marker:
            print(
                f"  id={marker['id']} "
                f"tvec_m={np.round(marker['tvec_m'], 4).tolist()} "
                f"normal_camera={np.round(marker['normal_camera'], 5).tolist()} "
                f"angle_to_table_normal_deg={marker['angle_to_table_normal_deg']:.2f}"
            )
        else:
            print(f"  id={marker['id']} corners_px={np.round(marker['corners_px'], 1).tolist()}")
    plane = result.get("table_plane_camera")
    if plane is not None:
        print(
            "  table_plane_camera "
            f"normal={np.round(plane['normal'], 5).tolist()} "
            f"d_m={plane['d_m']:.5f} "
            f"mean_abs_error_m={plane['mean_abs_error_m']:.5f}"
        )
    reference_plane = result.get("table_plane_reference")
    if reference_plane is not None:
        print(
            f"  table_plane_in_id{reference_plane['reference_id']} "
            f"normal={np.round(reference_plane['normal'], 5).tolist()} "
            f"d_m={reference_plane['d_m']:.5f}"
        )
    relative_pose = result.get("relative_pose")
    if relative_pose is not None:
        if "error" in relative_pose:
            print(
                f"  relative_pose id{relative_pose['target_id']} in id{relative_pose['reference_id']}: "
                f"{relative_pose['error']} detected_ids={relative_pose['detected_ids']}"
            )
        else:
            print(
                f"  relative_pose id{relative_pose['target_id']} in id{relative_pose['reference_id']} "
                f"translation_m={np.round(relative_pose['translation_m'], 4).tolist()} "
                f"distance_m={relative_pose['distance_m']:.4f} "
                f"rpy_rad={np.round(relative_pose['rpy_rad'], 4).tolist()}"
            )


def maybe_write_json(path: Optional[Path], result: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


def maybe_save_annotated(path: Optional[Path], annotated: np.ndarray) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), annotated):
        raise RuntimeError(f"failed to write annotated image: {path}")


def run_frame_stream(
    frames: Any,
    args: argparse.Namespace,
    detector: Any,
    dictionary: Any,
    camera_matrix: Optional[np.ndarray],
    dist_coeffs: Optional[np.ndarray],
) -> None:
    frame_idx = 0
    for frame in frames:
        annotated, result = estimate(
            frame,
            detector,
            dictionary,
            args.marker_length_m,
            camera_matrix,
            dist_coeffs,
            args.axis_scale,
            args.reference_id,
            args.target_id,
        )
        if frame_idx % max(args.print_every, 1) == 0:
            print_result(result)
            maybe_write_json(args.output_json, result)
            maybe_save_annotated(args.save_annotated, annotated)
        frame_idx += 1

        if args.show:
            cv2.imshow("aruco table detection", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break


def main() -> None:
    args = parse_args()
    detector, dictionary = make_detector(args.dict)
    camera_matrix, dist_coeffs = load_calibration(args)
    if camera_matrix is None:
        print("No camera intrinsics provided; reporting pixel detections only, not 3D table plane.")

    if args.image is not None:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise FileNotFoundError(f"could not read image: {args.image}")
        annotated, result = estimate(
            frame,
            detector,
            dictionary,
            args.marker_length_m,
            camera_matrix,
            dist_coeffs,
            args.axis_scale,
            args.reference_id,
            args.target_id,
        )
        print_result(result)
        maybe_write_json(args.output_json, result)
        maybe_save_annotated(args.save_annotated, annotated)
        if args.show:
            cv2.imshow("aruco table detection", annotated)
            cv2.waitKey(0)
        return

    if args.rosbridge_url is not None:
        frames = rosbridge_image_frames(args.rosbridge_url, args.ros_topic, args.ros_image_type, args.ros_throttle_ms)
        try:
            run_frame_stream(frames, args, detector, dictionary, camera_matrix, dist_coeffs)
        finally:
            cv2.destroyAllWindows()
        return

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"could not open camera index {args.camera}")

    def camera_frames():
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera read failed")
            yield frame

    try:
        run_frame_stream(camera_frames(), args, detector, dictionary, camera_matrix, dist_coeffs)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
