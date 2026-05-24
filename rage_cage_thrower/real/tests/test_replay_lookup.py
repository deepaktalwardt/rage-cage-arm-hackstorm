"""Tests for the recordings-manifest helpers in rage_cage_thrower.

These exercise the pure-python helpers (no rclpy) that select the
closest recorded trajectory for a given /cup_pose. Loaded directly
from real/rage_cage_thrower.py — we don't import the whole node module
because it pulls rclpy + agx_arm_msgs (container-only).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


_REAL_DIR = Path(__file__).resolve().parents[1]
_RECORDINGS_DIR = _REAL_DIR.parent / "recordings"
_MANIFEST = _RECORDINGS_DIR / "manifest.csv"


def _import_helpers():
    """Import only the module-level helpers without dragging in rclpy."""
    # Stub out the rclpy-tainted imports so the module body can execute
    # on the Mac. We only want the module-level helpers.
    fake_modules = [
        "rclpy",
        "rclpy.callback_groups",
        "rclpy.executors",
        "rclpy.node",
        "rclpy.qos",
        "rcl_interfaces.msg",
        "geometry_msgs.msg",
        "sensor_msgs.msg",
        "std_msgs.msg",
        "std_srvs.srv",
        "agx_arm_msgs.msg",
    ]
    for name in fake_modules:
        sys.modules.setdefault(name, type(sys)(name))
    # Minimal symbols the imports try to grab off these stubs.
    sys.modules["rclpy.callback_groups"].MutuallyExclusiveCallbackGroup = object
    sys.modules["rclpy.executors"].MultiThreadedExecutor = object
    sys.modules["rclpy.node"].Node = object
    sys.modules["rclpy.qos"].QoSDurabilityPolicy = object
    sys.modules["rclpy.qos"].QoSProfile = object
    sys.modules["rcl_interfaces.msg"].ParameterDescriptor = object
    sys.modules["rcl_interfaces.msg"].ParameterType = object
    sys.modules["geometry_msgs.msg"].PoseStamped = object
    sys.modules["sensor_msgs.msg"].JointState = object
    sys.modules["std_msgs.msg"].String = object
    sys.modules["std_srvs.srv"].Empty = object
    sys.modules["std_srvs.srv"].SetBool = object
    sys.modules["std_srvs.srv"].Trigger = object
    sys.modules["agx_arm_msgs.msg"].MoveMITMsg = object

    spec = importlib.util.spec_from_file_location(
        "rage_cage_thrower_helpers",
        _REAL_DIR / "rage_cage_thrower.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def helpers():
    return _import_helpers()


def test_manifest_loads_and_converts_pedestal_to_top_z(helpers) -> None:
    """All 27 entries load; top_z = pedestal + SIM_CUP_HEIGHT (0.12)."""
    assert _MANIFEST.exists(), f"missing test fixture: {_MANIFEST}"
    entries = helpers.load_recordings_manifest(_MANIFEST)
    assert len(entries) == 27  # 3 x * 3 y * 3 z grid
    # Every entry's z column should land on {0.12, 0.195, 0.27}.
    z_values = sorted({round(e[2], 4) for e in entries})
    assert z_values == [0.12, 0.195, 0.27]


def test_closest_matches_exact_recording(helpers) -> None:
    entries = helpers.load_recordings_manifest(_MANIFEST)
    # Nominal cup at (0.85, 0.0, 0.12) = top-of-cup for pedestal=0, x=0.85, y=0.
    path, dist = helpers.find_closest_recording((0.85, 0.0, 0.12), entries)
    assert dist == pytest.approx(0.0, abs=1e-6)
    assert path.name == "cup_0.850_p0.000_z0.000.csv"


def test_closest_picks_nearest_when_inexact(helpers) -> None:
    entries = helpers.load_recordings_manifest(_MANIFEST)
    # Slightly off — should still snap to nominal grid neighbor.
    path, dist = helpers.find_closest_recording((0.84, 0.01, 0.13), entries)
    assert path.name == "cup_0.850_p0.000_z0.000.csv"
    # Distance ≈ sqrt(0.01² + 0.01² + 0.01²) ≈ 0.0173
    assert dist == pytest.approx(0.01732, abs=1e-3)


def test_closest_returns_distance_for_threshold_check(helpers) -> None:
    """The caller uses the returned distance to enforce a max threshold;
    confirm a far-away cup pose yields a distance larger than 0.10m."""
    entries = helpers.load_recordings_manifest(_MANIFEST)
    # 50cm out of plane.
    path, dist = helpers.find_closest_recording((0.85, 0.0, 0.62), entries)
    assert dist > 0.10  # caller would refuse this
    assert path.name.startswith("cup_0.850_p0.000_z")  # closest x/y on grid


def test_bad_cup_xyz_shape_raises(helpers) -> None:
    entries = helpers.load_recordings_manifest(_MANIFEST)
    with pytest.raises(ValueError, match="3-vector"):
        helpers.find_closest_recording((0.85, 0.0), entries)
