#!/usr/bin/env python3
"""Interactive control shell for the rage_cage_thrower node.

One long-lived rclpy client that talks to the node's services. Commands:

    rage> home          -> /home_arm
    rage> throw         -> /throw_trigger
    rage> replay        -> /replay_trajectory (uses replay_path ROS param)
    rage> dance         -> /dance_throw (sweep joint1, replay, return home)
    rage> open          -> /open_gripper
    rage> close         -> /close_gripper
    rage> enable        -> /enable_arm
    rage> disable       -> /disable_arm
    rage> cup X Y Z     -> publish /cup_pose (top-of-cup, arm-base frame)
    rage> status        -> service availability
    rage> exit          -> quit

Run inside the rage_cage_thrower container (rclpy is container-only):

    docker exec -it rage_cage_thrower bash
    source /opt/ros/jazzy/setup.bash
    python3 /ws/real/repl.py
"""

from __future__ import annotations

import cmd
import sys
import threading
import time
from pathlib import Path

# Ensure /ws is on sys.path so `real.controller` resolves if anything we
# transitively import needs it.
_WS = Path(__file__).resolve().parents[1]
if str(_WS) not in sys.path:
    sys.path.insert(0, str(_WS))

import rclpy  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSDurabilityPolicy, QoSProfile  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402

SERVICE_NAMES = {
    "home": "/home_arm",
    "throw": "/throw_trigger",
    "replay": "/replay_trajectory",
    "dance": "/dance_throw",
    "open": "/open_gripper",
    "close": "/close_gripper",
    "enable": "/enable_arm",
    "disable": "/disable_arm",
}


class _ReplNode(Node):
    def __init__(self) -> None:
        super().__init__("rage_cage_repl")
        # Note: don't name this `self.clients` — that's already a property
        # on rclpy.Node and shadowing it raises AttributeError.
        self.service_clients = {
            label: self.create_client(Trigger, name)
            for label, name in SERVICE_NAMES.items()
        }
        latched_qos = QoSProfile(
            depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.cup_pub = self.create_publisher(
            PoseStamped, "/cup_pose", latched_qos,
        )


class RageShell(cmd.Cmd):
    intro = (
        "rage_cage_thrower control shell. Type '?' or 'help' for commands, "
        "'exit' or Ctrl-D to quit.\n"
    )
    prompt = "rage> "

    def __init__(self, node: _ReplNode) -> None:
        super().__init__()
        self.node = node
        # Background thread spins the node so service-response futures resolve
        # while the main thread blocks on cmd input.
        self._spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        self._spin.start()

    # ---------- helpers ----------

    def _call(self, label: str, timeout: float = 20.0) -> None:
        client = self.node.service_clients[label]
        if not client.wait_for_service(timeout_sec=1.0):
            print(f"  service {SERVICE_NAMES[label]} unavailable")
            return
        future = client.call_async(Trigger.Request())
        deadline = time.time() + timeout
        while not future.done():
            if time.time() > deadline:
                print(f"  {label}: timed out after {timeout:.0f}s")
                return
            time.sleep(0.05)
        result = future.result()
        ok = "OK" if result.success else "FAIL"
        print(f"  [{ok}] {result.message}")

    # ---------- commands ----------

    def do_home(self, _arg: str) -> None:
        """Send the arm to rage_home pose. Blocks until convergence."""
        self._call("home", timeout=15.0)

    def do_throw(self, _arg: str) -> None:
        """Run the full throw cycle. Requires /cup_pose to have been set."""
        self._call("throw", timeout=20.0)

    def do_replay(self, _arg: str) -> None:
        """Open-loop replay of a recorded sim trajectory. Set the file path
        first with: `ros2 param set /rage_cage_thrower replay_path <path>`
        (relative paths resolve from /ws)."""
        self._call("replay", timeout=20.0)

    def do_dance(self, _arg: str) -> None:
        """Dance throw: home, sweep joint1 7-10s, replay (with joint1
        retargeted to a random angle from `dance_target_angles_deg`),
        then auto-home. Requires `replay_path` to be set."""
        self._call("dance", timeout=30.0)

    def do_open(self, _arg: str) -> None:
        """Open the gripper at the current arm pose."""
        self._call("open", timeout=5.0)

    def do_close(self, _arg: str) -> None:
        """Close the gripper to the HOLD position at the current arm pose."""
        self._call("close", timeout=5.0)

    def do_enable(self, _arg: str) -> None:
        """Send an enable command to the piper_ros driver."""
        self._call("enable", timeout=5.0)

    def do_disable(self, _arg: str) -> None:
        """Disable arm motors. Safe lifecycle stop; for emergencies use the
        driver's /emergency_stop service directly."""
        self._call("disable", timeout=5.0)

    def do_cup(self, arg: str) -> None:
        """cup X Y Z  — publish /cup_pose (top-of-cup point, arm-base frame).

        Example:  cup 0.85 0.0 0.12   # NOMINAL_CUP_XY at table height
        """
        parts = arg.split()
        if len(parts) != 3:
            print("  usage: cup X Y Z")
            return
        try:
            x, y, z = (float(p) for p in parts)
        except ValueError:
            print("  X Y Z must be floats")
            return
        msg = PoseStamped()
        msg.header.frame_id = "arm_base"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        self.node.cup_pub.publish(msg)
        print(f"  /cup_pose -> ({x:.3f}, {y:.3f}, {z:.3f})")

    def do_status(self, _arg: str) -> None:
        """Show whether each service is currently available."""
        for label, name in SERVICE_NAMES.items():
            ok = self.node.service_clients[label].wait_for_service(timeout_sec=0.2)
            print(f"  {name:<18} {'available' if ok else 'UNAVAILABLE'}")

    def do_exit(self, _arg: str) -> bool:
        """Exit the shell."""
        print("bye.")
        return True

    do_quit = do_exit
    do_EOF = do_exit  # Ctrl-D


def main() -> None:
    rclpy.init()
    node = _ReplNode()
    try:
        RageShell(node).cmdloop()
    except KeyboardInterrupt:
        print()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
