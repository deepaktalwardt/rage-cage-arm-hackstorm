"""Real-arm deployment code for the rage_cage_thrower ROS2 node.

Layer 1 modules (`controller`, `stub_env`) have no rclpy import and run
on the Mac for unit tests. Layer 2 (`rage_cage_thrower`) is the rclpy
node and only runs inside the container.

See `docs/plans/2026-05-22-ros2-inference-node-design.md`.
"""
