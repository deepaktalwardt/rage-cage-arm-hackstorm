from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="rage_cage_perception",
                executable="perception_node",
                name="perception_node",
                output="screen",
                parameters=[
                    {
                        "image_topic": "/camera/d435i/color/image_raw",
                        "camera_info_topic": "/camera/d435i/color/camera_info",
                        "table_marker_id": 4,
                        "marker_length_m": 0.04,
                    }
                ],
            )
        ]
    )
