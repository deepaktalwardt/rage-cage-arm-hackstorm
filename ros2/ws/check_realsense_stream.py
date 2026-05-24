import time

import rclpy
from sensor_msgs.msg import Image


TOPICS = {
    "color": "/camera/d435i/color/image_raw",
    "depth": "/camera/d435i/depth/image_rect_raw",
}


def main():
    rclpy.init()
    node = rclpy.create_node("realsense_stream_check")
    counts = {name: 0 for name in TOPICS}
    last = {}

    def make_callback(name):
        def callback(msg):
            counts[name] += 1
            last[name] = {
                "width": msg.width,
                "height": msg.height,
                "encoding": msg.encoding,
            }

        return callback

    for name, topic in TOPICS.items():
        node.create_subscription(Image, topic, make_callback(name), 10)

    start = time.time()
    while time.time() - start < 6.0:
        rclpy.spin_once(node, timeout_sec=0.2)

    for name in TOPICS:
        print(f"{name}: frames={counts[name]} last={last.get(name)}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
