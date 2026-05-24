#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
import math

class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')

        self.lidar_ranges = None

        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)

        self.timer = self.create_timer(0.5, self.timer_callback)

    def lidar_callback(self, msg):
        self.lidar_ranges = msg.ranges   

    def timer_callback(self):
        if self.lidar_ranges is None:
            self.get_logger().warn("No LiDAR data yet")
            return

        ranges = self.lidar_ranges

        mid = len(ranges) // 2
        front = ranges[mid]

        if not math.isfinite(front):
            self.get_logger().warn("Front data invalid")
            return

        step = max(1, len(ranges) // 36)

        sample = [
            f"{ranges[i]:.2f}"
            for i in range(0, len(ranges), step)
            if math.isfinite(ranges[i])
        ]

        self.get_logger().info(
            f"Front: {front:.2f} m | Sample: {sample}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()