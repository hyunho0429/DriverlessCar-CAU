#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
import matplotlib.pyplot as plt
import numpy as np
import math

class LidarVisualizer(Node):
    def __init__(self):
        super().__init__('lidar_visualizer')

        self.ranges = None

        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)

        # Matplotlib 설정 (고정 스케일)
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)

        # 🔴 여기만 변경 (plot → scatter)
        self.lidar_points = self.ax.scatter([], [], s=5)

        # 차량 중심
        self.ax.plot(0, 0, 'ro')

        # 전방 방향 (위쪽)
        self.ax.plot([0, 0], [0, 2], 'r-')

        plt.ion()
        plt.show()

        self.create_timer(0.2, self.timer_callback)

    def lidar_callback(self, msg):
        self.ranges = msg.ranges

    def timer_callback(self):
        if self.ranges is None:
            self.get_logger().warn("No LiDAR data yet")
            return

        ranges = self.ranges

        # NaN 처리 
        valid = np.array([
            d if math.isfinite(d) else np.nan
            for d in ranges
        ])

        # 각도 보정 (0=왼쪽, 90=전방)
        angles = np.deg2rad(np.arange(len(valid)) - 90)

        x = -valid * np.cos(angles)
        y = -valid * np.sin(angles)

        indices = np.arange(len(valid))
        colors = np.full(len(valid), 'b', dtype=object)  # 기본: 파란색

        # 색상 구간 설정
        colors[(indices >= 0) & (indices < 45)] = 'r'           # 🔴 빨강
        colors[(indices >= 45) & (indices < 90)] = 'g'          # 🟢 초록
        colors[(indices >= 90) & (indices < 270)] = 'b'         # 🔵 파랑
        colors[(indices >= 270) & (indices < 315)] = 'orange'   # 🟠 주황
        colors[(indices >= 315) & (indices < 360)] = 'purple'   # 🟣 보라

        # 🔴 scatter 업데이트 (set_data → set_offsets)
        self.lidar_points.set_offsets(np.c_[x, y])
        self.lidar_points.set_color(colors)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # 전방 거리
        front_candidates = [
            d for d in ranges[85:95]
            if math.isfinite(d)
        ]

        if not front_candidates:
            return

        front = min(front_candidates)
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
    node = LidarVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()