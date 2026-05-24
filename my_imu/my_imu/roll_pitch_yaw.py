#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from rclpy.qos import qos_profile_sensor_data
import transforms3d
import math

class ImuNode(Node):
    def __init__(self):
        super().__init__('imu_print')

        self.imu_msg = None        

        self.subscription = self.create_subscription(
            Imu, '/imu', self.imu_callback, qos_profile_sensor_data)

        self.timer = self.create_timer(1.0, self.timer_callback)  

    def imu_callback(self, msg):        
        self.imu_msg = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]

    def timer_callback(self):        
        if self.imu_msg is None:
            self.get_logger().warn("No IMU data yet")
            return

        x, y, z, w = self.imu_msg

        # transforms3d는 [w, x, y, z]
        quat = [w, x, y, z]

        roll, pitch, yaw = transforms3d.euler.quat2euler(quat)

        # 라디안을 도 값으로
        roll = math.degrees(roll)
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)

        self.get_logger().info(
            f'Roll: {roll:.2f}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}'
        )

def main(args=None):
    rclpy.init(args=args)    
    node = ImuNode()
    
    try:        
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()