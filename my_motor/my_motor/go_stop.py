#!/usr/bin/env python3

import rclpy
import time
from rclpy.node import Node
from xycar_msgs.msg import XycarMotor

class DriverNode(Node):
    def __init__(self):
        super().__init__('driver')
        self.motor_publisher = self.create_publisher(XycarMotor, 'xycar_motor', 1)
        self.motor_msg = XycarMotor()
        
        self.drive(angle=0,speed=0)  # 모터를 초기상태(핸들 똑바로 정지)로 설정
        time.sleep(2)
        
        self.get_logger().info('----- Xycar self-driving node started -----')

    def drive(self, angle, speed):
        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)
        self.motor_publisher.publish(self.motor_msg)

    def main_loop(self):
        while rclpy.ok():
        
            for _ in range(15):
                self.drive(angle=0,speed=0)
                time.sleep(0.1)

            for _ in range(15):
                self.drive(angle=0,speed=5)
                time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)
    node = DriverNode()

    try:
        node.main_loop()
    except KeyboardInterrupt:
        pass
    finally:
        driver_node.drive(angle=0,speed=0)
        driver_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
