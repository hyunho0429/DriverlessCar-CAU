#!/usr/bin/env python3
# -*- coding: utf-8 -*- 1
#=============================================
# 본 프로그램은 자이트론에서 제작한 것입니다.
# 상업라이센스에 의해 제공되므로 무단배포 및 상업적 이용을 금합니다.
# 교육과 실습 용도로만 사용가능하며 외부유출은 금지됩니다.
#=============================================
import rclpy, time, cv2, os, math
import numpy as np
from rclpy.node import Node
from xycar_msgs.msg import XycarMotor
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from cv_bridge import CvBridge
from track_drive.lane_detector import LaneDetector
from track_drive.bev_lane_detector import BevLaneDetector

#=============================================
# ROS2 Node 클래스 정의
#=============================================
class TrackDriverNode(Node):

    #=============================================
    # 클래스 생성 초기화 함수
    #=============================================
    def __init__(self):

        super().__init__('driver')
        self.get_logger().info('----- Xycar self-driving node started -----')

        self.image = None
        self.motor_msg = XycarMotor()
        self.lidar_ranges = None
        self.bridge = CvBridge()

        self.target_lane = 2   # 주행 차선: 1(왼쪽) or 2(오른쪽)

        # 두 detector 동시 사용: front(근거리 가중) + BEV(근거리 가중)
        self._front_det = LaneDetector(target_lane=self.target_lane, use_lookahead=False)
        self._bev_det   = BevLaneDetector(target_lane=self.target_lane, use_lookahead=False)

        # 제어 파라미터
        self.kp_f = 0.65   # front: 640px 기준 offset
        self.kd_f = 0.18
        self.kp_b = 0.80   # BEV: 400px 기준 offset
        self.kd_b = 0.20
        self.w_f  = 0.65   # 퓨전 가중치: front (코너 반응 강함)
        self.w_b  = 0.35   # 퓨전 가중치: BEV  (직선 정밀도 기여)
        self.base_speed = 8.0

        self._prev_off_f = 0.0
        self._prev_off_b = 0.0
        self._prev_angle = 0.0
        self._prev_speed = 0.0

        # ROS2 Publisher & Subscriber 설정
        self.motor_pub = self.create_publisher(XycarMotor, 'xycar_motor', 10)

        self.sub_front = self.create_subscription(
            Image, '/usb_cam/image_raw/front', self.cam_callback, qos_profile_sensor_data)

        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos_profile_sensor_data)

        self.get_logger().info("Track Driver Node Initialized")

    #=============================================
    # 카메라 토픽을 수신하는 콜백 함수
    #=============================================
    def cam_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self._process()

    def lidar_callback(self, msg):
        self.lidar_ranges = msg.ranges

    def _process(self):
        if self.image is None:
            return

        r_f = self._front_det.detect(self.image)
        r_b = self._bev_det.detect(self.image)

        f_ok = r_f['lane_detected']
        b_ok = r_b['lane_detected']

        if not f_ok and not b_ok:
            # 둘 다 실패 → 직전 명령 유지
            angle, speed = self._prev_angle, self._prev_speed
            self.get_logger().warn(
                f"[__] both lost — holding  angle={angle:+.1f}  speed={speed:.1f}"
            )
        else:
            weighted_angles = []

            if f_ok:
                off_f = r_f['lane_center_offset']
                d_f = off_f - self._prev_off_f
                self._prev_off_f = off_f
                a_f = float(np.clip(-(self.kp_f * off_f + self.kd_f * d_f), -90.0, 90.0))
                weighted_angles.append((a_f, self.w_f))

            if b_ok:
                off_b = r_b['lane_center_offset']
                d_b = off_b - self._prev_off_b
                self._prev_off_b = off_b
                a_b = float(np.clip(-(self.kp_b * off_b + self.kd_b * d_b), -90.0, 90.0))
                weighted_angles.append((a_b, self.w_b))

            # 감지된 것만으로 가중 평균
            total_w = sum(w for _, w in weighted_angles)
            angle = sum(a * w for a, w in weighted_angles) / total_w

            speed_ratio = max(0.20, 1.0 - abs(angle) / 90.0)
            speed = self.base_speed * speed_ratio
            self._prev_angle = angle
            self._prev_speed = speed

            src = ('F' if f_ok else '_') + ('B' if b_ok else '_')
            f_str = f"f={r_f['lane_center_offset']:+.0f}" if f_ok else "f=X"
            b_str = f"b={r_b['lane_center_offset']:+.0f}" if b_ok else "b=X"
            self.get_logger().info(
                f"[{src}] angle={angle:+.1f}  speed={speed:.1f}  {f_str}  {b_str}"
            )

        # 디버그 창
        dbg_f = self._front_det.draw_debug(self.image, r_f)
        dbg_b = self._bev_det.draw_debug(self.image, r_b)
        cv2.imshow("front", dbg_f)
        cv2.imshow("bev",   dbg_b)
        cv2.waitKey(1)

        self.drive(angle, speed)

    #=============================================
    # 모터제어 토픽을 발행하는 Publisher 함수
    #=============================================
    def drive(self, angle, speed):
        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)
        self.motor_pub.publish(self.motor_msg)

    #=============================================
    # 메인 루프
    #=============================================
    def main_loop(self):
        self.get_logger().info("======================================")
        self.get_logger().info("  S T A R T    D R I V I N G ...      ")
        self.get_logger().info("======================================")
        rclpy.spin(self)

#=============================================
# 메인 함수
#=============================================
def main(args=None):

    rclpy.init(args=args)
    node = TrackDriverNode()

    try:
        node.main_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.drive(angle=0, speed=0)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
