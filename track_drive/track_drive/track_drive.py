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
        
        # 상수값 및 초기값 설정
        self.image = None  # 카메라 토픽 데이터를 저장할 변수
        self.motor_msg = XycarMotor()  # 모터토픽 메시지
        self.lidar_ranges = None
        self.bridge = CvBridge()

        self.target_lane = 2   # 주행 차선: 1(왼쪽) or 2(오른쪽)
        # 'lookahead' = 전면뷰 + 먼곳 가중치
        # 'bev'       = BEV + 가까운곳 가중치
        # 'both'      = BEV + 먼곳 가중치
        self.detector_mode = 'lookahead'
        self.lane_detector = self._make_detector(self.detector_mode, self.target_lane)

        # 제어 파라미터 — 시뮬레이터 결과에 따라 조정
        # lookahead: 640px 기준 offset, bev/both: 400px 기준 offset (스케일 보정됨)
        _kp_table = {'lookahead': 0.50, 'bev': 0.80, 'both': 0.80}
        _kd_table = {'lookahead': 0.12, 'bev': 0.20, 'both': 0.20}
        self.kp = _kp_table[self.detector_mode]
        self.kd = _kd_table[self.detector_mode]
        self.base_speed = 8.0  # 직선 기본 속도
        self._prev_offset = 0.0
        
        # ROS2 Publisher & Subscriber 설정
        self.motor_pub = self.create_publisher(XycarMotor,'xycar_motor',10)
        
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

        result = self.lane_detector.detect(self.image)
        offset = result['lane_center_offset']

        # PD 제어: D항이 offset 변화 속도를 감지해 과보정 억제
        d_offset = offset - self._prev_offset
        self._prev_offset = offset
        # Xycar angle 범위: ±100
        angle = float(np.clip(-(self.kp * offset + self.kd * d_offset), -90.0, 90.0))

        # 조향각 기반 속도 감소: angle이 클수록(=코너) 속도 감소, 최소 20%
        speed_ratio = max(0.20, 1.0 - abs(angle) / 90.0)
        speed = self.base_speed * speed_ratio

        self.get_logger().info(
            f"offset={offset:+.1f}px  angle={angle:+.1f}  speed={speed:.1f}"
            f"  {self.lane_detector.last_elapsed_ms:.1f}ms"
            f"  lane={result['current_lane']}  warn={result['solid_line_warning']}"
        )

        dbg = self.lane_detector.draw_debug(self.image, result)
        cv2.imshow("track_drive", dbg)
        cv2.waitKey(1)

        self.drive(angle, speed)
      
    @staticmethod
    def _make_detector(mode, target_lane):
        if mode == 'lookahead':
            return LaneDetector(target_lane=target_lane, use_lookahead=True)
        if mode == 'bev':
            return BevLaneDetector(target_lane=target_lane, use_lookahead=False)
        if mode == 'both':
            return BevLaneDetector(target_lane=target_lane, use_lookahead=True)
        raise ValueError(f"unknown detector_mode: {mode}")

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
        # main_loop() 함수를 호출하여 실행합니다.
        node.main_loop()
    except KeyboardInterrupt:
        # 사용자 인터럽트 (Ctrl+C)가 발생하면 예외를 처리합니다.
        pass
    finally:
        # 노드를 종료하고 ROS2를 정리합니다.
        node.drive(angle=0, speed=0)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

