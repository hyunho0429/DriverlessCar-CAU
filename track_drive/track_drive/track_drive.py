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

        self.target_lane = 2   # 시작 주행 차선: 1(왼쪽) or 2(오른쪽)
        self.current_lane = self.target_lane

        # 1/2차선 detector를 모두 돌린 뒤, 더 신뢰도 높은 차선을 상태 기반으로 선택
        self._front_dets = {
            1: LaneDetector(target_lane=1, use_lookahead=False),
            2: LaneDetector(target_lane=2, use_lookahead=False),
        }
        self._bev_dets = {
            1: BevLaneDetector(target_lane=1, use_lookahead=False),
            2: BevLaneDetector(target_lane=2, use_lookahead=False),
        }

        # 제어 파라미터 (비선형 PD)
        # angle = -(kp*e + kq*e*|e| + kd*de)
        #   kp: 선형항 — 작은 offset에서 기본 반응
        #   kq: 2차항  — 큰 offset(코너)에서 급격히 증폭, 작은 offset에선 거의 0
        #   kd: 미분항 — 과보정 억제
        self.kp_f = 0.40   # front 선형 (640px)
        self.kq_f = 0.004  # front 2차 (offset=100 → +40 추가)
        self.kd_f = 0.18
        self.kp_b = 0.50   # BEV 선형 (400px)
        self.kq_b = 0.005  # BEV 2차 (offset=80 → +32 추가)
        self.kd_b = 0.20
        self.w_f  = 0.65   # 퓨전 가중치: front
        self.w_b  = 0.35   # 퓨전 가중치: BEV
        self.base_speed = 8.0

        self._prev_off_f = {1: 0.0, 2: 0.0}
        self._prev_off_b = {1: 0.0, 2: 0.0}
        self._prev_angle = 0.0
        self._prev_speed = 0.0
        self._candidate_lane = None
        self._candidate_count = 0
        self._switch_confirm_frames = 4
        self._switch_margin = 0.25
        self._min_switch_score = 0.55

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

        results = {}
        scores = {}
        for lane in (1, 2):
            r_f = self._front_dets[lane].detect(self.image)
            r_b = self._bev_dets[lane].detect(self.image)
            results[lane] = (r_f, r_b)
            scores[lane] = self._lane_score(r_f, r_b)

        self._update_current_lane(scores)

        r_f, r_b = results[self.current_lane]

        f_ok = r_f['lane_detected']
        b_ok = r_b['lane_detected']

        if not f_ok and not b_ok:
            # 둘 다 실패 → 직전 명령 유지
            angle, speed = self._prev_angle, self._prev_speed
            self.get_logger().warn(
                f"[L{self.current_lane} __] selected lane lost — holding  "
                f"angle={angle:+.1f}  speed={speed:.1f}  "
                f"score=({scores[1]:.2f},{scores[2]:.2f})"
            )
        else:
            raw_sum = 0.0
            total_w = 0.0

            if f_ok:
                off_f = r_f['lane_center_offset']
                d_f = off_f - self._prev_off_f[self.current_lane]
                self._prev_off_f[self.current_lane] = off_f
                raw_f = -(self.kp_f * off_f
                          + self.kq_f * off_f * abs(off_f)
                          + self.kd_f * d_f)
                raw_sum += raw_f * self.w_f
                total_w += self.w_f

            if b_ok:
                off_b = r_b['lane_center_offset']
                d_b = off_b - self._prev_off_b[self.current_lane]
                self._prev_off_b[self.current_lane] = off_b
                raw_b = -(self.kp_b * off_b
                          + self.kq_b * off_b * abs(off_b)
                          + self.kd_b * d_b)
                raw_sum += raw_b * self.w_b
                total_w += self.w_b

            # 합산 후 한 번만 clip
            angle = float(np.clip(raw_sum / total_w, -90.0, 90.0))

            speed_ratio = max(0.20, 1.0 - abs(angle) / 90.0)
            speed = self.base_speed * speed_ratio
            self._prev_angle = angle
            self._prev_speed = speed

            src = ('F' if f_ok else '_') + ('B' if b_ok else '_')
            f_str = f"f={r_f['lane_center_offset']:+.0f}" if f_ok else "f=X"
            b_str = f"b={r_b['lane_center_offset']:+.0f}" if b_ok else "b=X"
            self.get_logger().info(
                f"[L{self.current_lane} {src}] angle={angle:+.1f}  speed={speed:.1f}  "
                f"{f_str}  {b_str}  score=({scores[1]:.2f},{scores[2]:.2f})"
            )

        # 디버그 창
        dbg_f = self._front_dets[self.current_lane].draw_debug(self.image, r_f)
        dbg_b = self._bev_dets[self.current_lane].draw_debug(self.image, r_b)
        cv2.imshow("front", dbg_f)
        cv2.imshow("bev",   dbg_b)
        cv2.waitKey(1)

        self.drive(angle, speed)

    def _lane_score(self, r_f, r_b):
        score = 0.0
        if r_f['lane_detected']:
            score += self.w_f * self._source_score(r_f.get('source'))
        if r_b['lane_detected']:
            score += self.w_b * self._source_score(r_b.get('source'))
        return float(score)

    @staticmethod
    def _source_score(source):
        return {
            'both': 1.0,
            'left+w': 0.75,
            'right+w': 0.75,
            'left_only': 0.45,
            'right_only': 0.45,
        }.get(source, 0.0)

    def _update_current_lane(self, scores):
        other_lane = 1 if self.current_lane == 2 else 2
        current_score = scores[self.current_lane]
        other_score = scores[other_lane]

        should_consider = (
            other_score >= self._min_switch_score and
            other_score >= current_score + self._switch_margin
        )

        if not should_consider:
            self._candidate_lane = None
            self._candidate_count = 0
            return

        if self._candidate_lane != other_lane:
            self._candidate_lane = other_lane
            self._candidate_count = 1
            return

        self._candidate_count += 1
        if self._candidate_count < self._switch_confirm_frames:
            return

        old_lane = self.current_lane
        self.current_lane = other_lane
        self._candidate_lane = None
        self._candidate_count = 0
        self.get_logger().warn(
            f"lane switch: L{old_lane} -> L{self.current_lane}  "
            f"score=({scores[1]:.2f},{scores[2]:.2f})"
        )

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
