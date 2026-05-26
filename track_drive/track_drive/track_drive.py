#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# 본 프로그램은 자이트론에서 제작한 것입니다.
# 상업라이센스에 의해 제공되므로 무단배포 및 상업적 이용을 금합니다.
# 교육과 실습 용도로만 사용가능하며 외부유출은 금지됩니다.
#=============================================
#=============================================
# Xycar self-driving node
#
# 핵심:
# - LaneDetector: front image offset
# - BevLaneDetector: near_offset / far_offset / curve_offset
# - S자 구간에서는 far_offset, curve_offset을 이용해 미리 조향
# - 한 차선 고정 모드
#=============================================

import rclpy, time, cv2, os, math
import numpy as np

from rclpy.node import Node
from xycar_msgs.msg import XycarMotor
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

from track_drive.lane_detector import LaneDetector
from track_drive.bev_lane_detector import BevLaneDetector


class TrackDriverNode(Node):

    def __init__(self):

        super().__init__("driver")
        self.get_logger().info("----- Xycar self-driving node started -----")

        self.image = None
        self.motor_msg = XycarMotor()
        self.lidar_ranges = None
        self.bridge = CvBridge()

        #=============================================
        # 차선 고정
        #=============================================
        self.target_lane = 2
        self.current_lane = self.target_lane

        #=============================================
        # Detector
        #=============================================
        self._front_dets = {
            1: LaneDetector(
                target_lane=1,
                roi_top_ratio=0.52,
                use_lookahead=True,
                smoothing_window=3
            ),
            2: LaneDetector(
                target_lane=2,
                roi_top_ratio=0.52,
                use_lookahead=True,
                smoothing_window=3
            ),
        }

        self._bev_dets = {
            1: BevLaneDetector(
                target_lane=1,
                use_lookahead=True,
                smoothing_window=3
            ),
            2: BevLaneDetector(
                target_lane=2,
                use_lookahead=True,
                smoothing_window=3
            ),
        }

        #=============================================
        # Front PD gain
        #=============================================
        self.kp_f = 0.38
        self.kq_f = 0.0045
        self.kd_f = 0.14

        #=============================================
        # BEV near/far gain
        #=============================================
        # near_offset: 현재 차량 가까운 차선 중심
        # far_offset : 앞쪽 차선 중심
        # curve_offset = far - near
        #
        # S자에서는 far_offset과 curve_offset이 먼저 변하므로
        # 이 값들을 조향에 넣어 미리 꺾게 한다.
        self.kp_b_near = 0.30
        self.kp_b_far = 0.80
        self.kq_b_far = 0.0060
        self.k_curve = 0.45
        self.kd_b = 0.10

        # front / BEV fusion
        self.w_f = 0.35
        self.w_b = 0.65

        #=============================================
        # S자 감지 기준
        #=============================================
        self.s_curve_curve_th = 28.0
        self.s_curve_far_th = 45.0

        #=============================================
        # 속도
        #=============================================
        self.straight_speed = 10.0
        self.base_speed = 7.0
        self.mid_curve_speed = 4.5
        self.hard_curve_speed = 3.0
        self.s_curve_speed = 4.0

        self.straight_angle_th = 6.0
        self.straight_offset_th_f = 25.0
        self.straight_offset_th_b = 18.0

        #=============================================
        # 조향 변화량 제한
        #=============================================
        self.max_angle_step_normal = 8.0
        self.max_angle_step_s_curve = 14.0

        #=============================================
        # 내부 상태
        #=============================================
        self._prev_off_f = {1: 0.0, 2: 0.0}
        self._prev_near_b = {1: 0.0, 2: 0.0}
        self._prev_angle = 0.0
        self._prev_speed = 0.0

        # 자동 차선 변경은 현재 사용 안 함
        self._candidate_lane = None
        self._candidate_count = 0
        self._switch_confirm_frames = 4
        self._switch_margin = 0.25
        self._min_switch_score = 0.55

        #=============================================
        # ROS2 pub/sub
        #=============================================
        self.motor_pub = self.create_publisher(
            XycarMotor,
            "xycar_motor",
            10
        )

        self.sub_front = self.create_subscription(
            Image,
            "/usb_cam/image_raw/front",
            self.cam_callback,
            qos_profile_sensor_data
        )

        self.sub_lidar = self.create_subscription(
            LaserScan,
            "/scan",
            self.lidar_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info("Track Driver Node Initialized")

    #=============================================
    # callbacks
    #=============================================
    def cam_callback(self, data):

        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self._process()

    def lidar_callback(self, msg):

        self.lidar_ranges = msg.ranges

    #=============================================
    # 조향 변화량 제한
    #=============================================
    def _limit_angle_rate(self, target_angle, s_curve_mode):

        max_step = self.max_angle_step_s_curve if s_curve_mode else self.max_angle_step_normal

        diff = target_angle - self._prev_angle

        if diff > max_step:
            diff = max_step

        elif diff < -max_step:
            diff = -max_step

        return float(self._prev_angle + diff)

    #=============================================
    # 속도 계산
    #=============================================
    def _compute_speed(self, angle, r_f, r_b, f_ok, b_ok, s_curve_mode):

        abs_angle = abs(angle)

        off_f = abs(r_f["lane_center_offset"]) if f_ok else 999.0
        near_b = abs(r_b.get("near_offset", r_b["lane_center_offset"])) if b_ok else 999.0
        far_b = abs(r_b.get("far_offset", r_b["lane_center_offset"])) if b_ok else 999.0

        if s_curve_mode:
            return float(self.s_curve_speed)

        is_straight = (
            abs_angle <= self.straight_angle_th and
            (not f_ok or off_f <= self.straight_offset_th_f) and
            (not b_ok or near_b <= self.straight_offset_th_b) and
            (not b_ok or far_b <= self.straight_offset_th_b * 1.5)
        )

        if is_straight:
            return float(self.straight_speed)

        if abs_angle < 15.0:
            return float(self.base_speed)

        elif abs_angle < 32.0:
            return float(self.mid_curve_speed)

        else:
            return float(self.hard_curve_speed)

    #=============================================
    # 메인 처리
    #=============================================
    def _process(self):

        if self.image is None:
            return

        results = {}
        scores = {}

        # 1차선/2차선 detector 모두 실행
        for lane in (1, 2):

            r_f = self._front_dets[lane].detect(self.image)
            r_b = self._bev_dets[lane].detect(self.image)

            results[lane] = (r_f, r_b)
            scores[lane] = self._lane_score(r_f, r_b)

        #=============================================
        # 한 차선 고정 모드
        #=============================================
        self.current_lane = self.target_lane
        self._candidate_lane = None
        self._candidate_count = 0

        r_f, r_b = results[self.current_lane]

        f_ok = r_f["lane_detected"]
        b_ok = r_b["lane_detected"]

        if not f_ok and not b_ok:

            angle = self._prev_angle
            speed = self._prev_speed
            s_curve_mode = False

            self.get_logger().warn(
                f"[L{self.current_lane} __] lane lost — holding "
                f"angle={angle:+.1f}, speed={speed:.1f}"
            )

        else:

            raw_sum = 0.0
            total_w = 0.0

            s_curve_mode = False

            #=============================================
            # Front PD
            #=============================================
            if f_ok:

                off_f = r_f["lane_center_offset"]

                d_f = off_f - self._prev_off_f[self.current_lane]
                self._prev_off_f[self.current_lane] = off_f

                raw_f = -(
                    self.kp_f * off_f
                    + self.kq_f * off_f * abs(off_f)
                    + self.kd_f * d_f
                )

                raw_sum += raw_f * self.w_f
                total_w += self.w_f

            #=============================================
            # BEV near/far S-curve control
            #=============================================
            if b_ok:

                near_b = r_b.get("near_offset", r_b["lane_center_offset"])
                far_b = r_b.get("far_offset", r_b["lane_center_offset"])
                curve_b = r_b.get("curve_offset", far_b - near_b)

                d_near = near_b - self._prev_near_b[self.current_lane]
                self._prev_near_b[self.current_lane] = near_b

                s_curve_mode = (
                    abs(curve_b) >= self.s_curve_curve_th or
                    abs(far_b) >= self.s_curve_far_th
                )

                raw_b = -(
                    self.kp_b_near * near_b
                    + self.kp_b_far * far_b
                    + self.kq_b_far * far_b * abs(far_b)
                    + self.k_curve * curve_b
                    + self.kd_b * d_near
                )

                raw_sum += raw_b * self.w_b
                total_w += self.w_b

            #=============================================
            # angle 계산
            #=============================================
            if total_w <= 0.0:
                angle = self._prev_angle

            else:
                angle_raw = float(np.clip(raw_sum / total_w, -90.0, 90.0))

                angle = self._limit_angle_rate(
                    target_angle=angle_raw,
                    s_curve_mode=s_curve_mode
                )

                angle = float(np.clip(angle, -90.0, 90.0))

            #=============================================
            # speed 계산
            #=============================================
            speed = self._compute_speed(
                angle=angle,
                r_f=r_f,
                r_b=r_b,
                f_ok=f_ok,
                b_ok=b_ok,
                s_curve_mode=s_curve_mode
            )

            self._prev_angle = angle
            self._prev_speed = speed

            #=============================================
            # 로그
            #=============================================
            src = ("F" if f_ok else "_") + ("B" if b_ok else "_")

            f_str = f"f={r_f['lane_center_offset']:+.0f}" if f_ok else "f=X"

            if b_ok:
                b_str = (
                    f"near={r_b.get('near_offset', 0):+.0f} "
                    f"far={r_b.get('far_offset', 0):+.0f} "
                    f"curv={r_b.get('curve_offset', 0):+.0f}"
                )
            else:
                b_str = "b=X"

            mode_str = "S" if s_curve_mode else "_"

            self.get_logger().info(
                f"[L{self.current_lane} {src} {mode_str}] "
                f"angle={angle:+.1f} speed={speed:.1f} "
                f"{f_str} {b_str} "
                f"score=({scores[1]:.2f},{scores[2]:.2f})"
            )

        #=============================================
        # debug windows
        #=============================================
        dbg_f = self._front_dets[self.current_lane].draw_debug(self.image, r_f)
        dbg_b = self._bev_dets[self.current_lane].draw_debug(self.image, r_b)

        cv2.imshow("front", dbg_f)
        cv2.imshow("bev", dbg_b)
        cv2.waitKey(1)

        self.drive(angle, speed)

    #=============================================
    # lane score
    #=============================================
    def _lane_score(self, r_f, r_b):

        score = 0.0

        if r_f["lane_detected"]:
            score += self.w_f * self._source_score(r_f.get("source"))

        if r_b["lane_detected"]:
            score += self.w_b * self._source_score(r_b.get("source"))

        return float(score)

    @staticmethod
    def _source_score(source):

        return {
            "both": 1.0,
            "left+w": 0.75,
            "right+w": 0.75,
            "left_only": 0.45,
            "right_only": 0.45,
        }.get(source, 0.0)

    #=============================================
    # 자동 차선 변경 함수: 현재는 호출하지 않음
    #=============================================
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
            f"lane switch: L{old_lane} -> L{self.current_lane} "
            f"score=({scores[1]:.2f},{scores[2]:.2f})"
        )

    #=============================================
    # motor publish
    #=============================================
    def drive(self, angle, speed):

        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)

        self.motor_pub.publish(self.motor_msg)

    #=============================================
    # main loop
    #=============================================
    def main_loop(self):

        self.get_logger().info("======================================")
        self.get_logger().info("  S T A R T    D R I V I N G ...      ")
        self.get_logger().info("======================================")

        rclpy.spin(self)


def main(args=None):

    rclpy.init(args=args)
    node = TrackDriverNode()

    try:
        node.main_loop()

    except KeyboardInterrupt:
        pass

    finally:
        try:
            if rclpy.ok():
                node.drive(angle=0.0, speed=0.0)
                time.sleep(0.1)

        except Exception as e:
            print(f"Shutdown publish skipped: {e}")

        cv2.destroyAllWindows()
        node.destroy_node()

        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
