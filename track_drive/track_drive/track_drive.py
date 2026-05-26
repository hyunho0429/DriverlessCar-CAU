#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# Xycar self-driving node
#
# 핵심:
# - Camera lane following fallback
# - Lidar sparse midpoint cone following priority mode
# - 직진 구간에서는 target_y≈0이면 직진
# - 곡선 라바콘에서는 짧은 lookahead + 낮은 deadband + 강한 steer gain으로 조향 반응 증가
# - 좌우 라바콘이 모두 안 보이는 시작 초반에는 짧게 startup straight target 사용
# - 이후에는 한쪽 라바콘만 보여도 임시 중앙 target 생성
#=============================================

import rclpy
import time
import cv2
import numpy as np

from rclpy.node import Node
from xycar_msgs.msg import XycarMotor
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

from track_drive.lane_detector import LaneDetector
from track_drive.bev_lane_detector import BevLaneDetector
from track_drive.lidar_cone_controller import LidarConeController


class TrackDriverNode(Node):

    def __init__(self):

        super().__init__("driver")
        self.get_logger().info("----- Xycar self-driving node started -----")

        self.image = None
        self.scan_msg = None
        self.lidar_ranges = None

        self.motor_msg = XycarMotor()
        self.bridge = CvBridge()

        #=============================================
        # 차선 고정
        #=============================================
        self.target_lane = 2
        self.current_lane = self.target_lane

        #=============================================
        # Camera detector
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
        # Camera fallback control gain
        #=============================================
        self.kp_f = 0.38
        self.kq_f = 0.0045
        self.kd_f = 0.14

        self.kp_b_near = 0.30
        self.kp_b_far = 0.80
        self.kq_b_far = 0.0060
        self.k_curve = 0.45
        self.kd_b = 0.10

        self.w_f = 0.35
        self.w_b = 0.65

        #=============================================
        # Camera S-curve detection
        #=============================================
        self.s_curve_curve_th = 28.0
        self.s_curve_far_th = 45.0

        #=============================================
        # Camera fallback speed
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
        # Steering rate limit
        #=============================================
        self.max_angle_step_normal = 8.0

        # 곡선 라바콘에서 조향 변화가 늦지 않게 증가
        self.max_angle_step_s_curve = 35.0

        #=============================================
        # Lidar sparse midpoint cone controller
        #=============================================
        self.USE_LIDAR_CONE_DRIVE = True
        self.lidar_cone_controller = LidarConeController()

        # 라이다 컨트롤러에 넘겨줄 시작 이후 프레임 카운트
        # 원래 track_drive.py 안에 있던 상태값인데, 모듈 분리 과정에서 빠지면
        # _process()에서 AttributeError가 발생한다.
        self._startup_frame_count = 0

        #=============================================
        # Internal state
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
    # Callbacks
    #=============================================
    def cam_callback(self, data):

        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self._process()


    def lidar_callback(self, msg):

        self.scan_msg = msg
        self.lidar_ranges = msg.ranges


    #=============================================
    # Camera steering rate limit
    #=============================================
    def _limit_angle_rate(self, target_angle, s_curve_mode=False):

        max_step = self.max_angle_step_s_curve if s_curve_mode else self.max_angle_step_normal

        diff = target_angle - self._prev_angle

        if diff > max_step:
            diff = max_step

        elif diff < -max_step:
            diff = -max_step

        return float(self._prev_angle + diff)



    #=============================================
    # Camera fallback speed
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

        self._startup_frame_count += 1

        #=============================================
        # 0. Lidar sparse midpoint mode 우선 적용
        #=============================================
        if self.USE_LIDAR_CONE_DRIVE:

            lidar_result = self.lidar_cone_controller.compute_control(
                scan_msg=self.scan_msg,
                startup_frame_count=self._startup_frame_count
            )

            if lidar_result.get("debug_image") is not None:
                cv2.imshow("lidar_cones", lidar_result["debug_image"])

            if not lidar_result["valid"]:
                self.get_logger().info(
                    f"[LIDAR WAIT {lidar_result['source']}] "
                    f"pts={lidar_result['num_points']} "
                    f"L={lidar_result['num_left']} R={lidar_result['num_right']} "
                    f"startup={self._startup_frame_count}"
                )

            # valid가 충분히 연속으로 들어왔을 때만 라이다 주행 적용
            if lidar_result["valid"] and lidar_result["confirmed"]:

                angle = lidar_result["angle"]
                speed = lidar_result["speed"]
                target = lidar_result["target"]

                self._prev_angle = angle
                self._prev_speed = speed

                self.get_logger().info(
                    f"[LIDAR SPARSE {lidar_result['source']}] "
                    f"angle={angle:+.1f} speed={speed:.1f} "
                    f"target=({target[0]:.2f},{target[1]:+.2f}) "
                    f"pts={lidar_result['num_points']} "
                    f"L={lidar_result['num_left']} R={lidar_result['num_right']} "
                    f"valid_cnt={lidar_result['valid_cnt']}"
                )

                cv2.imshow("front_raw", self.image)
                cv2.waitKey(1)

                self.drive(angle, speed)
                return

        #=============================================
        # 1. Lidar target이 없으면 기존 camera lane following
        #=============================================
        results = {}
        scores = {}

        for lane in (1, 2):

            r_f = self._front_dets[lane].detect(self.image)
            r_b = self._bev_dets[lane].detect(self.image)

            results[lane] = (r_f, r_b)
            scores[lane] = self._lane_score(r_f, r_b)

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
                f"[CAM L{self.current_lane} __] lane lost — holding "
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
                    abs(curve_b) >= self.s_curve_curve_th
                    or abs(far_b) >= self.s_curve_far_th
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

                angle_raw = float(
                    np.clip(
                        raw_sum / total_w,
                        -90.0,
                        90.0
                    )
                )

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
                f"[CAM L{self.current_lane} {src} {mode_str}] "
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

        if self.lidar_cone_controller.debug_image is not None:
            cv2.imshow("lidar_cones", self.lidar_cone_controller.debug_image)

        cv2.waitKey(1)

        self.drive(angle, speed)


    #=============================================
    # Lane score
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
            other_score >= self._min_switch_score
            and other_score >= current_score + self._switch_margin
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
    # Motor publish
    #=============================================
    def drive(self, angle, speed):

        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)

        self.motor_pub.publish(self.motor_msg)


    #=============================================
    # Main loop
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