#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# Lidar cone controller module
#
# track_drive.py에서 라바콘/라이다 주행 관련 로직을 분리한 파일.
# - /scan LaserScan -> 전방 ROI 점 필터링
# - 좌/우 라바콘 후보 분리
# - sparse midpoint target 생성
# - target -> steering/speed 계산
# - lidar_cones debug image 생성
#=============================================

import cv2
import math
import numpy as np
from collections import deque


class LidarConeController:

    def __init__(self):
        #=============================================
        # Lidar sparse midpoint cone following settings
        #=============================================
        # 가까운 라바콘도 잡되, 0.12m 같은 차체 반사는 제거
        self.MIN_CONE_DIST = 0.0
        self.MAX_CONE_DIST = 8.0

        # 전방 + 측전방까지 넓게 사용
        self.FRONT_ANGLE_LIMIT = 115.0

        # 차량 좌표계 ROI
        # x: 차량 전방
        # y: 라이다 좌우 방향
        #
        # 현재 시뮬레이터 기준:
        # y < 0 : 실제 왼쪽 라바콘
        # y > 0 : 실제 오른쪽 라바콘
        self.X_MIN = 0.20
        self.X_MAX = 7.0
        self.Y_LIMIT = 3.5

        # 정면 근처 점도 너무 많이 버리지 않도록 완화
        self.CENTER_Y_IGNORE = 0.05

        #=============================================
        # 곡선 대응용 lookahead
        #=============================================
        # 기존 2.2는 곡선에서 너무 멀어서 직진처럼 보일 수 있음.
        self.LOOKAHEAD_X = 1.30
        self.LOOKAHEAD_WINDOW = 1.6
        self.SIDE_SELECT_N = 4

        # 라바콘 통로 폭 sanity check
        self.CONE_WIDTH_MIN = 0.30
        self.CONE_WIDTH_MAX = 6.0

        # 곡선에서는 한쪽 라바콘만 보이는 순간이 있으므로 허용
        self.ALLOW_SINGLE_SIDE_PATH = True
        self.CONE_LANE_HALF_WIDTH = 0.68

        #=============================================
        # Start straight fallback target
        #=============================================
        # 시작 직후만 짧게 직진 보조.
        # 좌우 라바콘이 한 번이라도 동시에 잡히면 이후에는 꺼짐.
        self.ALLOW_START_STRAIGHT_TARGET = True
        self.START_STRAIGHT_FRAMES = 25
        self.START_STRAIGHT_MIN_POINTS = 1
        self._seen_both_sides_once = False

        # 라이다 valid 확인
        self.LIDAR_CONFIRM_FRAMES = 2
        self._lidar_valid_count = 0

        #=============================================
        # 곡선 조향을 죽이지 않도록 deadband 축소
        #=============================================
        self.LIDAR_TARGET_Y_DEADBAND = 0.03
        self.LIDAR_ANGLE_DEADBAND = 0.0

        #=============================================
        # Steering rate limit
        #=============================================
        self.max_angle_step_s_curve = 35.0

        #=============================================
        # 라이다 조향
        #=============================================
        # target_y < 0 : 실제 왼쪽 target
        # target_y > 0 : 실제 오른쪽 target
        # target 위치는 맞는데 차가 반대로 꺾으면 이 값을 +1.0으로 바꿔.
        self.LIDAR_STEER_SIGN = -1.0
        self.LIDAR_STEER_GAIN = 3.5
        self.LIDAR_TARGET_Y_GAIN = 1.65
        self.LIDAR_STEER_NONLINEAR_GAIN = 0.065
        self.LIDAR_MAX_STEER = 60.0

        # 라이다 주행 속도
        self.LIDAR_STRAIGHT_SPEED = 8.0
        self.LIDAR_CURVE_SPEED = 3.0
        self.LIDAR_HARD_CURVE_SPEED = 2.5

        # 라이다 smoothing 상태
        self._lidar_angle_history = deque(maxlen=1)
        self._lidar_target_y_history = deque(maxlen=2)
        self._prev_lidar_angle = 0.0

        #=============================================
        # Lidar debug view
        #=============================================
        # False:
        #   현재 y<0이 실제 왼쪽이면 화면 왼쪽에도 초록색이 나오도록 표시.
        # True:
        #   디버그 화면만 좌우 반전.
        #
        # 주행 로직에는 영향 없음.
        self.LIDAR_DEBUG_FLIP_X = False

        self.debug_image = None

    #=============================================
    # Public API
    #=============================================
    def compute_control(self, scan_msg, startup_frame_count):
        """LaserScan을 받아 라바콘 주행 angle/speed/debug 정보를 반환한다."""

        result = {
            "valid": False,
            "confirmed": False,
            "angle": 0.0,
            "speed": 0.0,
            "target": None,
            "source": "none",
            "num_points": 0,
            "num_left": 0,
            "num_right": 0,
            "valid_cnt": self._lidar_valid_count,
            "left_points": [],
            "right_points": [],
            "debug_image": self.debug_image,
        }

        if scan_msg is None:
            self._reset_on_invalid()
            result["valid_cnt"] = self._lidar_valid_count
            return result

        points = self._filter_lidar_points(scan_msg)
        left_points, right_points = self._split_left_right_cones(points)

        target, source = self._make_sparse_midpoint_target(
            left_points=left_points,
            right_points=right_points,
            startup_frame_count=startup_frame_count
        )

        result["num_points"] = len(points)
        result["num_left"] = len(left_points)
        result["num_right"] = len(right_points)
        result["target"] = target
        result["source"] = source
        result["left_points"] = left_points
        result["right_points"] = right_points

        self._draw_lidar_debug(
            points=points,
            left_points=left_points,
            right_points=right_points,
            target=target,
            source=source,
            startup_frame_count=startup_frame_count
        )
        result["debug_image"] = self.debug_image

        if target is None:
            self._reset_on_invalid()
            result["valid_cnt"] = self._lidar_valid_count
            return result

        target_x, target_y = target

        if target_x < self.X_MIN:
            self._reset_on_invalid()
            result["valid_cnt"] = self._lidar_valid_count
            return result

        if abs(target_y) < self.LIDAR_TARGET_Y_DEADBAND:
            target_y = 0.0

        # target_y < 0: 실제 왼쪽
        # target_y > 0: 실제 오른쪽
        #
        # 조향이 약하게 나오는 문제 대응:
        # 1) target_y 자체를 증폭해서 가까운 S자 곡선에 더 민감하게 반응
        # 2) heading이 커질수록 nonlinear 항을 추가해서 급커브에서 더 강하게 조향
        control_y = target_y * self.LIDAR_TARGET_Y_GAIN

        target_heading_deg = math.degrees(
            math.atan2(control_y, target_x)
        )

        angle_linear = self.LIDAR_STEER_GAIN * target_heading_deg
        angle_nonlinear = (
            self.LIDAR_STEER_NONLINEAR_GAIN
            * target_heading_deg
            * abs(target_heading_deg)
        )

        angle_raw = self.LIDAR_STEER_SIGN * (
            angle_linear + angle_nonlinear
        )

        angle_raw = float(
            np.clip(
                angle_raw,
                -self.LIDAR_MAX_STEER,
                self.LIDAR_MAX_STEER
            )
        )

        if abs(angle_raw) < self.LIDAR_ANGLE_DEADBAND:
            angle_raw = 0.0

        self._lidar_angle_history.append(angle_raw)
        angle_smooth = float(np.mean(self._lidar_angle_history))

        angle_cmd = self._limit_lidar_angle_rate(
            target_angle=angle_smooth
        )

        angle_cmd = float(
            np.clip(
                angle_cmd,
                -self.LIDAR_MAX_STEER,
                self.LIDAR_MAX_STEER
            )
        )

        abs_angle = abs(angle_cmd)

        if abs_angle < 10.0:
            speed = self.LIDAR_STRAIGHT_SPEED
        elif abs_angle < 25.0:
            speed = self.LIDAR_CURVE_SPEED
        else:
            speed = self.LIDAR_HARD_CURVE_SPEED

        self._lidar_valid_count += 1
        confirmed = self._lidar_valid_count >= self.LIDAR_CONFIRM_FRAMES

        # 기존 track_drive.py와 동일하게, 실제 라이다 주행이 적용되는 시점에
        # 다음 rate limit 기준 조향각을 갱신한다.
        if confirmed:
            self._prev_lidar_angle = angle_cmd

        result["valid"] = True
        result["confirmed"] = confirmed
        result["angle"] = angle_cmd
        result["speed"] = speed
        result["target"] = (target_x, target_y)
        result["valid_cnt"] = self._lidar_valid_count

        return result

    #=============================================
    # Internal helpers
    #=============================================
    def _reset_on_invalid(self):
        self._lidar_valid_count = 0
        self._lidar_angle_history.clear()
        self._lidar_target_y_history.clear()
        self._prev_lidar_angle = 0.0

    def _limit_lidar_angle_rate(self, target_angle):
        max_step = self.max_angle_step_s_curve
        diff = target_angle - self._prev_lidar_angle

        if diff > max_step:
            diff = max_step
        elif diff < -max_step:
            diff = -max_step

        return float(self._prev_lidar_angle + diff)

    @staticmethod
    def _normalize_deg(deg):
        deg = deg % 360.0

        if deg > 180.0:
            deg -= 360.0

        return deg

    #=============================================
    # /scan -> 전방 ROI 라이다 점 필터링
    #=============================================
    def _filter_lidar_points(self, scan_msg):
        points = []

        if scan_msg is None:
            return points

        for i, r in enumerate(scan_msg.ranges):

            if not np.isfinite(r):
                continue

            # 너무 가까운 차체 반사/너무 먼 배경 제거
            if r < self.MIN_CONE_DIST or r > self.MAX_CONE_DIST:
                continue

            theta = scan_msg.angle_min + i * scan_msg.angle_increment
            deg = self._normalize_deg(math.degrees(theta))

            # 전방 + 측전방 범위만 사용
            if abs(deg) > self.FRONT_ANGLE_LIMIT:
                continue

            rad = math.radians(deg)

            # 차량 기준 좌표
            # x: 전방
            # y: 라이다 좌우 방향
            #
            # 현재 시뮬레이터에서는 y < 0이 실제 왼쪽으로 보임.
            x = r * math.cos(rad)
            y = r * math.sin(rad)

            # 전방 ROI
            if x < self.X_MIN or x > self.X_MAX:
                continue

            # 좌우 폭 ROI
            if abs(y) > self.Y_LIMIT:
                continue

            points.append((x, y, r, deg))

        return points

    #=============================================
    # 좌/우 라바콘 분리
    #=============================================
    def _split_left_right_cones(self, points):
        left_points = []
        right_points = []

        for x, y, r, deg in points:

            # 정면 근처는 좌우 판단이 애매하므로 제외
            if abs(y) < self.CENTER_Y_IGNORE:
                continue

            # 현재 시뮬레이터 기준:
            # y < 0 -> 실제 왼쪽 라바콘
            # y > 0 -> 실제 오른쪽 라바콘
            if y < 0.0:
                left_points.append((x, y, r, deg))
            else:
                right_points.append((x, y, r, deg))

        return left_points, right_points

    #=============================================
    # 희소 라이다 점 기반 중앙 target 생성
    #=============================================
    def _make_sparse_midpoint_target(self, left_points, right_points, startup_frame_count):

        #=============================================
        # 좌우 중 하나라도 안 보이는 경우
        #=============================================
        if len(left_points) == 0 or len(right_points) == 0:

            total_points = len(left_points) + len(right_points)

            # 시작 직후에는 임시 직진 target 생성
            # 단, 좌우가 한 번이라도 동시에 잡힌 뒤에는 사용하지 않음
            if (
                self.ALLOW_START_STRAIGHT_TARGET
                and not self._seen_both_sides_once
                and startup_frame_count <= self.START_STRAIGHT_FRAMES
                and total_points >= self.START_STRAIGHT_MIN_POINTS
            ):
                return (
                    (float(self.LOOKAHEAD_X), 0.0),
                    f"startup_straight L{len(left_points)} R{len(right_points)}"
                )

            # 한쪽만 쓰는 모드
            if not self.ALLOW_SINGLE_SIDE_PATH:
                return None, f"need_both_sides L{len(left_points)} R{len(right_points)}"

            # 왼쪽 라바콘만 보이는 경우
            if len(left_points) > 0:
                arr = np.array(
                    [[p[0], p[1]] for p in left_points],
                    dtype=np.float32
                )

                order = np.argsort(np.abs(arr[:, 0] - self.LOOKAHEAD_X))
                arr = arr[order]

                # 실제 왼쪽 라바콘은 y < 0.
                # 중앙은 y 증가 방향.
                y = float(arr[0, 1] + self.CONE_LANE_HALF_WIDTH)

                # 곡선에서 한쪽 라바콘만 볼 때도 smoothing 적용
                if abs(y) < self.LIDAR_TARGET_Y_DEADBAND:
                    y = 0.0

                self._lidar_target_y_history.append(y)
                y = float(np.mean(self._lidar_target_y_history))

                return (
                    (float(self.LOOKAHEAD_X), y),
                    "left_only_sparse"
                )

            # 오른쪽 라바콘만 보이는 경우
            if len(right_points) > 0:
                arr = np.array(
                    [[p[0], p[1]] for p in right_points],
                    dtype=np.float32
                )

                order = np.argsort(np.abs(arr[:, 0] - self.LOOKAHEAD_X))
                arr = arr[order]

                # 실제 오른쪽 라바콘은 y > 0.
                # 중앙은 y 감소 방향.
                y = float(arr[0, 1] - self.CONE_LANE_HALF_WIDTH)

                if abs(y) < self.LIDAR_TARGET_Y_DEADBAND:
                    y = 0.0

                self._lidar_target_y_history.append(y)
                y = float(np.mean(self._lidar_target_y_history))

                return (
                    (float(self.LOOKAHEAD_X), y),
                    "right_only_sparse"
                )

            return None, "no_points"

        #=============================================
        # 좌우 라바콘이 둘 다 보이는 경우
        #=============================================
        def select_side_points(points):
            arr = np.array(
                [[p[0], p[1]] for p in points],
                dtype=np.float32
            )

            # LOOKAHEAD_X 근처 점 우선
            dist_to_lookahead = np.abs(arr[:, 0] - self.LOOKAHEAD_X)
            order = np.argsort(dist_to_lookahead)
            arr = arr[order]

            # lookahead window 안의 점 사용
            near = arr[
                np.abs(arr[:, 0] - self.LOOKAHEAD_X)
                <= self.LOOKAHEAD_WINDOW
            ]

            # window 안에 아무것도 없으면 LOOKAHEAD_X와 가까운 순서 몇 개 사용
            if len(near) == 0:
                near = arr[:min(self.SIDE_SELECT_N, len(arr))]
            else:
                near = near[:min(self.SIDE_SELECT_N, len(near))]

            sx = float(np.median(near[:, 0]))
            sy = float(np.median(near[:, 1]))

            return sx, sy, len(near)

        lx, ly, ln = select_side_points(left_points)
        rx, ry, rn = select_side_points(right_points)

        # 좌우가 동시에 한 번이라도 잡히면 startup straight fallback 종료
        self._seen_both_sides_once = True

        width = abs(ry - ly)

        # 폭이 이상한 경우
        if width < self.CONE_WIDTH_MIN or width > self.CONE_WIDTH_MAX:

            # 시작 초반이면서 아직 좌우가 안정적으로 잡힌 적이 없을 때만 직진 fallback
            # 위에서 seen_both가 True로 바뀌었으므로 사실상 거의 안 쓰이지만 안전장치로 둠.
            if (
                self.ALLOW_START_STRAIGHT_TARGET
                and not self._seen_both_sides_once
                and startup_frame_count <= self.START_STRAIGHT_FRAMES
            ):
                return (
                    (float(self.LOOKAHEAD_X), 0.0),
                    f"startup_straight_bad_width={width:.2f}"
                )

            return None, f"bad_width={width:.2f}"

        # 좌우 라바콘의 중앙 y
        target_y = 0.5 * (ly + ry)

        # 중요:
        # target_x는 실제 라바콘 평균 x가 아니라 고정 lookahead로 둔다.
        # 가까운 점 하나 때문에 atan2가 과도하게 커지는 것을 막는다.
        target_x = self.LOOKAHEAD_X

        # 직진 구간 deadband
        if abs(target_y) < self.LIDAR_TARGET_Y_DEADBAND:
            target_y = 0.0

        # target_y smoothing
        self._lidar_target_y_history.append(target_y)
        target_y = float(np.mean(self._lidar_target_y_history))

        return (
            (float(target_x), float(target_y)),
            f"sparse_midpoint L{ln} R{rn}"
        )

    #=============================================
    # 라이다 디버그 top-view 이미지
    #=============================================
    def _draw_lidar_debug(self, points, left_points, right_points, target, source="none", startup_frame_count=0):
        width = 500
        height = 500
        scale = 65.0

        img = np.zeros((height, width, 3), dtype=np.uint8)

        origin_x = width // 2
        origin_y = height - 40

        x_min_px = int(origin_y - self.X_MIN * scale)
        x_max_px = int(origin_y - self.X_MAX * scale)

        if self.LIDAR_DEBUG_FLIP_X:
            y_left_px = int(origin_x + self.Y_LIMIT * scale)
            y_right_px = int(origin_x - self.Y_LIMIT * scale)
        else:
            y_left_px = int(origin_x - self.Y_LIMIT * scale)
            y_right_px = int(origin_x + self.Y_LIMIT * scale)

        cv2.rectangle(
            img,
            (min(y_left_px, y_right_px), x_max_px),
            (max(y_left_px, y_right_px), x_min_px),
            (60, 60, 60),
            1
        )

        # 차량 위치 및 진행 방향
        cv2.circle(
            img,
            (origin_x, origin_y),
            6,
            (255, 255, 255),
            -1
        )

        cv2.arrowedLine(
            img,
            (origin_x, origin_y),
            (origin_x, origin_y - 45),
            (255, 255, 255),
            2
        )

        def to_pixel(x, y):
            # 주행 로직의 y값은 그대로 두고, 디버그 화면만 좌우 반전 가능.
            if self.LIDAR_DEBUG_FLIP_X:
                px = int(origin_x - y * scale)
            else:
                px = int(origin_x + y * scale)

            py = int(origin_y - x * scale)

            return px, py

        # 전체 후보 점: 회색
        for x, y, r, deg in points:
            px, py = to_pixel(x, y)
            cv2.circle(img, (px, py), 2, (80, 80, 80), -1)

        # 왼쪽 후보: 초록색
        for x, y, r, deg in left_points:
            px, py = to_pixel(x, y)
            cv2.circle(img, (px, py), 5, (0, 255, 0), -1)

        # 오른쪽 후보: 빨간색
        for x, y, r, deg in right_points:
            px, py = to_pixel(x, y)
            cv2.circle(img, (px, py), 5, (0, 0, 255), -1)

        # lookahead x line
        lx1, ly1 = to_pixel(self.LOOKAHEAD_X, -self.Y_LIMIT)
        lx2, ly2 = to_pixel(self.LOOKAHEAD_X, self.Y_LIMIT)
        cv2.line(img, (lx1, ly1), (lx2, ly2), (100, 100, 255), 1)

        # target: 파란색
        if target is not None:
            tx, ty = target
            px, py = to_pixel(tx, ty)

            cv2.circle(img, (px, py), 8, (255, 0, 0), -1)
            cv2.line(img, (origin_x, origin_y), (px, py), (255, 0, 0), 2)

        cv2.putText(
            img,
            "Lidar sparse midpoint",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            img,
            "green=left, red=right, blue=target",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.putText(
            img,
            f"dist={self.MIN_CONE_DIST:.2f}-{self.MAX_CONE_DIST:.1f}m angle=+-{self.FRONT_ANGLE_LIMIT:.0f}deg",
            (10, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.putText(
            img,
            f"lookahead_x={self.LOOKAHEAD_X:.2f} window={self.LOOKAHEAD_WINDOW:.1f} flip={self.LIDAR_DEBUG_FLIP_X}",
            (10, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.putText(
            img,
            f"source={source} startup={startup_frame_count} seen_both={self._seen_both_sides_once}",
            (10, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        self.debug_image = img
