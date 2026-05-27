#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# Lidar cone controller module (v2 — clustering + mode-aware control)
#
# 파이프라인:
#   /scan
#     -> [1] _filter_lidar_points   거리/각도/ROI 필터
#     -> [2] _cluster_cones         인접 점을 라바콘 객체로 묶음
#     -> [3] _split_left_right_cones  좌/우 라바콘 분리
#     -> [4] _x_binned_midpoints    bin별 좌/우 페어링 -> 통로 중점
#     -> [5] _detect_mode           std(midpoint y) 로 STRAIGHT/CURVE 판정
#     -> [6] _select_target_by_mode 모드별 lookahead로 target 결정
#     -> [7] _compute_steering_by_mode 모드별 게인으로 angle 산출
#     -> [8] _compute_speed_by_mode 모드별 속도 결정
#=============================================

import cv2
import math
import numpy as np
from collections import deque, namedtuple


# 클러스터링 결과로 만들어지는 라바콘 1개
# x, y: 차량 기준 좌표 [m]
# n_points: 이 라바콘에 속한 라이다 점 개수
# range: 차량으로부터의 거리 (점들의 평균) [m]
Cone = namedtuple("Cone", ["x", "y", "n_points", "range"])


class LidarConeController:

    def __init__(self):
        #=============================================
        # [1] 점 필터링 파라미터
        #=============================================
        # 차체 반사(인덱스 114~246에서 약 0.12m)를 완전히 제외하려면
        # MIN_CONE_DIST 가 0.30 이상이어야 안전하다.
        self.MIN_CONE_DIST = 0.35
        self.MAX_CONE_DIST = 8.0

        # 차체에 가려지지 않는 시야는 |deg|<115도지만, 가까운 측방 라바콘을
        # 놓치지 않도록 120도까지 허용. MIN_CONE_DIST=0.35가 차체 반사를 계속 차단.
        self.FRONT_ANGLE_LIMIT = 120.0

        # 차량 좌표계 ROI
        # x: 전방, y: 좌우 (sim 기준 y<0 이 실제 왼쪽)
        # X_MIN을 0.10으로 낮춰 차 바로 앞 라바콘도 포함 (차체 반사는 MIN_CONE_DIST가 차단)
        self.X_MIN = 0.10
        self.X_MAX = 7.0
        self.Y_LIMIT = 3.5

        # 정면 정중앙은 좌/우 판별이 애매하므로 제외
        self.CENTER_Y_IGNORE = 0.05

        #=============================================
        # [2] 라바콘 클러스터링 파라미터
        #=============================================
        # 1.3m 거리의 라바콘은 약 8개 점에 걸쳐 잡힘.
        # 인접 점이 다음을 모두 만족하면 같은 라바콘으로 묶는다.
        self.CLUSTER_ANGLE_GAP = 3.0   # 각도 차 임계 [deg]
        self.CLUSTER_RANGE_GAP = 0.30  # 거리 차 임계 [m]
        self.CLUSTER_MIN_POINTS = 2    # 라바콘 인정 최소 점 개수

        #=============================================
        # [4] x-binned pairing 파라미터
        #=============================================
        # 같은 x 구간에 들어온 좌/우 라바콘 짝을 찾아 통로 중점 산출.
        self.X_BIN_MIN = 0.4
        self.X_BIN_MAX = 2.4
        self.X_BIN_STEP = 0.4
        self.X_BIN_HALF = 0.30

        # 라바콘 통로 폭 sanity check (실측 트랙 4.0~4.5m 기준)
        self.CONE_WIDTH_MIN = 2.0
        self.CONE_WIDTH_MAX = 8.0

        # 한쪽만 보일 때 폴백 허용 (곡선에서 흔함)
        self.ALLOW_SINGLE_SIDE_PATH = True

        # 차선 절반폭(half-width) 동적 추정 — 양쪽 페어가 잡힐 때마다 EMA 갱신
        self.LANE_HALF_WIDTH_INIT = 2.2     # 트랙 4.4m 가정
        self.LANE_WIDTH_EMA_ALPHA = 0.15
        self._lane_half_width_est = self.LANE_HALF_WIDTH_INIT

        # Single-side 회피 부스트: 한쪽 라바콘에 가까울수록 반대쪽으로 강하게 꺾기
        self.SINGLE_SIDE_AVOID_THRESHOLD = 1.5   # 이 거리부터 부스트 시작 [m]
        self.SINGLE_SIDE_AVOID_MAX_BOOST = 2.0   # 0m 거리 최대 부스트 배수

        # single-side 폴백에서 사용하는 점 개수
        self.SIDE_SELECT_N = 4

        #=============================================
        # [5] 모드 판정 파라미터
        #=============================================
        # midpoint y의 표준편차로 직선/곡선 판정
        # 9Hz x 3프레임 = 약 0.34초 안에 모드 확정
        self.MODE_STD_STRAIGHT = 0.10  # std < 0.10 -> STRAIGHT
        self.MODE_STD_CURVE = 0.25     # std < 0.25 -> CURVE_ENTRY, 이상 -> CURVE
        self.MODE_CONFIRM_FRAMES = 3
        self._current_mode = "STRAIGHT"
        self._mode_candidate = "STRAIGHT"
        self._mode_confirm_count = 0

        #=============================================
        # [6] 모드별 lookahead [m]
        #=============================================
        self.LOOKAHEAD_BY_MODE = {
            "STRAIGHT":    1.50,
            "CURVE_ENTRY": 1.10,
            "CURVE":       0.85,
        }

        #=============================================
        # [7] 모드별 조향 게인
        # (kp_linear, kq_nonlinear, target_y_gain)
        #=============================================
        self.STEER_GAINS_BY_MODE = {
            "STRAIGHT":    (2.8, 0.04, 1.4),
            "CURVE_ENTRY": (3.8, 0.08, 1.6),
            "CURVE":       (4.5, 0.12, 1.8),
        }

        #=============================================
        # [8] 모드별 속도 [m/s]
        #=============================================
        # 디버깅 단계에서 절반 속도 — 안정화 후 다시 올릴 것
        self.SPEED_BY_MODE = {
            "STRAIGHT":    4.0,
            "CURVE_ENTRY": 2.5,
            "CURVE":       1.8,
        }

        #=============================================
        # 조향 공통: deadband, clip, rate limit
        #=============================================
        self.LIDAR_TARGET_Y_DEADBAND = 0.03
        self.LIDAR_ANGLE_DEADBAND = 0.0
        # sim은 ±100 범위를 받음 (실측: 명령 ±100이 물리 바퀴 약 ±20°)
        # 60으로 clip하면 sim이 받을 수 있는 한계의 60%만 쓰는 셈 → 100으로
        self.LIDAR_MAX_STEER = 100.0

        # rate limit: 한 프레임에 최대 변할 수 있는 angle [deg]
        # 9Hz 기준 60 deg/frame = 약 540 deg/s — target 큰 변화에 빠르게 catch-up
        self.max_angle_step_s_curve = 60.0

        # target_y < 0 일 때 차가 반대로 꺾으면 +1.0으로 뒤집기
        self.LIDAR_STEER_SIGN = -1.0

        #=============================================
        # Startup 직진 폴백 (시작 직후 한 번 동안만)
        #=============================================
        self.ALLOW_START_STRAIGHT_TARGET = True
        self.START_STRAIGHT_FRAMES = 25
        self.START_STRAIGHT_MIN_POINTS = 1
        self._seen_both_sides_once = False

        #=============================================
        # 라이다 valid 게이팅
        #=============================================
        self.LIDAR_CONFIRM_FRAMES = 2
        self._lidar_valid_count = 0

        #=============================================
        # smoothing 상태
        #=============================================
        self._lidar_angle_history = deque(maxlen=1)
        self._lidar_target_y_history = deque(maxlen=2)
        self._prev_lidar_angle = 0.0

        #=============================================
        # debug
        #=============================================
        # True: 디버그 화면만 좌우 반전 (주행 로직 무관)
        self.LIDAR_DEBUG_FLIP_X = False
        self.debug_image = None

    #=============================================
    # Public API
    #=============================================
    def compute_control(self, scan_msg, startup_frame_count):
        """LaserScan을 받아 라바콘 주행 angle/speed/debug 정보를 반환."""

        result = {
            "valid": False,
            "confirmed": False,
            "angle": 0.0,
            "speed": 0.0,
            "target": None,
            "source": "none",
            "mode": self._current_mode,
            "num_points": 0,
            "num_left": 0,
            "num_right": 0,
            "num_cones_left": 0,
            "num_cones_right": 0,
            "target_y_std": 0.0,
            "valid_cnt": self._lidar_valid_count,
            "left_points": [],
            "right_points": [],
            "debug_image": self.debug_image,
        }

        if scan_msg is None:
            self._reset_on_invalid()
            result["valid_cnt"] = self._lidar_valid_count
            return result

        # [1] 점 필터링
        points = self._filter_lidar_points(scan_msg)

        # [2] 라바콘 클러스터링
        cones = self._cluster_cones(points)

        # [3] 좌/우 분리
        left_cones, right_cones = self._split_left_right_cones(cones)

        # [4] x-bin pairing -> midpoints
        midpoints = self._x_binned_midpoints(left_cones, right_cones)

        # [5] 모드 판정
        target_y_std = self._compute_target_y_std(midpoints)
        mode = self._detect_mode(midpoints, target_y_std)

        # 결과 기본 채우기 (target 만들기 전이라도 디버그용)
        # 좌/우 분리에 사용된 raw 점은 디버그 용도로만 별도 유지
        left_points, right_points = self._split_left_right_points(points)
        result["num_points"] = len(points)
        result["num_left"] = len(left_points)
        result["num_right"] = len(right_points)
        result["num_cones_left"] = len(left_cones)
        result["num_cones_right"] = len(right_cones)
        result["left_points"] = left_points
        result["right_points"] = right_points
        result["target_y_std"] = float(target_y_std)
        result["mode"] = mode

        # [6] target 결정 (모드별 lookahead)
        target, source = self._select_target_by_mode(
            mode=mode,
            midpoints=midpoints,
            left_cones=left_cones,
            right_cones=right_cones,
            startup_frame_count=startup_frame_count,
        )
        result["target"] = target
        result["source"] = source

        # 디버그 이미지는 항상 그림
        self._draw_lidar_debug(
            points=points,
            left_cones=left_cones,
            right_cones=right_cones,
            midpoints=midpoints,
            target=target,
            source=source,
            mode=mode,
            target_y_std=target_y_std,
            startup_frame_count=startup_frame_count,
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

        # [7] 조향각 산출
        angle_cmd = self._compute_steering_by_mode(
            mode=mode, target_x=target_x, target_y=target_y
        )

        # [8] 속도 산출
        speed = self._compute_speed_by_mode(mode=mode)

        # valid confirm 게이팅
        self._lidar_valid_count += 1
        confirmed = self._lidar_valid_count >= self.LIDAR_CONFIRM_FRAMES

        # 라이다가 실제 주행에 반영되는 시점에만 rate limit 기준 갱신
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
    # [1] 점 필터링
    #=============================================
    def _filter_lidar_points(self, scan_msg):
        points = []
        if scan_msg is None:
            return points

        for i, r in enumerate(scan_msg.ranges):
            if not np.isfinite(r):
                continue
            if r < self.MIN_CONE_DIST or r > self.MAX_CONE_DIST:
                continue

            theta = scan_msg.angle_min + i * scan_msg.angle_increment
            deg = self._normalize_deg(math.degrees(theta))

            # 전방 + 측전방 시야만 (차체 가림 영역 제외)
            if abs(deg) > self.FRONT_ANGLE_LIMIT:
                continue

            rad = math.radians(deg)
            x = r * math.cos(rad)
            y = r * math.sin(rad)

            if x < self.X_MIN or x > self.X_MAX:
                continue
            if abs(y) > self.Y_LIMIT:
                continue

            points.append((x, y, r, deg))

        return points

    #=============================================
    # [2] 라바콘 클러스터링
    #=============================================
    def _cluster_cones(self, points):
        """인접한 점들을 묶어 개별 라바콘 객체 리스트로 변환.
        각도 기준 정렬 후 1D 그리디 클러스터링."""
        if len(points) == 0:
            return []

        # deg 기준 오름차순 정렬
        sorted_pts = sorted(points, key=lambda p: p[3])

        clusters = []
        current = [sorted_pts[0]]

        for prev, curr in zip(sorted_pts[:-1], sorted_pts[1:]):
            d_deg = abs(curr[3] - prev[3])
            d_range = abs(curr[2] - prev[2])
            if d_deg < self.CLUSTER_ANGLE_GAP and d_range < self.CLUSTER_RANGE_GAP:
                current.append(curr)
            else:
                clusters.append(current)
                current = [curr]
        clusters.append(current)

        cones = []
        for cl in clusters:
            if len(cl) < self.CLUSTER_MIN_POINTS:
                continue
            xs = [p[0] for p in cl]
            ys = [p[1] for p in cl]
            rs = [p[2] for p in cl]
            cones.append(Cone(
                x=float(np.mean(xs)),
                y=float(np.mean(ys)),
                n_points=len(cl),
                range=float(np.mean(rs)),
            ))
        return cones

    #=============================================
    # [3] 좌/우 분리 (클러스터 기준)
    #=============================================
    def _split_left_right_cones(self, cones):
        left_cones, right_cones = [], []
        for c in cones:
            if abs(c.y) < self.CENTER_Y_IGNORE:
                continue
            if c.y < 0.0:
                left_cones.append(c)
            else:
                right_cones.append(c)
        return left_cones, right_cones

    def _split_left_right_points(self, points):
        """디버그/로깅용으로만 사용 (모드 판정/조향은 클러스터 기준)."""
        left, right = [], []
        for p in points:
            if abs(p[1]) < self.CENTER_Y_IGNORE:
                continue
            if p[1] < 0.0:
                left.append(p)
            else:
                right.append(p)
        return left, right

    #=============================================
    # [4] x-binned pairing -> midpoints
    #=============================================
    def _x_binned_midpoints(self, left_cones, right_cones):
        """각 x bin에서 좌/우 라바콘이 모두 있으면 (mid_x, mid_y) 추출.
        width sanity 통과한 것만 반환."""
        if len(left_cones) == 0 or len(right_cones) == 0:
            return []

        left_arr = np.array([[c.x, c.y] for c in left_cones], dtype=np.float32)
        right_arr = np.array([[c.x, c.y] for c in right_cones], dtype=np.float32)

        midpoints = []
        widths = []
        x_centers = np.arange(
            self.X_BIN_MIN, self.X_BIN_MAX + 1e-6, self.X_BIN_STEP
        )
        for x_c in x_centers:
            L_mask = np.abs(left_arr[:, 0] - x_c) <= self.X_BIN_HALF
            R_mask = np.abs(right_arr[:, 0] - x_c) <= self.X_BIN_HALF
            if not L_mask.any() or not R_mask.any():
                continue
            ly_med = float(np.median(left_arr[L_mask, 1]))
            ry_med = float(np.median(right_arr[R_mask, 1]))
            width = abs(ry_med - ly_med)
            if width < self.CONE_WIDTH_MIN or width > self.CONE_WIDTH_MAX:
                continue
            midpoints.append((float(x_c), 0.5 * (ly_med + ry_med)))
            widths.append(width)

        # 양쪽 페어가 1개 이상이면 lane half-width EMA 갱신
        if len(widths) >= 1:
            current_half = float(np.median(widths)) * 0.5
            a = self.LANE_WIDTH_EMA_ALPHA
            self._lane_half_width_est = (
                a * current_half + (1.0 - a) * self._lane_half_width_est
            )

        return midpoints

    #=============================================
    # [5] 모드 판정 (std + 히스테리시스)
    #=============================================
    def _compute_target_y_std(self, midpoints):
        if len(midpoints) < 2:
            return 0.0
        ys = np.array([m[1] for m in midpoints], dtype=np.float32)
        return float(np.std(ys))

    def _detect_mode(self, midpoints, target_y_std):
        # bin이 부족하면 모드 유지 (깜빡임 방지)
        if len(midpoints) < 2:
            return self._current_mode

        if target_y_std < self.MODE_STD_STRAIGHT:
            raw_mode = "STRAIGHT"
        elif target_y_std < self.MODE_STD_CURVE:
            raw_mode = "CURVE_ENTRY"
        else:
            raw_mode = "CURVE"

        if raw_mode == self._current_mode:
            self._mode_candidate = raw_mode
            self._mode_confirm_count = 0
            return self._current_mode

        # 다른 모드 후보 등장
        if self._mode_candidate != raw_mode:
            self._mode_candidate = raw_mode
            self._mode_confirm_count = 1
            return self._current_mode

        self._mode_confirm_count += 1
        if self._mode_confirm_count >= self.MODE_CONFIRM_FRAMES:
            self._current_mode = raw_mode
            self._mode_confirm_count = 0
        return self._current_mode

    #=============================================
    # [6] target 결정 (모드별 lookahead)
    #=============================================
    def _select_target_by_mode(self, mode, midpoints, left_cones, right_cones, startup_frame_count):
        target_x_lookahead = self.LOOKAHEAD_BY_MODE[mode]

        # midpoint가 충분히 있으면 가중평균
        if len(midpoints) > 0:
            self._seen_both_sides_once = True
            arr = np.array(midpoints, dtype=np.float32)
            weights = 1.0 / (np.abs(arr[:, 0] - target_x_lookahead) + 0.3)
            target_y = float(np.average(arr[:, 1], weights=weights))

            if abs(target_y) < self.LIDAR_TARGET_Y_DEADBAND:
                target_y = 0.0
            self._lidar_target_y_history.append(target_y)
            target_y = float(np.mean(self._lidar_target_y_history))

            return (
                (float(target_x_lookahead), float(target_y)),
                f"binned_midpoint n={len(midpoints)} mode={mode}"
            )

        # midpoint 없음 — startup 직진 폴백
        total_cones = len(left_cones) + len(right_cones)
        if (
            self.ALLOW_START_STRAIGHT_TARGET
            and not self._seen_both_sides_once
            and startup_frame_count <= self.START_STRAIGHT_FRAMES
            and total_cones >= self.START_STRAIGHT_MIN_POINTS
        ):
            return (
                (float(target_x_lookahead), 0.0),
                f"startup_straight L{len(left_cones)} R{len(right_cones)}"
            )

        # single-side 폴백
        if not self.ALLOW_SINGLE_SIDE_PATH:
            return None, f"need_both_sides L{len(left_cones)} R{len(right_cones)}"
        if len(left_cones) == 0 and len(right_cones) == 0:
            return None, "no_cones"

        use_left = len(left_cones) >= len(right_cones)
        if use_left and len(left_cones) > 0:
            y, closest_r, boost = self._single_side_target_y(
                left_cones, sign=+1.0
            )
            src = (f"left_only_avoid n={len(left_cones)} "
                   f"r={closest_r:.2f} boost={boost:.2f}")
        elif len(right_cones) > 0:
            y, closest_r, boost = self._single_side_target_y(
                right_cones, sign=-1.0
            )
            src = (f"right_only_avoid n={len(right_cones)} "
                   f"r={closest_r:.2f} boost={boost:.2f}")
        else:
            return None, "no_cones"

        if abs(y) < self.LIDAR_TARGET_Y_DEADBAND:
            y = 0.0
        self._lidar_target_y_history.append(y)
        y = float(np.mean(self._lidar_target_y_history))
        return ((float(target_x_lookahead), float(y)), src)

    def _single_side_target_y(self, cones, sign):
        """한쪽 라바콘만 보일 때:
        가장 가까운 라바콘을 기준으로 동적 lane half-width 만큼 반대쪽으로 target 설정.
        가까울수록 부스트를 적용해 회피 강도 ↑.

        sign = +1.0 : 왼쪽만 보임 (target은 오른쪽 = +y 방향)
        sign = -1.0 : 오른쪽만 보임 (target은 왼쪽 = -y 방향)

        반환: (target_y, closest_range, boost)
        """
        arr = np.array(
            [[c.x, c.y, c.range] for c in cones], dtype=np.float32
        )
        # range 기준 가장 가까운 라바콘
        closest_idx = int(np.argmin(arr[:, 2]))
        closest_y = float(arr[closest_idx, 1])
        closest_r = float(arr[closest_idx, 2])

        # 거리 비례 회피 부스트
        thr = self.SINGLE_SIDE_AVOID_THRESHOLD
        if closest_r < thr:
            ratio = max(0.0, 1.0 - closest_r / thr)
            boost = 1.0 + ratio * (self.SINGLE_SIDE_AVOID_MAX_BOOST - 1.0)
        else:
            boost = 1.0

        # 반대쪽 라바콘이 있다고 가정한 위치
        # = 가까운 라바콘 y 좌표 + 반대 방향 (sign) × half_width × boost
        target_y = closest_y + sign * self._lane_half_width_est * boost
        return float(target_y), closest_r, float(boost)

    #=============================================
    # [7] 조향각 산출 (모드별 게인)
    #=============================================
    def _compute_steering_by_mode(self, mode, target_x, target_y):
        kp, kq, y_gain = self.STEER_GAINS_BY_MODE[mode]

        control_y = target_y * y_gain
        heading_deg = math.degrees(math.atan2(control_y, target_x))

        angle_linear = kp * heading_deg
        angle_nonlinear = kq * heading_deg * abs(heading_deg)
        angle_raw = self.LIDAR_STEER_SIGN * (angle_linear + angle_nonlinear)
        angle_raw = float(np.clip(
            angle_raw, -self.LIDAR_MAX_STEER, self.LIDAR_MAX_STEER
        ))

        if abs(angle_raw) < self.LIDAR_ANGLE_DEADBAND:
            angle_raw = 0.0

        self._lidar_angle_history.append(angle_raw)
        angle_smooth = float(np.mean(self._lidar_angle_history))

        angle_cmd = self._limit_lidar_angle_rate(angle_smooth)
        angle_cmd = float(np.clip(
            angle_cmd, -self.LIDAR_MAX_STEER, self.LIDAR_MAX_STEER
        ))
        return angle_cmd

    #=============================================
    # [8] 속도 결정 (모드별)
    #=============================================
    def _compute_speed_by_mode(self, mode):
        return float(self.SPEED_BY_MODE[mode])

    #=============================================
    # 공통 헬퍼
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
    # 디버그 top-view
    #=============================================
    def _draw_lidar_debug(self, points, left_cones, right_cones,
                          midpoints, target, source, mode,
                          target_y_std, startup_frame_count):
        width = 540
        height = 540
        scale = 65.0
        img = np.zeros((height, width, 3), dtype=np.uint8)

        origin_x = width // 2
        origin_y = height - 40

        # ROI 박스
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
            (60, 60, 60), 1
        )

        # 차량 위치 + 진행 방향 화살표
        cv2.circle(img, (origin_x, origin_y), 6, (255, 255, 255), -1)
        cv2.arrowedLine(
            img, (origin_x, origin_y),
            (origin_x, origin_y - 45), (255, 255, 255), 2
        )

        def to_pixel(x, y):
            if self.LIDAR_DEBUG_FLIP_X:
                px = int(origin_x - y * scale)
            else:
                px = int(origin_x + y * scale)
            py = int(origin_y - x * scale)
            return px, py

        # 원본 점은 회색
        for p in points:
            px, py = to_pixel(p[0], p[1])
            cv2.circle(img, (px, py), 2, (80, 80, 80), -1)

        # 클러스터 = 큰 동그라미. 좌 초록, 우 빨강.
        def draw_cones(cones, color):
            for c in cones:
                px, py = to_pixel(c.x, c.y)
                cv2.circle(img, (px, py), 9, color, 2)
                cv2.putText(
                    img, str(c.n_points),
                    (px - 5, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )
        draw_cones(left_cones, (0, 255, 0))
        draw_cones(right_cones, (0, 0, 255))

        # midpoints = 작은 노란 점
        for (mx, my) in midpoints:
            px, py = to_pixel(mx, my)
            cv2.circle(img, (px, py), 4, (0, 255, 255), -1)

        # 현재 모드의 lookahead 가로선
        lookahead_x = self.LOOKAHEAD_BY_MODE.get(mode, 1.3)
        lx1, ly1 = to_pixel(lookahead_x, -self.Y_LIMIT)
        lx2, ly2 = to_pixel(lookahead_x, self.Y_LIMIT)
        cv2.line(img, (lx1, ly1), (lx2, ly2), (180, 180, 255), 1)

        # target = 파란 원 + 라인
        if target is not None:
            tx, ty = target
            px, py = to_pixel(tx, ty)
            cv2.circle(img, (px, py), 8, (255, 0, 0), -1)
            cv2.line(img, (origin_x, origin_y), (px, py), (255, 0, 0), 2)

        # 모드 헤더 — 색상 mode별 다르게
        mode_color = {
            "STRAIGHT":    (0, 255, 0),
            "CURVE_ENTRY": (0, 255, 255),
            "CURVE":       (0, 0, 255),
        }.get(mode, (255, 255, 255))
        cv2.putText(
            img, f"MODE: {mode}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2
        )

        # 보조 정보
        cv2.putText(
            img,
            f"std={target_y_std:.3f}  thr(S<{self.MODE_STD_STRAIGHT}|C<{self.MODE_STD_CURVE})",
            (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )
        cv2.putText(
            img,
            f"lookahead={lookahead_x:.2f}  source={source}",
            (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )
        cv2.putText(
            img,
            f"cones L{len(left_cones)} R{len(right_cones)}  midpoints={len(midpoints)}",
            (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )
        cv2.putText(
            img,
            f"dist={self.MIN_CONE_DIST:.2f}-{self.MAX_CONE_DIST:.1f}m  "
            f"startup={startup_frame_count} seen_both={self._seen_both_sides_once}",
            (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )
        cv2.putText(
            img,
            f"lane_half_est={self._lane_half_width_est:.2f}m "
            f"(track ~{2.0 * self._lane_half_width_est:.2f}m)",
            (10, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1
        )
        cv2.putText(
            img,
            "green=left cones, red=right cones, yellow=midpoints, blue=target",
            (10, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1
        )

        self.debug_image = img
