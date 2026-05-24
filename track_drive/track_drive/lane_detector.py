#!/usr/bin/env python3
"""Lane detector for the Xytron Unity simulator (Kookmin qualifier track).

Lane layout:  [white] | Lane 1 | [yellow dashed] | Lane 2 | [white]

Pure CV (HSV mask + column histogram) with no ROS dependency.
Run directly with one or more image paths to see the debug overlay:

    python3 lane_detector.py screenshot.png
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np


class LaneDetector:
    def __init__(
        self,
        target_lane=2,
        roi_top_ratio=0.60,
        smoothing_window=5,
        lower_yellow=(20, 150, 150),
        upper_yellow=(35, 255, 255),
        lower_white=(0, 0, 190),
        upper_white=(180, 40, 255),
    ):
        self.target_lane = target_lane      # 1 or 2
        self.roi_top_ratio = roi_top_ratio
        self.lower_yellow = np.array(lower_yellow, dtype=np.uint8)
        self.upper_yellow = np.array(upper_yellow, dtype=np.uint8)
        self.lower_white = np.array(lower_white, dtype=np.uint8)
        self.upper_white = np.array(upper_white, dtype=np.uint8)

        self._offset_history = deque(maxlen=smoothing_window)
        self._last_offset = 0.0
        self._last_lane_width = None

        self.last_elapsed_ms = 0.0
        self._dbg = {}

    def detect(self, frame):
        t0 = time.time()
        h, w = frame.shape[:2]
        image_center_x = w / 2.0

        y0 = int(h * self.roi_top_ratio)
        roi = frame[y0:, :]
        rh, rw = roi.shape[:2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        white_mask  = cv2.inRange(hsv, self.lower_white,  self.upper_white)

        # 점선 갭 연결
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        white_mask  = cv2.morphologyEx(white_mask,  cv2.MORPH_CLOSE, kernel)

        # 각 선의 x좌표 추출
        yellow_x      = self._line_x(yellow_mask)
        left_white_x  = self._line_x(white_mask, x_max=rw // 2)
        right_white_x = self._line_x(white_mask, x_min=rw // 2)

        # 목표 차선에 따라 경계 선택
        # Lane 1: 왼쪽 흰선 ~ 노란선
        # Lane 2: 노란선   ~ 오른쪽 흰선
        if self.target_lane == 1:
            left_x, right_x = left_white_x, yellow_x
        else:
            left_x, right_x = yellow_x, right_white_x

        lane_center_x, source = self._lane_center(left_x, right_x, rw)
        lane_detected = lane_center_x is not None

        if lane_center_x is None:
            # 감지 실패: 직전 값 유지
            offset = self._last_offset
        else:
            # 차량 편차 = 카메라 중심 - 차선 중심
            raw = image_center_x - lane_center_x
            self._offset_history.append(raw)
            offset = float(np.mean(self._offset_history))
            self._last_offset = offset

        solid_line_warning = self._white_near_center(white_mask, rw)

        self.last_elapsed_ms = (time.time() - t0) * 1000.0
        self._dbg = dict(
            roi_origin_y=y0,
            roi_shape=(rh, rw),
            yellow_x=yellow_x,
            left_white_x=left_white_x,
            right_white_x=right_white_x,
            left_x=left_x,
            right_x=right_x,
            lane_center_x=lane_center_x,
            source=source,
        )

        return {
            'lane_center_offset': float(offset),
            'lane_detected': bool(lane_detected),
            'solid_line_warning': bool(solid_line_warning),
            'current_lane': self.target_lane,
        }

    # --- helpers --------------------------------------------------------

    @staticmethod
    def _line_x(mask, x_min=0, x_max=None):
        """컬럼 히스토그램으로 라인 x좌표 추출.
        하단에 가까울수록 가중치를 높여 가까운 차선 위치에 더 집중."""
        if x_max is None:
            x_max = mask.shape[1]
        region = mask[:, x_min:x_max].astype(np.float32)
        # 하단 행일수록 가중치 높음 (0.5 ~ 1.0)
        row_w = np.linspace(0.5, 1.0, region.shape[0])[:, np.newaxis]
        hist = (region * row_w).sum(axis=0)
        if hist.max() == 0:
            return None
        indices = np.arange(len(hist))
        return x_min + float(np.sum(hist * indices) / hist.sum())

    def _lane_center(self, left_x, right_x, width):
        if left_x is not None and right_x is not None:
            self._last_lane_width = right_x - left_x
            return (left_x + right_x) / 2.0, 'both'
        if left_x is not None and self._last_lane_width:
            return left_x + self._last_lane_width / 2.0, 'left+w'
        if right_x is not None and self._last_lane_width:
            return right_x - self._last_lane_width / 2.0, 'right+w'
        # 차선 폭 미확보: 프레임 끝을 경계로 추정
        if left_x is not None:
            return (left_x + width) / 2.0, 'left_only'
        if right_x is not None:
            return right_x / 2.0, 'right_only'
        return None, 'none'

    @staticmethod
    def _white_near_center(white_mask, width, band_ratio=0.15):
        if white_mask.sum() == 0:
            return False
        mid = width // 2
        band = int(width * band_ratio)
        in_band = white_mask[:, mid - band: mid + band].sum()
        return bool(in_band > white_mask.sum() * 0.4)

    # --- debug ----------------------------------------------------------

    def draw_debug(self, frame, result):
        out = frame.copy()
        h, w = out.shape[:2]
        y0   = self._dbg.get('roi_origin_y', 0)
        rh, rw = self._dbg.get('roi_shape', (h - y0, w))
        ref_y = y0 + rh - 1

        # ROI 경계
        cv2.rectangle(out, (0, y0), (w - 1, h - 1), (0, 255, 255), 1)

        # 노란선 (노란 원)
        yx = self._dbg.get('yellow_x')
        if yx is not None:
            cv2.circle(out, (int(yx), ref_y), 8, (0, 215, 255), -1)

        # 흰선 좌 (초록 원) / 우 (빨간 원)
        lwx = self._dbg.get('left_white_x')
        rwx = self._dbg.get('right_white_x')
        if lwx is not None:
            cv2.circle(out, (int(lwx), ref_y), 8, (0, 255, 0), -1)
        if rwx is not None:
            cv2.circle(out, (int(rwx), ref_y), 8, (0, 0, 255), -1)

        # 계산된 차선 중심 (흰 세로선)
        cx = self._dbg.get('lane_center_x')
        if cx is not None:
            cv2.line(out, (int(cx), y0), (int(cx), h - 1), (255, 255, 255), 2)

        # 카메라 중심 (파란 세로선)
        cv2.line(out, (w // 2, y0), (w // 2, h - 1), (255, 0, 0), 1)

        hud = [
            f"[lane {self.target_lane}]  offset = {result['lane_center_offset']:+7.2f} px"
            f"   ({self._dbg.get('source')})",
            f"detected = {result['lane_detected']}"
            f"   warn = {result['solid_line_warning']}",
            f"time = {self.last_elapsed_ms:5.2f} ms",
        ]
        for i, line in enumerate(hud):
            cv2.putText(out, line, (10, 22 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, line, (10, 22 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return out


def _cli():
    ap = argparse.ArgumentParser(description="LaneDetector standalone tuner")
    ap.add_argument("images", nargs="+", help="image file path(s)")
    ap.add_argument("--lane", type=int, default=2, choices=[1, 2],
                    help="target lane (default: 2)")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    det = LaneDetector(target_lane=args.lane)
    for path in args.images:
        frame = cv2.imread(path)
        if frame is None:
            print(f"[skip] cannot read: {path}")
            continue
        result = det.detect(frame)
        print(f"{path}: {result}  [{det.last_elapsed_ms:.2f} ms]")
        if args.no_show:
            continue
        dbg = det.draw_debug(frame, result)
        cv2.imshow("LaneDetector", dbg)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _cli()
