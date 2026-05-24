#!/usr/bin/env python3
"""Bird's-Eye-View 차선 감지기 (Xytron Unity 시뮬레이터).

원본 프레임을 perspective warp 으로 위에서 본 시점(BEV)으로 펴고,
색상 마스크 + 컬럼 히스토그램으로 차선 x 좌표를 찾는다.

LaneDetector와 동일한 dict 인터페이스를 출력하므로 track_drive.py 에서 교체만 하면 됨.

단독 실행 (사다리꼴 4점 캘리브레이션 / BEV 시각화):
    python3 bev_lane_detector.py screenshot.png
    python3 bev_lane_detector.py screenshot.png --lookahead --lane 2
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np


class BevLaneDetector:
    def __init__(
        self,
        target_lane=2,
        # 원본 이미지(640x480 기준) 도로 사다리꼴: (좌하, 좌상, 우상, 우하)
        # 실측값: y=290 흰선(L=184,R=440), y=415 흰선(L=10,R=630)
        src_pts=((10, 415), (180, 290), (445, 290), (630, 415)),
        bev_size=(400, 400),    # (width, height) — BEV 출력 크기
        use_lookahead=False,    # True: BEV 중상단 가중치 ↑
        smoothing_window=5,
        lower_yellow=(20, 150, 150),
        upper_yellow=(35, 255, 255),
        lower_white=(0, 0, 190),
        upper_white=(180, 40, 255),
    ):
        self.target_lane = target_lane
        self.use_lookahead = use_lookahead
        self.bev_size = (int(bev_size[0]), int(bev_size[1]))

        self.src_pts = np.float32(src_pts)
        w, h = self.bev_size
        # 좌하, 좌상, 우상, 우하 순서로 대응
        self.dst_pts = np.float32([[0, h], [0, 0], [w, 0], [w, h]])
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)

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
        bw, bh = self.bev_size

        # 1. Perspective warp → BEV
        bev = cv2.warpPerspective(frame, self.M, (bw, bh))

        # 2. HSV 마스크
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        white_mask = cv2.inRange(hsv, self.lower_white, self.upper_white)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        # 3. 각 라인 x좌표
        yellow_x = self._line_x(yellow_mask)
        left_white_x = self._line_x(white_mask, x_max=bw // 2)
        right_white_x = self._line_x(white_mask, x_min=bw // 2)

        # 4. 차선 경계 선택
        if self.target_lane == 1:
            left_x, right_x = left_white_x, yellow_x
        else:
            left_x, right_x = yellow_x, right_white_x

        lane_center_x, source = self._lane_center(left_x, right_x, bw)
        lane_detected = lane_center_x is not None

        bev_center_x = bw / 2.0
        if lane_center_x is None:
            offset = self._last_offset
        else:
            raw = bev_center_x - lane_center_x
            self._offset_history.append(raw)
            offset = float(np.mean(self._offset_history))
            self._last_offset = offset

        solid_line_warning = self._white_near_center(white_mask, bw)

        self.last_elapsed_ms = (time.time() - t0) * 1000.0
        self._dbg = dict(
            bev=bev,
            yellow_x=yellow_x,
            left_white_x=left_white_x,
            right_white_x=right_white_x,
            left_x=left_x,
            right_x=right_x,
            lane_center_x=lane_center_x,
            bev_center_x=bev_center_x,
            source=source,
        )

        return {
            'lane_center_offset': float(offset),
            'lane_detected': bool(lane_detected),
            'solid_line_warning': bool(solid_line_warning),
            'current_lane': self.target_lane,
        }

    # --- helpers --------------------------------------------------------

    def _line_x(self, mask, x_min=0, x_max=None):
        """BEV의 컬럼 히스토그램. use_lookahead=True면 상단(=먼 곳) 가중치 ↑."""
        if x_max is None:
            x_max = mask.shape[1]
        region = mask[:, x_min:x_max].astype(np.float32)
        if self.use_lookahead:
            row_w = np.linspace(1.0, 0.3, region.shape[0])[:, np.newaxis]
        else:
            row_w = np.linspace(0.3, 1.0, region.shape[0])[:, np.newaxis]
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
        # 사다리꼴 표시
        pts = self.src_pts.astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 255), 2)

        bev = self._dbg.get('bev')
        if bev is None:
            return out
        bev_out = bev.copy()
        bh = bev_out.shape[0]

        for x, color in [
            (self._dbg.get('yellow_x'),       (0, 215, 255)),
            (self._dbg.get('left_white_x'),   (0, 255, 0)),
            (self._dbg.get('right_white_x'),  (0, 0, 255)),
        ]:
            if x is not None:
                cv2.line(bev_out, (int(x), 0), (int(x), bh - 1), color, 2)

        cx = self._dbg.get('lane_center_x')
        if cx is not None:
            cv2.line(bev_out, (int(cx), 0), (int(cx), bh - 1), (255, 255, 255), 2)
        bcx = int(self._dbg.get('bev_center_x', bev.shape[1] // 2))
        cv2.line(bev_out, (bcx, 0), (bcx, bh - 1), (255, 0, 0), 1)

        # 입력 + BEV 가로로 연결
        target_h = out.shape[0]
        scale = target_h / bh
        bev_scaled = cv2.resize(bev_out, (int(bev_out.shape[1] * scale), target_h))
        combined = np.hstack([out, bev_scaled])

        mode = "BEV+lookahead" if self.use_lookahead else "BEV"
        hud = [
            f"[{mode} | lane {self.target_lane}]  offset = {result['lane_center_offset']:+7.2f}",
            f"detected = {result['lane_detected']}  warn = {result['solid_line_warning']}"
            f"  ({self._dbg.get('source')})",
            f"time = {self.last_elapsed_ms:5.2f} ms",
        ]
        for i, line in enumerate(hud):
            cv2.putText(combined, line, (10, 22 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(combined, line, (10, 22 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return combined


def _cli():
    ap = argparse.ArgumentParser(description="BevLaneDetector standalone tuner")
    ap.add_argument("images", nargs="+", help="image file path(s)")
    ap.add_argument("--lane", type=int, default=2, choices=[1, 2])
    ap.add_argument("--lookahead", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    det = BevLaneDetector(target_lane=args.lane, use_lookahead=args.lookahead)
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
        cv2.imshow("BevLaneDetector", dbg)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _cli()
