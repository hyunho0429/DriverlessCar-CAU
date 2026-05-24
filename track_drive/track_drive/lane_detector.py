#!/usr/bin/env python3
"""Lane detector for the Xytron Unity simulator (Kookmin qualifier track).

Pure CV (HSV mask + Hough) with no ROS dependency so HSV ranges and ROI can
be tuned against static screenshots before being wired into track_drive.py.

Run directly with one or more image paths to see the debug overlay:

    python3 lane_detector.py screenshot1.png screenshot2.png
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np


class LaneDetector:
    def __init__(
        self,
        roi_top_ratio=0.60,
        smoothing_window=5,
        lower_yellow=(20, 150, 150),
        upper_yellow=(35, 255, 255),
        lower_white=(0, 0, 190),
        upper_white=(180, 40, 255),
    ):
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
        ref_y = rh - 1

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        white_mask = cv2.inRange(hsv, self.lower_white, self.upper_white)

        # 점선 갭을 메워서 끊긴 노란선을 연결
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        combined = cv2.bitwise_or(yellow_mask, white_mask)

        edges = cv2.Canny(combined, 50, 150)
        segments = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=20, minLineLength=15, maxLineGap=40,
        )
        left_x, right_x = self._fit_left_right(segments, rw, ref_y)

        lane_center_x, source = self._lane_center(left_x, right_x, rw)
        lane_detected = lane_center_x is not None

        if lane_center_x is None:
            # No data this frame: hold last value (brief: never emit garbage).
            offset = self._last_offset
        else:
            # Sign per brief: deviation of the *vehicle* from the lane center.
            # Camera is body-fixed, so vehicle x in image == image_center_x;
            # therefore offset = image_center_x - lane_center_x (not the reverse).
            raw = image_center_x - lane_center_x
            self._offset_history.append(raw)
            offset = float(np.mean(self._offset_history))
            self._last_offset = offset

        current_lane = self._classify_lane(yellow_mask, rw)
        solid_line_warning = self._white_near_center(white_mask, rw)

        self.last_elapsed_ms = (time.time() - t0) * 1000.0
        self._dbg = dict(
            roi_origin_y=y0,
            roi_shape=(rh, rw),
            left_x=left_x,
            right_x=right_x,
            lane_center_x=lane_center_x,
            source=source,
            segments=segments,
        )

        return {
            'lane_center_offset': float(offset),
            'lane_detected': bool(lane_detected),
            'solid_line_warning': bool(solid_line_warning),
            'current_lane': current_lane,
        }

    # --- helpers --------------------------------------------------------

    def _fit_left_right(self, segments, width, ref_y):
        if segments is None:
            return None, None

        left_pts, right_pts = [], []
        mid = width / 2.0
        for seg in segments:
            x1, y1, x2, y2 = seg[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.2:
                continue
            x_mean = (x1 + x2) / 2.0
            # Image y grows downward, so the left lane line has negative slope.
            if slope < 0 and x_mean < mid:
                left_pts.extend([(x1, y1), (x2, y2)])
            elif slope > 0 and x_mean > mid:
                right_pts.extend([(x1, y1), (x2, y2)])

        return (
            self._fit_x_at_y(left_pts, ref_y),
            self._fit_x_at_y(right_pts, ref_y),
        )

    @staticmethod
    def _fit_x_at_y(pts, ref_y):
        if len(pts) < 2:
            return None
        ys = np.array([p[1] for p in pts], dtype=np.float32)
        xs = np.array([p[0] for p in pts], dtype=np.float32)
        if np.unique(ys).size < 2:
            return float(np.mean(xs))
        # Fit x = m*y + c so near-vertical lane lines stay well-conditioned.
        m, c = np.polyfit(ys, xs, 1)
        return float(m * ref_y + c)

    def _lane_center(self, left_x, right_x, width):
        if left_x is not None and right_x is not None:
            self._last_lane_width = right_x - left_x
            return (left_x + right_x) / 2.0, 'both'
        if left_x is not None and self._last_lane_width:
            return left_x + self._last_lane_width / 2.0, 'left+w'
        if right_x is not None and self._last_lane_width:
            return right_x - self._last_lane_width / 2.0, 'right+w'
        # No remembered width yet: assume the missing side sits at the frame edge.
        if left_x is not None:
            return (left_x + width) / 2.0, 'left_only'
        if right_x is not None:
            return right_x / 2.0, 'right_only'
        return None, 'none'

    @staticmethod
    def _classify_lane(yellow_mask, width):
        if yellow_mask.sum() == 0:
            return None
        peak_x = int(np.argmax(yellow_mask.sum(axis=0)))
        # Yellow center on the right side of the camera => we are in left lane (1).
        return 1 if peak_x > width // 2 else 2

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
        y0 = self._dbg.get('roi_origin_y', 0)
        rh, rw = self._dbg.get('roi_shape', (h - y0, w))

        cv2.rectangle(out, (0, y0), (w - 1, h - 1), (0, 255, 255), 1)

        segments = self._dbg.get('segments')
        if segments is not None:
            for seg in segments:
                x1, y1, x2, y2 = seg[0]
                cv2.line(out, (x1, y0 + y1), (x2, y0 + y2), (180, 180, 180), 1)

        ref_y = y0 + rh - 1
        if self._dbg.get('left_x') is not None:
            cv2.circle(out, (int(self._dbg['left_x']), ref_y), 8, (0, 255, 0), -1)
        if self._dbg.get('right_x') is not None:
            cv2.circle(out, (int(self._dbg['right_x']), ref_y), 8, (0, 0, 255), -1)
        if self._dbg.get('lane_center_x') is not None:
            cx = int(self._dbg['lane_center_x'])
            cv2.line(out, (cx, y0), (cx, h - 1), (0, 255, 0), 2)
        cv2.line(out, (w // 2, y0), (w // 2, h - 1), (255, 0, 0), 1)

        hud = [
            f"offset = {result['lane_center_offset']:+7.2f} px"
            f"   ({self._dbg.get('source')})",
            f"lane   = {result['current_lane']}"
            f"   detected = {result['lane_detected']}"
            f"   warn = {result['solid_line_warning']}",
            f"time   = {self.last_elapsed_ms:5.2f} ms",
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
    ap.add_argument("--no-show", action="store_true",
                    help="skip cv2.imshow (headless tuning)")
    args = ap.parse_args()

    det = LaneDetector()
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
        if k == 27:  # ESC
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _cli()
