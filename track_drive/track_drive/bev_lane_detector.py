#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bird's-Eye-View 차선 감지기 (Xytron Unity 시뮬레이터).

원본 프레임을 perspective warp 으로 위에서 본 시점(BEV)으로 펴고,
색상 마스크 + 컬럼 히스토그램으로 차선 x 좌표를 찾는다.

LaneDetector와 동일한 dict 인터페이스를 출력하며, 추가로 near/far band
offset과 curve offset을 반환해 S자/곡선 구간 선제 조향에 사용한다.
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

        #=============================================
        # BEV warp source points
        #=============================================
        src_pts=((32, 434), (174, 319), (466, 319), (608, 434)),

        bev_size=(400, 400),
        use_lookahead=True,
        smoothing_window=3,
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

        # dst_pts 순서도 src_pts와 동일하게:
        # 좌하, 좌상, 우상, 우하
        self.dst_pts = np.float32([
            [0, h],
            [0, 0],
            [w, 0],
            [w, h]
        ])

        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)

        self.lower_yellow = np.array(lower_yellow, dtype=np.uint8)
        self.upper_yellow = np.array(upper_yellow, dtype=np.uint8)
        self.lower_white = np.array(lower_white, dtype=np.uint8)
        self.upper_white = np.array(upper_white, dtype=np.uint8)

        self._offset_history = deque(maxlen=smoothing_window)
        self._near_history = deque(maxlen=smoothing_window)
        self._far_history = deque(maxlen=smoothing_window)

        self._last_offset = 0.0
        self._last_near_offset = 0.0
        self._last_far_offset = 0.0
        self._last_lane_width = None

        self.last_elapsed_ms = 0.0
        self._dbg = {}


    #=============================================
    # 메인 detect 함수
    #=============================================
    def detect(self, frame):

        t0 = time.time()

        bw, bh = self.bev_size

        #=============================================
        # 1. Perspective warp → BEV
        #=============================================
        bev = cv2.warpPerspective(frame, self.M, (bw, bh))

        #=============================================
        # 2. HSV color mask
        #=============================================
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(
            hsv,
            self.lower_yellow,
            self.upper_yellow
        )

        white_mask = cv2.inRange(
            hsv,
            self.lower_white,
            self.upper_white
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        yellow_mask = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        #=============================================
        # 3. 전체 BEV 기준 offset
        #=============================================
        center_result = self._detect_offset_in_band(
            yellow_mask,
            white_mask,
            y_min=0,
            y_max=bh
        )

        #=============================================
        # 4. near / far offset 분리
        #=============================================
        # BEV 좌표계:
        # y=0   : 먼 곳
        # y=bh  : 가까운 곳
        #
        # far_result:
        #   앞쪽 차선 중심. S자 진입을 미리 감지.
        #
        # near_result:
        #   차량 가까운 차선 중심. 현재 위치 안정화.
        far_result = self._detect_offset_in_band(
            yellow_mask,
            white_mask,
            y_min=int(bh * 0.10),
            y_max=int(bh * 0.45)
        )

        near_result = self._detect_offset_in_band(
            yellow_mask,
            white_mask,
            y_min=int(bh * 0.55),
            y_max=int(bh * 0.95)
        )

        lane_detected = center_result["lane_center_x"] is not None

        #=============================================
        # 5. 전체 offset smoothing
        #=============================================
        if center_result["offset"] is None:
            offset = self._last_offset
        else:
            self._offset_history.append(center_result["offset"])
            offset = float(np.mean(self._offset_history))
            self._last_offset = offset

        #=============================================
        # 6. near offset smoothing
        #=============================================
        if near_result["offset"] is None:
            near_offset = self._last_near_offset
        else:
            self._near_history.append(near_result["offset"])
            near_offset = float(np.mean(self._near_history))
            self._last_near_offset = near_offset

        #=============================================
        # 7. far offset smoothing
        #=============================================
        if far_result["offset"] is None:
            far_offset = self._last_far_offset
        else:
            self._far_history.append(far_result["offset"])
            far_offset = float(np.mean(self._far_history))
            self._last_far_offset = far_offset

        #=============================================
        # 8. curve offset
        #=============================================
        # far와 near 차이가 크면 앞쪽 차선 형태가 현재 차선과 다르다는 뜻.
        # S자/곡선 진입 판단에 사용.
        curve_offset = far_offset - near_offset

        solid_line_warning = self._white_near_center(white_mask, bw)

        self.last_elapsed_ms = (time.time() - t0) * 1000.0

        self._dbg = dict(
            bev=bev,
            yellow_mask=yellow_mask,
            white_mask=white_mask,
            center=center_result,
            near=near_result,
            far=far_result,
            offset=offset,
            near_offset=near_offset,
            far_offset=far_offset,
            curve_offset=curve_offset,
        )

        return {
            "lane_center_offset": float(offset),
            "near_offset": float(near_offset),
            "far_offset": float(far_offset),
            "curve_offset": float(curve_offset),
            "lane_detected": bool(lane_detected),
            "source": center_result["source"],
            "solid_line_warning": bool(solid_line_warning),
            "current_lane": self.target_lane,
        }


    #=============================================
    # 특정 y band에서 lane offset 계산
    #=============================================
    def _detect_offset_in_band(self, yellow_mask, white_mask, y_min, y_max):

        bw = yellow_mask.shape[1]
        bev_center_x = bw / 2.0

        yellow_band = yellow_mask[y_min:y_max, :]
        white_band = white_mask[y_min:y_max, :]

        yellow_x = self._line_x(yellow_band)

        left_white_x = self._line_x(
            white_band,
            x_min=0,
            x_max=bw // 2
        )

        right_white_x = self._line_x(
            white_band,
            x_min=bw // 2,
            x_max=bw
        )

        #=============================================
        # target lane별 차선 경계 선택
        #=============================================
        # Lane 1:
        #   왼쪽 흰선 ~ 노란선
        #
        # Lane 2:
        #   노란선 ~ 오른쪽 흰선
        if self.target_lane == 1:
            left_x = left_white_x
            right_x = yellow_x
        else:
            left_x = yellow_x
            right_x = right_white_x

        lane_center_x, source = self._lane_center(left_x, right_x, bw)

        if lane_center_x is None:
            offset = None
        else:
            # 기존 LaneDetector와 같은 부호 체계 유지
            # offset > 0:
            #   차선 중심이 BEV 중심보다 왼쪽에 있음
            #
            # offset < 0:
            #   차선 중심이 BEV 중심보다 오른쪽에 있음
            offset = bev_center_x - lane_center_x

        return {
            "yellow_x": yellow_x,
            "left_white_x": left_white_x,
            "right_white_x": right_white_x,
            "left_x": left_x,
            "right_x": right_x,
            "lane_center_x": lane_center_x,
            "offset": offset,
            "source": source,
            "y_min": y_min,
            "y_max": y_max,
        }


    #=============================================
    # mask에서 x좌표 추출
    #=============================================
    def _line_x(self, mask, x_min=0, x_max=None):

        if x_max is None:
            x_max = mask.shape[1]

        region = mask[:, x_min:x_max].astype(np.float32)

        if region.size == 0:
            return None

        # use_lookahead=True:
        #   band 내부에서 위쪽 row에 더 큰 가중치.
        # use_lookahead=False:
        #   band 내부에서 아래쪽 row에 더 큰 가중치.
        if self.use_lookahead:
            row_w = np.linspace(1.0, 0.3, region.shape[0])[:, np.newaxis]
        else:
            row_w = np.linspace(0.3, 1.0, region.shape[0])[:, np.newaxis]

        hist = (region * row_w).sum(axis=0)

        if hist.max() <= 0:
            return None

        indices = np.arange(len(hist), dtype=np.float32)

        return x_min + float(np.sum(hist * indices) / np.sum(hist))


    #=============================================
    # lane center 계산
    #=============================================
    def _lane_center(self, left_x, right_x, width):

        if left_x is not None and right_x is not None:
            self._last_lane_width = right_x - left_x
            return (left_x + right_x) / 2.0, "both"

        if left_x is not None and self._last_lane_width:
            return left_x + self._last_lane_width / 2.0, "left+w"

        if right_x is not None and self._last_lane_width:
            return right_x - self._last_lane_width / 2.0, "right+w"

        if left_x is not None:
            return (left_x + width) / 2.0, "left_only"

        if right_x is not None:
            return right_x / 2.0, "right_only"

        return None, "none"


    #=============================================
    # 중앙 근처 흰선 경고
    #=============================================
    @staticmethod
    def _white_near_center(white_mask, width, band_ratio=0.15):

        if white_mask.sum() == 0:
            return False

        mid = width // 2
        band = int(width * band_ratio)

        in_band = white_mask[:, mid - band: mid + band].sum()

        return bool(in_band > white_mask.sum() * 0.4)


    #=============================================
    # 디버그 화면
    #=============================================
    def draw_debug(self, frame, result):

        out = frame.copy()

        # 원본 이미지에 src_pts 사다리꼴 표시
        pts = self.src_pts.astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 255), 2)

        bev = self._dbg.get("bev")

        if bev is None:
            return out

        bev_out = bev.copy()
        bh, bw = bev_out.shape[:2]

        # far / near band 표시
        for key, color in [
            ("far", (255, 0, 255)),
            ("near", (0, 255, 255)),
        ]:
            band = self._dbg.get(key)

            if band is None:
                continue

            y_min = band["y_min"]
            y_max = band["y_max"]

            cv2.rectangle(
                bev_out,
                (0, y_min),
                (bw - 1, y_max),
                color,
                1
            )

            cx = band["lane_center_x"]

            if cx is not None:
                cy = int((y_min + y_max) / 2)

                cv2.circle(
                    bev_out,
                    (int(cx), cy),
                    6,
                    color,
                    -1
                )

        # 전체 center line 표시
        center = self._dbg.get("center")

        if center is not None:
            cx = center["lane_center_x"]

            if cx is not None:
                cv2.line(
                    bev_out,
                    (int(cx), 0),
                    (int(cx), bh - 1),
                    (255, 255, 255),
                    2
                )

        # BEV 중심선
        cv2.line(
            bev_out,
            (bw // 2, 0),
            (bw // 2, bh - 1),
            (255, 0, 0),
            1
        )

        # 입력 이미지 + BEV 이미지 가로 연결
        target_h = out.shape[0]
        scale = target_h / bh

        bev_scaled = cv2.resize(
            bev_out,
            (int(bev_out.shape[1] * scale), target_h)
        )

        combined = np.hstack([out, bev_scaled])

        hud = [
            f"[BEV lane {self.target_lane}] off={result['lane_center_offset']:+.1f}",
            f"near={result['near_offset']:+.1f}  far={result['far_offset']:+.1f}  curve={result['curve_offset']:+.1f}",
            f"detected={result['lane_detected']}  source={result['source']}  time={self.last_elapsed_ms:.1f}ms",
        ]

        for i, line in enumerate(hud):
            cv2.putText(
                combined,
                line,
                (10, 22 + 22 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                combined,
                line,
                (10, 22 + 22 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return combined


def _cli():

    ap = argparse.ArgumentParser(description="BEV Lane Detector standalone tuner")

    ap.add_argument("images", nargs="+")
    ap.add_argument("--lane", type=int, default=2, choices=[1, 2])
    ap.add_argument("--no-show", action="store_true")

    args = ap.parse_args()

    det = BevLaneDetector(target_lane=args.lane)

    for path in args.images:

        frame = cv2.imread(path)

        if frame is None:
            print(f"[skip] cannot read: {path}")
            continue

        result = det.detect(frame)
        print(f"{path}: {result}")

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
