#!/usr/bin/env python3
"""ROI 내 픽셀을 클릭하면 BGR / HSV 값을 출력합니다.
노란선, 흰선, 도로를 각각 클릭해서 실제 HSV 범위 확인

실행:
    python3 hsv_probe.py 이미지경로.jpg
"""

# 확인용코드입니다!!!

import sys
import cv2
import numpy as np

img = cv2.imread(sys.argv[1])
if img is None:
    print("이미지를 열 수 없습니다:", sys.argv[1])
    sys.exit(1)

h, w = img.shape[:2]
roi_top = int(h * 0.60)
roi = img[roi_top:, :].copy()
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

print(f"이미지 크기: {w}x{h}  |  ROI: y={roi_top}~{h}")
print("ROI 창에서 노란선 / 흰선 / 도로를 클릭하세요. ESC = 종료\n")

def on_mouse(event, x, y, flags, _):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    bgr = roi[y, x]
    h_val = hsv[y, x]
    print(f"  클릭 ({x},{y})  BGR={bgr}  HSV={h_val}")

cv2.namedWindow("hsv_probe")
cv2.setMouseCallback("hsv_probe", on_mouse)

while True:
    cv2.imshow("hsv_probe", roi)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
