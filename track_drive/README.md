# track_drive

Xytron Unity 시뮬레이터용 자율주행 노드.  
전방 카메라(Front Perspective) + 버드아이뷰(BEV) **듀얼 디텍터 퓨전** 방식으로 차선을 인식하고, **비선형 PD 제어**로 조향합니다.

## 실행

```bash
# ROS 2 워크스페이스 루트에서
colcon build --symlink-install --packages-select track_drive
source install/setup.bash
ros2 run track_drive track_drive
```

## 파일 구조

| 파일 | 설명 |
| --- | --- |
| `track_drive.py` | ROS 2 노드. 센서 퓨전, 비선형 PD 제어, 자동 차선 전환 |
| `lane_detector.py` | 전방 카메라 뷰 기반 차선 검출 (HSV 마스크 + 컬럼 히스토그램) |
| `bev_lane_detector.py` | Bird's-Eye View 변환 후 차선 검출 (Perspective Warp + 히스토그램) |
| `hsv_probe.py` | 이미지 클릭으로 BGR/HSV 값 확인하는 유틸리티 |

## 구조 요약

```
카메라 영상 (640×480)
    │
    ├─ LaneDetector (전방 원근 뷰, ROI 하단 60%)
    │    └─ HSV 마스크 → 컬럼 히스토그램 → 차선 x좌표
    │
    └─ BevLaneDetector (원근 변환 → 400×400 탑다운 뷰)
         └─ HSV 마스크 → 컬럼 히스토그램 → 차선 x좌표
    │
    ▼
 센서 퓨전 (가중 평균: Front 65% + BEV 35%)
    │
    ▼
 비선형 PD 제어: angle = -(kp·e + kq·e·|e| + kd·de)
    │
    ▼
 /xycar_motor 토픽 발행 (angle, speed)
```

## 주요 기능

### 듀얼 디텍터 퓨전
- 1/2차선 각각에 대해 Front + BEV 검출기를 동시 구동 (총 4개)
- 두 뷰의 조향값을 가중 평균하여 최종 조향각 산출
- 한쪽 뷰가 실패해도 나머지 뷰로 주행 지속 (Fail-safe)

### 비선형 PD 조향 제어
- **선형항 (kp·e)**: 작은 오차에서 기본 반응
- **2차항 (kq·e·|e|)**: 큰 오차(코너)에서 급격히 증폭, 직선에선 거의 0
- **미분항 (kd·de)**: 과보정(지그재그) 억제

### 자동 차선 전환
- 양쪽 차선의 검출 신뢰도(score)를 매 프레임 비교
- 상대 차선이 일정 마진 이상 우수하면 4프레임 연속 확인 후 차선 전환
- 노이즈에 의한 오전환 방지를 위한 히스테리시스 적용

## 파라미터 조정

`track_drive.py` 내 `__init__` 에서 수정:

```python
self.target_lane = 2        # 시작 차선 (1: 왼쪽, 2: 오른쪽)
self.kp_f = 0.40            # Front 선형 이득
self.kq_f = 0.004           # Front 2차 이득
self.kd_f = 0.18            # Front 미분 이득
self.base_speed = 8.0       # 직선 기본 속도
self.w_f  = 0.65            # 퓨전 가중치 (Front)
self.w_b  = 0.35            # 퓨전 가중치 (BEV)
```

## 사용된 OpenCV 기술

- **색상 변환**: `cvtColor` (BGR → HSV)
- **색상 마스킹**: `inRange` (노란선/흰선 이진 분할)
- **모폴로지**: `morphologyEx` (MORPH_CLOSE, 점선 갭 연결)
- **원근 변환**: `getPerspectiveTransform` + `warpPerspective` (BEV 생성)

## 디버그

실행 시 두 개의 OpenCV 창이 표시됩니다:
- **front**: 전방 카메라 뷰 + 차선 검출 오버레이
- **bev**: 버드아이뷰 + 차선 검출 오버레이

단독 테스트:
```bash
python3 track_drive/lane_detector.py screenshot.png --lane 2
python3 track_drive/bev_lane_detector.py screenshot.png --lane 2
python3 track_drive/hsv_probe.py screenshot.png
```
