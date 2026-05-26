# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A collection of ROS 2 (ament) packages for the Xycar / Xytron Unity self-driving simulator. The directory is not a workspace itself — it is meant to be cloned (or symlinked) under a ROS 2 workspace's `src/` and built with `colcon`. The Unity sim and the ROS 2 nodes communicate over the ROS-TCP-Connector bridge that lives in `ROS-TCP-Endpoint-main-ros2/`.

## Build & run

`xycar_msgs` (CMake / `rosidl`) must be built first — every other package depends on its `XycarMotor` / `XycarUltrasonic` messages. `colcon` resolves this automatically; just don't try to build a single dependent package without it.

```bash
# from your ROS 2 workspace root (this dir under src/)
colcon build --symlink-install
source install/setup.bash
```

Common runtime entry points (all `ros2 run <pkg> <exe>`):

| Package | Executables |
| --- | --- |
| `my_motor` | `go`, `go_stop` |
| `my_cam` | `cam_viewer` |
| `my_imu` | `roll_pitch_yaw` |
| `my_lidar` | `lidar_scan`, `lidar_viewer` |
| `track_drive` | `track_drive` |
| `kookmin9_viewer` | `test_viewer` |
| `ros_tcp_endpoint` | `default_server_endpoint` (use `ros2 launch ros_tcp_endpoint endpoint.py`) |

The Unity bridge launch binds `ROS_IP=0.0.0.0` and `ROS_TCP_PORT=10000` (see `ROS-TCP-Endpoint-main-ros2/launch/endpoint.py`). The sim must be configured to connect to those.

Tests use `pytest` via `colcon test` (ament_python lint tests are declared in each `package.xml` but the demo nodes have no real unit tests).

## Topic & message contract

All sensor/control packages here speak to the **same** topic set published by the Unity sim — keep names consistent if you add new nodes:

- `/usb_cam/image_raw/{front,left,right,behind}` — `sensor_msgs/Image`
- `/scan` — `sensor_msgs/LaserScan`
- `/imu` — `sensor_msgs/Imu`
- `/xycar_motor` — `xycar_msgs/XycarMotor` (`{header, float32 angle, float32 speed}`)

Sim-side control input bounds (enforced in `kookmin9_viewer/test_viewer.py`): `speed ∈ [-50, 50]`, `angle ∈ [-100, 100]`.

**Image encoding mismatch — easy gotcha.** The Unity sim publishes images as `rgb8`. `kookmin9_viewer/test_viewer.py` reads `msg.data` directly as a numpy buffer and **rejects anything that isn't `rgb8`** (`_on_image`). The other consumers (`my_cam/cam_viewer.py`, `track_drive/track_drive.py`) use `cv_bridge.imgmsg_to_cv2(data, "bgr8")`, which silently converts. If you write a new viewer, pick a lane and document it; mixing them is what causes "black image" bugs.

Sensor subscriptions use `qos_profile_sensor_data` (BEST_EFFORT) in `my_*` nodes; the kookmin viewer uses the default reliable QoS with depth 10. Match the publisher's QoS or the subscription silently drops.

## Architecture map

- **`xycar_msgs/`** — CMake/`rosidl` package. Only message defs (`msg/XycarMotor.msg`, `msg/XycarUltrasonic.msg`). All Python packages list it as a runtime dep.
- **`my_motor/`, `my_cam/`, `my_imu/`, `my_lidar/`** — single-purpose demo nodes, one file each. Useful as starting templates; their pattern is "subscribe with sensor QoS → cache latest in `self.x` → periodic `create_timer` callback prints/displays".
- **`track_drive/`** — 자율주행 패키지. 아래 세 모듈로 구성됨:
  - **`lane_detector.py`** — 전면뷰 HSV 차선 감지기. ROI(하단 40%) + 컬럼 히스토그램. `use_lookahead` 플래그로 행 가중치 방향 전환.
  - **`bev_lane_detector.py`** — Bird's Eye View 차선 감지기. Perspective warp → HSV 마스크 → 컬럼 히스토그램. `src_pts` 보정값은 실측 기반(640×480 기준).
  - **`track_drive.py`** — ROS2 노드. 두 detector를 동시에 돌려 결과를 융합해 `XycarMotor`를 발행.
- **`kookmin9_viewer/test_viewer.py`** — the most useful single node for sim verification. matplotlib UI with 4 camera tiles, polar LiDAR plot, IMU arrows, and W/A/S/D control. Runs `rclpy.spin` on a background thread (matplotlib must own the main thread). It disables matplotlib's default keymap (`save`, `quit`, `pan`, …) to free `s`, `q`, `p`, etc. for vehicle control — preserve that block if you add keys. Publishes `/xycar_motor` at 10 Hz as a heartbeat (the sim treats prolonged silence as a control timeout).
- **`ROS-TCP-Endpoint-main-ros2/`** — vendored Unity ROS-TCP-Connector endpoint. Don't modify casually; it's a snapshot of Unity-Technologies/ROS-TCP-Endpoint and most edits belong upstream.

## track_drive 상세

### 차선 구조 (Kookmin 트랙)
```
[흰선] | 1차선 | [노란 점선] | 2차선 | [흰선]
```
- `target_lane = 1`: 왼쪽 흰선 ~ 노란선
- `target_lane = 2`: 노란선 ~ 오른쪽 흰선 (기본값)

### HSV 파라미터 (실측값)
- 노란선: H=[20-35], S=[150-255], V=[150-255]
- 흰선:   H=[0-180], S=[0-40],   V=[190-255]

### BEV 사다리꼴 src_pts (640×480 기준, 실측 보정)
```python
src_pts = ((10, 415), (220, 260), (390, 260), (630, 415))
#           좌하        좌상        우상        우하
# y=260: 흰선 L≈225, R≈387  (y=250 이상은 배경 노이즈)
# y=415: 흰선 L≈10,  R≈630
```

### 제어 알고리즘: 비선형 PD + 퓨전
두 detector(front, BEV)를 매 프레임 동시 실행해 가중 합산 후 clip:

```
angle = clip(raw_f * w_f + raw_b * w_b, -90, 90)

raw = -(kp * e + kq * e * |e| + kd * de)
         선형항     2차항(코너 증폭)   미분항
```

파라미터:

| | front (640px) | BEV (400px) |
|--|--|--|
| kp | 0.40 | 0.50 |
| kq | 0.004 | 0.005 |
| kd | 0.18 | 0.20 |
| 융합 가중치 | 0.65 | 0.35 |

kq 효과: offset=20px → 각도 거의 그대로 (직선 안정), offset=100px → +40° 추가 (코너 증폭)

속도: `speed = base_speed * max(0.20, 1.0 - abs(angle) / 90.0)`

### 감지 실패 처리
`lane_detected=False`가 front/BEV **둘 다** 반환되면 직전 angle/speed를 그대로 발행 (`_prev_angle`, `_prev_speed`). 로그에 `[__] both lost — holding` 출력.

### 디버그 창
`cam_callback` → `_process()` → `cv2.imshow("front", ...)` + `cv2.imshow("bev", ...)` 두 창이 뜸. 각 창에서 감지된 차선 중심선, offset, source를 실시간으로 확인 가능.

### 독립 실행 (시뮬레이터 없이 이미지 튜닝)
```bash
python3 track_drive/track_drive/lane_detector.py 이미지.png --lane 2
python3 track_drive/track_drive/bev_lane_detector.py 이미지.png --lane 2
python3 track_drive/track_drive/hsv_probe.py 이미지.png  # 픽셀 클릭 → HSV 확인
```

## 주요 주의사항

- `my_motor/go.py`, `go_stop.py`의 `finally` 블록에서 `driver_node`를 참조하나 실제 변수명은 `node` — 잠재적 버그, 해당 파일 수정 시 같이 고칠 것.
- 코드베이스는 한국어/영어 혼용. 파일 수정 시 기존 언어 스타일을 유지할 것.
- `track_drive/track_drive.py` 헤더에 Xytron 상업 라이선스 명시. 시뮬레이터/수업 용도 외 배포 금지.
- BEV `src_pts` 변경 시 반드시 실측 이미지로 검증할 것 — 직선 도로 가정 기반 보정이므로 카메라 마운트 위치가 달라지면 무효화됨.
- 융합 계산 시 clip은 **합산 후 한 번만** 적용해야 함. 개별 clip 후 평균하면 포화된 신호가 희석됨.
