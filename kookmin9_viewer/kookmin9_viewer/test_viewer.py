"""Xytron Unity 시뮬레이터 검증용 통합 viewer.

토픽:
  - /usb_cam/image_raw/{front,left,right,behind}  sensor_msgs/Image (rgb8)
  - /scan                                          sensor_msgs/LaserScan
  - /imu                                           sensor_msgs/Imu
  - /xycar_motor                                   xycar_msgs/XycarMotor (publish)

조작 키 (matplotlib 창 포커스):
  W/S        speed ±1
  A/D        angle ±5
  Space      정지 (speed=0, angle=0)
  Q          speed=0 (각도 유지)
  ESC        종료

실행:
  ros2 run kookmin9_viewer test_viewer
"""

import math
import threading
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image, LaserScan, Imu
from xycar_msgs.msg import XycarMotor

import matplotlib
matplotlib.use("TkAgg")
# matplotlib 의 default 단축키 (s=save, q=quit, p=pan, l=log, f=fullscreen 등) 가
# 우리 차량 조작 키와 충돌. 충돌 가능한 키맵 모두 비움.
for _k in ("save", "quit", "quit_all", "fullscreen", "home", "back", "forward",
           "pan", "zoom", "grid", "grid_minor", "yscale", "xscale", "copy",
           "help"):
    if f"keymap.{_k}" in matplotlib.rcParams:
        matplotlib.rcParams[f"keymap.{_k}"] = []
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation


# ---------- 설정 상수 -------------------------------------------------------

CAMERA_TOPICS = {
    "front":  "/usb_cam/image_raw/front",
    "left":   "/usb_cam/image_raw/left",
    "right":  "/usb_cam/image_raw/right",
    "behind": "/usb_cam/image_raw/behind",
}

SCAN_TOPIC = "/scan"
IMU_TOPIC = "/imu"
XYCAR_TOPIC = "/xycar_motor"

# 차량 조종 입력 한계 (Unity VehicleControlSubscriber 가 받는 범위)
SPEED_MIN, SPEED_MAX = -50.0, 50.0
ANGLE_MIN, ANGLE_MAX = -100.0, 100.0
SPEED_STEP = 1.0
ANGLE_STEP = 5.0

# matplotlib animation 주기 (ms). 카메라/스캔 표시 부담 고려.
ANIM_INTERVAL_MS = 100  # 10 FPS


# ---------- 공유 상태 -------------------------------------------------------

@dataclass
class Latest:
    """ROS callback 이 채우고 matplotlib animation 이 읽는 공유 상태."""
    cam_front: np.ndarray = None
    cam_left: np.ndarray = None
    cam_right: np.ndarray = None
    cam_behind: np.ndarray = None
    scan_ranges: np.ndarray = None
    scan_angle_min: float = 0.0
    scan_angle_inc: float = 0.0
    scan_range_max: float = 100.0
    imu_rpy: tuple = (0.0, 0.0, 0.0)


# ---------- ROS 노드 --------------------------------------------------------

class TestViewerNode(Node):
    def __init__(self, latest: Latest):
        super().__init__("kookmin9_viewer")
        self.latest = latest
        self._lock = threading.Lock()

        # 카메라 4개 구독
        for slot, topic in CAMERA_TOPICS.items():
            self.create_subscription(
                Image, topic,
                lambda msg, slot=slot: self._on_image(slot, msg),
                10,
            )

        self.create_subscription(LaserScan, SCAN_TOPIC, self._on_scan, 10)
        self.create_subscription(Imu, IMU_TOPIC, self._on_imu, 10)

        self._motor_pub = self.create_publisher(XycarMotor, XYCAR_TOPIC, 10)

        self.target_speed = 0.0
        self.target_angle = 0.0
        # 10Hz heartbeat publish — 시뮬 측 timeout 회피 + 조작감 향상
        self.create_timer(0.1, self._publish_motor)

        self.get_logger().info("kookmin9_viewer 시작. matplotlib 창에서 W/A/S/D 조작.")

    # --- subscribe callbacks ------------------------------------------------

    def _on_image(self, slot: str, msg: Image):
        if msg.encoding != "rgb8":
            return
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        except ValueError:
            return
        with self._lock:
            setattr(self.latest, f"cam_{slot}", arr)

    def _on_scan(self, msg: LaserScan):
        with self._lock:
            self.latest.scan_ranges = np.asarray(msg.ranges, dtype=np.float32)
            self.latest.scan_angle_min = msg.angle_min
            self.latest.scan_angle_inc = msg.angle_increment
            self.latest.scan_range_max = msg.range_max

    def _on_imu(self, msg: Imu):
        q = msg.orientation
        roll, pitch, yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)
        with self._lock:
            self.latest.imu_rpy = (roll, pitch, yaw)

    # --- publish ------------------------------------------------------------

    def _publish_motor(self):
        msg = XycarMotor()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.speed = float(self.target_speed)
        msg.angle = float(self.target_angle)
        self._motor_pub.publish(msg)

    # --- 외부 control API (메인 thread 의 키 이벤트가 호출) ------------------

    def adjust_speed(self, delta: float):
        self.target_speed = float(np.clip(self.target_speed + delta, SPEED_MIN, SPEED_MAX))

    def adjust_angle(self, delta: float):
        self.target_angle = float(np.clip(self.target_angle + delta, ANGLE_MIN, ANGLE_MAX))

    def stop(self):
        self.target_speed = 0.0
        self.target_angle = 0.0

    def speed_zero(self):
        self.target_speed = 0.0


# ---------- 유틸 ------------------------------------------------------------

def quaternion_to_euler(x, y, z, w):
    """ROS 표준 ZYX (yaw-pitch-roll) → roll(X), pitch(Y), yaw(Z) [rad]."""
    # roll
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    # yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


# ---------- matplotlib UI ---------------------------------------------------

class Viewer:
    def __init__(self, node: TestViewerNode, latest: Latest):
        self.node = node
        self.latest = latest

        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title("kookmin9_viewer")
        gs = GridSpec(3, 4, figure=self.fig, height_ratios=[1, 1, 0.8])

        # 카메라 4개 (좌측 2x2)
        self.ax_cam = {
            "front":  self.fig.add_subplot(gs[0, 0]),
            "left":   self.fig.add_subplot(gs[0, 1]),
            "right":  self.fig.add_subplot(gs[1, 0]),
            "behind": self.fig.add_subplot(gs[1, 1]),
        }
        self.cam_im = {}
        for slot, ax in self.ax_cam.items():
            ax.set_title(f"camera/{slot}")
            ax.set_xticks([])
            ax.set_yticks([])
            self.cam_im[slot] = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

        # LiDAR polar (우상단 2x2)
        self.ax_scan = self.fig.add_subplot(gs[0:2, 2:4], projection="polar")
        self.ax_scan.set_title("/scan (LaserScan)")
        self.ax_scan.set_theta_zero_location("N")
        self.ax_scan.set_theta_direction(-1)
        self.scan_line, = self.ax_scan.plot([], [], 'r.', markersize=2)
        self.ax_scan.set_ylim(0, 30)

        # IMU + control (하단 wide row 분할)
        self.ax_imu = self.fig.add_subplot(gs[2, 0:2])
        self.ax_imu.set_xlim(-1.2, 1.2)
        self.ax_imu.set_ylim(-1.2, 1.2)
        self.ax_imu.set_aspect("equal")
        self.ax_imu.set_xticks([])
        self.ax_imu.set_yticks([])
        self.ax_imu.set_title("/imu — roll/pitch/yaw")
        # 화살표 3개: roll(빨강), pitch(녹색), yaw(파랑)
        self._arrows = []  # 매 frame 새로 그림 (Quiver 갱신이 번거로움)
        self._imu_text = self.ax_imu.text(
            -1.15, 1.05, "", fontsize=9, family="monospace",
            verticalalignment="top",
        )

        # control panel (하단 우측)
        self.ax_ctrl = self.fig.add_subplot(gs[2, 2:4])
        self.ax_ctrl.set_xticks([])
        self.ax_ctrl.set_yticks([])
        self.ax_ctrl.set_title("/xycar_motor — control")
        self._ctrl_text = self.ax_ctrl.text(
            0.02, 0.95, "", fontsize=10, family="monospace",
            transform=self.ax_ctrl.transAxes,
            verticalalignment="top",
        )

        # 키 입력
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # 애니메이션
        self.anim = FuncAnimation(
            self.fig, self._update, interval=ANIM_INTERVAL_MS,
            blit=False, cache_frame_data=False,
        )

    def _on_key(self, event):
        k = event.key
        if k in ("w", "up"):
            self.node.adjust_speed(+SPEED_STEP)
        elif k in ("s", "down"):
            self.node.adjust_speed(-SPEED_STEP)
        elif k in ("a", "left"):
            self.node.adjust_angle(-ANGLE_STEP)
        elif k in ("d", "right"):
            self.node.adjust_angle(+ANGLE_STEP)
        elif k == " ":
            self.node.stop()
        elif k == "q":
            self.node.speed_zero()
        elif k == "escape":
            plt.close(self.fig)

    def _update(self, _frame):
        # 카메라
        for slot, im in self.cam_im.items():
            arr = getattr(self.latest, f"cam_{slot}")
            if arr is not None:
                im.set_data(arr)

        # LiDAR
        ranges = self.latest.scan_ranges
        if ranges is not None and ranges.size > 0:
            n = ranges.size
            angles = self.latest.scan_angle_min + np.arange(n) * self.latest.scan_angle_inc
            valid = np.isfinite(ranges) & (ranges < self.latest.scan_range_max - 1e-3)
            self.scan_line.set_data(angles[valid], ranges[valid])
            self.ax_scan.set_ylim(0, max(5.0, float(np.percentile(ranges[valid], 95)) if valid.any() else 30.0))

        # IMU
        roll, pitch, yaw = self.latest.imu_rpy
        for art in self._arrows:
            art.remove()
        self._arrows = []
        # roll, pitch, yaw 각각 단위벡터를 X-Y 평면에 투영
        # roll = X-axis 회전 → red
        # pitch = Y-axis 회전 → green
        # yaw = Z-axis 회전 → blue
        L = 0.9
        self._arrows.append(self.ax_imu.arrow(
            0, 0, L * math.cos(roll), L * math.sin(roll),
            color="red", width=0.02, head_width=0.06, length_includes_head=True))
        self._arrows.append(self.ax_imu.arrow(
            0, 0, L * math.cos(pitch + math.pi/2), L * math.sin(pitch + math.pi/2),
            color="green", width=0.02, head_width=0.06, length_includes_head=True))
        self._arrows.append(self.ax_imu.arrow(
            0, 0, L * math.cos(yaw), L * math.sin(yaw),
            color="blue", width=0.02, head_width=0.06, length_includes_head=True))
        self._imu_text.set_text(
            f"roll  (red)   = {math.degrees(roll):+7.2f} deg\n"
            f"pitch (green) = {math.degrees(pitch):+7.2f} deg\n"
            f"yaw   (blue)  = {math.degrees(yaw):+7.2f} deg"
        )

        # Control
        self._ctrl_text.set_text(
            f"speed = {self.node.target_speed:+6.2f}    angle = {self.node.target_angle:+6.2f}\n"
            f"\n"
            f"  W / S   : speed +/- {SPEED_STEP}\n"
            f"  A / D   : angle +/- {ANGLE_STEP}\n"
            f"  Space   : stop (speed=0, angle=0)\n"
            f"  Q       : speed=0 (keep angle)\n"
            f"  ESC     : exit"
        )

    def show(self):
        plt.show()


# ---------- main ------------------------------------------------------------

def main():
    rclpy.init()
    latest = Latest()
    node = TestViewerNode(latest)

    # rclpy spin 을 백그라운드 thread 에서. matplotlib 는 메인 thread.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    viewer = Viewer(node, latest)
    try:
        viewer.show()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
