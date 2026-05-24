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
- **`track_drive/track_drive.py`** — the integrated self-driving node template. Subscribes to camera + lidar, publishes `XycarMotor`. Currently runs a stub `main_loop` (alternating stop / forward) inside a blocking `while rclpy.ok()` — meaning **callbacks don't run during the loop** because `rclpy.spin` is never called. Real driving logic added here needs to either restructure around timers or interleave `rclpy.spin_once()`.
- **`kookmin9_viewer/test_viewer.py`** — the most useful single node for sim verification. matplotlib UI with 4 camera tiles, polar LiDAR plot, IMU arrows, and W/A/S/D control. Runs `rclpy.spin` on a background thread (matplotlib must own the main thread). It disables matplotlib's default keymap (`save`, `quit`, `pan`, …) to free `s`, `q`, `p`, etc. for vehicle control — preserve that block if you add keys. Publishes `/xycar_motor` at 10 Hz as a heartbeat (the sim treats prolonged silence as a control timeout).
- **`ROS-TCP-Endpoint-main-ros2/`** — vendored Unity ROS-TCP-Connector endpoint. Don't modify casually; it's a snapshot of Unity-Technologies/ROS-TCP-Endpoint and most edits belong upstream.

## Notes on the existing code

- `my_motor/go.py` and `my_motor/go_stop.py` reference an undefined `driver_node` in their `finally` blocks (should be `node`). This is a latent bug — fix if you touch those files for any other reason, otherwise leave alone.
- The codebase is bilingual (Korean comments throughout). Match the existing language when editing a file.
- `track_drive/track_drive.py`'s header notes a commercial license restriction from Xytron; only edit it for the simulator/coursework use it was distributed for.
