#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class BevPointPicker(Node):

    def __init__(self):
        super().__init__('bev_point_picker')

        self.bridge = CvBridge()
        self.image = None
        self.frozen = False

        self.points = []

        self.bev_width = 400
        self.bev_height = 400

        self.sub = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self.image_callback,
            qos_profile_sensor_data
        )

        cv2.namedWindow("front")
        cv2.setMouseCallback("front", self.mouse_callback)

        self.get_logger().info("BEV point picker started.")
        self.get_logger().info("Click order: LEFT-BOTTOM, LEFT-TOP, RIGHT-TOP, RIGHT-BOTTOM")
        self.get_logger().info("Keys: r=reset, f=freeze/unfreeze, q=quit")

    def image_callback(self, msg):
        if self.frozen:
            return

        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.image is None:
            return

        if len(self.points) >= 4:
            return

        self.points.append((x, y))
        print(f"point {len(self.points)} = ({x}, {y})")

        if len(self.points) == 4:
            self.print_result()
            self.show_bev_preview()

    def draw_points(self, frame):
        out = frame.copy()

        labels = ["LB", "LT", "RT", "RB"]

        for i, p in enumerate(self.points):
            x, y = p
            cv2.circle(out, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(
                out,
                labels[i],
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        if len(self.points) == 4:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(out, [pts], True, (0, 255, 255), 2)

        info = [
            "Click order: 1 LB, 2 LT, 3 RT, 4 RB",
            "r: reset   f: freeze/unfreeze   q: quit",
            f"points: {len(self.points)}/4"
        ]

        for i, text in enumerate(info):
            cv2.putText(
                out,
                text,
                (10, 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                3,
                cv2.LINE_AA
            )
            cv2.putText(
                out,
                text,
                (10, 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return out

    def print_result(self):
        print("\n======================================")
        print("Copy this into bev_lane_detector.py")
        print("Order: left-bottom, left-top, right-top, right-bottom")
        print("--------------------------------------")
        print(
            "src_pts=(({}, {}), ({}, {}), ({}, {}), ({}, {}))".format(
                self.points[0][0], self.points[0][1],
                self.points[1][0], self.points[1][1],
                self.points[2][0], self.points[2][1],
                self.points[3][0], self.points[3][1],
            )
        )
        print("======================================\n")

    def show_bev_preview(self):
        if self.image is None or len(self.points) != 4:
            return

        src_pts = np.float32(self.points)

        dst_pts = np.float32([
            [0, self.bev_height],
            [0, 0],
            [self.bev_width, 0],
            [self.bev_width, self.bev_height],
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        bev = cv2.warpPerspective(
            self.image,
            M,
            (self.bev_width, self.bev_height)
        )

        # BEV 중심선 표시
        cv2.line(
            bev,
            (self.bev_width // 2, 0),
            (self.bev_width // 2, self.bev_height - 1),
            (255, 0, 0),
            1
        )

        cv2.imshow("bev_preview", bev)

    def main_loop(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)

            if self.image is not None:
                debug = self.draw_points(self.image)
                cv2.imshow("front", debug)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                self.points = []
                cv2.destroyWindow("bev_preview")
                cv2.namedWindow("bev_preview")
                print("reset points")

            elif key == ord('f'):
                self.frozen = not self.frozen
                print(f"frozen = {self.frozen}")

            elif key == ord('q') or key == 27:
                break


def main(args=None):
    rclpy.init(args=args)
    node = BevPointPicker()

    try:
        node.main_loop()

    except KeyboardInterrupt:
        pass

    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()