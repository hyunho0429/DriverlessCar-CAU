#!/usr/bin/env python3

import cv2
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge

class CamViewerNode(Node):
    def __init__(self):
        super().__init__('cam_viewer')

        self.bridge = CvBridge()

        self.images = {
            "front": None,
            "back": None,
            "left": None,
            "right": None
        }

        # Subscribers
        self.sub_front = self.create_subscription(
            Image, '/usb_cam/image_raw/front', self.img_callback1, qos_profile_sensor_data)

        self.sub_back = self.create_subscription(
            Image, '/usb_cam/image_raw/behind', self.img_callback2, qos_profile_sensor_data)

        self.sub_left = self.create_subscription(
            Image, '/usb_cam/image_raw/left', self.img_callback3 , qos_profile_sensor_data)

        self.sub_right = self.create_subscription(
            Image, '/usb_cam/image_raw/right', self.img_callback4, qos_profile_sensor_data)

        # Timer (30 FPS)
        self.timer = self.create_timer(0.03, self.process_images)

    def img_callback1(self, data):
        self.images["front"] = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def img_callback2(self, data):
        self.images["back"] = self.bridge.imgmsg_to_cv2(data, "bgr8")
    
    def img_callback3(self, data):
        self.images["left"] = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def img_callback4(self, data):        
        self.images["right"] = self.bridge.imgmsg_to_cv2(data, "bgr8")        

    def process_images(self):

        if any(v is None for v in self.images.values()):
            return
            
        h, w = 240, 320

        f = cv2.resize(self.images["front"], (w, h))
        b = cv2.resize(self.images["back"],  (w, h))
        l = cv2.resize(self.images["left"],  (w, h))
        r = cv2.resize(self.images["right"], (w, h))

        top = np.hstack((f, r))
        bottom = np.hstack((l, b))
        combined = np.vstack((top, bottom))

        cv2.imshow("4 Cameras", combined)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = CamViewerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()