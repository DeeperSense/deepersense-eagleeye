#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class ImageFlip:

    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/oculus/drawn_sonar_rect", Image, self.callback)
        self.image_pub = rospy.Publisher("/oculus/drawn_sonar_rect_flipped", Image, queue_size=1)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
            return

        # Flip image (0 - horizontal, 1 - vertical, -1 - both)
        flipped_img = cv2.flip(cv_image, 1)

        try:
            msg = self.bridge.cv2_to_imgmsg(flipped_img, encoding="bgr8")
        except CvBridgeError as e:
            print(e)
            return
        
        # copy header from original image
        msg.header = data.header
        

        self.image_pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('image_flip')
    img_flp_obj = ImageFlip()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")