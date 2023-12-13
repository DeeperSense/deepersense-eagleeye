#!/usr/bin/env python3

import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

rospy.init_node('dual', anonymous=True)
image_pub = rospy.Publisher("dual_image", Image, queue_size=10)
bridge = CvBridge()

def callback(data):
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

    processed_image = bridge.cv2_to_imgmsg(cv_image, "bgr8")
    image_pub.publish(processed_image)
    print("published")


image_sub = rospy.Subscriber("/sparus2/camera/image_raw", Image, callback)

rospy.spin()
