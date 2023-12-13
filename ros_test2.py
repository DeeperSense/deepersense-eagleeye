import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

def callback(msg):
    print("Received an image!")
    bridge = CvBridge()

    try:
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # img = cv2.resize(img, (640, 480))
        cv2.imshow("Image", img)
        cv2.waitKey(1)

        # Re-convert the processed image to Image msg and publish
        img_msg = bridge.cv2_to_imgmsg(img, encoding='bgr8')
        pub.publish(img_msg)
    except CvBridgeError as e:
        print(e)

rospy.init_node('image_subscriber')
sub = rospy.Subscriber('/sparus2/camera/image_raw', Image, callback)
pub = rospy.Publisher('output_image_topic', Image, queue_size=10)
rospy.spin()