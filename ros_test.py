import rospy
import cv2
import cv_bridge

from sensor_msgs.msg import Image

def callback(msg):
    print("Received an image!")
    bridge = cv_bridge.CvBridge()
    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    img = cv2.resize(img, (640, 480))
    cv2.imshow("Image", img)
    cv2.waitKey(1)

print('Start')
rospy.init_node('image_subscriber')
print('Node initialized')
sub = rospy.Subscriber('/camera/image_raw', Image, callback)
# sub = rospy.Subscriber('/sparus2/camera/image_raw', Image, callback)
# sub = rospy.Subscriber('/sparus2/FLS/Img_color/bone/mono', Image, callback)
rospy.spin()

