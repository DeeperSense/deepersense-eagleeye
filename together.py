import rospy
import message_filters
from sensor_msgs.msg import Image

def callback(image1, image2):
    print('cbd')


def listener():

    rospy.init_node('listener', anonymous=True)

    image1_sub = message_filters.Subscriber('/camera/image_raw', Image)
    image2_sub = message_filters.Subscriber('/oculus/drawn_sonar', Image)

    queue_size = 10
    slop = 0.1
    ts = message_filters.ApproximateTimeSynchronizer([image1_sub, image2_sub], queue_size, slop, allow_headerless=True)
    ts.registerCallback(callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()