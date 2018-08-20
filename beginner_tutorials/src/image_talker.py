#!/usr/bin/env python
from __future__ import print_function

#import roslib
#roslib.load_manifest('beginner_tutorials')
import sys
import rospy
import numpy as np
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic",Image)

    self.bridge = CvBridge()

  def send(self):
    cv_image = np.zeros((64, 64, 3), dtype=np.uint8)
    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)
      
if __name__ == '__main__':
    ic = image_converter()
    rospy.init_node('image_talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        ic.send()
        rate.sleep()
