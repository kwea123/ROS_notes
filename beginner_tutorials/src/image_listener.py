#!/usr/bin/env python
from __future__ import print_function

#import roslib
#roslib.load_manifest('beginner_tutorials')
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String

class image_converter:

  def __init__(self):

    self.image_sub = rospy.Subscriber("picamera", String, self.callback)
    self.n_frames = 0

  def callback(self,data):
    
    cv_image = cv2.imdecode(np.fromstring(data.data, dtype=np.uint8), -1)
    #cv2.imwrite('image%02d.jpg'%self.n_frames, cv_image)
    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)
    self.n_frames += 1

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
