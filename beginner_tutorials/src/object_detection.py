#!/usr/bin/env python
import rospy
from std_msgs.msg import String

import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
import time
import cv2
import yaml # load label_dict

MODEL_PATH = '/home/ubuntu/workspace/models/research/object_detection/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'

LABEL_DICT_PATH = os.path.join(os.path.dirname(__file__), 'coco_label_dict.txt')

IMAGE_TOPIC = 'picamera'

class Object_Detector():
	def __init__(self, model_path, label_dict_path, subscribed_topic):
		self.__load_model(model_path, label_dict_path)
		print('model loaded')
		self.__image_sub = rospy.Subscriber(subscribed_topic, String, self.callback)
		self.n_frames = 0

	def __load_model(self, model_path, label_dict_path):

		# load tf model
		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(model_path, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

		config = tf.ConfigProto()
		config.gpu_options.allow_growth= True

		with self.detection_graph.as_default():
			self.sess = tf.Session(config=config, graph=self.detection_graph)
			self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
			self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
			self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
			self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
			self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

		# load label_dict
		with open(label_dict_path, 'r') as f:
			self.label_dict = yaml.load(f)
		
		# warmup
		self.detect_image(np.ones((600, 600, 3)))

	def detect_image(self, image_np):
		image_w, image_h = image_np.shape[1], image_np.shape[0]
    
		# Actual detection.
		t = time.time()
		(boxes, scores, classes, num) = self.sess.run(
		  [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
		  feed_dict={self.image_tensor: np.expand_dims(image_np, axis=0)})
		print('detection time :', time.time()-t)
		# Visualization of the results of a detection.
		for i, box in enumerate(boxes[scores>0.5]):
		    top_left = (int(box[1]*image_w), int(box[0]*image_h))
		    bottom_right = (int(box[3]*image_w), int(box[2]*image_h))
		    cv2.rectangle(image_np, top_left, bottom_right, (0,255,0), 3)
		    cv2.putText(image_np, self.label_dict[int(classes[0,i])], top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

		return image_np
	
	def detect_images_from_paths(self, list_of_image_paths):
		for image_path in list_of_image_paths:
			image_np = np.array(Image.open(image_path))
			image_np = object_detector.detect_image(image_np)
			cv2.imshow('image', image_np[...,::-1])
			cv2.waitKey(1)

	def callback(self, data):
		cv_image = cv2.imdecode(np.fromstring(data.data, dtype=np.uint8), -1)
		cv_image = self.detect_image(cv_image)
		cv2.imshow('image', cv_image)
		cv2.waitKey(1)

if __name__ == '__main__':
	object_detector = Object_Detector(MODEL_PATH, LABEL_DICT_PATH, IMAGE_TOPIC)

	#TEST_IMAGE_PATHS = ['/home/ubuntu/image%02d.jpg'%i for i in range(60)]
	#object_detector.detect_images_from_paths(TEST_IMAGE_PATHS)
	rospy.init_node('object_detector', anonymous=True)
	rospy.spin()
	time.sleep(1)
	cv2.destroyAllWindows()
