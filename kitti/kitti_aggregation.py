#!/usr/bin/env python
import numpy as np
import sys

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2

DATA_PATH = '/home/ubuntu/data/kitti/RawData/2011_09_26/2011_09_26_drive_0009_sync/velodyne_points/data/'

import pandas as pd
import numpy as np
import yaml

with open('/home/ubuntu/data/kitti/RawData/2011_09_26/calib_velo_to_cam.txt', 'r') as f:
    yml = yaml.load(f)

R_velo_to_cam2 = np.array([float(i) for i in yml['R'].split(' ')]).reshape(3, 3)
T_velo_to_cam2 = np.array([float(i) for i in yml['T'].split(' ')]).reshape(3, 1)

Tr_velo_to_cam2 = np.vstack((np.hstack([R_velo_to_cam2, T_velo_to_cam2]), [0, 0, 0, 1]))

COLUMN_NAMES = ['frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
                'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']
df = pd.read_csv('/home/ubuntu/data/kitti/tracking/training/label_02/0001.txt', header=None, sep=' ')
df.columns = COLUMN_NAMES
df = df[df['track_id']>=0]

SEQUENCE_LENGTH = 447

def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
	"""
	Return : 3xn in cam2 coordinate
	"""
	R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
	x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
	y_corners = [0,0,0,0,-h,-h,-h,-h]
	z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
	corners_3d_cam2 = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
	corners_3d_cam2[0,:] += x
	corners_3d_cam2[1,:] += y
	corners_3d_cam2[2,:] += z
	return corners_3d_cam2

def cam2_3d_to_velo(corners_3d_cam2):
	"""
	Input : 3xn in cam2 coordinate
	Return : 4xn in velo coordinate
	"""
	return np.linalg.inv(Tr_velo_to_cam2).dot(np.r_[corners_3d_cam2, np.ones((1, 8))])

LINES = [[0, 1], [1, 2], [2, 3], [3, 0]] # lower face
LINES+= [[4, 5], [5, 6], [6, 7], [7, 4]] # upper face
LINES+= [[4, 0], [5, 1], [6, 2], [7, 3]] # connect lower face and upper face
LINES+= [[4, 1], [5, 0]] # front face

def publish_3dbox(marker_pub, corners_3d_velo, track_id, publish_id=True):
	marker = Marker()
	marker.header.frame_id = "map"
	marker.header.stamp = rospy.Time.now()

	marker.id = track_id
	marker.action = Marker.ADD
	marker.lifetime = rospy.Duration(0.1)
	marker.type = Marker.LINE_LIST
	marker.text = "blablabla"

	marker.color.r = 0.0
	marker.color.g = 1.0
	marker.color.b = 1.0
	marker.color.a = 1.0

	marker.scale.x = 0.1

	marker.points = []
	for l in LINES:
		p1 = corners_3d_velo[l[0]]
		marker.points.append(Point(p1[0], p1[1], p1[2]))
		p2 = corners_3d_velo[l[1]]
		marker.points.append(Point(p2[0], p2[1], p2[2]))
	marker_pub.publish(marker)
	
	if publish_id:
		text_marker = Marker()
		text_marker.header.frame_id = "map"
		text_marker.header.stamp = rospy.Time.now()

		text_marker.id = track_id + 1000
		text_marker.action = Marker.ADD
		text_marker.lifetime = rospy.Duration(0.1)
		text_marker.type = Marker.TEXT_VIEW_FACING

		p4 = corners_3d_velo[4]

		text_marker.pose.position.x = p4[0]
		text_marker.pose.position.y = p4[1]
		text_marker.pose.position.z = p4[2] + 0.5

		text_marker.text = str(track_id)

		text_marker.scale.x = 1
		text_marker.scale.y = 1
		text_marker.scale.z = 1

		text_marker.color.r = 0.0
		text_marker.color.g = 1.0
		text_marker.color.b = 1.0
		text_marker.color.a = 1.0
		marker_pub.publish(text_marker)

	rospy.loginfo("box %s published"%track_id)

def publish_car_fov(marker_pub):
	marker = Marker()
	marker.header.frame_id = "map"
	marker.header.stamp = rospy.Time.now()

	#marker.id = i-10
	marker.action = Marker.ADD
	marker.lifetime = rospy.Duration()
	marker.type = Marker.LINE_STRIP

	marker.color.r = 0.0
	marker.color.g = 1.0
	marker.color.b = 0.0
	marker.color.a = 1.0
	marker.scale.x = 0.2

	marker.points = []
	marker.points.append(Point(10, -10, 0))
	marker.points.append(Point(0, 0, 0))
	marker.points.append(Point(10, 10, 0))
	marker_pub.publish(marker)

if __name__ == '__main__':
	frame = 0
	rospy.init_node('kitti_pointcloud_node', anonymous=True)
	pcl_pub = rospy.Publisher("kitti_pointcloud", PointCloud2, queue_size=10)
	box_pub = rospy.Publisher("kitti_3dbox", Marker, queue_size=10)
	rate = rospy.Rate(10)

	while not rospy.is_shutdown():
		#header
		header = std_msgs.msg.Header()
		header.stamp = rospy.Time.now()
		header.frame_id = 'map'
		
		#create pcl from points
		points = np.fromfile(DATA_PATH+'%010d.bin'%frame, dtype=np.float32).reshape(-1, 4)
		# if we want the intensity
		fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('i', 12, PointField.FLOAT32, 1)]
		pcl_msg = pcl2.create_cloud(header, fields, points)
		# # if we don't want to display the intensity
		# pcl_msg = pcl2.create_cloud_xyz32(header, points[:, :3])
		
		#publish    
		rospy.loginfo("pointcloud published")
		pcl_pub.publish(pcl_msg)

		# Publish 3d box
		df_frame = df[df['frame']==frame]
		df_frame.reset_index(inplace=True, drop=True)

		for i in range(len(df_frame)):
			corners_3d_cam2 = compute_3d_box_cam2(*np.array(df_frame.loc[i, ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']]))
			corners_3d_velo = cam2_3d_to_velo(corners_3d_cam2)
			corners_3d_velo = corners_3d_velo.T # 8x4

			publish_3dbox(box_pub, corners_3d_velo, df_frame.loc[i, 'track_id'], publish_id=True)

		# Publish car FOV
		publish_car_fov(box_pub)

		frame += 1
		frame %= SEQUENCE_LENGTH
		rate.sleep()
