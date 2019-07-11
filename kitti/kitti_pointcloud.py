#!/usr/bin/env python
import numpy as np
import sys

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2

SEQUENCE_NUMBER = 0
SEQUENCE_LENGTH = 154

DETECTION_COLOR_MAP = {'Car': (255,255,0), 'Pedestrian': (0, 226, 255), 'Cyclist': (141, 40, 255)} # color for detection, in format bgr

def publish_3dbox(box3d_pub, corners_3d_velos, object_types=None):
    """
    Publish 3d boxes in velodyne coordinate, with color specified by object_types
    If object_types is None, set all color to cyan
    corners_3d_velos : list of (8, 4) 3d corners
    """
    marker_array = MarkerArray()
    for i, corners_3d_velo in enumerate(corners_3d_velos):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()

        marker.id = i
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(LIFETIME)
        marker.type = Marker.LINE_LIST

        if object_types is None:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
        else:
            b, g, r = DETECTION_COLOR_MAP[object_types[i]]
            marker.color.r = r/255.0
            marker.color.g = g/255.0
            marker.color.b = b/255.0

        marker.color.a = 1.0

        marker.scale.x = 0.1

        marker.points = []
        for l in LINES:
            p1 = corners_3d_velo[l[0]]
            marker.points.append(Point(p1[0], p1[1], p1[2]))
            p2 = corners_3d_velo[l[1]]
            marker.points.append(Point(p2[0], p2[1], p2[2]))
        marker_array.markers.append(marker)
        
	box3d_pub.publish(marker_array)

def read_tracking(path):
    df = pd.read_csv(path, header=None, sep=' ')
    df.columns = ['frame', 'type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top',
'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']
    df = df[df['track_id']>=0] # remove DontCare objects
    df.loc[df.type.isin(['Bus', 'Truck', 'Van', 'Tram']), 'type'] = 'Car' # Set all vehicle type to Car
    df = df[df.type.isin(['Car', 'Pedestrian', 'Cyclist'])]
    return df

if __name__ == '__main__':
	frame = 0
	rospy.init_node('kitti_pointcloud_node', anonymous=True)
	pcl_pub = rospy.Publisher("kitti_pointcloud", PointCloud2, queue_size=10)
    box3d_pub = rospy.Publisher('kitti_3dboxes', MarkerArray, queue_size=10)
	rate = rospy.Rate(10)

    corners_3d_velos = []
	while not rospy.is_shutdown():
		#header
		header = std_msgs.msg.Header()
		header.stamp = rospy.Time.now()
		header.frame_id = 'map'
		
		#create pcl from points
		points = np.fromfile('/home/ubuntu/data/tracking/velodyne/%04d/%06d.bin'%(SEQUENCE_NUMBER, frame), dtype=np.float32).reshape(-1, 4)
		# if we want the intensity
		fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('i', 12, PointField.FLOAT32, 1)]
		pcl_msg = pcl2.create_cloud(header, fields, points)
		# # if we don't want to display the intensity
		# pcl_msg = pcl2.create_cloud_xyz32(header, points[:, :3])
		
		#publish    
		rospy.loginfo("publishing pointcloud.. !")
		pcl_pub.publish(pcl_msg)
		frame += 1
		frame %= SEQUENCE_LENGTH
		rate.sleep()
