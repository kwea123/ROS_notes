#!/usr/bin/env python2
import numpy as np
import sys
import pandas as pd
import cv2
from collections import deque
import time
import struct
import yaml

import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
import tf as ros_tf
from cv_bridge import CvBridge, CvBridgeError

from kitti_util import *

TRACKING = True
DETECTION = False
BACK = False # cars behind
DEPTH2POINTS = False # use points that are generated from depth map
DISPLAY_POINT_COLOR = DEPTH2POINTS

RAWDATA = False
# DATE = '2011_09_26'
# DATANAME = DATE+'_drive_%04d_sync'%51
# SEQUENCE_LENGTH = 438

KISOKEN = False
# START_FRAME = 5473
# END_FRAME = 5873
# SEQUENCE_NAME = '180301_0011_%010d_%010d'%(START_FRAME, END_FRAME)

PRIUS = False
# START_FRAME = 1
# SEQUENCE_NAME = '20180626_%04d'%0

KISOKEN2 = False
# START_FRAME = 0
# END_FRAME = 3144
# SEQUENCE_NAME = 'drive_%04d_sync'%0

KISOKEN_P = False
# START_FRAME = 9100
# END_FRAME = 15500
# SEQUENCE_NAME = 'drive_%04d_sync'%2

VSYS = False
# SEQUENCE_NAME = 'drive_%04d'%7

VSYS2 = False
# SEQUENCE_NAME = 'drv00_2019-03-12-13-37-20'
# SEQUENCE_NAME = 'drv01_2019-02-19-09-50-00'

VSYS_URA = False
# SEQUENCE_NAME = '20190703-17-11-28'

RX = False
# SEQUENCE_NAME = 'rx_to_label'

SEQUENCE_LENGTH = 8000

DATASET = 'testing'
SEQUENCE_NUMBER = 14

SIXTEEN_LINES = False

AB3DMOT = True
SHIGE = False # shigenaka detection results
SHIGE_DETECTOR = 'pointpillars'

DETECTION_COLOR_MAP = {'Car': (255, 255, 0), 'Truck': (80, 127, 255),
                        'Pedestrian': (0, 226, 0), 'Cyclist': (141, 40, 255),
                        'tlr': (0, 255, 255), 'far_object': (180, 105, 255)} # color for detection, in format bgr

LINES = [[0, 1], [1, 2], [2, 3], [3, 0]] # lower face
LINES+= [[4, 5], [5, 6], [6, 7], [7, 4]] # upper face
LINES+= [[4, 0], [5, 1], [6, 2], [7, 3]] # connect lower face and upper face
# LINES+= [[4, 1], [5, 0]] # front face

FRAME_ID = 'map'
RATE = 10
LIFETIME = 1.0/RATE

# for tracking only
MIN_TRACKED_FRAMES = 3

def rgb_to_float32(r, g, b):
    """
    Input : r, g, b integer values in range [0, 255]
    Output : The same number in float32 format
    """
    rgb_uint32 = (r<<16) + (g<<8) + b
    return struct.unpack('f', struct.pack('I', rgb_uint32))[0]

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3+), box3d: (8,3+) '''
    box3d_roi_inds = in_hull(pc[:, :3], box3d)
    return pc[box3d_roi_inds], box3d_roi_inds

def publish_camera(cam_pub, bridge, image, boxes_2d=None, object_types=None, scale=1.0):
    if boxes_2d is not None:
        for i, box in enumerate(boxes_2d):
            top_left = int(box[0]), int(box[1])
            bottom_right = int(box[2]), int(box[3])
            if object_types is None:
                cv2.rectangle(image, top_left, bottom_right, (255,255,0), 2)
            else:
                cv2.rectangle(image, top_left, bottom_right, DETECTION_COLOR_MAP[object_types[i]], 2)
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale)
    cam_pub.publish(bridge.cv2_to_imgmsg(image, "bgr8"))

def publish_ego_car(ego_car_pub, fov=np.pi/4, l=10, velo_height=1.73):
    """
    Publish left and right 45 degree FOV lines and ego car model mesh
    """
    marker_array = MarkerArray()

    marker = Marker()
    marker.header.frame_id = FRAME_ID
    marker.header.stamp = rospy.Time.now()

    marker.id = 0
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration()
    marker.type = Marker.LINE_STRIP

    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 0.5
    marker.scale.x = 0.2

    marker.points = []
    marker.points.append(Point(l, -l*np.tan(fov), 0))
    marker.points.append(Point(0, 0, 0))
    marker.points.append(Point(l, l*np.tan(fov), 0))

    marker_array.markers.append(marker)

    mesh_marker = Marker()
    mesh_marker.header.frame_id = FRAME_ID
    mesh_marker.header.stamp = rospy.Time.now()

    mesh_marker.id = -1
    mesh_marker.lifetime = rospy.Duration()
    mesh_marker.type = Marker.MESH_RESOURCE
    mesh_marker.mesh_resource = "package://kitti/bmw_x5/BMW X5 4.dae"

    mesh_marker.pose.position.x = 0.0
    mesh_marker.pose.position.y = 0.0
    mesh_marker.pose.position.z = -velo_height

    q = ros_tf.transformations.quaternion_from_euler(np.pi/2, 0, np.pi)
    mesh_marker.pose.orientation.x = q[0]
    mesh_marker.pose.orientation.y = q[1]
    mesh_marker.pose.orientation.z = q[2]
    mesh_marker.pose.orientation.w = q[3]

    mesh_marker.color.r = 1.0
    mesh_marker.color.g = 1.0
    mesh_marker.color.b = 1.0
    mesh_marker.color.a = 1.0

    mesh_marker.scale.x = 0.9
    mesh_marker.scale.y = 0.9
    mesh_marker.scale.z = 0.9

    marker_array.markers.append(mesh_marker)
    ego_car_pub.publish(marker_array)

def publish_3dbox(box3d_pub, corners_3d_velos, object_types=None, track_ids=None, tracked_frames=None):
    """
    Publish 3d boxes in velodyne coordinate, with color specified by object_types
    If object_types is None, set all color to cyan
    corners_3d_velos : list of (8, 3) 3d corners
    """
    marker_array = MarkerArray()
    for i, corners_3d_velo in enumerate(corners_3d_velos):
        if (tracked_frames is not None and tracked_frames[i] < MIN_TRACKED_FRAMES) or (track_ids is not None and track_ids[i]<0):
            continue
        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = rospy.Time.now()

        marker.id = -i if track_ids is None else track_ids[i] + 100000
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(LIFETIME)
        marker.type = Marker.LINE_LIST

        if object_types is None:
            marker.color.r = 255/255.0
            marker.color.g = 0/255.0
            marker.color.b = 0/255.0
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

        if track_ids is not None:
            track_id = track_ids[i]
            text_marker = Marker()
            text_marker.header.frame_id = 'map'
            text_marker.header.stamp = rospy.Time.now()

            text_marker.id = track_id + 1000
            text_marker.action = Marker.ADD
            text_marker.lifetime = rospy.Duration(LIFETIME)
            text_marker.type = Marker.TEXT_VIEW_FACING

            # p4 = corners_3d_velo[4] # upper front left corner
            p = np.mean(corners_3d_velo, axis=0) # center

            text_marker.pose.position.x = p[0]
            text_marker.pose.position.y = p[1]
            text_marker.pose.position.z = p[2] + 1

            if int(track_id) >= 20000:
                track_id_ = int(track_id) - 20000
            elif int(track_id) >= 10000:
                track_id_ = int(track_id) - 10000
            else:
                track_id_ = int(track_id)
            text_marker.text = str(track_id_%100)

            text_marker.scale.x = 1.0
            text_marker.scale.y = 1.0
            text_marker.scale.z = 1.0

            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0

            text_marker.color.a = 1.0
            marker_array.markers.append(text_marker)
        
    box3d_pub.publish(marker_array)

def publish_pointcloud(pcl_pub, points, display_point_color=False):
    #header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'map'

    if display_point_color:
        # if we want the intensity
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgb', 12, PointField.FLOAT32, 1)]
        pcl_msg = pcl2.create_cloud(header, fields, points)
    else:
        pcl_msg = pcl2.create_cloud_xyz32(header, points[:, :3])
    pcl_pub.publish(pcl_msg)

def read_detection(path, gt=False):
    df = pd.read_csv(path, header=None, sep=' ')
    if gt:
        df.columns = ['frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 
                  'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']
        df = df[df.type.isin(['Car', 'Pedestrian'])]
    else:
        df.columns = ['frame', 'type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 
                  'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y', 'score']
    # df.loc[df.type.isin(['Bus', 'Truck', 'Van', 'Tram']), 'type'] = 'Car' # Set all vehicle type to Car
        # df = df[df.type.isin(['Car', 'Pedestrian', 'Cyclist'])]
    # df = df[((df['type']=='Car')&(df['score']>0.2))|((df['type']=='Pedestrian')&(df['score']>0.3))|
    #         ((df['type']=='Cyclist')&(df['score']>0.5))]
    return df

def read_tracking(path, gt=False):
    df = pd.read_csv(path, header=None, sep=' ')
    if gt:
        df.columns = ['frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 
                  'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']
    else:
        df.columns = ['frame', 'track_id', 'type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 
                  'bbox_right', 'bbox_bottom', 'height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y', 'score']
    # df.loc[df.type.isin(['Bus', 'Truck', 'Van', 'Tram']), 'type'] = 'Car' # Set all vehicle type to Car
    # df = df[df.type.isin(['Car', 'Pedestrian', 'Cyclist'])]
    # df = df[((df['type']=='Car')&(df['score']>0.3))|((df['type']=='Pedestrian')&(df['score']>0.3))|
    #         ((df['type']=='Cyclist')&(df['score']>0.5))]
    return df

def read_imu(path):
    df = pd.read_csv(path, header=None, delim_whitespace=True)
    df.columns = ['lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 'vn', 've', 'vf', 'vl', 'vu', 'ax', 'ay', 'az', 'af',
                    'al', 'au', 'wx', 'wy', 'wz', 'wf', 'wl', 'wu', 'posacc', 'velacc', 'navstat', 'numsats', 'posmode',
                    'velmode', 'orimode']
    return df

def read_nav(nav_path):
    with open(nav_path, 'r') as f:
        nav = yaml.load(f)
        
    lat, lon, alt = nav['latitude'], nav['longitude'], nav['latitude']
    roll, pitch, yaw = nav['roll']*np.pi/180, -nav['pitch']*np.pi/180, (90-nav['heading'])*np.pi/180
    vn, ve, vu = nav['north_vel'], nav['east_vel'], -nav['down_vel']
    
    df = pd.DataFrame(columns=['lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 'vn', 've', 'vf', 'vl', 'vu', 'ax', 'ay', 'az', 'af',
                    'al', 'au', 'wx', 'wy', 'wz', 'wf', 'wl', 'wu', 'posacc', 'velacc', 'navstat', 'numsats', 'posmode',
                    'velmode', 'orimode'])
    df.loc[0] = -1000
    df[['lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 'vn', 've', 'vu']] = np.array([lat, lon, alt, roll, pitch, yaw, vn, ve, vu])
    return df.loc[0]

def read_oxts(gps, imu, prev_imu_data=None, interval=0.1):
    """
    reads gps and imu from diffrent paths, and returns a kitti-format oxts dataframe
    """
    def quaternion_to_euler(x, y, z, w):
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        t2 = np.clip(2.0 * (w * y - z * x), -1, 1)
        pitch = np.arcsin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)

        return roll, pitch, yaw
    
    def read_gps_ros(filename):
        with open(filename, 'r') as f:
            gps = f.readlines()
        lat, lon, alt = float(gps[7][10:]), float(gps[8][11:]), float(gps[9][10:])
        return lat, lon, alt

    def read_imu_ros(filename):
        with open(filename, 'r') as f:
            imu = f.readlines()
        roll, pitch, yaw = quaternion_to_euler(float(imu[5][4:]), float(imu[6][4:]), float(imu[7][4:]), float(imu[8][4:]))
        ax, ay, az = float(imu[34][4:]), float(imu[35][4:]), float(imu[36][4:])
        return roll, pitch, yaw, ax, ay, az
    
    lat, lon, alt = read_gps_ros(gps)
    roll, pitch, yaw, ax, ay, az = read_imu_ros(imu)
    df = pd.DataFrame(columns=['lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 'vn', 've', 'vf', 'vl', 'vu', 'ax', 'ay', 'az', 'af',
                    'al', 'au', 'wx', 'wy', 'wz', 'wf', 'wl', 'wu', 'posacc', 'velacc', 'navstat', 'numsats', 'posmode',
                    'velmode', 'orimode'])
    df.loc[0] = -1000
    df[['lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 'vf', 'vl', 'vu', 'ax', 'ay', 'az']] = np.array([lat, lon, alt, roll, pitch, yaw, 0, 0, 0, ax, ay, az])
    if prev_imu_data is not None:
        # vf, vl, vu require previous accelerations
        df[['vf', 'vl', 'vu']] = np.array(prev_imu_data[['vf', 'vl', 'vu']]) + interval*np.array(prev_imu_data[['ax', 'ay', 'az']])
    return df.loc[0]

def read_can(can_path, prev_can=None):
    with open(can_path, 'r') as f:
        can = yaml.load(f)
        
    df = pd.DataFrame(columns=['lat', 'lon', 'alt', 'roll', 'pitch', 'yaw', 'vn', 've', 'vf', 'vl', 'vu', 'ax', 'ay', 'az', 'af',
                    'al', 'au', 'wx', 'wy', 'wz', 'wf', 'wl', 'wu', 'posacc', 'velacc', 'navstat', 'numsats', 'posmode',
                    'velmode', 'orimode'])
    df.loc[0] = -1000
    vn, ve = can['vehicleSpeed']/3.6, 0
    if prev_can is not None:
        yaw = can['yawRate']*0.1*np.pi/180.0 + prev_can.yaw
    else:
        yaw = 0
    df[['yaw', 'vn', 've']] = np.array([yaw, vn, ve])
    return df.loc[0]

def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    """
    Return : 3x8 in cam2 coordinate
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

def compute_displacement(prev_imu_data, imu_data, mode='imu'):
    def compute_great_circle_distance(lat1, lon1, lat2, lon2):
        """
        Compute the great circle distance from two gps data
        Input   : latitudes and longitudes in degree
        Output  : distance in meter
        """
        delta_sigma = float(np.sin(lat1*np.pi/180)*np.sin(lat2*np.pi/180)+ \
                            np.cos(lat1*np.pi/180)*np.cos(lat2*np.pi/180)*np.cos(lon1*np.pi/180-lon2*np.pi/180))
        if np.abs(delta_sigma) > 1:
            return 0.0
        return 6371000.0 * np.arccos(delta_sigma)

    if mode == 'imu':
        return np.linalg.norm(prev_imu_data[['vn', 've']]+imu_data[['vn', 've']]) * 0.05
    elif mode == 'gps':
        return compute_great_circle_distance(prev_imu_data.lat, prev_imu_data.lon,
                                                         imu_data.lat, imu_data.lon)

def publish_trajectory(tracker_pub, objects_to_track, track_ids):
    marker_array = MarkerArray()

    for track_id in track_ids: # for each object
        if track_id in objects_to_track:
            obj = objects_to_track[track_id]
            if track_id < 0 or obj.tracked_frames < MIN_TRACKED_FRAMES: # only show objects tracked for more than MIN_TRACKED_FRAMES frames
                continue
            marker = Marker()
            marker.header.frame_id = FRAME_ID
            marker.header.stamp = rospy.Time.now()

            marker.id = track_id + 10000
            marker.action = Marker.ADD
            marker.lifetime = rospy.Duration(LIFETIME)
            marker.type = Marker.LINE_STRIP

            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            marker.scale.x = 0.1

            marker.points = []
            obj_z = objects_to_track[track_id].z
            for p in objects_to_track[track_id].locations:
                marker.points.append(Point(p[0], p[1], obj_z))

            marker_array.markers.append(marker)

    tracker_pub.publish(marker_array)

def publish_center(tracker_pub, centers):
    marker_array = MarkerArray()

    for center in centers: # for each object
        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = rospy.Time.now()

        marker.id = 0
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(LIFETIME)
        marker.type = Marker.SPHERE

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.8
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = 0

        marker_array.markers.append(marker)

    tracker_pub.publish(marker_array)

def publish_cuboid(tracker_pub, centers):
    marker_array = MarkerArray()

    for center in centers: # for each object
        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = rospy.Time.now()

        marker.id = 50
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(LIFETIME)
        marker.type = Marker.LINE_LIST

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        marker.scale.x = 0.1

        x,y,z = center[0],center[1],10
        l = 2.5

        corners_3d_velo = np.zeros((8, 3))
        corners_3d_velo[0] = [x+l, y+l, -z]
        corners_3d_velo[1] = [x+l, y-l, -z]
        corners_3d_velo[2] = [x-l, y-l, -z]
        corners_3d_velo[3] = [x-l, y+l, -z]
        corners_3d_velo[4] = [x+l, y+l, z]
        corners_3d_velo[5] = [x+l, y-l, z]
        corners_3d_velo[6] = [x-l, y-l, z]
        corners_3d_velo[7] = [x-l, y+l, z]

        marker.points = []
        for l in LINES:
            p1 = corners_3d_velo[l[0]]
            marker.points.append(Point(p1[0], p1[1], p1[2]))
            p2 = corners_3d_velo[l[1]]
            marker.points.append(Point(p2[0], p2[1], p2[2]))
        marker_array.markers.append(marker)

    tracker_pub.publish(marker_array)


class Object():
    def __init__(self, center, z, max_length):
        self.locations = deque(maxlen=max_length)
        self.locations.appendleft(center)
        self.z = z
        self.max_length = max_length
        self.tracked_frames = 1
        self.disappeared_frames = 0

    def update(self, displacement, yaw_change):
        """
        Update the locations of the object
        """
        R = np.array([[np.cos(yaw_change), -np.sin(yaw_change)], [np.sin(yaw_change), np.cos(yaw_change)]])
        self.locations = deque(np.array(self.locations).dot(R)+np.array([-displacement, 0]))

    # def is_full(self):
    #     return len(self.locations) >= self.max_length//2

class Tracker():
    def __init__(self, tracker_pub, max_length=20, log=False):
        self.tracker_pub = tracker_pub
        self.max_length = max_length
        self.prev_imu_data = None
        self.objects_to_track = {}
        self.log = log

    def update(self, imu_data, track_ids, centers, zs):
        if self.prev_imu_data is None: # if it's the very first frame
            for track_id, center, z in zip(track_ids, centers, zs):
                self.objects_to_track[track_id] = Object(center, z, max_length=self.max_length)
        else:
            displacement = compute_displacement(self.prev_imu_data, imu_data, mode='gps')
            yaw_change = float(imu_data.yaw - self.prev_imu_data.yaw)
            # print(displacement, yaw_change)

            for track_id in self.objects_to_track: # for all tracked objects, update locations according to imu
                self.objects_to_track[track_id].update(displacement, yaw_change)

            for track_id, center, z in zip(track_ids, centers, zs):
                if track_id not in self.objects_to_track: # if it's a new object
                    self.objects_to_track[track_id] = Object(center, z, max_length=self.max_length)
                else: # existing object tracked
                    self.objects_to_track[track_id].locations.appendleft(center)
                    self.objects_to_track[track_id].z = z
                    self.objects_to_track[track_id].tracked_frames += 1
                    self.objects_to_track[track_id].disappeared_frames = 0

            # remove objects disappeared more than 20 frames
            track_ids_to_delete = []
            for track_id in self.objects_to_track:
                if track_id not in track_ids:
                    self.objects_to_track[track_id].disappeared_frames += 1
                    if self.objects_to_track[track_id].disappeared_frames > 20:
                        track_ids_to_delete += [track_id]
            for track_id in track_ids_to_delete:
                self.objects_to_track.pop(track_id)

        self.prev_imu_data = imu_data
        return [self.objects_to_track[track_id].tracked_frames for track_id in track_ids]

    def reset(self):
        self.prev_imu_data = None
        self.objects_to_track.clear()

    def publish(self, track_ids):
        publish_trajectory(self.tracker_pub, self.objects_to_track, track_ids)

def compute_center_of_box(corners_3d_velo):
    return np.mean(corners_3d_velo, axis=0)

if __name__ == '__main__':
    if KISOKEN or PRIUS or KISOKEN2 or KISOKEN_P:
        frame = START_FRAME
    else:
        frame = 0
    rospy.init_node('kitti_pointcloud_node', anonymous=True)
    pcl_pubs = {}
    for i in range(4):
        pcl_pubs[i] = rospy.Publisher("kitti_pointcloud%d"%i, PointCloud2, queue_size=10)
    cam_pub = rospy.Publisher('kitti_cam', Image, queue_size=10)
    bridge = CvBridge()
    ego_car_pub = rospy.Publisher('kitti_ego_car', MarkerArray, queue_size=10)
    box3d_pub = rospy.Publisher('kitti_3dboxes', MarkerArray, queue_size=10)
    box3d_gt_pub = rospy.Publisher('kitti_3dboxes_gt', MarkerArray, queue_size=10)
    tracker_pub = rospy.Publisher('kitti_trajectories', MarkerArray, queue_size=10)
    rate = rospy.Rate(RATE)

    # velodyne_reader = VelodyneReader()
    tracker = Tracker(tracker_pub, max_length=20)

    # read calibration and detection/tracking result
    if RAWDATA:
        calib = Calibration('/home/kwea123/hdd/kwea123/dataset/RawData/%s'%DATE, from_video=True)
        if TRACKING:
            df_tracking = read_tracking('/home/kwea123/workspace/tracking/evaluations/RawData/%s.txt'%DATANAME)
        elif DETECTION:
            df_tracking = read_detection('/home/kwea123/workspace/frustum-pointnets/train/detection_results_tracking/RawData/%s.txt'%DATANAME)
        SEQUENCE_LENGTH = len(os.listdir('/home/kwea123/hdd/kwea123/dataset/RawData/%s/%s/oxts/data'%(DATE, DATANAME)))
    elif KISOKEN:
        calib = Calibration('/home/kwea123/workspace/frustum-pointnets/dataset/kisoken/tracking/', from_video=True)
        df_tracking = read_detection('/home/kwea123/workspace/frustum-pointnets/train/detection_results_tracking/kisoken/%s.txt'%SEQUENCE_NAME)
    elif PRIUS:
        calib = Calibration('/home/kwea123/hdd/kwea123/dataset/prius/', from_video=True)
        if TRACKING:
            df_tracking = read_tracking('/home/kwea123/workspace/tracking/evaluations/prius/%s.txt'%(SEQUENCE_NAME))
    elif KISOKEN2:
        calib = Calibration('/home/kwea123/hdd/kwea123/dataset/kisoken2/', from_video=True)
        if TRACKING:
            df_tracking = read_tracking('/home/kwea123/workspace/tracking/evaluations/kisoken2/%s.txt'%SEQUENCE_NAME)
        elif DETECTION:
            df_tracking = read_detection('/home/kwea123/workspace/frustum-pointnets/train/detection_results_tracking/kisoken2/%s.txt'%SEQUENCE_NAME)
        df_imu = read_imu('/home/kwea123/data/tracking/training/oxts/%04d.txt'%0) # fake imu data, same for all frames
    elif KISOKEN_P:
        calib = Calibration('/home/kwea123/hdd/kwea123/dataset/kisoken_palace/', from_video=True)
        if TRACKING:
            df_tracking = read_tracking('/home/kwea123/workspace/tracking/evaluations/kisoken_palace/%s.txt'%SEQUENCE_NAME)
    elif VSYS:
        calib = Calibration('/home/kwea123/data/tracking/%s/calib/%04d.txt'%('training', 0))
        SEQUENCE_LENGTH = len(os.listdir('/home/kwea123/hdd/kwea123/dataset/vsys/%s/velodyne/data/'%(SEQUENCE_NAME)))
        df_tracking = read_detection('/home/kwea123/workspace/tracking/evaluations/vsys/%s.txt'%(SEQUENCE_NAME))
    elif VSYS2:
        calib = Calibration('/home/kwea123/hdd/kwea123/dataset/vsys2/', from_video=True)
        if TRACKING:
            df_tracking = read_tracking('/home/kwea123/workspace/tracking/evaluations/vsys2/%s.txt'%(SEQUENCE_NAME))
        elif DETECTION:
            df_tracking = read_detection('/home/kwea123/workspace/frustum-pointnets/train/detection_results_vsys2/%s.txt'%(SEQUENCE_NAME))
    elif VSYS_URA:
        calib = Calibration('/home/kwea123/hdd/kwea123/dataset/vsys_ura/', from_video=True)
        if TRACKING:
            df_tracking = read_tracking('/home/kwea123/workspace/tracking/evaluations/vsys_ura/%s.txt'%(SEQUENCE_NAME))
    elif RX:
        calib = Calibration('/home/kwea123/hdd/kwea123/dataset/rx/', from_video=True)
        if TRACKING:
            df_tracking = read_tracking('/home/kwea123/workspace/tracking/evaluations/rx/%s.txt'%(SEQUENCE_NAME))
    else: # tracking sequences
        calib = Calibration('/home/kwea123/data/tracking/%s/calib/%04d.txt'%(DATASET, SEQUENCE_NUMBER))
        df_imu = read_imu('/home/kwea123/data/tracking/%s/oxts/%04d.txt'%(DATASET, SEQUENCE_NUMBER))
        if TRACKING:
            if BACK:
                df_tracking = read_tracking('/home/kwea123/workspace/tracking/evaluations/%s/back%04d.txt'%(DATASET, SEQUENCE_NUMBER))
            elif SHIGE:
                df_tracking = read_tracking('/home/kwea123/workspace/tracking/evaluations/shige_detection/kitti/%s/%s/%04d.txt'%(DATASET, SHIGE_DETECTOR, SEQUENCE_NUMBER))
            elif AB3DMOT:
                df_tracking = read_tracking('/home/kwea123/workspace/tracking/evaluations/AB3DMOT/%04d.txt'%(SEQUENCE_NUMBER))
            else:
                df_tracking = read_tracking('/home/kwea123/workspace/tracking/evaluations/%s/%04d.txt'%(DATASET, SEQUENCE_NUMBER))
        elif DETECTION:
            if BACK:
                df_tracking = read_detection('/home/kwea123/workspace/frustum-pointnets/train/detection_results_tracking/%s/%04dback.txt'%(DATASET, SEQUENCE_NUMBER))
            else:
                df_tracking = read_detection('/home/kwea123/workspace/frustum-pointnets/train/detection_results_tracking/%s/%04d.txt'%(DATASET, SEQUENCE_NUMBER))

        # df_tracking_gt = read_detection('/home/kwea123/data/tracking/'+DATASET+'/label_02/%04d.txt'%SEQUENCE_NUMBER, gt=True)
        SEQUENCE_LENGTH = len(df_imu)

    # read image and point data
    prev_imu_data = None
    while not rospy.is_shutdown():
        if RAWDATA:       
            points = np.fromfile('/home/kwea123/hdd/kwea123/dataset/RawData/%s/%s/velodyne_points/data/%010d.bin'%(DATE, DATANAME, frame), dtype=np.float32).reshape(-1, 4)
            # points = points[::2]
            image = cv2.imread('/home/kwea123/hdd/kwea123/dataset/RawData/%s/%s/image_02/data/%010d.png'%(DATE, DATANAME, frame))
        elif KISOKEN:
            points = np.fromfile('/home/kwea123/workspace/frustum-pointnets/dataset/kisoken/tracking/velodyne/%s/%010d.bin'%(SEQUENCE_NAME, frame), dtype=np.float32).reshape(-1, 4)
            # points = points[::2]
            points[:, -1] /= 255.0
            image = cv2.imread('/home/kwea123/workspace/frustum-pointnets/dataset/kisoken/tracking/image/%s/%010d.png'%(SEQUENCE_NAME, frame))
        elif PRIUS:
            points = np.fromfile('/home/kwea123/hdd/kwea123/dataset/prius/%s/velodyne/data/%010d.bin'%(SEQUENCE_NAME, frame), dtype=np.float32).reshape(-1, 4)
            points = points[::4]
            image = cv2.imread('/home/kwea123/hdd/kwea123/dataset/prius/%s/stereo_left/data/%010d.png'%(SEQUENCE_NAME, frame))
        elif KISOKEN2:
            points = np.fromfile('/home/kwea123/hdd/kwea123/dataset/kisoken2/%s/velodyne/data/%010d.bin'%(SEQUENCE_NAME, frame), dtype=np.float32).reshape(-1, 4)
            points = points[(points[:, 0]>=1)&(points[:, 0]<=40)]
            # points = points[::2]
            image = cv2.imread('/home/kwea123/hdd/kwea123/dataset/kisoken2/%s/stereo_left/data/%010d.png'%(SEQUENCE_NAME, frame))
        elif KISOKEN_P:
            points = np.fromfile('/home/kwea123/hdd/kwea123/dataset/kisoken_palace/%s/velodyne/data/%010d.bin'%(SEQUENCE_NAME, frame), dtype=np.float32).reshape(-1, 4)
            points = points[(points[:, 0]>=1)]
            points = points[::5]
            image = cv2.imread('/home/kwea123/hdd/kwea123/dataset/kisoken_palace/%s/stereo_left/data/%010d.png'%(SEQUENCE_NAME, frame))
        elif VSYS:
            points = np.fromfile('/home/kwea123/hdd/kwea123/dataset/vsys/%s/velodyne/data/%010d.bin'%(SEQUENCE_NAME, frame), dtype=np.float32).reshape(-1, 4)
            points = points[points[:, 2]<2.0]
            points[:, 2] -= 1.0
            image = np.zeros((960, 1280, 3), dtype=np.uint8)
        elif VSYS2:
            points = np.fromfile('/home/kwea123/hdd/kwea123/dataset/vsys2/%s/velodyne_front/data/%010d.bin'%(SEQUENCE_NAME, frame), dtype=np.float32).reshape(-1, 4)
            # BACK VELO
            if os.path.exists('/home/kwea123/hdd/kwea123/dataset/vsys2/%s/velodyne_back/data/%010d.bin'%(SEQUENCE_NAME, frame)):
                points2 = np.fromfile('/home/kwea123/hdd/kwea123/dataset/vsys2/%s/velodyne_back/data/%010d.bin'%(SEQUENCE_NAME, frame), dtype=np.float32).reshape(-1, 4)
                points2 = np.concatenate([calib.project_velo_to_velo(points2[:,:3], name='Tr_veloBack_to_veloFront'), points2[:,-1:]], axis=-1)
                points = np.concatenate([points, points2], axis=0)

            points = points[(points[:, 0]>=1)]
            image = cv2.imread('/home/kwea123/hdd/kwea123/dataset/vsys2/%s/tss2_c2c4/data/%010d.png'%(SEQUENCE_NAME, frame))
        elif VSYS_URA:
            points = np.fromfile('/home/kwea123/hdd/kwea123/dataset/vsys_ura/%s/velodyne_center/data/%010d.bin'%(SEQUENCE_NAME, frame), dtype=np.float32).reshape(-1, 4)
            if os.path.exists('/home/kwea123/hdd/kwea123/dataset/vsys_ura/%s/velodyne_front/data/%010d.bin'%(SEQUENCE_NAME, frame)):
                points2 = np.fromfile('/home/kwea123/hdd/kwea123/dataset/vsys_ura/%s/velodyne_front/data/%010d.bin'%(SEQUENCE_NAME, frame), dtype=np.float32).reshape(-1, 4)
                points2 = np.concatenate([calib.project_velo_to_velo(points2[:,:3], name='Tr_veloFront_to_veloCenter'), points2[:,-1:]], axis=-1)
                points = np.concatenate([points, points2], axis=0)
            # if os.path.exists('/home/kwea123/hdd/kwea123/dataset/vsys_ura/%s/velodyne_left/data/%010d.bin'%(SEQUENCE_NAME, frame)):
            #     points3 = np.fromfile('/home/kwea123/hdd/kwea123/dataset/vsys_ura/%s/velodyne_left/data/%010d.bin'%(SEQUENCE_NAME, frame), dtype=np.float32).reshape(-1, 4)
            #     points3 = np.concatenate([calib.project_velo_to_velo(points3[:,:3], name='Tr_veloLeft_to_veloCenter'), points3[:,-1:]], axis=-1)
            #     # points3 = points3[points3[:, 2]>-2]
            #     # points = np.concatenate([points, points2], axis=0)
            # if os.path.exists('/home/kwea123/hdd/kwea123/dataset/vsys_ura/%s/velodyne_right/data/%010d.bin'%(SEQUENCE_NAME, frame)):
            #     points4 = np.fromfile('/home/kwea123/hdd/kwea123/dataset/vsys_ura/%s/velodyne_right/data/%010d.bin'%(SEQUENCE_NAME, frame), dtype=np.float32).reshape(-1, 4)
            #     points4 = np.concatenate([calib.project_velo_to_velo(points4[:,:3], name='Tr_veloRight_to_veloCenter'), points4[:,-1:]], axis=-1)
                # points4 = points4[points4[:, 2]>-2]
                # points = np.concatenate([points, points2], axis=0)
            image = cv2.imread('/home/kwea123/hdd/kwea123/dataset/vsys_ura/%s/tss3_center/data/%010d.png'%(SEQUENCE_NAME, frame))
            points = points[(points[:,0]>-1)]
            if image is None:
                image = np.zeros((1424, 2896, 3), dtype=np.uint8)
        elif RX:
            points = np.fromfile('/home/kwea123/hdd/kwea123/dataset/rx/%s/velodyne_center/data/%010d.bin'%(SEQUENCE_NAME, frame), dtype=np.float32).reshape(-1, 4)
            image = cv2.imread('/home/kwea123/hdd/kwea123/dataset/rx/%s/tss2_c2c4/data/%010d.png'%(SEQUENCE_NAME, frame))
            points = points[(points[:, 0]>=-1)]
            points = points[::2]
        else: # KITTI tracking sequences
            if DEPTH2POINTS:
                points = np.load('/home/kwea123/hdd/kwea123/dataset/depth2points/%06d.npy'%32)
                points = points[points[:,2]<1]
                points2 = np.fromfile('/home/kwea123/hdd/kwea123/dataset/KITTI/object/training/velodyne/%06d.bin'%32, dtype=np.float32).reshape(-1, 4)
                points2 = points2[points2[:,0]>0]
                # points = np.load('/home/kwea123/data/tracking/%s/depth2points/%04d/%06d.npy'%(DATASET, SEQUENCE_NUMBER, frame))
                # points = points[points[:, 2]<2]
            else:
                points = np.fromfile('/home/kwea123/data/tracking/%s/velodyne/%04d/%06d.bin'%(DATASET, SEQUENCE_NUMBER, frame), dtype=np.float32).reshape(-1, 4)
                points = points[::2]
                # points = points[points[:, 0]>=-8]
                # points = np.fromfile('/home/kwea123/data/tracking/'+DATASET+'/velodyne/%04d/%06d_4lines.npy'%(SEQUENCE_NUMBER, frame), dtype=np.float32).reshape(-1, 4)
            image = cv2.imread('/home/kwea123/data/tracking/%s/image_02/%04d/%06d.png'%(DATASET, SEQUENCE_NUMBER, frame))

        if SIXTEEN_LINES:
            velodyne_reader.set_point_cloud(points)
            # points = velodyne_reader.select_lines(2, 3, 16)
            points = velodyne_reader.select_lines(0, 2, 32)

        if DETECTION or TRACKING:
            df_tracking_frame = df_tracking[df_tracking['frame']==frame]
            df_tracking_frame.reset_index(inplace=True, drop=True)

            corners_3d_velos = []
            centers = []
            zs = []
            params = np.array(df_tracking_frame[['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
            for i in range(len(df_tracking_frame)):
                corners_3d_cam2 = compute_3d_box_cam2(*params[i])
                corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T) # 8x3
                corners_3d_velos += [corners_3d_velo]
                center_velo = compute_center_of_box(corners_3d_velo)
                centers += [center_velo[:2]]
                zs += [center_velo[-1]]
                # x,y,z = centers[0][0],centers[0][1],10
                # l = 2.5
                # corners_3d_velo[0] = [x+l, y+l, -z]
                # corners_3d_velo[1] = [x+l, y-l, -z]
                # corners_3d_velo[2] = [x-l, y-l, -z]
                # corners_3d_velo[3] = [x-l, y+l, -z]
                # corners_3d_velo[4] = [x+l, y+l, z]
                # corners_3d_velo[5] = [x+l, y-l, z]
                # corners_3d_velo[6] = [x-l, y-l, z]
                # corners_3d_velo[7] = [x-l, y+l, z]
                # box3droi_pc_velo, box3d_roi_inds = extract_pc_in_box3d(points, corners_3d_velo[:, :3])

            if 'df_tracking_gt' in locals():
                df_tracking_frame_gt = df_tracking_gt[df_tracking_gt['frame']==frame]
                df_tracking_frame_gt.reset_index(inplace=True, drop=True)

                corners_3d_velos_gt = []
                params = np.array(df_tracking_frame_gt[['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']])
                for i in range(len(df_tracking_frame_gt)):
                    corners_3d_cam2 = compute_3d_box_cam2(*params[i])
                    corners_3d_velo = calib.project_rect_to_velo(corners_3d_cam2.T) # 8x3
                    corners_3d_velos_gt += [corners_3d_velo]
            
            # read imu data
            if TRACKING:
                if RAWDATA:
                    df_imu_frame = read_imu('/home/kwea123/hdd/kwea123/dataset/RawData/%s/%s/oxts/data/%010d.txt'%(DATE, DATANAME, frame))
                elif KISOKEN2:
                    # fake imu data (same for every frame)
                    df_imu_frame = df_imu.loc[0]
                elif PRIUS:
                    df_imu_frame = read_imu('/home/kwea123/hdd/kwea123/dataset/prius/%s/poslv_ros/data/%010d.txt'%(SEQUENCE_NAME, frame)).loc[0]
                elif KISOKEN_P:
                    df_imu_frame = read_can('/home/kwea123/hdd/kwea123/dataset/kisoken_palace/%s/priuscan/data/%010d.txt'%(SEQUENCE_NAME, frame), prev_imu_data)
                elif VSYS2:
                    df_imu_frame = read_nav('/home/kwea123/hdd/kwea123/dataset/vsys2/%s/navsol/data/%010d.txt'%(SEQUENCE_NAME, frame))
                elif VSYS_URA:
                    df_imu_frame = read_oxts('/home/kwea123/hdd/kwea123/dataset/vsys_ura/%s/gps/data/%010d.txt'%(SEQUENCE_NAME, frame),
                                             '/home/kwea123/hdd/kwea123/dataset/vsys_ura/%s/imu/data/%010d.txt'%(SEQUENCE_NAME, frame))
                elif RX:
                    try:
                        df_imu_frame = read_can('/home/kwea123/hdd/kwea123/dataset/rx/%s/rx_can/data/%010d.txt'%(SEQUENCE_NAME, frame), prev_imu_data)
                    except:
                        df_imu_frame = read_imu('/home/kwea123/hdd/kwea123/dataset/rx/%s/poslv_ros/data/%010d.txt'%(SEQUENCE_NAME, frame)).loc[0]
                else:
                    df_imu_frame = df_imu.loc[frame]
                track_ids = df_tracking_frame['track_id']
                tracked_frames = tracker.update(df_imu_frame, track_ids, centers, zs)

        # #publish    
        rospy.loginfo("publishing frame %d"%frame)
        if DETECTION or TRACKING: # publish 2d box on image
            publish_camera(cam_pub, bridge, image, np.array(df_tracking_frame[['bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom']]), df_tracking_frame['type'], scale=0.25 if VSYS_URA else 1.0)
        else: # publish only image
            publish_camera(cam_pub, bridge, image)

        # publish_center(box3d_pub, centers)
        # publish_cuboid(box3d_pub, centers)
        # points[:, 3] = rgb_to_float32(255, 255, 255)
        # points[box3d_roi_inds, 3] = rgb_to_float32(255, 0, 0)
        # publish_pointcloud(pcl_pub, points, display_point_color=True)
        
        if TRACKING: # publish 3d box with ID
            publish_3dbox(box3d_pub, corners_3d_velos, df_tracking_frame['type'], track_ids, tracked_frames=tracked_frames)
            if 'df_tracking_gt' in locals():
                publish_3dbox(box3d_gt_pub, corners_3d_velos_gt, track_ids=None)
            tracker.publish(track_ids)
        elif DETECTION: # publish only 3d box
            publish_3dbox(box3d_pub, corners_3d_velos, df_tracking_frame['type'])
        publish_ego_car(ego_car_pub, fov=np.pi/6 if VSYS2 or RX else np.pi/4, l=20, velo_height=2.2 if VSYS_URA else 1.73)
        publish_pointcloud(pcl_pubs[0], points, display_point_color=DISPLAY_POINT_COLOR)
        # publish_pointcloud(pcl_pubs[1], points2, display_point_color=False)
        # publish_pointcloud(pcl_pubs[2], points3, display_point_color=False)
        # publish_pointcloud(pcl_pubs[3], points4, display_point_color=False)

        # proceed to next frame
        frame += 1
        if 'df_imu_frame' in locals():
            prev_imu_data = df_imu_frame
        if KISOKEN or KISOKEN2 or KISOKEN_P:
            if frame == END_FRAME:
                frame = START_FRAME
                tracker.reset()
        else:
            if frame == SEQUENCE_LENGTH:
                frame = 0
                tracker.reset()
                break
        rate.sleep()