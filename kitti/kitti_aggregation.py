#!/usr/bin/env python
import numpy as np
import yaml
import os
import struct
import time

import rospy

from publish_utils import *
from kitti_data_utils import *

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

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3+), box3d: (8,3+) '''
    box3d_roi_inds = in_hull(pc[:,:3], box3d)
    return pc[box3d_roi_inds], box3d_roi_inds

def rgb_to_float32(r, g, b):
    """
    Input : r, g, b integer values in range [0, 255]
    Output : The same number in float32 format
    """
    rgb_uint32 = (r<<16) + (g<<8) + b
    return struct.unpack('f', struct.pack('I', rgb_uint32))[0]

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


DATA_PATH = '/home/ubuntu/data/kitti/RawData/2011_09_26/2011_09_26_drive_0014_sync/'

with open('/home/ubuntu/data/kitti/RawData/2011_09_26/calib_velo_to_cam.txt', 'r') as f:
    yml = yaml.load(f)

R_velo_to_cam2 = np.array([float(i) for i in yml['R'].split(' ')]).reshape(3, 3)
T_velo_to_cam2 = np.array([float(i) for i in yml['T'].split(' ')]).reshape(3, 1)
Tr_velo_to_cam2 = np.vstack((np.hstack([R_velo_to_cam2, T_velo_to_cam2]), [0, 0, 0, 1]))

RANDOM_COLORS = [np.random.randint(255, size=3) for _ in range(1000)]
COLOR_WHITE = rgb_to_float32(255, 255, 255)

class Localizer():
    def __init__(self, loc_pub, max_length=20, log=False):
        self.loc_pub = loc_pub
        self.prev_imu_data = None
        self.prev_locations = []
        self.max_length = max_length
        self.log = log

    def update(self, imu_data):
        if self.prev_imu_data is not None:
            displacement = (compute_great_circle_distance(self.prev_imu_data.lat, self.prev_imu_data.lon,
                                                               imu_data.lat, imu_data.lon))
            yaw = (imu_data.yaw - self.prev_imu_data.yaw)
            
            # R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]) # 2x2 rotation matrix of -yaw
            for i in range(len(self.prev_locations)):
                loc = self.prev_locations[i]
                newlocx = np.cos(yaw) * loc[0] + np.sin(yaw) * loc[1] - displacement
                newlocy = -np.sin(yaw)* loc[0] + np.cos(yaw) * loc[1]
                self.prev_locations[i] = [newlocx, newlocy]

        self.prev_locations = [[0, 0]] + self.prev_locations
        if len(self.prev_locations) > self.max_length:
            self.prev_locations = self.prev_locations[:self.max_length]
        self.prev_imu_data = imu_data

    def reset(self):
        """
        Empty the locations when the sequence has reached the end
        """
        self.prev_imu_data = None
        self.prev_locations[:] = []

    def publish(self):
        publish_location(self.loc_pub, self.prev_locations, log=self.log)

if __name__ == '__main__':

    frame = 0
    log = True # log info to the console or not
    # create node and publishers
    rospy.init_node('kitti_pointcloud_node', anonymous=True)
    cam_pub = rospy.Publisher('kitti_cam', Image, queue_size=10)
    bridge = CvBridge()
    pcl_pub = rospy.Publisher('kitti_pointcloud', PointCloud2, queue_size=10)
    marker_pub = rospy.Publisher('kitti_carFOV', Marker, queue_size=10)
    markers_pub = rospy.Publisher('kitti_3dboxes', MarkerArray, queue_size=10)
    imu_pub = rospy.Publisher('kitti_imu', Imu, queue_size=10)
    gps_pub = rospy.Publisher('kitti_gps', NavSatFix, queue_size=10)
    loc_pub = rospy.Publisher('kitti_loc', Marker, queue_size=10)
    rate = rospy.Rate(10)

    df_tracking = read_tracking('/home/ubuntu/data/kitti/tracking/training/label_02/0004.txt')
    sequence_length = max(df_tracking['frame'])
    
    localizer = Localizer(loc_pub)

    while not rospy.is_shutdown():

        # read camera data of the current frame
        image = read_camera(os.path.join(DATA_PATH, 'image_02/data/%010d.png'%frame))

        # extract tracking data of the current frame
        df_tracking_frame = df_tracking[df_tracking['frame']==frame]
        df_tracking_frame.reset_index(inplace=True, drop=True)

        # read imu data of the current frame 
        df_imu_frame = read_imu(os.path.join(DATA_PATH, 'oxts/data/%010d.txt'%frame))

        # read point cloud of the current frame
        point_cloud = read_velodyne(os.path.join(DATA_PATH, 'velodyne_points/data/%010d.bin'%frame))
        # downsample the point cloud
        point_cloud = point_cloud[::2]
        # set default point cloud color to white
        point_cloud[:, 3] = COLOR_WHITE

        # read 2d and 3d boxes
        borders_2d_cam2s = []
        object_types = []
        corners_3d_velos = []
        for i in range(len(df_tracking_frame)):

            borders_2d_cam2 = np.array(df_tracking_frame.loc[i, ['bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom']])
            borders_2d_cam2s += [borders_2d_cam2]
            object_types += [df_tracking_frame.loc[i, 'type']]

            corners_3d_cam2 = compute_3d_box_cam2(*np.array(df_tracking_frame.loc[i, ['height', 'width', 'length', 'pos_x', 'pos_y', 'pos_z', 'rot_y']]))
            corners_3d_velo = cam2_3d_to_velo(corners_3d_cam2).T # 8x4
            corners_3d_velos += [corners_3d_velo]
            
            # # set different color for point cloud in each object (slow)
            # box3droi_pc_velo, box3d_roi_inds = extract_pc_in_box3d(point_cloud, corners_3d_velo[:, :3])
            # point_cloud[box3d_roi_inds, 3] = rgb_to_float32(*RANDOM_COLORS[df_tracking_frame.loc[i, 'track_id']])

        # update the localizer
        localizer.update(df_imu_frame)

        # publish location
        localizer.publish()
        # publish camera image
        publish_camera(cam_pub, bridge, image, borders_2d_cam2s, object_types, log=log)
        # publish point cloud
        publish_point_cloud(pcl_pub, point_cloud, format='xyzrgb', log=log)
        # publish 3d boxes
        publish_3dbox(markers_pub, corners_3d_velos, np.array(df_tracking_frame['track_id']), object_types, publish_id=True, log=log)
        # # publish imu
        # publish_imu(imu_pub, df_imu_frame, log=log)
        # # publish gps
        # publish_gps(gps_pub, df_imu_frame, log=log)
        # publish car FOV
        publish_car_fov(marker_pub)

        frame += 1
        if frame == sequence_length: # if the sequence has reached the end
            frame = 0
            localizer.reset()
            rospy.loginfo("sequence reset !")

        rate.sleep()
