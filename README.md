# ROS_notes

## Use ROS to visualize kitti data

First install [kitti2bag](https://github.com/tomas789/kitti2bag).
Follow the construction to create a rosbag.

Follow [this](https://www.youtube.com/watch?v=e0r4uKK1zkk) to visualize the image and lidar points on `rviz`.

[here](https://github.com/tomas789/kitti2bag/blob/master/bin/kitti2bag) is some source code of how to create ros msgs.

## Sending video over ROS

It is faster to encode the video stream to bytes (e.g. in mjpeg), then send the bytes, instead of sending an image array.
