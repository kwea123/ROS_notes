# ROS_notes

## Use ROS to visualize kitti data

First install [kitti2bag](https://github.com/tomas789/kitti2bag).
Follow the construction to create a rosbag.

Follow [this](https://www.youtube.com/watch?v=e0r4uKK1zkk) to visualize the image and lidar points on `rviz`.

[here](https://github.com/tomas789/kitti2bag/blob/master/bin/kitti2bag) is some source code of how to create ros msgs.

## Sending video over ROS

It is faster to encode the video stream to bytes (e.g. in mjpeg), then send the bytes, instead of sending an image array.

## ROS markers

[Here](https://www.robotech-note.com/entry/2018/04/15/221524) is a tutorial on how to create basic markers using python.

## ROS synchronization

Use `chrony` to synchronize across machines.
[tutorial](https://qiita.com/ngkazu/items/916f476985fa3e3f2951)


## ROS with Gazebo

[Installation](http://gazebosim.org/tutorials?tut=ros_installing)

[Problem1](http://answers.gazebosim.org/question/14237/problem-with-launching-the-gazebo/)

### Turtlebot tutorial

[1](http://r17u.hatenablog.com/entry/2017/06/11/213942)
[2](http://wiki.ros.org/turtlebot_gazebo/Tutorials/indigo/Make%20a%20map%20and%20navigate%20with%20it)
[3](http://learn.turtlebot.com/2015/02/03/6/)
[4](http://robot.isc.chubu.ac.jp/?p=969)
