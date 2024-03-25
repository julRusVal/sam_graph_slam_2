#!/usr/bin/env python3

from typing import Tuple
from gtsam import Pose3, Rot3

# ROS imports
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time

"""
Set of functions for converting between GTSAM and ROS objects

ROS -> GTSAM
- ros_pose_to_gtsam_pose3_and_stamp()

GTSAM -> ROS
- gtsam_pose3_to_ros_pose3()
"""


def ros_pose_to_gtsam_pose3_and_stamp(pose: PoseStamped) -> Tuple[Pose3, Time]:
    """
    Converts a a ROS PoseStamped into a GTSAM Pose3, also returning the time stamp
    :param pose:
    :return:
    """
    rot3 = Rot3.Quaternion(w=pose.pose.orientation.w,
                           x=pose.pose.orientation.x,
                           y=pose.pose.orientation.y,
                           z=pose.pose.orientation.z)

    stamp = pose.header.stamp
    return Pose3(rot3, [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]), stamp


def gtsam_pose3_to_ros_pose3(pose: Pose3, stamp: Time = None, frame_id: str = None) -> PoseStamped:
    """
    Combines a GTSAM Pose, time stamp, and frame_id into a ROS PoseStamped object
    :param pose:
    :param stamp:
    :param frame_id:
    :return:
    """
    quaternion_wxyz = pose.rotation().toQuaternion()
    translation = pose.translation()

    new_stamped_pose = PoseStamped()
    if stamp is not None:
        new_stamped_pose.header.stamp = stamp
    if frame_id is not None:
        new_stamped_pose.header.frame_id = frame_id
    new_stamped_pose.pose.position.x = translation[0]
    new_stamped_pose.pose.position.y = translation[1]
    new_stamped_pose.pose.position.z = translation[2]
    new_stamped_pose.pose.orientation.x = quaternion_wxyz.x()
    new_stamped_pose.pose.orientation.y = quaternion_wxyz.y()
    new_stamped_pose.pose.orientation.z = quaternion_wxyz.z()
    new_stamped_pose.pose.orientation.w = quaternion_wxyz.w()

    return new_stamped_pose
