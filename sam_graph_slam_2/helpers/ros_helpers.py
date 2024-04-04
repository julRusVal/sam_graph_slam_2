#!/usr/bin/env python3

# ROS imports
from builtin_interfaces.msg import Time as Stamp
from rclpy.time import Time as rcl_Time

"""
Set of functions for helping with ROS
Time
- ros_time_to_seconds(): Converts ROS Time to seconds
"""


def ros_time_to_secs(time: Stamp) -> float:
    """
    Converts ROS Time to seconds
    :param time:
    :return:
    """
    return time.sec + time.nanosec / 1e9


def rcl_time_to_secs(time: rcl_Time) -> float:
    """
    Converts rcl Time to seconds. It appears to me that the time message and object formats are a little different.
    rcl_Time only exposes the nanoseconds.
    :param time: float, in seconds
    :return:
    """
    return time.nanoseconds / 1e9


def rcl_times_delta_secs(time_a: rcl_Time, time_b: rcl_Time) -> float:
    """
    Computes delta seconds between two ROS Times
    :param time_a:
    :param time_b:
    :return:
    """
    return abs(time_a.nanoseconds - time_b.nanoseconds) / 1e9


def rcl_time_to_stamp(time: rcl_Time) -> Stamp:
    """
    Converts rcl Time to stamp
    :param time:
    :return:
    """
    stamp = Stamp()
    stamp.sec = int(time.nanoseconds // 1e9)
    stamp.nanosec = int(time.nanoseconds % 1e9)
    return stamp
