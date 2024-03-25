#!/usr/bin/env python3

# ROS imports
from builtin_interfaces.msg import Time
from rclpy.time import Time as rcl_Time

"""
Set of functions for helping with ROS
Time
- ros_time_to_seconds(): Converts ROS Time to seconds
"""


def ros_time_to_seconds(time: Time) -> float:
    """
    Converts ROS Time to seconds
    :param time:
    :return:
    """
    return time.sec + time.nanosec / 1e9


def ros_times_delta_seconds(time_a: rcl_Time, time_b: rcl_Time) -> float:
    """
    Computes delta seconds between two ROS Times
    :param time_a:
    :param time_b:
    :return:
    """
    return abs(time_a.nanoseconds - time_b.nanoseconds) / 1e9
