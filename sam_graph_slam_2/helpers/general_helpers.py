#!/usr/bin/env python3

import os
import shutil
import math
import numpy as np
import csv
from enum import Enum

"""
General helper functions

ROS1 version was called sam_slam_helpers.py
"""


# ===== General stuff =====
def overwrite_directory(directory_path):
    if os.path.isdir(directory_path):
        shutil.rmtree(directory_path)
        os.makedirs(directory_path)
    else:
        os.makedirs(directory_path)


def read_csv_to_array(file_path):
    """
    Reads a CSV file and returns the contents as a 2D Numpy array.

    Parameters:
        file_path (str): The path to the CSV file to be read.

    Returns:
        numpy.ndarray: The contents of the CSV file as a 2D Numpy array.
    """
    try:
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            data = [row for row in reader]

        f.close()
        return np.array(data, dtype=np.float64)
    except OSError:
        return -1


def read_csv_to_list(file_path):
    try:
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            data = [row for row in reader]

        f.close()
        return data
    except OSError:
        return -1


def write_array_to_csv(file_path, data_array):
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in data_array:
            writer.writerow(row)


def get_enum_name_or_value(enum_class, value):
    try:
        enum_element = enum_class(value)
        return enum_element.name
    except ValueError:
        return value


# ===== General geometric function =====
def calculate_angle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    angle = math.atan2(dy, dx)
    return angle


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def calculate_distances(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    # Check if arrays have the same number of points
    if array1.shape != array2.shape:
        print("Size mismatch between the arrays.")
        return None

    # Calculate Euclidean distances
    distances = np.sqrt(np.sum((array1 - array2) ** 2, axis=1))

    return distances


def calculate_center(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """
    Calculate the center point of the provided coordinates
    :param x1: x coordinate point 1
    :param y1: y coordinate point 1
    :param x2: x coordinate point 2
    :param y2: y coordinate point 2
    :return: (2,) numpy array containing the x and y coordinates of the center
    """
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    return np.array((x_center, y_center), dtype=np.float64)


def closest_point_distance_to_line_segment(A, B, P):
    """
    Calculates the closest point, Q, on a line, defined by A and B, with point P.
    Also returns  the distance between P and Q.

    :param A: [x, y] start of line segment
    :param B: [x, y] end of line segment
    :param P: [x, y] point of interest
    :return: [Qx, Qy], distance
    """
    ABx = B[0] - A[0]
    ABy = B[1] - A[1]
    APx = P[0] - A[0]
    APy = P[1] - A[1]

    dot_product = ABx * APx + ABy * APy
    length_squared_AB = ABx * ABx + ABy * ABy

    t = dot_product / length_squared_AB

    if t < 0:
        Qx, Qy = A[0], A[1]
    elif t > 1:
        Qx, Qy = B[0], B[1]
    else:
        Qx = A[0] + t * ABx
        Qy = A[1] + t * ABy

    QPx = P[0] - Qx
    QPy = P[1] - Qy

    distance = math.sqrt(QPx * QPx + QPy * QPy)

    return [Qx, Qy], distance


def calculate_line_point_distance(x1: float, y1: float,
                                  x2: float, y2: float,
                                  x3: float, y3: float) -> float:
    """
    points 1 and 2 form a line segment, point 3 is
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param x3:
    :param y3:
    :return:
    """
    if x1 == x2 and y1 == y2:
        return -1

    # Calculate the length of the line segment
    line_mag_sqrd = (x2 - x1) ** 2 + (y2 - y1) ** 2
    u = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / line_mag_sqrd

    if 0 < u < 1.0:
        x_perpendicular = x1 + u * (x2 - x1)
        y_perpendicular = y1 + u * (y2 - y1)

        return math.sqrt((x_perpendicular - x3) ** 2 + (y_perpendicular - y3) ** 2)

    else:
        # Calculate the distance from the third point to each endpoint of the line segment
        distance_line_end_1 = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        distance_line_end_2 = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)

        # Find the minimum distance
        min_distance = min(distance_line_end_1, distance_line_end_2)

        return min_distance


def angle_between_rads(target_angle: float, source_angle: float) -> float:
    """
    Return the smallest angle between the two provided angles. Every angle is in radians and the
    output is bound between +/- pi
    :param target_angle:
    :param source_angle:
    :return:
    """
    # Bound the angle [-pi, pi]
    target_angle = math.remainder(target_angle, 2 * np.pi)
    source_angle = math.remainder(source_angle, 2 * np.pi)

    diff_angle = target_angle - source_angle

    if diff_angle > np.pi:
        diff_angle = diff_angle - 2 * np.pi
    elif diff_angle < -1 * np.pi:
        diff_angle = diff_angle + 2 * np.pi

    return diff_angle


def calc_pose_error(array_test: np.ndarray, array_true: np.ndarray):
    """
    Calculate the error between two arrays of equal size.
    x: [:,0]
    y: [:,1]
    theta: [:,2]
    """
    # Positional error
    pos_error = array_test[:, :2] - array_true[:, :2]

    theta_error = np.zeros((array_true.shape[0], 1))
    for i in range(array_true.shape[0]):
        theta_error[i] = angle_between_rads(array_test[i, 2], array_true[i, 2])

    return np.hstack((pos_error, theta_error))


# ===== 3d geometry utilities =====
def projectPixelTo3dRay(u, v, cx, cy, fx, fy):
    """
    From ROS-perception
    https://github.com/ros-perception/vision_opencv/blob/rolling/image_geometry/image_geometry/cameramodels.py
    Returns the unit vector which passes from the camera center to through rectified pixel (u, v),
    using the camera :math:`P` matrix.
    This is the inverse of :math:`project3dToPixel`.
    """
    x = (u - cx) / fx
    y = (v - cy) / fy
    norm = math.sqrt(x * x + y * y + 1)
    x /= norm
    y /= norm
    z = 1.0 / norm
    return x, y, z


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.
    Author: AutomaticAddison.com

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qw, qx, qy, qz: The orientation in quaternion [w, x,y,z] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qw, qx, qy, qz]


# ===== General Math function =====
def ceiling_division(n, d):
    return -(n // -d)


class odometry_data:
    def __init__(self, x, y, z, q_w, q_x, q_y, q_z, roll, pitch, depth, image_id=-1):
        self.x = x
        self.y = y
        self.z = z
        self.q_w = q_w
        self.q_x = q_x
        self.q_y = q_y
        self.q_z = q_z
        self.roll = roll
        self.pitch = pitch
        self.depth = depth
        self.image_id = image_id
