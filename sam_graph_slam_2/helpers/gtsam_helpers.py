#!/usr/bin/env python3

import numpy as np
import gtsam
import networkx as nx
import matplotlib.pyplot as plt

"""
GTSAM helper function
"""

from sam_graph_slam_2.helpers.general_helpers import get_quaternion_from_euler

# === Pose2 ===
def create_Pose2(input_pose):
    """
    Create a GTSAM Pose3 from the recorded poses in the form:
    [x,y,z,q_w,q_x,q_,y,q_z]
    """
    rot3 = gtsam.Rot3.Quaternion(input_pose[3], input_pose[4], input_pose[5], input_pose[6])
    rot3_yaw = rot3.yaw()
    # GTSAM Pose2: x, y, theta
    return gtsam.Pose2(input_pose[0], input_pose[1], rot3_yaw)


def pose2_list_to_nparray(pose_list):
    out_array = np.zeros((len(pose_list), 3))

    for i, pose2 in enumerate(pose_list):
        out_array[i, :] = pose2.x(), pose2.y(), pose2.theta()

    return out_array


# === Pose3 ====
def create_Pose3(input_pose):
    """
    Create a GTSAM Pose3 from the recorded poses in the form:
    [x,y,z,q_w,q_x,q_,y,q_z]
    """
    rot3 = gtsam.Rot3.Quaternion(input_pose[3], input_pose[4], input_pose[5], input_pose[6])
    return gtsam.Pose3(r=rot3, t=np.array((input_pose[0], input_pose[1], input_pose[2]), dtype=np.float64))


def convert_poses_to_Pose3(poses):
    """
    Poses is is of the form: [[x,y,z,q_w,q_x,q_,y,q_z]]
    """
    pose3s = []
    for pose in poses:
        pose3s.append(create_Pose3(pose))

    return pose3s


def apply_transformPoseFrom(pose3s, transform):
    """
    pose3s: [gtsam.Pose3]
    transform: gtsam.Pose3

    Apply the transform given in local coordinates, result is expressed in the world coords
    """
    transformed_pose3s = []
    for pose3 in pose3s:
        transformed_pose3 = pose3.transformPoseFrom(transform)
        transformed_pose3s.append(transformed_pose3)

    return transformed_pose3s


def merge_into_Pose3(input_pose2, input_rpd):
    """
    Create a GTSAM Pose3 from the recorded poses in the form:
    [x,y,z,q_w,q_x,q_,y,q_z]
    """

    # Calculate rotation component of Pose3
    # The yaw is provided by the pose2 and is combined with roll and pitch
    q_from_rpy = get_quaternion_from_euler(input_rpd[0],
                                           input_rpd[1],
                                           input_pose2[2])

    rot3 = gtsam.Rot3.Quaternion(q_from_rpy[0], q_from_rpy[1], q_from_rpy[2], q_from_rpy[3])

    # Calculate translation component of Pose3
    trans = np.array((input_pose2[0], input_pose2[1], input_rpd[2]), dtype=np.float64)

    return gtsam.Pose3(r=rot3, t=trans)


def show_simple_graph_2d(graph, x_keys, b_keys, values, label):
    """
    Show Graph of Pose2 and Point2 elements
    This function does not display data association colors

    """
    plot_limits = [-12.5, 12.5, -5, 20]

    # Check for buoys
    if b_keys is None:
        b_keys = {}

    # Initialize network
    G = nx.Graph()
    for i in range(graph.size()):
        factor = graph.at(i)
        for key_id, key in enumerate(factor.keys()):
            # Test if key corresponds to a pose
            if key in x_keys.values():
                pos = (values.atPose2(key).x(), values.atPose2(key).y())
                G.add_node(key, pos=pos, color='black')

            # Test if key corresponds to points
            elif key in b_keys.values():
                pos = (values.atPoint2(key)[0], values.atPoint2(key)[1])
                G.add_node(key, pos=pos, color='yellow')
            else:
                print('There was a problem with a factor not corresponding to an available key')

            # Add edges that represent binary factor: Odometry or detection
            for key_2_id, key_2 in enumerate(factor.keys()):
                if key != key_2 and key_id < key_2_id:
                    # detections will have key corresponding to a landmark
                    if key in b_keys.values() or key_2 in b_keys.values():
                        G.add_edge(key, key_2, color='red')
                    else:
                        G.add_edge(key, key_2, color='blue')

    # ===== Plot the graph using matplotlib =====
    # Matplotlib options
    fig, ax = plt.subplots()
    plt.title(f'Factor Graph\n{label}')
    ax.set_aspect('equal', 'box')
    plt.axis(plot_limits)
    plt.grid(True)
    plt.xticks(np.arange(plot_limits[0], plot_limits[1] + 1, 2.5))

    # Networkx Options
    pos = nx.get_node_attributes(G, 'pos')
    e_colors = nx.get_edge_attributes(G, 'color').values()
    n_colors = nx.get_node_attributes(G, 'color').values()
    options = {'node_size': 25, 'width': 3, 'with_labels': False}

    # Plot
    nx.draw_networkx(G, pos, edge_color=e_colors, node_color=n_colors, **options)
    np.arange(plot_limits[0], plot_limits[1] + 1, 2.5)
    plt.show()