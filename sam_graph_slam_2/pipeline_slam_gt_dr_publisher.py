#!/usr/bin/env python3

import gtsam
import numpy as np

import rclpy
from rclpy import time
from rclpy.node import Node

from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

try:
    # ROS2 IMPORTS
    from .helpers.gtsam_ros_helpers import ros_pose_to_gtsam_pose3_and_stamp, gtsam_pose3_to_ros_pose3
    from .helpers.ros_helpers import ros_time_to_seconds
except:
    from helpers.gtsam_ros_helpers import ros_pose_to_gtsam_pose3_and_stamp, gtsam_pose3_to_ros_pose3
    from helpers.ros_helpers import ros_time_to_seconds

"""
Implements dr and gt publisher - dr based on gt
gtsam objects, Pose3, are used to maintain the dr pose

This has been ported to ROS2

Work remaining:
- Topic names should be gathers from a central source
- noise parameters should be controlled via a launch file/parameters
"""


class PipelineSimDrGtPublisher(Node):
    def __init__(self):
        super().__init__('pipeline_sim_dr_gt_publisher')
        self.get_logger().info('Created: Pipeline Sim Dr Gt Publisher')

        # ===== Topics =====
        self.gt_topic = '/sam/sim/odom'
        self.imu_topic = '/sam0/core/imu'
        self.pub_dr_topic = '/dr_odom'
        self.pub_gt_topic = '/gt_odom'

        # ===== Frame and tf stuff =====
        self.map_frame = 'odom'  # 'map'
        self.robot_frame = 'sam0_base_link'
        self.dr_tf_frame = 'dr_frame'  # Currently intended for the detector saving dr poses with pointcloud data

        # Set up TF broadcaster
        self.tf_br = TransformBroadcaster(self)

        # Set up TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize variables
        self.current_gt_pose = None  # PoseStamped
        self.last_gt_pose = None  # PoseStamped

        self.gt_between = None
        self.dr_pose3 = None
        self.dr_stamp = None
        self.dr_frame = None  # parent frame of the dr odometry

        # Subscribe to simulated IMU topic
        # self.imu_subscriber = rospy.Subscriber(self.imu_topic, Imu, self.imu_callback)  #ROS1
        self.imu_subscriber = self.create_subscription(msg_type=Imu, topic=self.imu_topic, callback=self.imu_callback,
                                                       qos_profile=10)

        # Publishers
        # self.dr_publisher = rospy.Publisher(self.pub_dr_topic, Odometry, queue_size=10)  # ROS1
        # self.gt_publisher = rospy.Publisher(self.pub_gt_topic, Odometry, queue_size=10)  # ROS1
        self.dr_publisher = self.create_publisher(msg_type=Odometry, topic=self.pub_dr_topic, qos_profile=10)
        self.gt_publisher = self.create_publisher(msg_type=Odometry, topic=self.pub_gt_topic, qos_profile=10)

        # DR noise parameters
        self.add_noise = True
        self.bound_depth = True
        self.bound_pitch_roll = True

        # Initialization noise
        self.init_position_sigmas = np.array([1.0, 1.0, 1.0])  # x, y, z
        self.init_rotation_sigmas = np.array([np.pi / 1e3, np.pi / 1e3, np.pi / 50])  # roll, pitch, yaw

        # Step noise
        self.delta_position_sigmas = np.array([0.001, 0.001, 0.001])  # x, y, z - per second
        self.delta_rotation_sigmas = np.array([np.pi / 1e5, np.pi / 1e5, np.pi / 1e2])  # roll, pitch, yaw - per second

        self.depth_sigma = 0.1

        # ===== logging settings =====

    def imu_callback(self, imu_msg: Imu):
        # for simulated data current pose is ground truth
        self.update_current_pose_world_frame()

        # initial
        if self.last_gt_pose is None and self.current_gt_pose is not None:
            # generate current dr
            self.dr_pose3, self.dr_stamp = ros_pose_to_gtsam_pose3_and_stamp(self.current_gt_pose)
            self.dr_frame = self.map_frame

            if self.add_noise:
                self.add_noise_to_initial_pose()

            # Publish dr and gt
            self.publish_dr_pose()
            self.publish_gt_pose()

        # Normal running condition
        elif self.last_gt_pose is not None and self.current_gt_pose is not None:
            self.update_dr_pose3()

            # Publish dr and gt
            self.publish_dr_pose()
            self.publish_gt_pose()

    def update_current_pose_world_frame(self):
        # Get the current transform from the base_link to the world frame
        try:
            # ( to_frame, from_frame, ...
            transform = self.tf_buffer.lookup_transform(self.map_frame,
                                                        self.robot_frame,
                                                        rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(f'Could not transform {self.map_frame} to {self.robot_frame}: {ex}')
            return

        # Define gt pose based on the above transform
        gt_pose = PoseStamped()

        gt_pose.header.stamp = transform.header.stamp
        gt_pose.header.frame_id = self.map_frame
        gt_pose.pose.position.x = transform.transform.translation.x
        gt_pose.pose.position.y = transform.transform.translation.y
        gt_pose.pose.position.z = transform.transform.translation.z
        gt_pose.pose.orientation.x = transform.transform.rotation.x
        gt_pose.pose.orientation.y = transform.transform.rotation.y
        gt_pose.pose.orientation.z = transform.transform.rotation.z
        gt_pose.pose.orientation.w = transform.transform.rotation.w

        self.last_gt_pose = self.current_gt_pose
        self.current_gt_pose = gt_pose

    def add_noise_to_initial_pose(self):
        if self.dr_pose3 is None:
            return
        # Translational noise
        position_noise = np.random.normal(0, self.init_position_sigmas)

        # Rotational noise
        roll_noise, pitch_noise, yaw_noise = np.random.normal(0, self.init_rotation_sigmas)
        # Still confused by ypr vs rpy :)
        rotation_noise = gtsam.Rot3.Ypr(yaw_noise, pitch_noise, roll_noise)

        noise_pose3 = gtsam.Pose3(rotation_noise, position_noise)

        noisy_pose3 = self.dr_pose3.compose(noise_pose3)
        self.dr_pose3 = noisy_pose3

        return

    def update_dr_pose3(self):
        # calculate between
        init_pose3, init_time = ros_pose_to_gtsam_pose3_and_stamp(self.last_gt_pose)
        final_pose3, final_time = ros_pose_to_gtsam_pose3_and_stamp(self.current_gt_pose)

        between_pose3 = init_pose3.between(final_pose3)
        # dt = (final_time - init_time).to_sec()  # Old ROS1 w/ rospy approach
        dt = ros_time_to_seconds(final_time) - ros_time_to_seconds(init_time)

        new_dr_pose = self.dr_pose3.compose(between_pose3)

        if self.add_noise:
            noise_pose3 = self.return_step_noise_pose3(dt)
            new_dr_pose = new_dr_pose.compose(noise_pose3)

        if self.bound_depth:
            # Determine depth values
            depth_noise = np.random.normal(0, self.depth_sigma)
            bounded_noisy_depth = init_pose3.z() + depth_noise

            # Update the depth, z, value
            new_translation = new_dr_pose.translation()
            new_translation[2] = bounded_noisy_depth

            # Reform Pose3
            rotation = new_dr_pose.rotation()
            new_dr_pose = gtsam.Pose3(rotation, new_translation)

        if self.bound_pitch_roll:
            # determine pitch and roll values
            roll_noise, pitch_noise, _ = np.random.normal(0, self.init_rotation_sigmas)  # roll, pitch, yaw
            _, pitch_current, roll_current = final_pose3.rotation().ypr()
            bounded_pitch = pitch_current + pitch_noise
            bounded_roll = roll_current + roll_noise
            additive_yaw = new_dr_pose.rotation().yaw()

            # form the newly bounded Rot3
            bounded_rotation = gtsam.Rot3.Ypr(additive_yaw, bounded_pitch, bounded_roll)

            # Reform Pose3
            translation = new_dr_pose.translation()
            new_dr_pose = gtsam.Pose3(bounded_rotation, translation)

        self.dr_pose3 = new_dr_pose
        self.dr_stamp = final_time

    def return_step_noise_pose3(self, dt):

        roll_noise, pitch_noise, yaw_noise = np.random.normal(0, self.delta_rotation_sigmas * np.sqrt(dt))  # r, p, y
        x_noise, y_noise, z_noise = np.random.normal(0, self.delta_position_sigmas * np.sqrt(dt))

        rotation_noise = gtsam.Rot3.Ypr(y=yaw_noise, p=pitch_noise, r=roll_noise)  # yaw, pitch, roll
        translation_noise = np.array((x_noise, y_noise, z_noise))  # OLD: gtsam.Point3(x_noise, y_noise, z_noise)
        pose_noise = gtsam.Pose3(rotation_noise, translation_noise)

        return pose_noise

    def publish_gt_pose(self):

        # Create an Odometry message for the Dead Reckoning pose
        gt_pose_msg = Odometry()
        gt_pose_msg.header = self.current_gt_pose.header
        gt_pose_msg.pose.pose = self.current_gt_pose.pose

        # Publish the Dead Reckoning pose
        self.gt_publisher.publish(gt_pose_msg)

    def publish_dr_pose(self):
        dr_pose_msg = Odometry()
        current_dr_pose = gtsam_pose3_to_ros_pose3(self.dr_pose3, self.dr_stamp, self.dr_frame)
        dr_pose_msg.header = current_dr_pose.header
        dr_pose_msg.pose.pose = current_dr_pose.pose

        # Publish the Dead Reckoning pose
        self.dr_publisher.publish(dr_pose_msg)
        self.publish_dr_transform(dr_pose_msg=dr_pose_msg)

    def publish_dr_transform(self, dr_pose_msg: Odometry):
        """
        Publish the transform of the dr pose
        :param dr_pose_msg: Odometry of the dr pose
        :return:
        """
        # Generate content for transform
        dr_timestamp = dr_pose_msg.header.stamp
        dr_frame = dr_pose_msg.header.frame_id
        dr_trans_x, dr_trans_y, dr_trans_z = (dr_pose_msg.pose.pose.position.x,
                                              dr_pose_msg.pose.pose.position.y,
                                              dr_pose_msg.pose.pose.position.z)
        dr_rot_x, dr_rot_y, dr_rot_z, dr_rot_w = (dr_pose_msg.pose.pose.orientation.x,
                                                  dr_pose_msg.pose.pose.orientation.y,
                                                  dr_pose_msg.pose.pose.orientation.z,
                                                  dr_pose_msg.pose.pose.orientation.w)

        # Form transform for the tf2_ros broadcaster
        dr_t = TransformStamped()
        dr_t.header.stamp = dr_timestamp
        dr_t.header.frame_id = dr_frame
        dr_t.child_frame_id = self.dr_tf_frame
        dr_t.transform.translation.x = dr_trans_x
        dr_t.transform.translation.y = dr_trans_y
        dr_t.transform.translation.z = dr_trans_z
        dr_t.transform.rotation.x = dr_rot_x
        dr_t.transform.rotation.y = dr_rot_y
        dr_t.transform.rotation.z = dr_rot_z
        dr_t.transform.rotation.w = dr_rot_w

        # Attempt to broadcast
        try:
            self.tf_br.sendTransform(dr_t)

        except Exception as e:
            self.get_logger().error(f"Transform broadcast error: {ros_time_to_seconds(dr_timestamp)}")


def main(args=None):
    rclpy.init(args=args)
    pipeline_sim_dr_gt_publisher = PipelineSimDrGtPublisher()
    rclpy.spin(pipeline_sim_dr_gt_publisher)
    pipeline_sim_dr_gt_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
