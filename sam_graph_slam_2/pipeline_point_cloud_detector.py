#!/usr/bin/env python3

import os

import numpy as np
import open3d as o3d  # Used for processing point cloud data


# ROS
import rclpy
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile

import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# from tf.transformations import ...  # ROS1
from tf_transformations import (quaternion_matrix, compose_matrix, euler_from_quaternion, inverse_matrix,
                                quaternion_from_euler)
import sensor_msgs_py.point_cloud2 as pc2  # Utility for handling point clouds

# Messages
# import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped, Quaternion, Pose
from visualization_msgs.msg import Marker
from vision_msgs.msg import ObjectHypothesisWithPose, Detection2DArray, Detection2D

# Imports from other smarc packages
# I guess this info should be kept somewhere other than sss_object_detection...
# For now, I'm going to copy the contents to the local detector package
# from sss_object_detection.consts import ObjectID

# Imports from this package
# See the note above about the
try:
    from .detectors.consts import ObjectID
    from .detectors.process_pointcloud2 import ProcessPointCloud2
    # from .detectors import process_pointcloud2  # this was not working when launched from ros2
    from .helpers.ros_helpers import ros_time_to_seconds, ros_times_delta_seconds
except:
    from detectors.consts import ObjectID
    from .detectors.process_pointcloud2 import ProcessPointCloud2
    from detectors import process_pointcloud2  # This was working
    from helpers.ros_helpers import ros_time_to_seconds, ros_times_delta_seconds


'''
Basic detector with the ability to save point clouds for offline processing and analysis
INPUT:


OUTPUT:
1) Publish detection
Topic: /{robot_name}/payload/sidescan/detection_hypothesis'
Message type and format:
- Detection2DArray message with the pipeline detection coordinates.
    - Detection2d[]
        - ObjectHypothesisWithPose[]
            - hypothesis: ObjectHypothesis
                - class_id: string - Uses the names defined in ObjectID to identify
                - score: float - Can indicate the sequence ID of the data that generated the data or the true confidence
                                 score.
            - pose: PoseWithCovariance - stores the pose of the corresponding detection
            
2) Publish detection marker
Topic: /detection_marker
Message type: Marker
'''


def stamped_transform_to_homogeneous_matrix(transform_stamped: TransformStamped):
    # Extract translation and quaternion components from the TransformStamped message
    translation = [transform_stamped.transform.translation.x,
                   transform_stamped.transform.translation.y,
                   transform_stamped.transform.translation.z]
    quaternion = [transform_stamped.transform.rotation.x,
                  transform_stamped.transform.rotation.y,
                  transform_stamped.transform.rotation.z,
                  transform_stamped.transform.rotation.w]

    # Create a 4x4 homogeneous transformation matrix
    homogeneous_matrix = compose_matrix(
        translate=translation,
        angles=euler_from_quaternion(quaternion)
    )

    return homogeneous_matrix


def check_homogeneous_matrix(homogeneous_matrix):
    R = homogeneous_matrix[0:3, 0:3]
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0):
        print("Homogeneous matrix not pure rotation matrix")


def stamped_transform_to_rotation_matrix(transform_stamped: TransformStamped):
    # Convert quaternion to rotation matrix using tf2
    rotation_matrix = quaternion_matrix([transform_stamped.transform.rotation.x,
                                         transform_stamped.transform.rotation.y,
                                         transform_stamped.transform.rotation.z,
                                         transform_stamped.transform.rotation.w])

    return rotation_matrix


class PointCloudDetector(Node):
    def __init__(self, topic: str = '/sam0/mbes/odom/bathy_points',
                 save_data: bool = False,
                 save_location: str = '',
                 save_timeout: float = 10):
        super().__init__('point_cloud_detector')

        # Set up TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # === Names ===
        # Robot name
        self.robot_name = 'sam'

        # Subscription topic name(s)
        self.point_cloud_topic = topic

        # publisher topic name(s)
        self.detection_topic = f'/{self.robot_name}/payload/sidescan/detection_hypothesis'
        self.detection_marker_topic = '/detection_marker'

        # Data frame setting
        self.data_frame = 'odom'  # 'map' from given mbes data
        self.robot_frame = 'sam0'  # 'sam0_base_link' transform into robot frame
        self.dr_tf_frame = 'dr_frame'  # frame of dr, used to transform mbes pointcloud

        # Verbose
        # TODO figure out parameters in ROS2
        # self.verbose_detector = self.declare_parameter('verbose_detector', False)
        # assert isinstance(self.verbose_detector, bool)
        self.verbose_detector = True

        # Data recording setting
        self.save_data = save_data
        # self.save_timeout = rospy.Duration.from_sec(save_timeout)  # ROS1
        self.save_timeout = save_timeout  # save time  out
        self.last_data_time = None
        self.data_written = False
        self.stacked_pc_original = None
        self.stacked_pc_transformed = None
        self.stacked_world_to_robot_transform = None
        self.stacked_robot_to_world_transform = None

        # Data saving settings
        if os.path.isdir(save_location):
            self.save_location = save_location + '/'
        else:
            self.save_location = ''

        # Detector setting
        # TODO figure out parameters in ROS2
        # self.min_update_time = self.declare_parameter('detector_min_update_time', False)
        # assert isinstance(self.verbose_detector, bool)
        self.min_update_time = 1.0
        print(f"Detector min_update_time: {self.min_update_time}")
        self.confidence_dummy = 0.5  # Confidence is just a dummy value for now

        # Set up subscriptions
        # rospy.Subscriber(self.point_cloud_topic, pc2.PointCloud2, self.point_cloud_callback, queue_size=1)  # ROS1

        point_cloud_qos_profile = rclpy.qos.QoSProfile(depth=1)  # The hope is to only process the most recent data
        self.pc_sub = self.create_subscription(msg_type=PointCloud2, topic=self.point_cloud_topic,
                                               callback=self.point_cloud_callback, qos_profile=point_cloud_qos_profile)

        # Set up a timer to check for data saving periodically
        # rospy.Timer(self.save_timeout, self.save_timer_callback)  # ROS1
        self.save_timer = self.create_timer(timer_period_sec=self.save_timeout,
                                            callback=self.save_timer_callback)

        # Set up publishers for the detections
        # 1) message for the online sam slam listener
        # 2) marker for visualization
        # self.detection_marker_pub = rospy.Publisher('/detection_marker', Marker, queue_size=10)
        # self.detection_pub = rospy.Publisher(f'/{self.robot_name}/payload/sidescan/detection_hypothesis',
        #                                      Detection2DArray,
        #                                      queue_size=2)
        self.detection_pub = self.create_publisher(msg_type=Detection2DArray,
                                                   topic=self.detection_topic,
                                                   qos_profile=10)

        self.detection_marker_pub = self.create_publisher(msg_type=Marker,
                                                          topic=self.detection_marker_topic,
                                                          qos_profile=10)

    def point_cloud_callback(self, msg):
        # time_now = rospy.Time.now()  # ROS1
        time_now = self.get_clock().now()

        # handle first message and regulate the rate that message are processed
        if self.last_data_time is None:
            print("Pointcloud2 data received")
            self.last_data_time = time_now
        # TODO Fix time comparison
        # elif (time_now - self.last_data_time).to_sec() < self.min_update_time:
        elif ros_times_delta_seconds(time_now, self.last_data_time) < self.min_update_time:
            return
        else:
            # Update the last received data time
            self.last_data_time = time_now

        # Find the transform to move pointcloud from odom/map to the robots frame
        try:
            # ( to_frame, from_frame, ...
            # transform = self.tf_buffer.lookup_transform(self.robot_frame,
            #                                             self.data_frame,
            #                                             rospy.Time(0),
            #                                             rospy.Duration(1, int(0.1 * 1e9)))

            transform = self.tf_buffer.lookup_transform(target_frame=self.robot_frame,
                                                        source_frame=self.data_frame,
                                                        time=Time(),
                                                        timeout=Duration(seconds=0.25))

        except (LookupException, ConnectivityException, ExtrapolationException):
            self.get_logger().info("Failed to look up transform")
            print("Failed to look up transform")
            return

        # Convert the ROS transform to a 4x4 homogeneous transform
        homogeneous_transform = stamped_transform_to_homogeneous_matrix(transform)
        check_homogeneous_matrix(homogeneous_transform)
        inverse_homogeneous_transform = inverse_matrix(homogeneous_transform)

        # Convert PointCloud2 message to NumPy array
        pc_list = pc2.read_points_list(msg, field_names=["x", "y", "z"], skip_nans=True)  # this data comes out
        pc_array = np.array(pc_list)

        # Convert pointcloud to od3 format
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(pc_array)

        # Apply transform to point cloud
        pc_o3d.transform(homogeneous_transform)

        # save both the original and transformed data
        pc_array_transformed = np.asarray(pc_o3d.points)

        # Process pointcloud data
        # process_start_time = rospy.Time.now()  # ROS1
        process_start_time = self.get_clock().now()
        detector = ProcessPointCloud2(pc_array_transformed)
        # process_end_time = rospy.Time.now()  # ROS1
        process_end_time = self.get_clock().now()

        if self.verbose_detector:
            process_time_seconds = ros_times_delta_seconds(process_end_time, process_start_time)
            print(f"Detector - Processing time {process_time_seconds:.3f} secs")

        if detector.detection_coords_world.size != 3:
            print("No detection!!")
        else:
            # publish a marker indicating the position of the pipeline detection, world coords
            detection_homo = np.vstack([detector.detection_coords_world.reshape(3, 1),
                                        np.array([1])])

            detection_world = np.matmul(inverse_homogeneous_transform, detection_homo)

            # self.publish_detection_marker(detector.detection_coords_world, self.robot_frame)
            self.publish_detection_marker(detection_world, self.data_frame)

            # Debugging print outs
            # print(f"Raw: {detector.detection_coords_world}")
            # print(f"Transformed: {detection_world}")

            # Publish the message for SLAM
            self.publish_mbes_pipe_detection(detector.detection_coords_world, self.confidence_dummy, msg.header.stamp)
        # # Store the point cloud data, original and transformed
        if self.save_data:
            # TODO can these get out of sync? check?
            # Original
            if self.stacked_pc_original is None:
                self.stacked_pc_original = pc_array
            else:
                self.stacked_pc_original = np.dstack([self.stacked_pc_original, pc_array])

            # Transformed
            if self.stacked_pc_transformed is None:
                self.stacked_pc_transformed = pc_array_transformed
            else:
                self.stacked_pc_transformed = np.dstack([self.stacked_pc_transformed, pc_array_transformed])

            # record the transform
            if self.stacked_world_to_robot_transform is None:
                self.stacked_world_to_robot_transform = homogeneous_transform
            else:
                self.stacked_world_to_robot_transform = np.dstack(
                    [self.stacked_world_to_robot_transform, homogeneous_transform])

            if self.stacked_robot_to_world_transform is None:
                self.stacked_robot_to_world_transform = inverse_homogeneous_transform
            else:
                self.stacked_robot_to_world_transform = np.dstack(
                    [self.stacked_robot_to_world_transform, inverse_homogeneous_transform])

    def save_timer_callback(self):
        # Check if no new data has been received for the specified interval
        if self.last_data_time is None:
            return
        # if rospy.Time.now() - self.last_data_time > self.save_timeout:
        delta_time_seconds = ros_times_delta_seconds(time_a=self.get_clock().now(),
                                                     time_b=self.last_data_time)
        if delta_time_seconds > self.save_timeout:
            # Save point cloud data to a CSV file
            if self.save_data:
                self.save_stacked_point_cloud()
            # rospy.signal_shutdown("Script shutting down")  # ROS1
            self.get_logger().info("Detector node shutting down")
            rclpy.shutdown()

    def save_stacked_point_cloud(self):
        if self.stacked_pc_original is None or self.stacked_pc_transformed is None:
            print("No point cloud data to save.")
        elif self.stacked_world_to_robot_transform is None or self.stacked_robot_to_world_transform is None:
            print("No transforms to save.")
        else:
            original_name = f"{self.save_location}pc_original.npy"
            transformed_name = f"{self.save_location}pc_transformed.npy"
            world_to_local_name = f"{self.save_location}world_to_local.npy"
            local_to_world_name = f"{self.save_location}local_to_world.npy"

            np.save(original_name, self.stacked_pc_original)
            np.save(transformed_name, self.stacked_pc_transformed)
            np.save(world_to_local_name, self.stacked_world_to_robot_transform)
            np.save(local_to_world_name, self.stacked_robot_to_world_transform)

            self.data_written = True

            print(f"Stacked point cloud saved")

        if self.data_written:
            # rospy.signal_shutdown("Script shutting down")  # ROS1
            self.get_logger().info("Detector node shutting down")
            rclpy.shutdown()

    def publish_detection_marker(self, detection, frame):
        """
        Publishes some markers for debugging estimated detection location and the DA location
        """
        # output in xyzw order
        heading_quaternion = quaternion_from_euler(0, 0, 0)
        heading_quaternion_type = Quaternion()
        heading_quaternion_type.x = heading_quaternion[0]
        heading_quaternion_type.y = heading_quaternion[1]
        heading_quaternion_type.z = heading_quaternion[2]
        heading_quaternion_type.w = heading_quaternion[3]

        detection_flat = detection.reshape(-1, )

        # Estimated detection location
        marker = Marker()
        marker.header.frame_id = frame
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.id = 0
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0
        marker.pose.position.x = detection_flat[0]
        marker.pose.position.y = detection_flat[1]
        marker.pose.position.z = detection_flat[2]
        marker.pose.orientation = heading_quaternion_type
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.6
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.detection_marker_pub.publish(marker)
        # print("Published marker")

    def publish_mbes_pipe_detection(self, detection_coords,
                                    detection_confidence,
                                    stamp,
                                    seq_id=None):

        """
        Publishes message for mbes detection given [x, y, z] coords.

        :param detection_coords: numpy array of x, y, z coordinates
        :param detection_confidence: float confidence, in some cases this is used for data association (but not here)
        :param stamp: stamp of the message
        :param seq_id: seq_id can be used for data association
        :return:
        """

        # Convert detection coords into a Pose()
        detection_flat = detection_coords.reshape(-1, )
        detection_pose = Pose()
        detection_pose.position.x = detection_flat[0]
        detection_pose.position.y = detection_flat[1]
        detection_pose.position.z = detection_flat[2]

        # Form the message
        # Detection2DArray()
        # -> list of Detection2D()
        # ---> list of ObjectHypothesisWithPose()
        # -----> Contains: id, score, pose
        # pose is PoseWithCovariance()

        detection_array_msg = Detection2DArray()
        detection_array_msg.header.frame_id = self.robot_frame
        detection_array_msg.header.stamp = stamp

        # Individual detection, Detection2D, currently only a single detection per call will be published
        detection_msg = Detection2D()
        detection_msg.header = detection_array_msg.header

        # Define single ObjectHypothesisWithPose
        object_hypothesis_pose_msg = ObjectHypothesisWithPose()

        # set the hypothesis: class id and score
        object_hypothesis_pose_msg.hypothesis.class_id = ObjectID.PIPE.name

        if seq_id is not None:
            object_hypothesis_pose_msg.hypothesis.score = seq_id
        else:
            object_hypothesis_pose_msg.hypothesis.score = detection_confidence

        # Set the pose
        object_hypothesis_pose_msg.pose.pose = detection_pose

        # Append the to form a complete message
        detection_msg.results.append(object_hypothesis_pose_msg)
        detection_array_msg.detections.append(detection_msg)

        self.detection_pub.publish(detection_array_msg)
        return


def main(topic: str = None, save_data: bool = False, save_location: str = '',
         save_timeout: float = 5.0, args=None, ):
    rclpy.init(args=args)
    pipeline_point_cloud_detector = PointCloudDetector(topic=topic,
                                                       save_data=save_data,
                                                       save_location=save_location,
                                                       save_timeout=save_timeout)
    rclpy.spin(pipeline_point_cloud_detector)
    pipeline_point_cloud_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main(topic='/sam0/mbes/odom/bathy_points',
         save_data=True,
         save_location='/home/julian/porting_test_data',
         save_timeout=5)
