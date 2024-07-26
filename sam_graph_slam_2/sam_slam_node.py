#!/usr/bin/env python3
import os.path
import sys
import ast
from functools import partial

# cv_bridge and cv2 to convert and save images
# from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2

# ROS
import rclpy
from rclpy.time import Time as rcl_Time
from rclpy.duration import Duration as rcl_Duration
from rclpy.node import Node

# TF stuff
import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster
# from tf2_ros import TransformException

# Transform conversion
from tf_transformations import euler_from_quaternion, quaternion_multiply  # quaternion_from_euler
import tf2_geometry_msgs

# ROS messages
# from std_msgs.msg import Time
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Pose, PoseStamped  # TwistStamped
from sensor_msgs.msg import Image, CameraInfo
from smarc_msgs.msg import Sidescan

# Topics and frames
from sam_msgs.msg import Topics as SamTopics
from sam_msgs.msg import Links as SamLinks
from dead_reckoning_msgs.msg import Topics as DRTopics
from sam_graph_slam_2_msgs.msg import Topics as GraphSlamTopics

try:  # ROS 2
    # Detector
    # from sss_object_detection.consts import ObjectID
    from .detectors.consts import ObjectID

    # General Helpers
    # rom .helpers.gtsam_helpers import show_simple_graph_2d
    from .helpers.general_helpers import write_array_to_csv, overwrite_directory
    from .helpers.general_helpers import get_enum_name_or_value

    # ROS helpers
    from .helpers.ros_helpers import rcl_time_to_secs, rcl_times_delta_secs

    # SLAM imports
    from .slam_packages.sam_slam_graphs import online_slam_2d
    from .slam_packages.sam_slam_graphs import analyze_slam

except:  # Starting node directly
    # Detector
    # from sss_object_detection.consts import ObjectID
    from detectors.consts import ObjectID

    # General Helpers
    # rom .helpers.gtsam_helpers import show_simple_graph_2d
    from helpers.general_helpers import write_array_to_csv, overwrite_directory
    from helpers.general_helpers import get_enum_name_or_value

    # ROS helpers
    from helpers.ros_helpers import rcl_time_to_secs, rcl_times_delta_secs

    # SLAM imports
    from slam_packages.sam_slam_graphs import online_slam_2d
    from slam_packages.sam_slam_graphs import analyze_slam


def imgmsg_to_cv2(img_msg):
    """
    Its assumed that the input image is rgb, opencv expects bgr
    """
    dtype = np.dtype("uint8")  # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3),
                              dtype=dtype, buffer=img_msg.data)
    # flip converts rgb to bgr
    image_opencv = np.flip(image_opencv, axis=2)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv


class SamSlamListener(Node):
    """
    This class defines the behavior of the slam_listener node
    Dead reckoning (dr) and ground truth data is saved [x, y, z, q_w, q_x, q_y, q_z] in the map frame
    Note I about the gt data, there are two potential sources of gt data
    - the topic: /sam/sim/odom
    - the frame attached to the simulation: gt/sam/base_link (currently used)
    Note II about the gt data, I have tried to transform all the poses to the map frame but even after this I need to
    invert the sign of the x-axis and corrected_heading = pi - original_heading

    Detections are saved in two lists. {There is no need for both}
    detections format: [x_map, y_map, z_map, q_w, q_x, q_y, q_z, corresponding dr id, score]
    detections_graph format: [x_map, y_map, z_map, x_rel, y_rel, z_vel, corresponding dr id]
    rope_detections_graph format: [x_map, y_map, z_map, x_rel, y_rel, z_vel, corresponding dr id]

    Online
    If an online_graph object is passed to the listener it will be updated at every detection
    - dr_callback: first update and odometry updates
    - det_callback: update
    - buoy_callback: send buoy info to online graph
    - time_check_callback: Save results when there is no longer any dr update

    """

    def __init__(self):
        super().__init__("sam_slam_listener_node")
        self.get_logger().info("Created: Sam Slam Listener")

        self.declare_node_parameters()

        # tf stuff
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ===== Scenario parameters =====
        self.simulated_data = self.get_parameter("simulated_data").value
        self.simulated_detections = self.get_parameter("simulated_detections").value
        self.algae_scenario = self.get_parameter("algae_scenario").value
        self.pipeline_scenario = self.get_parameter("pipeline_scenario").value
        self.record_gt = self.get_parameter("record_ground_truth").value

        # DR correction --Scenario specific--
        # TODO this will be different for the new simulator
        if self.algae_scenario:
            self.correct_dr = False
        elif self.pipeline_scenario:  # Simulated data for the pipeline scenario does not require correction
            self.correct_dr = False
        elif self.simulated_data:  # Simulated data for the stonefish algae scenario required correction
            self.correct_dr = True
        else:  # real data from the algae farm does not require correction
            self.correct_dr = False

        # ===== Graph input parameters =====
        # This setting controls how buoy detections are handled
        # True to use all buoy detections, otherwise detector time limits apply
        self.prioritize_buoy_detections = self.get_parameter("prioritize_buoy_detections").value
        # Provided information
        self.manual_associations = self.get_parameter("manual_associations").value
        # Rope detection usage
        self.rope_associations = self.get_parameter('rope_associations').value

        # ===== Topic names =====
        self.robot_name = self.get_parameter("robot_name").value

        # === Dead reckoning and ground truth ===
        # These values vary based on the scenario :(
        # Keeping this to show how it was set up in the past
        # 1) Pipeline scenario was in the simulator with data from an earlier or renamed topics generated by unity
        # 2) Simulated data was of the algae farm scene in stonefish -> this also required some processing of raw dr
        # 3) Real algae farm data !! :) !!
        # if self.pipeline_scenario:
        #     # TODO unify the naming convention of topics
        #     # self.buoy_topic = f'/{self.robot_name}/sim/marked_positions'  # See below, now using defined topics
        #     # self.rope_topic = f'/{self.robot_name}/sim/rope_outer_marker'  # See below, now using defined topics
        #     self.gt_odom_topic = f'/gt_odom'
        #     self.dr_odom_topic = f'/dr_odom'
        # elif self.simulated_data:
        #     # self.buoy_topic = f'/{self.robot_name}/sim/marked_positions'  # See below, now using defined topics
        #     # self.rope_topic = f'/{self.robot_name}/sim/rope_outer_marker'  # See below, now using defined topics
        #     self.gt_odom_topic = f'/{self.robot_name}/sim/odom'
        #     self.dr_odom_topic = f'/{self.robot_name}/dr/odom'
        # else:
        #     # self.buoy_topic = f'/{self.robot_name}/real/marked_positions'  # See below, now using defined topics
        #     # self.rope_topic = f'/{self.robot_name}/real/rope_outer_marker'  # See below, now using defined topics
        #     self.gt_odom_topic = f'/{self.robot_name}/dr/gps_odom'
        #     self.dr_odom_topic = f'/{self.robot_name}/dr/odom'

        self.buoy_topic = GraphSlamTopics.MAP_POINT_FEATURE_TOPIC
        self.rope_topic = GraphSlamTopics.MAP_LINE_FEATURE_TOPIC

        self.gt_odom_topic = GraphSlamTopics.GT_ODOM_TOPIC
        self.dr_odom_topic = GraphSlamTopics.DR_ODOM_TOPIC

        self.roll_topic = f'/{self.robot_name}/dr/roll'
        self.pitch_topic = f'/{self.robot_name}/dr/pitch'
        self.depth_topic = f'/{self.robot_name}/dr/depth'

        # === Detector topic
        self.det_topic = GraphSlamTopics.DETECTOR_HYPOTH_TOPIC
        # Old topic: f'/{self.robot_name}/payload/sidescan/detection_hypothesis'

        # === Sonar ===
        self.sss_topic = SamTopics.SIDESCAN_TOPIC
        # Old topic: f'/{self.robot_name}/payload/sidescan'

        # === Cameras ===
        if self.simulated_data:
            # Camera: down
            self.cam_down_image_topic = f'/{self.robot_name}/perception/csi_cam_0/camera/image_color'
            self.cam_down_info_topic = f'/{self.robot_name}/perception/csi_cam_0/camera/camera_info'
            # Camera: left
            self.cam_left_image_topic = f'/{self.robot_name}/perception/csi_cam_1/camera/image_color'
            self.cam_left_info_topic = f'/{self.robot_name}/perception/csi_cam_1/camera/camera_info'
            # Camera: right
            self.cam_right_image_topic = f'/{self.robot_name}/perception/csi_cam_2/camera/image_color'
            self.cam_right_info_topic = f'/{self.robot_name}/perception/csi_cam_2/camera/camera_info'

        else:
            # Camera: down
            self.cam_down_image_topic = f'/{self.robot_name}/payload/cam_down/image_raw'
            self.cam_down_info_topic = f'/{self.robot_name}/payload/cam_down/camera_info'
            # Camera: left
            self.cam_left_image_topic = f'/{self.robot_name}/payload/cam_port/image_raw'
            self.cam_left_info_topic = f'/{self.robot_name}/payload/cam_port/camera_info'
            # Camera: right
            self.cam_right_image_topic = f'/{self.robot_name}/payload/cam_starboard/image_raw'
            self.cam_right_info_topic = f'/{self.robot_name}/payload/cam_starboard/camera_info'

        # ===== Frame names =====
        # For the most part everything is transformed to the map frame
        self.map_frame = self.get_parameter("map_frame").value
        self.gt_frame_id = 'gt/' + self.robot_name + '/base_link'

        # ===== File paths for logging =====
        self.output_path = self.get_parameter("output_path").value
        self.get_logger().info(f"Reading: {self.output_path}")

        if self.output_path is None or not os.path.isdir(self.output_path):
            print("Invalid file path provided")

        if self.output_path[-1] != '/':
            self.output_path = self.output_path + '/'

        # Create folders for sensor data
        data_folders = ['left', 'right', 'down', 'sss']
        for data_folder in data_folders:
            overwrite_directory(self.output_path + data_folder)

        self.gt_poses_graph_file_path = self.output_path + 'gt_poses_graph.csv'
        self.dr_poses_graph_file_path = self.output_path + 'dr_poses_graph.csv'
        self.buoys_file_path = self.output_path + 'buoys.csv'

        # === Sonar ===
        # Currently detections are provided by the published buoy location
        self.detections_graph_file_path = self.output_path + 'detections_graph.csv'
        self.rope_detections_graph_file_path = self.output_path + 'rope_detections_graph.csv'
        self.associations_graph_file_path = self.output_path + 'associations_graph.csv'
        # self.sss_graph_file_path = self.output_path + 'detections_graph.csv'

        # === Camera ===
        # Down
        self.down_info_file_path = self.output_path + 'down_info.csv'
        self.down_gt_file_path = self.output_path + 'down_gt.csv'
        # Left
        self.left_info_file_path = self.output_path + 'left_info.csv'
        self.left_times_file_path = self.output_path + 'left_times.csv'
        self.left_gt_file_path = self.output_path + 'left_gt.csv'
        # Right
        self.right_info_file_path = self.output_path + 'right_info.csv'
        self.right_gt_file_path = self.output_path + 'right_gt.csv'

        # ===== SLAM and mapping material =====
        # Map elements - buoys and ropes
        self.buoys = []
        self.ropes = []
        self.ropes_by_buoy_ind = None
        self.define_ropes_by_inds = self.get_parameter("define_ropes_by_inds").value
        # Old settings -> use this in the launch if using these scenarios
        # old stonefish simulator: [[0, 4], [4, 2], [1, 5], [5, 3]]
        # Real Algae farm: [[0, 5], [1, 4], [2, 3]]
        # Pipeline simulator: [[0,1], [2,3], [0,2], [1,3]]
        # self.pipeline_lines = ast.literal_eval(self.get_parameter("pipeline_lines").value)

        if self.define_ropes_by_inds:
            # read the ropes_by_buoy_inds parameter
            # If it is initialized to an empty list pass None to the graph initializatio

            ropes_by_buoy_inds_list = ast.literal_eval(self.get_parameter("ropes_by_buoy_inds").value)
            if ropes_by_buoy_inds_list is None:
                self.ropes_by_buoy_ind = None
            elif len(ropes_by_buoy_inds_list) == 0:
                self.ropes_by_buoy_ind = None
            else:
                self.ropes_by_buoy_ind = ropes_by_buoy_inds_list
        else:
            self.ropes_by_buoy_inds_list = None

        # Online SLAM w/ iSAM2
        self.get_logger().info(f"Initializing online graph: {self.ropes_by_buoy_ind}")
        self.online_graph = online_slam_2d(path_name=self.output_path,  # file path of output
                                           ropes_by_buoy_ind=self.ropes_by_buoy_ind,
                                           node=self)

        # ===== Logging =====
        # Raw logging, occurs at the rate the data is received
        self.gt_poses = []
        self.dr_poses = []
        self.detections = []

        # TODO figure out how to handle this scenario
        # Originally roll, pitch, and depth were informed by the corresponding topics
        # These topics are not available for the simulated pipeline data
        self.use_raw_rpd = False
        self.rolls = []
        self.pitches = []
        self.depths = []

        # === Sonar logging ===
        self.sss_buffer_len = 10
        self.sss_data_len = 1000  # This is determined by the message
        self.sss_buffer = np.zeros((self.sss_buffer_len, 2 * self.sss_data_len), dtype=np.ubyte)

        # === Camera stuff ===
        # down
        self.down_gt = []
        self.down_info = []
        self.down_times = []
        # left
        self.left_gt = []
        self.left_info = []
        self.left_times = []
        # right
        self.right_gt = []
        self.right_info = []
        self.right_times = []

        # ===== Graph logging =====
        # TODO check comment for accuracy
        """
        dr_callback will update at a set rate will also record ground truth pose
        det_callback will update all three
        # Current format: [index w.r.t. dr_poses_graph[], x, y, z]
        """
        self.gt_poses_graph = []
        self.dr_poses_graph = []
        self.detections_graph = []
        self.rope_detections_graph = []
        self.associations_graph = []

        # ===== States and timings =====
        # TODO: make real gt work
        # if simulated_data:
        #     self.gt_updated = False  # for simulated data updates are skipped until gt is received
        # else:
        #     self.gt_updated = True

        self.gt_updated = False
        self.dr_updated = False
        self.roll_updated = False
        self.pitch_updated = False
        self.depth_updated = False
        self.buoy_updated = False
        self.rope_updated = False
        self.data_written = False
        self.image_received = False

        # self.gt_last_time = self.get_clock().now()
        # self.gt_timeout = 10.0  # Time out used to save data at end of simulation

        self.initialization_time = self.get_clock().now()

        self.dr_last_time = self.initialization_time
        # Time for limiting the rate that odometry factors are added to graph
        self.dr_update_time = self.get_parameter("dr_update_time").value
        self.dr_timeout = 10.0  # Time out used to save data at end of simulation

        self.detect_last_time = self.initialization_time
        # Time for limiting the rate that detection factors are added to graph
        self.detect_update_time = self.get_parameter("detect_update_time").value

        self.camera_last_time = self.initialization_time
        # Time for limiting the rate that pose with camera data are added to graph
        self.camera_update_time = self.get_parameter("camera_update_time").value
        # TODO change to some other form of ID not based on seq
        self.camera_last_seq = -1

        # Time for limiting the rate that pose with camera data are added to graph
        self.sss_update_time = self.get_parameter("sss_update_time").value
        self.sss_last_time = self.initialization_time - rcl_Duration(seconds=self.sss_update_time)
        self.current_sss_id = -1  # ROS1 used the sss header seq but now we just have a local counter

        # ===== Subscribers =====
        # Ground truth
        self.gt_subscriber = self.create_subscription(msg_type=Odometry, topic=self.gt_odom_topic,
                                                      callback=self.gt_callback, qos_profile=10)

        # Dead reckoning
        self.dr_subscriber = self.create_subscription(msg_type=Odometry, topic=self.dr_odom_topic,
                                                      callback=self.dr_callback, qos_profile=10)

        # Additional odometry topics: roll, pitch, depth
        self.roll_subscriber = self.create_subscription(msg_type=Float64, topic=self.roll_topic,
                                                        callback=self.roll_callback, qos_profile=10)

        self.pitch_subscriber = self.create_subscription(msg_type=Float64, topic=self.pitch_topic,
                                                         callback=self.pitch_callback, qos_profile=10)

        self.depth_subscriber = self.create_subscription(msg_type=Float64, topic=self.depth_topic,
                                                         callback=self.depth_callback, qos_profile=10)

        # Buoys
        self.buoy_subscriber = self.create_subscription(msg_type=MarkerArray, topic=self.buoy_topic,
                                                        callback=self.buoy_callback, qos_profile=10)

        # Ropes
        self.rope_subscriber = self.create_subscription(msg_type=MarkerArray, topic=self.rope_topic,
                                                        callback=self.rope_callback, qos_profile=10)

        # Detections
        self.det_subscriber = self.create_subscription(msg_type=Detection2DArray, topic=self.det_topic,
                                                       callback=self.det_callback, qos_profile=10)

        # Sonar
        self.sss_subscriber = self.create_subscription(msg_type=Sidescan, topic=self.sss_topic,
                                                       callback=self.sss_callback, qos_profile=10)

        # Cameras
        # Down camera
        self.cam_down_image_subscriber = self.create_subscription(msg_type=Image, topic=self.cam_down_image_topic,
                                                                  callback=partial(self.image_callback,
                                                                                   camera_id='down'),
                                                                  qos_profile=10)

        self.cam_down_info_subscriber = self.create_subscription(msg_type=CameraInfo, topic=self.cam_down_info_topic,
                                                                 callback=partial(self.info_callback,
                                                                                  camera_id='down'),
                                                                 qos_profile=10)

        # Left camera
        self.cam_left_image_subscriber = self.create_subscription(msg_type=Image, topic=self.cam_left_image_topic,
                                                                  callback=partial(self.image_callback,
                                                                                   camera_id='left'),
                                                                  qos_profile=10)

        self.cam_left_info_subscriber = self.create_subscription(msg_type=CameraInfo, topic=self.cam_left_info_topic,
                                                                 callback=partial(self.info_callback,
                                                                                  camera_id='left'),
                                                                 qos_profile=10)

        # Right camera
        self.cam_right_image_subscriber = self.create_subscription(msg_type=Image, topic=self.cam_right_image_topic,
                                                                   callback=partial(self.image_callback,
                                                                                    camera_id='right'),
                                                                   qos_profile=10)

        self.cam_right_info_subscriber = self.create_subscription(msg_type=CameraInfo, topic=self.cam_right_info_topic,
                                                                  callback=partial(self.info_callback,
                                                                                   camera_id='right'),
                                                                  qos_profile=10)

        self.time_check = self.create_timer(timer_period_sec=2.0, callback=self.time_check_callback)

        # ===== Verboseness parameters =====
        self.verbose_DRs = self.get_parameter('verbose_listener_DRs').value
        self.verbose_detections = self.get_parameter('verbose_listener_detections').value
        self.verbose_sonars = self.get_parameter('verbose_listener_sonars').value
        self.verbose_buoys = self.get_parameter('verbose_listener_buoys').value
        self.verbose_cameras = self.get_parameter('verbose_listener_cameras').value

        # ===== Analysis =====
        # The analysis class is separate from the online slam class within sam_slam_graphs.py
        self.analysis = None

        # Line colors
        line_colors = ast.literal_eval(self.get_parameter("line_colors").value)
        if len(line_colors) is not None and len(line_colors) > 0:
            self.line_colors = line_colors
        else:
            self.line_colors = None

        self.get_logger().info("Sam Slam Listener Initialized")

    def declare_node_parameters(self):
        """
        Declare the required parameters for the sam_slam_ros_classes node
        :return:
        """

        # ===== Scenario =====
        # TODO Figure out how to handle all the different scenarios
        # There is room for some improvement in this section
        # Maybe scenario should be some string idea and the other values would be set accordingly
        self.declare_parameter("simulated_data", False)  # real or simulated data
        self.declare_parameter("simulated_detections", False)  # Maybe not used???
        self.declare_parameter("algae_scenario", False)
        self.declare_parameter("pipeline_scenario", False)  # indicates
        self.declare_parameter("record_ground_truth", True)  # indicates if GT is available
        # TODO consider what the default state of things should be
        # TODO re-consider how to set up defaults, this was a little painful
        # default_pipeline_lines = "[[0, 1], [1, 2], [2, 3]]"  # pipeline scenario
        # self.declare_parameter("pipeline_lines",
        #                        default_pipeline_lines)  # Defines line features w.r.t. point landmark indices
        # default_pipeline_colors = "[[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]"
        # self.declare_parameter("pipeline_colors", default_pipeline_colors)
        # Using Algae defaults
        #
        # algae_colors = "[[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]"
        self.declare_parameter("define_ropes_by_inds", False)  # refers to the two methods of adding ropes
        self.declare_parameter("ropes_by_buoy_inds", "[[0,1], [2,3], [0,2], [1,3]]")  # Algae
        self.declare_parameter("line_colors",
                               "[[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]")  # Algae

        # ===== Graph input parameters =====
        self.declare_parameter("prioritize_buoy_detections", True)
        self.declare_parameter("manual_associations", False)
        self.declare_parameter("rope_associations", True)

        # ===== Topic names =====
        # for now these are mostly defined w.r.t the robot name and the scenario
        self.declare_parameter("robot_name", "sam0")

        # ===== Frame names =====
        # Todo this needs a better name
        self.declare_parameter("map_frame", "odom")

        # ===== Output path and parameters =====
        # TODO remove hard coded path
        self.declare_parameter("output_path", "/home/julian/testing_files")

        # ===== Timing =====
        # Raw sensor and detection inputs are metered
        self.declare_parameter("dr_update_time", 2.0)
        self.declare_parameter("detect_update_time", 0.5)
        self.declare_parameter("camera_update_time", 0.5)
        self.declare_parameter("sss_update_time", 5.0)

        # ===== Verbose output =====
        self.declare_parameter("verbose_listener_DRs", False)
        self.declare_parameter("verbose_listener_detections", True)
        self.declare_parameter("verbose_listener_sonars", False)
        self.declare_parameter("verbose_listener_buoys", False)
        self.declare_parameter("verbose_listener_cameras", False)

    # Subscriber callbacks
    def gt_callback(self, msg):
        """
        Call back for the ground truth subscription, msg is of type nav_msgs/Odometry.
        The data is saved in a list w/ format [x, y, z, q_w, q_x, q_y, q_z, time].
        Note the position of q_w, this is for compatibility with gtsam and matlab
        """
        if not self.gt_updated:
            print('Start recording ground truth')
        transformed_pose = self.transform_pose(msg.pose, from_frame=msg.header.frame_id, to_frame=self.map_frame)

        gt_position = transformed_pose.pose.position
        gt_quaternion = transformed_pose.pose.orientation
        gt_time = transformed_pose.header.stamp.sec

        self.gt_poses.append([gt_position.x, gt_position.y, gt_position.z,
                              gt_quaternion.w, gt_quaternion.x, gt_quaternion.y, gt_quaternion.z,
                              gt_time])

        self.gt_updated = True

    def dr_callback(self, msg):
        """
        Call back for the dead reckoning subscription, msg is of type nav_msgs/Odometry.
        WAS: The data is saved in a list w/ format [x, y, z, q_w, q_x, q_y, q_z].
        NOW: The data is saved in a list w/ format [x, y, z, q_w, q_x, q_y, q_z, roll, pitch, depth].
        Note the position of q_w, this is for compatibility with gtsam and matlab
        """
        # wait for raw roll, pitch, depth
        if self.use_raw_rpd:
            if False in [self.roll_updated, self.pitch_updated, self.depth_updated]:
                return

        # wait for gt (if this real)
        if not self.gt_updated:
            return

        # transform odom to the map frame
        transformed_dr_pose = self.transform_pose(msg.pose,
                                                  from_frame=msg.header.frame_id,
                                                  to_frame=self.map_frame)

        dr_position = transformed_dr_pose.pose.position
        dr_quaternion = transformed_dr_pose.pose.orientation

        if self.use_raw_rpd:
            curr_roll = self.rolls[-1]
            curr_pitch = self.pitches[-1]
            curr_depth = self.depths[-1]
        else:
            (curr_roll, curr_pitch, _) = euler_from_quaternion([dr_quaternion.x, dr_quaternion.y,
                                                                dr_quaternion.z, dr_quaternion.w])
            curr_depth = dr_position.z

        # Record dr poses in format compatible with GTSAM
        if self.correct_dr:
            # Correction of position
            corrected_x = -dr_position.x
            # Correction of orientation
            # Incorrect method
            # uncorrected_q = Quaternion(dr_quaternion.x, dr_quaternion.y, dr_quaternion.z, dr_quaternion.w)
            # uncorrected_rpy = euler_from_quaternion(dr_quaternion)
            # uncorrected_rpy = euler_from_quaternion([dr_quaternion.x, dr_quaternion.y, dr_quaternion.z, dr_quaternion.w])
            # corrected_y = np.pi - uncorrected_rpy[2]
            # corrected_q = quaternion_from_euler(uncorrected_rpy[0], uncorrected_rpy[1], corrected_y)
            #
            # Correct? method
            r_q = [0, -1, 0, 0]  # The correct orientation correction factor

            dr_q = [dr_quaternion.x, dr_quaternion.y, dr_quaternion.z, dr_quaternion.w]
            corrected_q = quaternion_multiply(r_q, dr_q)

            self.dr_poses.append([corrected_x, dr_position.y, dr_position.z,
                                  corrected_q[3], corrected_q[0], corrected_q[1], corrected_q[2],
                                  curr_roll, curr_pitch, curr_depth])

        else:
            self.dr_poses.append([dr_position.x, dr_position.y, dr_position.z,
                                  dr_quaternion.w, dr_quaternion.x, dr_quaternion.y, dr_quaternion.z,
                                  curr_roll, curr_pitch, curr_depth])

        # Conditions for updating dr: (1) first time or (2) stale data or (3) online graph is still uninitialized
        time_now = self.get_clock().now()
        first_time_cond = not self.dr_updated and self.gt_updated
        stale_data_cond = (self.dr_updated and
                           rcl_times_delta_secs(time_now, self.dr_last_time) > self.dr_update_time)

        if self.online_graph is not None:
            online_waiting_cond = self.gt_updated and self.online_graph.initial_pose_set is False
        else:
            online_waiting_cond = False

        if first_time_cond or stale_data_cond or online_waiting_cond:
            # Add to the dr and gt lists
            dr_pose = self.dr_poses[-1]
            self.dr_poses_graph.append(dr_pose)
            # TODO: make gt work
            if self.record_gt:
                gt_pose = self.gt_poses[-1]
                self.gt_poses_graph.append(gt_pose)
            else:
                gt_pose = self.dr_poses[-1]  # NOTE: dr is used in place of gt for real data

            # if self.simulated_data:
            #     gt_pose = self.gt_poses[-1]
            #     self.gt_poses_graph.append(gt_pose)
            # else:
            #     gt_pose = self.dr_poses[-1]  # NOTE: dr is used in place of gt for real data

            # Update time and state
            self.dr_last_time = time_now
            self.dr_updated = True

            # ===== Online first update =====
            if self.online_graph is None:
                return
            if self.online_graph.initial_pose_set is False and not self.online_graph.busy:
                print("DR - First update - x0")
                self.online_graph.add_first_pose(dr_pose, gt_pose)

            elif not self.online_graph.busy:
                if self.verbose_DRs:
                    print(f"DR - Odometry update - x{self.online_graph.current_x_ind + 1}")
                self.online_graph.online_update_queued(dr_pose, gt_pose)

            else:
                print('Busy condition found - DR')

    def roll_callback(self, msg):
        """
        update the list of rolls. These values are used for saving the complete dr pose.
        The complete pose info is needed as the estimate is only 2D for now!
        if self.use_raw_rpd is set to false this is done within the dead reckoning callback
        """
        if self.use_raw_rpd:
            self.rolls.append(msg.data)
            self.roll_updated = True

    def pitch_callback(self, msg):
        """
        update the list of pitches. These values are used for saving the complete dr pose.
        The complete pose info is needed as the estimate is only 2D for now!
        if self.use_raw_rpd is set to false this is done within the dead reckoning callback
        """
        if self.use_raw_rpd:
            self.pitches.append(msg.data)
            self.pitch_updated = True

    def depth_callback(self, msg):
        """
        update the list of depths. These values are used for saving the complete dr pose.
        The complete pose info is needed as the estimate is only 2D for now!
        if self.use_raw_rpd is set to false this is done within the dead reckoning callback
        """
        if self.use_raw_rpd:
            self.depths.append(msg.data)
            self.depth_updated = True

    def det_callback(self, msg):
        # Too verbose
        # if self.verbose_detections:
        #     self.get_logger().info("Detection Callback")
        # wait for raw roll, pitch, depth
        if self.use_raw_rpd:
            if False in [self.roll_updated, self.pitch_updated, self.depth_updated]:
                return

        # Check that dr and gt topics have received messages
        if False in [self.dr_updated, self.gt_updated]:
            return

        # Check that the map is available
        if False in [self.buoy_updated, self.rope_updated]:
            return

        # check elapsed time
        detect_time_now = self.get_clock().now()
        if rcl_times_delta_secs(detect_time_now, self.detect_last_time) < self.detect_update_time:
            detection_is_current = True
        else:
            detection_is_current = False

        # Check if detection is of a buoy
        detection_is_buoy = False
        for detection in msg.detections:
            # detection: vision_msgs/Detection2d
            for result in detection.results:
                # result type: vision_msgs/ObjectHypothesisWithPose
                # result fields:
                #   - vision_msgs/ObjectHypothesis hypothesis
                #   - geometry_msgs/PoseWithCovariance pose
                if result.hypothesis.class_id == ObjectID.BUOY.name:  # nadir:0 rope:1 buoy:2
                    detection_is_buoy = True
                    break
            if detection_is_buoy:
                break

        # Buoy detections can be fairly rare, so it might be desirable to use all buoy detections while
        # limiting the rate that rope detections are added to the graph
        if not self.prioritize_buoy_detections and detection_is_current:
            return

        if self.prioritize_buoy_detections and detection_is_current and not detection_is_buoy:
            return

        # Reset timer
        self.detect_last_time = self.get_clock().now()

        # Process detection
        for det_ind, detection in enumerate(msg.detections):
            for res_ind, result in enumerate(detection.results):
                # result type: vision_msgs/ObjectHypothesisWithPose
                # result fields:
                #   - vision_msgs/ObjectHypothesis hypothesis
                #   - geometry_msgs/PoseWithCovariance pose

                # detection type is specified by ObjectID
                detection_type = result.hypothesis.class_id

                # Seq was removed from header in ROS2
                # TODO find replacement for seq, maybe use timestamps
                # problem with time stamps as its not east to put them in np arrays -> could use structured array
                # detection score is currently used to communicate the seq_id corresponding to the current detection
                detection_seq_id = result.hypothesis.score

                # Pose in base_link, convert to map
                det_pose_base = result.pose.pose  # geometry_msgs/Pose
                # det_pose_map: geometry_msgs/PoseStamped
                det_pos_map = self.transform_pose(det_pose_base,
                                                  from_frame=msg.header.frame_id,
                                                  to_frame=self.map_frame)

                index = len(self.dr_poses_graph)

                # ===== Log data for the graph =====
                # First update dr and gr with the most current
                dr_pose = self.dr_poses[-1]
                self.dr_poses_graph.append(dr_pose)
                # TODO: make gt work
                if self.record_gt:
                    gt_pose = self.gt_poses[-1]
                    self.gt_poses_graph.append(gt_pose)
                else:
                    gt_pose = self.dr_poses[-1]  # NOTE: dr is used in place of gt for real data

                # if self.simulated_data:
                #     gt_pose = self.gt_poses[-1]
                #     self.gt_poses_graph.append(gt_pose)
                # else:
                #     gt_pose = self.dr_poses[-1]  # NOTE: dr is used in place of gt for real data

                # ===== Output =====
                if self.verbose_detections:
                    # detection type is now passed as a string
                    # detection_type_name = get_enum_name_or_value(ObjectID, detection_type)
                    print(f'Detection callback - Index:{index}  Type:{detection_type}')
                    self.get_logger().info(f'Detection callback - Index:{index}  Type:{detection_type}')
                # ===== Handle buoy detections
                if detection_type == ObjectID.BUOY.name:
                    # Log detection position
                    # Append [x_map,y_map,z_map, x_rel, y_rel, z_vel, id,score, index of dr_pose_graph]
                    self.detections_graph.append([det_pos_map.pose.position.x,
                                                  det_pos_map.pose.position.y,
                                                  det_pos_map.pose.position.z,
                                                  det_pose_base.position.x,
                                                  det_pose_base.position.y,
                                                  det_pose_base.position.z,
                                                  index])

                    # Data association
                    if self.manual_associations:
                        det_da = int(result.score)
                    else:
                        det_da = - ObjectID.BUOY.value  # -2

                    self.associations_graph.append([det_da])

                    # ===== Online detection update =====
                    if self.online_graph is None:
                        return
                    if not self.online_graph.busy:
                        if self.verbose_detections:
                            print(f"Detection update - Buoy - x{self.online_graph.current_x_ind + 1}")
                        self.online_graph.online_update_queued(dr_pose=dr_pose, gt_pose=gt_pose,
                                                               relative_detection=np.array(
                                                                   (det_pose_base.position.x,
                                                                    det_pose_base.position.y),
                                                                   dtype=np.float64),
                                                               da_id=det_da,
                                                               seq_id=detection_seq_id)
                    else:
                        print('Busy condition found - Detection - Buoy')

                # ===== Handle rope detections =====
                if detection_type == ObjectID.ROPE.name:
                    # TODO this statement prevents rope detections from being added to graph
                    # Log detection position
                    # Append [x_map,y_map,z_map, x_rel, y_rel, z_vel, id,score, index of dr_pose_graph]
                    self.rope_detections_graph.append([det_pos_map.pose.position.x,
                                                       det_pos_map.pose.position.y,
                                                       det_pos_map.pose.position.z,
                                                       det_pose_base.position.x,
                                                       det_pose_base.position.y,
                                                       det_pose_base.position.z,
                                                       index])

                    det_da = - ObjectID.ROPE.value  # -1

                    # ===== Online detection update =====
                    if self.online_graph is None:
                        return
                    if not self.online_graph.busy:
                        if self.verbose_detections:
                            print(f"Detection update - Rope - x{self.online_graph.current_x_ind + 1}")
                        self.online_graph.online_update_queued(dr_pose=dr_pose, gt_pose=gt_pose,
                                                               relative_detection=np.array(
                                                                   (det_pose_base.position.x,
                                                                    det_pose_base.position.y),
                                                                   dtype=np.float64),
                                                               da_id=det_da,
                                                               seq_id=detection_seq_id)
                    else:
                        print('Busy condition found - Detection - Rope')

                # ===== Handle pipe detections =====
                """
                Pipe updates should be handled in the same manner as ropes
                """
                if detection_type == ObjectID.PIPE.name:
                    # Log detection position
                    # Append [x_map,y_map,z_map, x_rel, y_rel, z_vel, id,score, index of dr_pose_graph]
                    self.rope_detections_graph.append([det_pos_map.pose.position.x,
                                                       det_pos_map.pose.position.y,
                                                       det_pos_map.pose.position.z,
                                                       det_pose_base.position.x,
                                                       det_pose_base.position.y,
                                                       det_pose_base.position.z,
                                                       index])

                    det_da = - ObjectID.PIPE.value  # -3

                    # ===== Online detection update =====
                    if self.online_graph is None:
                        return
                    if not self.online_graph.busy:
                        if self.verbose_detections:
                            print(f"Detection update - Pipe - x{self.online_graph.current_x_ind + 1}")
                        self.online_graph.online_update_queued(dr_pose=dr_pose, gt_pose=gt_pose,
                                                               relative_detection=np.array(
                                                                   (det_pose_base.position.x,
                                                                    det_pose_base.position.y),
                                                                   dtype=np.float64),
                                                               da_id=det_da,
                                                               seq_id=detection_seq_id)
                    else:
                        print('Busy condition found - Detection - Rope')

    def sss_callback(self, msg):
        """
        The sss callback is responsible for filling the sss_buffer.
        """
        # Update nodes sss id counter for every sss message, regardless of the nodes current state
        self.current_sss_id += 1

        # Only begin to process sss once Dead reckoning is available
        # There is also the option to use the independently published roll, pitch and depths
        # wait for raw roll, pitch, depth
        if self.use_raw_rpd:
            if False in [self.roll_updated, self.pitch_updated, self.depth_updated]:
                return

        # Check that dr and gt topics have received messages
        if False in [self.dr_updated, self.gt_updated]:
            return

        # Updated sss id counter used to identify data, mostly for post-processing
        sss_id = self.current_sss_id

        if self.verbose_sonars:
            print(f"sss frame: {sss_id}")

        # Record start time
        sss_time_now = self.get_clock().now()

        # Fill buffer regardless of other conditions
        port = np.array(bytearray(msg.port_channel), dtype=np.ubyte)
        stbd = np.array(bytearray(msg.starboard_channel), dtype=np.ubyte)
        meas = np.concatenate([np.flip(port), stbd])
        self.sss_buffer[1:, :] = self.sss_buffer[:-1, :]
        self.sss_buffer[0, :] = meas

        # Copy buffer
        sss_current = np.copy(self.sss_buffer)

        # check elapsed time
        if rcl_times_delta_secs(sss_time_now, self.sss_last_time) < self.sss_update_time:
            return

        dr_pose = self.dr_poses[-1]
        self.dr_poses_graph.append(dr_pose)
        # TODO: make gt work
        if self.record_gt:
            gt_pose = self.gt_poses[-1]
            self.gt_poses_graph.append(gt_pose)
        else:
            gt_pose = self.dr_poses[-1]  # NOTE: dr is used in place of gt for real data

        # if self.simulated_data:
        #     gt_pose = self.gt_poses[-1]
        #     self.gt_poses_graph.append(gt_pose)
        # else:
        #     gt_pose = self.dr_poses[-1]  # NOTE: dr is used in place of gt for real data

        if self.online_graph is not None:
            if not self.online_graph.busy:
                if self.online_graph.initial_pose_set is False:
                    print("SSS - First update w/ sss data")
                else:
                    if self.verbose_sonars:
                        print(f"SSS - Odometry and sss update - x{self.online_graph.current_x_ind + 1}")

                self.online_graph.online_update_queued(dr_pose, gt_pose,
                                                       relative_detection=None,
                                                       id_string=f'sss_{sss_id}')

                # Reset timer after successful completion
                self.sss_last_time = self.get_clock().now()

                # Write to 'disk'
                if self.output_path is None or not os.path.isdir(self.output_path):
                    print("Provide valid file path sss output")
                else:
                    save_path = self.output_path + f'/sss/{sss_id}.jpg'
                    cv2.imwrite(save_path, sss_current)

            else:
                print('Busy condition found - sss')

    def info_callback(self, msg, camera_id):
        if camera_id == 'down':
            if len(self.down_info) == 0:
                self.down_info.append(msg.K)
                self.down_info.append(msg.P)
                self.down_info.append([msg.width, msg.height])
        elif camera_id == 'left':
            if len(self.left_info) == 0:
                self.left_info.append(msg.K)
                self.left_info.append(msg.P)
                self.left_info.append([msg.width, msg.height])
        elif camera_id == 'right':
            if len(self.right_info) == 0:
                self.right_info.append(msg.K)
                self.right_info.append(msg.P)
                self.right_info.append([msg.width, msg.height])
        else:
            print('Unknown camera_id passed to info callback')

    def image_callback(self, msg, camera_id):
        """
        Callback for camera images, the same callback is used for all the cameras. This was designed around the
        simulator in which the three images were mostly synchronized. The hope was to record all the desired images and
        only add one node in the graph for each set. Not sure if this holds for the actual AUV.
        """
        # TODO use some sort of synchronizer
        # wait for raw roll, pitch, depth
        if self.use_raw_rpd:
            if False in [self.roll_updated, self.pitch_updated, self.depth_updated]:
                return

        # Check that dr and gt topics have received messages
        if False in [self.dr_updated, self.gt_updated]:
            return

        # Identifies frames
        # We want to save down, left , and right images of the same frame
        current_id = msg.header.seq

        # check elapsed time
        camera_time_now = self.get_clock().now()
        camera_stale = rcl_times_delta_secs(camera_time_now, self.camera_last_time) > self.camera_update_time

        if camera_stale or not self.image_received:
            self.camera_last_seq = current_id
            self.camera_last_time = self.get_clock().now()
            new_frame = True  # Used to only add one node to graph for each camera frame: down, left, and right
            self.image_received = True
            if self.verbose_cameras:
                print(f'New camera frame: {camera_id} - {current_id}')
        elif current_id != self.camera_last_seq:
            return
        else:
            new_frame = False  # Do not add a node to the graph
            if self.verbose_cameras:
                print(f'Current camera frame: {camera_id} - {current_id}')

        now_clock = self.get_clock().now()
        msg_stamp = msg.header.stamp

        dr_pose = self.dr_poses[-1]
        self.dr_poses_graph.append(dr_pose)
        # TODO: make gt work
        if self.record_gt:
            gt_pose = self.gt_poses[-1]
            self.gt_poses_graph.append(gt_pose)
        else:
            gt_pose = self.dr_poses[-1]  # NOTE: dr is used in place of gt for real data

        # if self.simulated_data:
        #     gt_pose = self.gt_poses[-1]
        #     self.gt_poses_graph.append(gt_pose)
        # else:
        #     gt_pose = self.dr_poses[-1]  # NOTE: dr is used in place of gt for real data

        # A node is only add to the graph when a new 'frame' is detected
        # Here frames are images from the different cameras with the same seq_id
        if new_frame:
            if self.verbose_cameras:
                print(f"Adding img-{current_id} to graph")

            if self.online_graph is not None:
                if not self.online_graph.busy:
                    if self.verbose_cameras and self.online_graph.initial_pose_set:
                        print(f"CAM - Odometry and camera update - {self.online_graph.current_x_ind + 1}")
                    else:
                        print("CAM - First update w/ camera data")
                    self.online_graph.online_update_queued(dr_pose, gt_pose,
                                                           relative_detection=None,
                                                           id_string=f'cam_{current_id}')
                else:
                    print('Busy condition found - camera')
        # === Debugging ===
        """
        This is only need to verify that the data is being recorded properly
        gt_pose is of the format [x, y, z, q_w, q_x, q_y, q_z, time]
        """
        pose_current = gt_pose[0:-1]
        pose_time = gt_pose[-1]
        pose_current.append(current_id)

        # Record debugging data
        if camera_id == 'down':
            # Record the ground truth and times
            self.down_gt.append(pose_current)
            self.down_times.append([rcl_time_to_secs(now_clock),
                                    msg_stamp.to_sec(),
                                    pose_time])
        elif camera_id == 'left':
            # Record the ground truth and times
            self.left_gt.append(pose_current)
            self.left_times.append([rcl_time_to_secs(now_clock),
                                    msg_stamp.to_sec(),
                                    pose_time])
        elif camera_id == 'right':
            # Record the ground truth and times
            self.right_gt.append(pose_current)
            self.right_times.append([rcl_time_to_secs(now_clock),
                                     msg_stamp.to_sec(),
                                     pose_time])
        else:
            print('Unknown camera_id passed to image callback')
            return

        # Display call back info
        if self.verbose_cameras:
            print(f'image callback - {camera_id}: {current_id}')

        # Convert to cv2 format
        cv2_img = imgmsg_to_cv2(msg)

        # Write to 'disk'
        if self.output_path is None or not os.path.isdir(self.output_path):
            print("Provide valid file path image output")
        else:
            save_path = self.output_path + f'/{camera_id}/{current_id}.jpg'
            cv2.imwrite(save_path, cv2_img)

        return

    def rope_callback(self, msg):
        """
        This was supposed to be a way of getting the rope info into graph
        I don't like the direction its taking !! :(
        :param msg:
        :return:
        """
        if not self.rope_updated:
            self.get_logger().info('Capturing Rope map positions')
            marker_count = len(msg.markers)
            if marker_count % 2 != 0:
                self.get_logger().info(f'Rope_callback: {marker_count} markers received')
                self.get_logger().info('Rope_callback: Received rope info was malformed, expected even number of points')
                return

            self.ropes = [None for i in range(marker_count // 2)]
            current_rope = 0
            for rope_ind, marker_ind in enumerate(range(0, marker_count, 2)):
                marker_start = msg.markers[marker_ind]
                marker_end = msg.markers[marker_ind + 1]
                marker_frame_id = marker_start.header.frame_id
                marker_id = rope_ind

                if self.map_frame in marker_frame_id:
                    self.ropes[marker_id] = [[marker_start.pose.position.x, marker_start.pose.position.y],
                                             [marker_end.pose.position.x, marker_end.pose.position.x]]

                else:
                    # Convert to frame of interest, most work done in map
                    transformed_start = self.transform_pose(marker_start.pose,
                                                            from_frame=marker_frame_id,
                                                            to_frame=self.map_frame)

                    transformed_end = self.transform_pose(marker_end.pose,
                                                          from_frame=marker_frame_id,
                                                          to_frame=self.map_frame)

                    self.ropes[marker_id] = [[transformed_start.pose.position.x, transformed_start.pose.position.y],
                                             [transformed_end.pose.position.x, transformed_end.pose.position.y]]

            if self.online_graph is not None and self.rope_associations:
                self.get_logger().info("Online: rope update")
                if self.online_graph.rope_setup(self.ropes) == 1:
                    self.rope_updated = True

    def buoy_callback(self, msg):
        # NOTE: The buoy publisher did not give each buoy a unique id.
        # Currently, these IDs are faked for simulated data. It might be necessary to change the publisher
        # to provide real IDs if the structure of the map, with ropes, is important.
        if not self.buoy_updated:
            marker_count = len(msg.markers)

            self.get_logger().info(f'Capturing buoy map positions: {marker_count}')
            self.buoys = [None for i in range(marker_count)]

            marker_id_current = 0

            for marker in msg.markers:
                # See note above about buoy IDs
                if self.pipeline_scenario or self.algae_scenario:
                    marker_id = int(marker.id)

                elif self.simulated_data:
                    marker_id = int(marker_id_current)
                    marker_id_current += 1
                else:
                    marker_id = int(marker.id)

                if self.map_frame in marker.header.frame_id:
                    self.buoys[marker_id] = [marker.pose.position.x, marker.pose.position.y, marker.pose.position.z]

                else:
                    # Convert to frame of interest, most work done in map
                    marker_pos_map = self.transform_pose(marker.pose,
                                                         from_frame=marker.header.frame_id,
                                                         to_frame=self.map_frame)
                    self.buoys[marker_id] = [marker_pos_map.pose.position.x, marker_pos_map.pose.position.y,
                                             marker_pos_map.pose.position.z]

            if self.online_graph is not None:
                self.get_logger().info("Online: buoy update")
                return_message = self.online_graph.buoy_setup(self.buoys)
                self.get_logger().info(f"Online: buoy update -> {return_message}")

            self.buoy_updated = True

    # Timer callback
    def time_check_callback(self):
        """
        Data saving and final analysis are triggered by this timer when the dr_timeout is exceeded
        :return:
        """
        if not self.dr_updated:
            return
        delta_t_secs = rcl_times_delta_secs(self.get_clock().now(), self.dr_last_time)
        if delta_t_secs >= self.dr_timeout and not self.data_written:
            print('Data written')
            self.write_data()
            self.data_written = True

            if self.online_graph is not None:
                # TODO Save final results
                print("Initializing analysis")
                self.analysis = analyze_slam(self.online_graph, self.output_path)
                self.analysis.save_for_sensor_processing()
                self.analysis.save_2d_poses()
                self.analysis.save_3d_poses()
                self.analysis.save_performance_metrics()
                self.analysis.save_rope_info()
                self.analysis.calculate_corresponding_points(debug=False)
                print("Analysis initialized")

                print("Producing final output")
                self.final_output()

        return

    def final_output(self):
        """
        Produces the final output of the analysis based on the current scenario.
        This should be called after the analysis
        :return:
        """
        if self.analysis is None:
            print("No analysis present")
            return

        if self.pipeline_scenario:
            self.analysis.visualize_final(plot_gt=True,
                                          plot_dr=True,
                                          plot_buoy=False,
                                          rope_colors=self.line_colors)

            self.analysis.visualize_online(plot_final=True,
                                           plot_correspondence=True,
                                           plot_buoy=False)

            self.analysis.plot_error_positions(gt_baseline=True,
                                               plot_dr=True,
                                               plot_online=True,
                                               plot_final=True)

        else:
            self.analysis.visualize_final(plot_gt=False,
                                          plot_dr=False)

            self.analysis.visualize_online(plot_final=True, plot_correspondence=True)
            self.analysis.plot_error_positions()

            self.analysis.show_buoy_info()

    # ===== Transforms and poses =====
    def transform_pose(self, pose: Pose | PoseStamped, from_frame: str, to_frame: str) -> PoseStamped:
        """
        Returns a PoseStamped with
        :param pose: Pose or PoseStamped
        :param from_frame: The initial frame of the input pose
        :param to_frame: The name of the target frame to transform the PoseStamped into.
        :return:
        """

        trans = self.wait_for_transform(from_frame=from_frame, to_frame=to_frame)

        if isinstance(pose, Pose):
            pose_to_transform = PoseStamped()
            pose_to_transform.header.frame_id = from_frame
            pose_to_transform.pose.position = pose.position
            pose_to_transform.pose.orientation = pose.orientation
            pose_transformed = tf2_geometry_msgs.do_transform_pose_stamped(pose_to_transform, trans)
        else:
            pose_transformed = tf2_geometry_msgs.do_transform_pose_stamped(pose, trans)

        return pose_transformed

    def wait_for_transform(self, from_frame, to_frame):
        """Wait for transform from from_frame to to_frame"""
        trans = None
        while trans is None:
            try:
                trans = self.tf_buffer.lookup_transform(to_frame,
                                                        from_frame,
                                                        rcl_Time(),
                                                        rcl_Duration(seconds=0.1))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as error:
                print('Failed to transform. Error: {}'.format(error))

        return trans

    def get_gt_trans_in_map(self):
        """
        Finds pose of the ground truth.
        First,the transform between the map and the ground truth frame.
        Second, the transform is applied to a null_pose located at the origin.
        Modifying the orientation of this pose might be need to prevent later
        processing on the ground truth
        Returns [ x, y, z, q_w, q_x, q_y, q_z]
        """

        trans = self.wait_for_transform(from_frame=self.gt_frame_id,
                                        to_frame=self.map_frame)

        null_pose = PoseStamped()
        null_pose.pose.orientation.w = 1.0
        pose_in_map = tf2_geometry_msgs.do_transform_pose_stamped(null_pose, trans)

        pose_list = [pose_in_map.pose.position.x,
                     pose_in_map.pose.position.y,
                     pose_in_map.pose.position.z,
                     pose_in_map.pose.orientation.w,
                     pose_in_map.pose.orientation.x,
                     pose_in_map.pose.orientation.y,
                     pose_in_map.pose.orientation.z]

        return pose_list

    # ===== Random utility methods =====
    def write_data(self):
        """
        Save all the relevant data
        """
        write_array_to_csv(self.dr_poses_graph_file_path, self.dr_poses_graph)
        write_array_to_csv(self.gt_poses_graph_file_path, self.gt_poses_graph)
        write_array_to_csv(self.detections_graph_file_path, self.detections_graph)
        write_array_to_csv(self.associations_graph_file_path, self.associations_graph)
        write_array_to_csv(self.buoys_file_path, self.buoys)

        # === Camera ===
        # Down
        write_array_to_csv(self.down_info_file_path, self.down_info)
        write_array_to_csv(self.down_gt_file_path, self.down_gt)
        # Left
        write_array_to_csv(self.left_info_file_path, self.left_info)
        write_array_to_csv(self.left_times_file_path, self.left_times)
        write_array_to_csv(self.left_gt_file_path, self.left_gt)
        # Right
        write_array_to_csv(self.right_info_file_path, self.right_info)
        write_array_to_csv(self.right_gt_file_path, self.right_gt)

        return


def main(args=None):
    rclpy.init(args=args)
    sam_slam_listener = SamSlamListener()
    try:
        rclpy.spin(sam_slam_listener)
    except KeyboardInterrupt:
        pass
        # sam_slam_listener.get_logger().info(f"Shutting down")
        # sam_slam_listener.destroy_node()
        # rclpy.shutdown()


if __name__ == '__main__':
    # This works but currently will default to the parameter values that are declared in
    # declare_node_parameters() and declare_graph_parameters()
    # TODO read yaml file for parameters, not sure how to do this outside of ROS
    main()
