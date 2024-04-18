#!/usr/bin/env python3

import os
import cv2 as cv
import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32
from smarc_msgs.msg import Sidescan
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose
from vision_msgs.msg import ObjectHypothesisWithPose, Detection2DArray, Detection2D
from cv_bridge import CvBridge, CvBridgeError  # will use to send the output back to ROS
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray

# Topics and links
from sam_msgs.msg import Topics as SamTopics
from sam_msgs.msg import Links as SamLinks
from dead_reckoning_msgs.msg import Topics as DRTopics
from sam_graph_slam_2_msgs.msg import Topics as GraphSlamTopics

# Ros
try:
    from .helpers.general_helpers import write_array_to_csv
    from .helpers.ros_helpers import rcl_times_delta_secs, ros_time_to_secs, rcl_time_to_stamp, float_time_to_stamp
    from .detectors.sss_image_detector import process_sss, concat_arrays_with_time_threshold, overlay_detections_simple
    from .detectors.consts import ObjectID, Side
except ImportError:
    from helpers.general_helpers import write_array_to_csv
    from helpers.ros_helpers import rcl_times_delta_secs, ros_time_to_secs, rcl_time_to_stamp, float_time_to_stamp
    from detectors.sss_image_detector import process_sss, concat_arrays_with_time_threshold, overlay_detections_simple
    from detectors.consts import ObjectID, Side

"""
Detector node for algae farm scenario using side scan sonar returns

Detector: image processing based on the side scan image. See, process_real_sss.py

detection publisher: there should be an example of this in sam_slam call sss_detection_img_proc_publisher.py
"""


class SSSDetector(Node):
    def __init__(self, namespace: str = None, time_out: float = 0.0, buffer_len: int = 0):
        super().__init__('algae_sss_detector_node', namespace=namespace)
        self.get_logger().info(f"Initialized sss_saver_node - namespace: {self.get_namespace()}")
        self.declare_node_parameters()

        # === General parameters ===
        self.robot_name = self.get_parameter("robot_name").value
        self.robot_frame = f"{self.robot_name}_{SamLinks.BASE_LINK}"

        # TODO Handle save output path better: logic and destination set by parameters
        self.output_path = "/home/julian/sss_detector_data"
        if not os.path.isdir(self.output_path):
            self.output_path = ""

        # SAM sensor topics
        self.sss_topic = SamTopics.SIDESCAN_TOPIC  # f"/{self.robot_name}/payload/sidescan"
        self.depth_topic = DRTopics.DR_DEPTH_TOPIC  # f"/{self.robot_name}/dr/depth"  # press_to_depth

        # Map info topic
        self.line_depth_topic = GraphSlamTopics.MAP_LINE_DEPTH_TOPIC

        # Initial output, just for checking
        self.get_logger().info(f"SSS topic: {self.sss_topic}")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")
        self.get_logger().info(f"Line depth topic: {self.line_depth_topic}")
        self.get_logger().info(f"Output path: {self.output_path}")

        # === Initialize state info ===
        # Sam states
        self.current_depth = 0  # Initialize to zero, depth node only publishes depths in a valid band of pressures

        # Data states
        # If timeout is not specified, default value of 5.0 is used
        if time_out != 0.0:
            self.time_out = time_out
        else:
            self.time_out = 5.0

        self.data_written = False
        self.write_data = self.get_parameter("write_data").value
        self.sss_last_time = self.get_clock().now()
        self.sss_count = 0  # number of sss measurements received
        self.step_counter = -1  # counter to trigger processing of window
        self.current_step_offset = -1  # true initial index of the current window
        self.port_width = None
        self.stbd_width = None

        # Saved data
        self.data = []  # list of all sss returns
        self.timestamps = []  # list of all data time stamps (floats)
        self.depths = []  # list of all data depth (floats)
        # All of these should be the same length

        # === Detector parameters ===
        self.det_window_size = 25  # number off sss returns to perform detection on
        self.det_step_size = 10  # number of sss returns to wait between processing windows
        self.max_range_ing = 175  # Detector parameter: determine max range of targets to consider for detection
        self.time_threshold = 0.01  # Detections within threshold of existing detections will not be added
        self.det_window = None  # (det_window_size x sss width) array of sss returns
        self.det_times = None  # (det_wind_size,) array of times, corresponding to the sss returns, float seconds
        self.det_depths = None  # (det_wind_size,) array of depths, corresponding to the sss returns, float meters
        self.det_initialized = False
        self.detected_buoys = None
        self.detected_ropes_port = None
        self.detected_ropes_star = None

        # convert the detection indices to spatial ranges
        self.sss_resolution = self.get_parameter("sss_resolution").value

        # === Map parameters ===
        self.line_depth = 0
        # TODO Allow the depth to be ignored?
        self.valid_depth = False
        self.buoy_depth = 0
        self.water_depth = 1000  #

        # === Output parameters ===
        # See overlay_detection_simple from sss_image_detector.py for a plotting example
        # buffer is used for output
        if buffer_len != 0:
            self.buffer_len = buffer_len
        else:
            self.buffer_len = 500
        self.output_window_name = "Sidescan returns"
        self.output_width = 1000
        self.output_initialized = False
        self.output = None  # np.zeros((buffer_len, 2 * sss_data_len), dtype=np.ubyte)

        self.circ_rad = 10
        self.circ_thick = 2
        self.rope_color = np.array([0, 255, 0], dtype=np.uint8)  # Use for images
        self.buoy_color = np.array([0, 0, 255], dtype=np.uint8)  # Use for images

        # ROS images
        self.bridge = CvBridge()

        # rviz markers
        # self.marker_pub = rospy.Publisher(f'/{robot_name}/real/marked_detections', MarkerArray, queue_size=10)

        self.marker_duration = 10000
        self.marker_scale = 1.0
        self.rope_color_float = np.array([0., 1., 0.])  # Use for markers
        self.buoy_color_float = np.array([0., 0., 1.])  # Use for markers
        self.marker_alpha = 0.5


        # === Subscriptions ===
        # SSS
        self.create_subscription(msg_type=Sidescan,
                                 topic=self.sss_topic,
                                 callback=self.sss_callback,
                                 qos_profile=10)

        # depth
        self.create_subscription(msg_type=PoseWithCovarianceStamped, topic=self.depth_topic,
                                 callback=self.depth_cb, qos_profile=10)

        # Map line feature depth
        self.create_subscription(msg_type=Float32, topic=self.line_depth_topic,
                                 callback=self.line_depth_callback, qos_profile=10)

        # === Publishers ===
        # Detection
        self.detection_pub = self.create_publisher(msg_type=Detection2DArray,
                                                   topic=GraphSlamTopics.DETECTOR_HYPOTH_TOPIC,
                                                   qos_profile=10)
        # Detection visualizations
        self.marker_pub = self.create_publisher(msg_type=MarkerArray, topic=GraphSlamTopics.DETECTOR_MARKER_TOPIC,
                                                qos_profile=10)

        self.raw_sss_pub = self.create_publisher(msg_type=Image, topic=GraphSlamTopics.DETECTOR_RAW_SSS_TOPIC,
                                                 qos_profile=10)

        self.marked_sss_pub = self.create_publisher(msg_type=Image, topic=GraphSlamTopics.DETECTOR_MARKED_SSS_TOPIC,
                                                    qos_profile=10)

        self.create_timer(timer_period_sec=self.time_out,
                          callback=self.time_out_callback)

    def declare_node_parameters(self):
        # TODO declare parameters
        #
        # det_window_size
        # det_step_size
        default_robot_name = "sam0"
        self.declare_parameter("robot_name", default_robot_name)
        self.declare_parameter("sss_resolution", 0.05)
        self.declare_parameter("write_data", False)
        pass

    def init_output(self):
        """
        Initialze the output array and initialize the output window
        :return:
        """
        if None in [self.port_width, self.stbd_width]:
            self.get_logger().info(f"Issue determining sss channel width")

        # Initialize Output
        cv.namedWindow(self.output_window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.output_window_name, self.output_width, self.buffer_len)

        self.output = np.zeros((self.buffer_len, self.port_width + self.stbd_width, 3), dtype=np.ubyte)

        self.output_initialized = True

    def line_depth_callback(self, msg):
        self.line_depth = msg.data
        self.valid_depth = True
        self.get_logger().info("Line depth set")

    def sss_callback(self, msg):
        """
        Read new sss data and add to existing imagery
        """
        if self.sss_count == 0:
            self.get_logger().info(f"SSS data received")
        self.sss_count += 1

        if self.sss_count >= self.det_window_size:
            self.step_counter += 1

        port = np.array(bytearray(msg.port_channel), dtype=np.ubyte)
        stbd = np.array(bytearray(msg.starboard_channel), dtype=np.ubyte)
        meas = np.concatenate([np.flip(port), stbd])
        measure_time_secs = ros_time_to_secs(msg.header.stamp)
        measure_depth = self.current_depth

        if not self.output_initialized:
            self.port_width = port.shape[0]
            self.stbd_width = stbd.shape[0]
            self.init_output()

        if not self.det_initialized:
            self.det_window = np.zeros((self.det_window_size, self.port_width + self.stbd_width), dtype=np.ubyte)
            self.det_times = np.zeros((self.det_window_size,))
            self.det_depths = np.zeros((self.det_window_size,))
            self.det_initialized = True

        # update the data
        # These maintain all returns, times, and depths for offline processing or checking
        self.data.append(meas)
        self.timestamps.append(measure_time_secs)
        self.depths.append(measure_depth)

        # Update the output
        self.output[1:, :, :] = self.output[:-1, :, :]  # shift data down
        self.output[0, :, :] = np.dstack((meas, meas, meas))

        # Update the detection window
        self.det_window[1:, :] = self.det_window[:-1, :]  # shift data down
        self.det_window[0, :] = meas

        self.det_times[1:] = self.det_times[:-1]
        self.det_times[0] = measure_time_secs

        self.det_depths[1:] = self.det_depths[:-1]
        self.det_depths[0] = measure_depth

        # Perform Detection at rate set by self.det_step_size
        if self.step_counter >= 0 and self.step_counter % self.det_step_size == 0:
            # Online_analysis object will express the detections of the current window
            # Must apply the window offset for visualizing the detections wrt complete sss returns
            # Not required for publishing as the time of the detections is accurate
            window_offset = self.step_counter
            online_analysis = process_sss(sss_data=self.det_window,
                                          sss_times=self.det_times,
                                          output_path="",
                                          start_ind=0, end_ind=0,
                                          max_range_ind=self.max_range_ing)

            online_analysis.perform_algae_farm_detection(show_output=False,
                                                         save_output=False)

            # ===== Updates =====
            # Updates the detected_buoys, detected_ropes_port, and detected_ropes_star
            # Adds new detections to the output buffer
            # Case I - no detections: returns None
            # Case II - detections: np array with the following rows[(row) index | seq ID | cross (column) index ]
            updated_buoys = None
            updated_port = None
            updated_star = None

            # === Buoy ===
            if online_analysis.final_bouys is not None:
                # apply offset
                offset_online_buoys = online_analysis.final_bouys.copy()
                # Format: [[data index, data time, range index]]
                # These values are offset by the window offset so that the values stored in
                # self.detected_buoys contains the true data index and not just the local window index
                offset_online_buoys[:, 0] += window_offset

                if self.detected_buoys is None:
                    self.detected_buoys = offset_online_buoys
                    updated_buoys = offset_online_buoys

                else:
                    self.detected_buoys, updated_buoys = concat_arrays_with_time_threshold(self.detected_buoys,
                                                                                           offset_online_buoys,
                                                                                           self.time_threshold)
                # Offset removed for plotting of the new detection
                if updated_buoys is not None:
                    updated_buoys[:, 0] -= window_offset

            # === Port ===
            if online_analysis.final_ropes_port is not None:
                offset_ropes_port = online_analysis.final_ropes_port.copy()
                # Format: [[data index, data time, range index]]
                offset_ropes_port[:, 0] += window_offset
                if self.detected_ropes_port is None:
                    self.detected_ropes_port = offset_ropes_port
                else:
                    self.detected_ropes_port, updated_port = concat_arrays_with_time_threshold(self.detected_ropes_port,
                                                                                               offset_ropes_port,
                                                                                               self.time_threshold)
                    # have to remove offset of the updated elements for plotting
                    if updated_port is not None:
                        updated_port[:, 0] -= window_offset

            # === Star ===
            if online_analysis.final_ropes_star is not None:
                offset_ropes_star = online_analysis.final_ropes_star.copy()
                # Format: [[data index, data time, range index]]
                offset_ropes_star[:, 0] += window_offset
                if self.detected_ropes_star is None:
                    self.detected_ropes_star = offset_ropes_star
                else:
                    self.detected_ropes_star, updated_star = concat_arrays_with_time_threshold(self.detected_ropes_star,
                                                                                               offset_ropes_star,
                                                                                               self.time_threshold)
                    # have to remove offset of the updated elements for plotting
                    if updated_star is not None:
                        updated_star[:, 0] -= window_offset

            self._overlay_detections_on_image(buoys=updated_buoys,
                                              port=updated_port,
                                              star=updated_star)

            detection_array_msg = self._construct_detection_array_msg(buoy_detections=updated_buoys,
                                                                      port_detections=updated_port,
                                                                      star_detections=updated_star,
                                                                      detection_depths=self.det_depths)

            if len(detection_array_msg.detections) > 0:
                self.get_logger().info(f"Publishing {len(detection_array_msg.detections)} detections")
                self._publish_sidescan_and_detection_images()
                self._publish_detection_marker(detection_array_msg)
                self.detection_pub.publish(detection_array_msg)

        # update time_out timer
        self.sss_last_time = self.get_clock().now()

        # Display sss data
        resized = cv.resize(self.output, (self.output_width, self.buffer_len), interpolation=cv.INTER_AREA)
        cv.imshow(self.output_window_name, resized)
        cv.waitKey(1)

    def depth_cb(self, msg):
        """
        Callback for depth. The pressure to depth conversion node will not report values out of the water.
        Within this detector depths are treated as positive values.
        :param msg:
        :return:
        """
        self.current_depth = abs(msg.pose.pose.position.z)

    def time_out_callback(self):
        if len(self.data) > 0 and not self.data_written:
            elapsed_time = rcl_times_delta_secs(self.get_clock().now(), self.sss_last_time)
            if elapsed_time > self.time_out and self.write_data:
                # debug
                overlay_detections_simple(image=np.array(self.data),
                                          buoys=self.detected_buoys,
                                          port=self.detected_ropes_port,
                                          star=self.detected_ropes_star)

                print('Saving data!')
                data_array = np.flipud(np.asarray(self.data))
                times_array = np.flipud(np.asarray(self.timestamps).reshape((-1, 1)))

                data_len = data_array.shape[0]
                times_len = times_array.shape[0]

                # Save sonar as csv
                data_path = os.path.join(self.output_path, f'sss_data_{data_len}.csv')
                write_array_to_csv(data_path, data_array)

                # Save sonar as jpg
                data_image = cv.cvtColor(data_array, cv.COLOR_RGB2BGR)
                image_path = os.path.join(self.output_path, f'sss_data_{data_len}.jpg')
                data_image.save(image_path)

                # Save seq ids to help with related the image back to the sensor readings
                time_path = os.path.join(self.output_path, f'sss_seqs_{times_len}.csv')
                write_array_to_csv(time_path, times_array)

                self.data_written = True

                # Output
                print('SSS recording complete!')
                print(f'Callback count : {self.sss_count} - Output length: {data_len} ')
                cv.destroyWindow(self.output_window_name)

            elif elapsed_time > self.time_out and not self.write_data:
                self.data_written = True
                print('sss Detector timeout')
                cv.destroyWindow(self.output_window_name)

    def _overlay_detections_on_image(self,
                                     buoys: np.ndarray | None = None,
                                     port: np.ndarray | None = None,
                                     star: np.ndarray | None = None) -> None:
        """
        Overlay detections on the output buffer image. Does not alter the input arrays.
        :param buoys:
        :param port:
        :param star:
        :return:
        """
        if buoys is not None:
            buoys_indices = buoys[:, [0, 2]].astype(int)
            for center in buoys_indices:
                # Color is reversed because cv assumes bgr??
                color = self.buoy_color[::-1]  # (color[0], color[1], color[2])
                # Colors must be given as an iterable of python ints
                color_list_int = [element.item() for element in color]
                cv.circle(self.output, (center[1], center[0]),
                          radius=self.circ_rad, color=color_list_int, thickness=self.circ_thick)

        # TODO thicken line to make it easier to see
        # EXAMPLE: self.detection_image[0:thickness, max(pos - 10, 0):min(pos + 10, self.channel_size * 2), :] = color
        if port is not None:
            port_copy = port.copy()
            port_copy[:, 2] = (self.port_width - 1) - port_copy[:, 2]
            port_copy = port_copy[:, [0, 2]].astype(int)
            self.output[port_copy[:, 0], port_copy[:, 1]] = self.rope_color

        if star is not None:
            star_copy = star.copy()
            star_copy[:, 2] = self.port_width + star_copy[:, 2]
            star_copy = star_copy[:, [0, 2]].astype(int)
            self.output[star_copy[:, 0], star_copy[:, 1]] = self.rope_color

    def _construct_detection_array_msg(self,
                                       buoy_detections: np.ndarray,
                                       port_detections: np.ndarray,
                                       star_detections: np.ndarray,
                                       detection_depths: np.ndarray,
                                       ):
        """
        Detections are performed on a sliding window of SSS returns.
        Detections are of the format: [[data index, data time, range index]]
        Buoy detections are combined, both port and starboard channels
        Rope detections are seperated by channel
        """
        # Message for the current batch of detections
        detection_array_msg = Detection2DArray()
        detection_array_msg.header.frame_id = self.robot_frame
        detection_array_msg.header.stamp = rcl_time_to_stamp(self.get_clock().now())

        combined_detection = [buoy_detections, port_detections, star_detections]
        combined_types = [ObjectID.BUOY.name, ObjectID.ROPE.name, ObjectID.ROPE.name]
        combined_channels = ['buoy', Side.PORT.name, Side.STARBOARD.name]

        # Add buoys to the detection array
        # Remember that port and starboard buoys are combined
        for detections, detection_type, channel in zip(combined_detection, combined_types, combined_channels):
            if detections is None:
                continue
            for detection in detections:
                detection_window_index = int(detection[0])
                detection_time = detection[1]

                # Rope cases
                if detection_type == ObjectID.ROPE.name:
                    detection_channel = channel
                    detection_range = detection[2] * self.sss_resolution

                # Buoy case
                else:
                    if detection[2] < self.port_width:
                        # Port side buoy detection
                        detection_range = (self.port_width - 1 - detection[2]) * self.sss_resolution
                        detection_channel = Side.PORT.name
                    else:
                        # Starboard side detection
                        detection_range = (detection[2] - self.port_width) * self.sss_resolution
                        detection_channel = Side.STARBOARD.name

                # Depth
                try:
                    detection_depth = detection_depths[detection_window_index]
                except KeyError:
                    self.get_logger().info("Out of range index for detection depth")
                    # Over the time frame of a window the depth shouldn't vary too much, maybe...
                    detection_depth = np.mean(detection_depths)

                # Check if the detection is valid based on object type and depths
                # Return is None if invalid
                detection_pose = self._detection_to_pose(detection_range=detection_range, depth=detection_depth,
                                                         channel=detection_channel, detection_type=detection_type)

                if detection_pose is None:
                    continue

                detection_msg = Detection2D()
                detection_msg.header.frame_id = self.robot_frame
                detection_msg.header.stamp = float_time_to_stamp(detection_time)

                # Define single ObjectHypothesisWithPose
                object_hypothesis_pose_msg = ObjectHypothesisWithPose()

                # set the hypothesis: class id and score
                object_hypothesis_pose_msg.hypothesis.class_id = detection_type  # string
                object_hypothesis_pose_msg.hypothesis.score = 1.0  # float

                # Set the pose
                object_hypothesis_pose_msg.pose.pose = detection_pose
                # object_hypothesis_pose_msg.pose.covariance = ....  # Unused

                # Append the to form a complete message
                detection_msg.results.append(object_hypothesis_pose_msg)
                detection_array_msg.detections.append(detection_msg)

        return detection_array_msg

    def _detection_to_pose(self, detection_range: float, depth: float,
                           channel: str, detection_type: str) -> Pose | None:
        """
        Given detected range, depth, channel, and type, determine if detection is valid

        :param detection_range: range of detection
        :param depth: depth at the time of the detection
        :param channel: either Side.PORT.name or Side.STARBOARD.name
        :param detection_type: either ObjectID.ROPE or ObjectID.BUOY
        :return: Pose for the detection if valid, otherwise None
        """

        if detection_type == ObjectID.ROPE.name:
            target_depth = self.line_depth
        elif detection_type == ObjectID.BUOY.name:
            target_depth = self.buoy_depth
        else:
            self.get_logger().info("Malformed call of _detection_to_pose(): detection_type")
            return None

        height_difference = target_depth - depth

        # Error 1) reported ranged is less than the height difference
        if abs(detection_range) <= height_difference:
            return None

        distance_2d = (detection_range ** 2 - height_difference ** 2) ** 0.5

        # Base_link points forward and to the left
        if channel == Side.PORT.name:
            detected_pose = Pose()
            detected_pose.position.y = distance_2d
        elif channel == Side.STARBOARD.name:
            detected_pose = Pose()
            detected_pose.position.y = - distance_2d
        else:
            self.get_logger().info("Malformed call of _detection_to_pose(): channel")
            return None

        return detected_pose

    def _publish_detection_marker(self, detection_message: Detection2DArray):
        if len(detection_message.detections) == 0:
            return

        detection_marker_array = MarkerArray()
        detection_count = 0

        for detection in detection_message.detections:
            for result in detection.results:
                marker = Marker()
                marker.header.frame_id = self.robot_frame  # detection.header.frame_id
                marker.header.stamp = rcl_time_to_stamp(self.get_clock().now())
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.id = detection_count
                marker.lifetime.sec = int(self.marker_duration)
                marker.pose.position.x = float(result.pose.pose.position.x)
                marker.pose.position.y = float(result.pose.pose.position.y)
                marker.pose.position.z = 0.
                marker.pose.orientation.x = 0.
                marker.pose.orientation.y = 0.
                marker.pose.orientation.z = 0.
                marker.pose.orientation.w = 1.
                marker.scale.x = float(self.marker_scale)
                marker.scale.y = float(self.marker_scale)
                marker.scale.z = float(self.marker_scale)
                if result.hypothesis.class_id == ObjectID.ROPE.name:
                    marker.color.r = self.rope_color_float[0]
                    marker.color.g = self.rope_color_float[1]
                    marker.color.b = self.rope_color_float[2]
                elif result.hypothesis.class_id == ObjectID.BUOY.name:
                    marker.color.r = self.buoy_color_float[0]
                    marker.color.g = self.buoy_color_float[1]
                    marker.color.b = self.buoy_color_float[2]
                else:
                    marker.color.r = 1.
                    marker.color.g = 0.
                    marker.color.b = 0.
                marker.color.a = float(self.marker_alpha)

                detection_marker_array.markers.append(marker)
                detection_count += 1

            print('Publishing detection markers')
            self.marker_pub.publish(detection_marker_array)

    def _publish_sidescan_and_detection_images(self):
        try:
            # TODO Publish raw sss this way ass well
            # self.sidescan_image_pub.publish(
            #     self.bridge.cv2_to_imgmsg(self.sidescan_image, "passthrough"))
            # Publish marked sss
            self.marked_sss_pub.publish(
                self.bridge.cv2_to_imgmsg(self.output, "passthrough"))
        except CvBridgeError as error:
            print('Error converting numpy array to img msg: {}'.format(error))


def main(arg=None, namespace=None):
    rclpy.init(args=arg)
    sss_detector = SSSDetector(namespace=namespace)
    try:
        rclpy.spin(sss_detector)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    # namespace set to "sam0" to allow for debugging
    # Topics assumed that node is namespaced properly
    main(namespace="sam0")
