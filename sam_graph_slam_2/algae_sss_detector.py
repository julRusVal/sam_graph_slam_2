#!/usr/bin/env python3

import os
import cv2 as cv
import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node

from smarc_msgs.msg import Sidescan

# Ros
try:
    from .helpers.general_helpers import write_array_to_csv
    from .helpers.ros_helpers import rcl_times_delta_secs
    from .detectors.sss_image_detector import process_sss, concat_arrays_with_time_threshold, overlay_detections_simple
except ImportError:
    from helpers.general_helpers import write_array_to_csv
    from helpers.ros_helpers import rcl_times_delta_secs, ros_time_to_secs
    from detectors.sss_image_detector import process_sss, concat_arrays_with_time_threshold, overlay_detections_simple

"""
Detector node for algae farm scenario using side scan sonar returns

Detector: image processing based on the side scan image. See, process_real_sss.py

detection publisher: there should be an example of this in sam_slam call sss_detection_img_proc_publisher.py
"""


class SSSDetector(Node):
    def __init__(self, time_out: float = 0.0, buffer_len: int = 0):
        super().__init__('algae_sss_detector_node')
        self.get_logger().info("Initialized sss_saver_node")
        self.declare_node_parameters()

        # Sonar parameters
        # TODO use parameters
        self.robot_name = "sam0"
        self.sss_topic = f"/{self.robot_name}/payload/sidescan"
        self.output_path = "/home/julian/sss_detector_data"
        if not os.path.isdir(self.output_path):
            self.output_path = ""

        if time_out != 0.0:
            self.time_out = time_out
        else:
            self.time_out = 5.0

        # TODO this should probably not be hardcoded and be set when a message is recieved
        sss_data_len = 1000  # This is determined by the message

        self.get_logger().info(f"SSS topic: {self.sss_topic}")
        self.get_logger().info(f"Output path: {self.output_path}")

        # Initialize state info
        self.data_written = False
        self.sss_last_time = self.get_clock().now()
        self.sss_count = 0  # number of sss measurements received
        self.step_counter = -1  # counter to trigger processing of window
        self.current_step_offset = -1  # true initial index of the current window
        self.port_width = None
        self.stbd_width = None

        # Saved data
        self.data = []
        self.timestamps = []  # list of data time stamps, seconds as floats

        # ===== Detector parameters =====
        self.det_window_size = 25  # number off sss returns to perform detection on
        self.det_step_size = 10  # number of sss returns to wait between processing windows
        self.max_range_ing = 175  # Detector parameter: determine max range of targets to consider for detection
        self.time_threshold = 0.01  # Detections within threshold of existing detections will not be added
        self.det_window = None  # (det_window_size x sss width) array of sss returns
        self.det_times = None  # (det_wind_size,) array of times, corresponding to the sss returns, float seconds
        self.det_initialized = False
        self.detected_buoys = None
        self.detected_ropes_port = None
        self.detected_ropes_star = None

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
        self.rope_color = np.array([0, 255, 0], dtype=np.uint8)
        self.buoy_color = np.array([0, 0, 255], dtype=np.uint8)

        # SSS data subscription
        self.create_subscription(msg_type=Sidescan,
                                 topic=self.sss_topic,
                                 callback=self.sss_callback,
                                 qos_profile=10)

        self.create_timer(timer_period_sec=self.time_out,
                          callback=self.time_out_callback)

    def declare_node_parameters(self):
        # TODO declare parameters
        #
        # det_window_size
        # det_step_size
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
        # cv.resizeWindow(self.output_window_name, self.port_width + self.stbd_width, self.buffer_len)
        cv.resizeWindow(self.output_window_name, self.output_width, self.buffer_len)

        self.output = np.zeros((self.buffer_len, self.port_width + self.stbd_width, 3), dtype=np.ubyte)

        self.output_initialized = True

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

        if not self.output_initialized:
            self.port_width = port.shape[0]
            self.stbd_width = stbd.shape[0]
            self.init_output()

        if not self.det_initialized:
            self.det_window = np.zeros((self.det_window_size, self.port_width + self.stbd_width), dtype=np.ubyte)
            self.det_times = np.zeros((self.det_window_size,))
            self.det_initialized = True

        # update the data
        self.data.append(meas)
        self.timestamps.append(measure_time_secs)

        # Update the output
        self.output[1:, :, :] = self.output[:-1, :, :]  # shift data down
        self.output[0, :, :] = np.dstack((meas, meas, meas))

        # Update the detection window
        self.det_window[1:, :] = self.det_window[:-1, :]  # shift data down
        self.det_window[0, :] = meas

        self.det_times[1:] = self.det_times[:-1]
        self.det_times[0] = measure_time_secs

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
            updated_buoys = None
            updated_port = None
            updated_star = None

            # === Buoy ===
            if online_analysis.final_bouys is not None:
                # apply offset
                offset_online_buoys = online_analysis.final_bouys.copy()
                # Format: [[data index, data time, range index]]
                offset_online_buoys[:, 0] += window_offset
                if self.detected_buoys is None:
                    self.detected_buoys = offset_online_buoys
                else:
                    self.detected_buoys, updated_buoys = concat_arrays_with_time_threshold(self.detected_buoys,
                                                                                           offset_online_buoys,
                                                                                           self.time_threshold)
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

            self.overlay_detections_on_image(buoys=updated_buoys,
                                             port=updated_port,
                                             star=updated_star)

        # update time_out timer
        self.sss_last_time = self.get_clock().now()


        # Display sss data
        resized = cv.resize(self.output, (self.output_width, self.buffer_len), interpolation=cv.INTER_AREA)
        cv.imshow(self.output_window_name, resized)
        cv.waitKey(1)

    def time_out_callback(self):
        if len(self.data) > 0 and not self.data_written:
            elapsed_time = rcl_times_delta_secs(self.get_clock().now(), self.sss_last_time)
            if elapsed_time > self.time_out:
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
                data_image = Image.fromarray(data_array)
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

    def overlay_detections_on_image(self,
                                    buoys : np.ndarray|None = None,
                                    port : np.ndarray|None = None,
                                    star : np.ndarray|None = None) -> None:
        """
        Overlay detections on the output buffer image
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

        if port is not None:
            port[:, 2] = (self.port_width - 1) - port[:, 2]
            port = port[:, [0, 2]].astype(int)
            self.output[port[:, 0], port[:, 1]] = self.rope_color

        if star is not None:
            star[:, 2] = self.port_width + star[:, 2]
            star = star[:, [0, 2]].astype(int)
            self.output[star[:, 0], star[:, 1]] = self.rope_color

def main(arg=None):
    rclpy.init(args=arg)
    sss_detector = SSSDetector()
    try:
        rclpy.spin(sss_detector)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    # sss_recorder = sss_recorder(10, 500)
    main()
