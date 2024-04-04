#!/usr/bin/env python3

import os
import cv2
import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node

from smarc_msgs.msg import Sidescan

# Ros
try:
    from .helpers.general_helpers import write_array_to_csv
    from .helpers.ros_helpers import rcl_times_delta_secs
except ImportError:
    from helpers.general_helpers import write_array_to_csv
    from helpers.ros_helpers import rcl_times_delta_secs, ros_time_to_secs

"""
Utility for viewing and saving the raw sidescan imagery produced by the sss_detector.
This is based on the view_sidescan.py found in smarc_utils/sss_viewer

Will save the all raw measurements and continue to show a buffers worth at all times.

This is the ROS 2 version of sss_raw_saver_node.py from the original sam_slam package.

The ROS 1 the sequence IDs were saved to disk for data association:
seq_ids[image index] -> sequence ID --> Then the appropriate callback could check for that sequence ID

ROS 2 has deprecated the sequence ID so now we'll use the time.

NOTE: data is flipped vertically, so that the most recent returns are plotted at the top, have the lowest index.
"""


class SSSSaver(Node):
    def __init__(self, time_out: float = 0.0, buffer_len: int = 0):
        super().__init__('util_sss_saver_node')
        self.get_logger().info("Initialized sss_saver_node")
        # Sonar parameters
        # TODO use parameters
        self.robot_name = "sam0"
        self.sss_topic = f"/{self.robot_name}/payload/sidescan"
        self.output_path = "/home/julian/sss_data"
        if not os.path.isdir(self.output_path):
            self.output_path = ""

        if time_out != 0.0:
            self.time_out = time_out
        else:
            self.time_out = 5.0

        if buffer_len != 0:
            self.buffer_len = buffer_len
        else:
            self.buffer_len = 500

        # TODO this should probably not be hardcoded and be set when a message is recieved
        sss_data_len = 1000  # This is determined by the message

        self.get_logger().info(f"SSS topic: {self.sss_topic}")
        self.get_logger().info(f"Output path: {self.output_path}")

        # Initialize state info
        self.data_written = False
        self.sss_last_time = self.get_clock().now()
        self.sss_count = 0
        self.port_width = None
        self.stbd_width = None

        self.output_window_name = "Sidescan returns"
        self.output_initialized = False
        self.output = None  # np.zeros((buffer_len, 2 * sss_data_len), dtype=np.ubyte)

        # Saved data
        self.data = []
        self.timestamps = []  # list of data time stamps, seconds as floats

        # SSS data subscription
        self.create_subscription(msg_type=Sidescan,
                                 topic=self.sss_topic,
                                 callback=self.sss_callback,
                                 qos_profile=10)

        self.create_timer(timer_period_sec=self.time_out,
                          callback=self.time_out_callback)

    def init_output(self):
        """
        Initialze the output array and initialize the output window
        :return:
        """
        if None in [self.port_width, self.stbd_width]:
            self.get_logger().info(f"Issue determining sss channel width")

        # Initialize Output
        cv2.namedWindow(self.output_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.output_window_name, self.port_width + self.stbd_width, self.buffer_len)

        self.output = np.zeros((self.buffer_len, self.port_width + self.stbd_width), dtype=np.ubyte)

        self.output_initialized = True

    def sss_callback(self, msg):
        """
        Read new sss data and add to existing imagery
        """
        if self.sss_count == 0:
            self.get_logger().info(f"SSS data received")
        self.sss_count += 1

        port = np.array(bytearray(msg.port_channel), dtype=np.ubyte)
        stbd = np.array(bytearray(msg.starboard_channel), dtype=np.ubyte)
        meas = np.concatenate([np.flip(port), stbd])

        if not self.output_initialized:
            self.port_width = port.shape[0]
            self.stbd_width = stbd.shape[0]
            self.init_output()

        # update the data
        self.data.append(meas)
        self.timestamps.append(ros_time_to_secs(msg.header.stamp))

        # Update the output
        self.output[1:, :] = self.output[:-1, :]  # shift data down
        self.output[0, :] = meas

        # update time_out timer
        self.sss_last_time = self.get_clock().now()

        # Display sss data
        resized = cv2.resize(self.output, (2 * 256, self.buffer_len), interpolation=cv2.INTER_AREA)
        cv2.imshow(self.output_window_name, resized)
        cv2.waitKey(1)

    def time_out_callback(self):
        if len(self.data) > 0 and not self.data_written:
            elapsed_time = rcl_times_delta_secs(self.get_clock().now(), self.sss_last_time)
            if elapsed_time > self.time_out:
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
                cv2.destroyWindow(self.output_window_name)


def main(arg=None):
    rclpy.init(args=arg)
    sss_saver = SSSSaver()
    try:
        rclpy.spin(sss_saver)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    # sss_recorder = sss_recorder(10, 500)
    main()
