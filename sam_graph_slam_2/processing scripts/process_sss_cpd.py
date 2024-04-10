#!/usr/bin/env python3

"""
Apply Change point detection to saved side scans. This is meant to be applied to offline.
This script needs to be modified to specify the correct sss data to process

Source: sss_cpd_process.py of sam_slam
"""

# %% Imports
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


from sam_graph_slam_2.detectors.consts import ObjectID
from sam_graph_slam_2.detectors.cpd_detector import CPDDetector


# %% Classes
class sss_detector:
    def __init__(self, file_name, start_ind, end_ind, max_nadir_ind, cpd_ratio=None):
        # SSS parameters
        self.resolution = 0.05
        # If this is set above 0 the nadir detection is not performed and is assumed to be at this index
        # TODO: figure out nadir detection, this is a little ugly
        self.max_nadir_ind = max_nadir_ind

        # Img info
        self.file_name = file_name
        self.img = cv.imread(file_name)  # cv.IMREAD_GRAYSCALE)
        self.height = self.img.shape[0]

        # Check the clipping indices
        if 0 < end_ind <= start_ind:
            print('Improper clipping indices!')

        if end_ind <= 0:
            self.end_ind = self.height - 1
        else:
            self.end_ind = min(self.height - 1, end_ind)

        if start_ind <= 0:
            self.start_ind = 0
        else:
            self.start_ind = min(self.height - 1, self.end_ind - 1, start_ind)

        # Apply Clipping
        self.img = self.img[self.start_ind:self.end_ind, :]

        self.height = self.img.shape[0]
        self.scan_width = self.img.shape[1] // 2

        self.port = np.flip(self.img[:, :self.scan_width], axis=1)
        self.starboard = self.img[:, self.scan_width:]

        # Detector
        self.port_detections = np.zeros((self.height, self.scan_width, 3))
        self.starboard_detections = np.zeros((self.height, self.scan_width, 3))

        if cpd_ratio is None:
            self.detector = CPDDetector()
        else:
            self.detector = CPDDetector(min_mean_diff_ratio=cpd_ratio)

        self.nadir_color = np.array([255, 0, 0], dtype=np.uint8)
        self.rope_color = np.array([0, 255, 0], dtype=np.uint8)
        self.buoy_color = np.array([0, 0, 255], dtype=np.uint8)

        self.nadir_label = 'Nadir'
        self. rope_label = 'Rope'
        self.buoy_label = 'Buoy'

        # The colors need to specified as floats, 0-1
        self.labels = {self.nadir_label: self.nadir_color/255.0,
                       self.rope_label: self.rope_color/255.0,
                       self.buoy_label: self.buoy_color/255.0}

        self.legend_handles = [mpatches.Patch(color=value,label=key) for key, value in self.labels.items()]

        # Plotting parameters
        self.plot_width = 0

    def perform_detection(self, side=0):
        """
        Note: this has the ability to use multiple detection methods, defined in cp_detector_local.py
        """
        # Select which side to perform detection on
        if side == 0:
            img_side = self.port
        else:
            img_side = self.starboard

        # Allocate array for detections
        # img_detections = np.zeros((img_side.shape[0], img_side.shape[1], 3), dtype=np.uint8)
        img_detections = np.copy(img_side).astype(np.uint8)

        for i, ping in enumerate(img_side):

            if self.max_nadir_ind > 0:
                ping_results = self.detector.detect_rope(ping, self.max_nadir_ind)
            else:
                ping_results = self.detector.detect(ping)

            if ObjectID.NADIR in ping_results.keys():
                img_detections[i, ping_results[ObjectID.NADIR]['pos'], :] = self.nadir_color

            if ObjectID.ROPE in ping_results.keys():
                img_detections[i, ping_results[ObjectID.ROPE]['pos'], :] = self.rope_color

            if ObjectID.BUOY in ping_results.keys():
                img_detections[i, ping_results[ObjectID.BUOY]['pos'], :] = self.buoy_color

        # Save results
        if side == 0:
            self.port_detections[:, :, :] = img_detections[:, :, :]
        else:
            self.starboard_detections[:, :, :] = img_detections[:, :, :]

    def plot_detections(self):
        """
        Plot the detections
        """
        # Form yellow center band, helps to separate the port and starboard returns
        band = np.ones((self.height, 5, 3), dtype=np.uint8) * 255
        band[:, :, 2] = 0

        if self.plot_width <= 0 or self.plot_width >= self.scan_width:
            final = np.hstack((np.flip(self.port_detections[:, :, :], axis=1),
                               band,
                               self.starboard_detections[:, :, :]))
        else:
            final = np.hstack((np.flip(self.port_detections[:, :self.plot_width, :], axis=1),
                               band,
                               self.starboard_detections[:, :self.plot_width, :]))

        plt.imshow(final)
        plt.legend(handles=self.legend_handles)
        plt.show()

    def plot_max(self):

        sss_max = np.maximum(self.port, self.starboard)
        plt.imshow(sss_max)
        plt.title('Channel Max Plot')
        plt.show()


if __name__ == '__main__':
    # %% Processing parameters
    data_file = '/home/julian/sss_data/sss_data_1450.jpg'  # 'data/sss_data_1.png'
    start_ind = 0  # 3400  # 3400  # 6300
    end_ind = 0  # 4600  # 5700  # 7200
    max_nadir_depth = 0

    show_max = True
    perform_detections = True

    # %% Process
    detector_test = sss_detector(data_file,
                                 start_ind=start_ind, end_ind=end_ind,
                                 max_nadir_ind=max_nadir_depth,
                                 cpd_ratio=1.0)  # 1.55 is the previous standard ration
    # %% Max
    """
    The idea here is to use both channels to find the nadir as, depending on the sss beam widths and orientations
    the two channels should detected the same approximate point as the nadir.
    """
    if show_max:
        detector_test.plot_max()

    # %% Main detections
    if perform_detections:
        detector_test.perform_detection(0)
        detector_test.perform_detection(1)
        detector_test.plot_detections()
