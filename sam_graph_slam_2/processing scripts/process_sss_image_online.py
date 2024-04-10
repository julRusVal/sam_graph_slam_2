#!/usr/bin/env python3

"""
Apply image processing techniques to assist in the detection of relevant features

This script is intended to process the real data collected at the algae farm.
"""

# %% Imports
import os
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

from sam_graph_slam_2.detectors.sss_image_detector import process_sss, concat_arrays_with_time_threshold, safe_vstack


def overlay_detections_simple(image: np.ndarray,
                              buoys: np.ndarray,
                              port: np.ndarray,
                              star: np.ndarray,
                              title: str | None = None):
    """
    buoy_detections is [data index, time, range index]

    :return:
    """
    # Plotting parameters
    circ_rad = 10
    circ_thick = 2
    rope_color = np.array([0, 255, 0], dtype=np.uint8)
    buoy_color = np.array([0, 0, 255], dtype=np.uint8)

    # Raw image converted to RGBd
    img_color = np.dstack((image, image, image))
    img_combined = np.copy(img_color)

    # Buoy
    if buoys is not None:
        buoys_indices = buoys[:, [0, 2]].astype(int)
        for center in buoys_indices:
            # Color is reversed because cv assumes bgr??
            color = buoy_color[::-1]  # (color[0], color[1], color[2])
            color_tup = (255, 0, 0)
            cv.circle(img_combined, (center[1], center[0]),
                      radius=circ_rad, color=color_tup, thickness=circ_thick)

    # Ropes
    # Port and star rope inds are purely range measurements
    # This required offsetting for correct plotting
    if port is not None:
        # (image.shape[1]//2 - 1)
        port[:, 2] = 1000 - port[:, 2]
    if star is not None:
        star[:, 2] = 1000 + star[:, 2]

    # combine port ans starboard detections
    ropes_combined = safe_vstack(port, star)

    if ropes_combined is not None:
        ropes = ropes_combined[:, [0, 2]].astype(int)
        img_combined[ropes[:, 0], ropes[:, 1]] = rope_color

    detect_fig, (ax1, ax2) = plt.subplots(1, 2)
    if title is None:
        detect_fig.suptitle(f'Detection overlay')
    else:
        detect_fig.suptitle(f'{title} Detection overlay')

    ax1.title.set_text('Original')
    ax1.imshow(img_color)

    ax2.title.set_text('Combined detections')
    ax2.imshow(img_combined)
    plt.gcf().set_dpi(300)
    plt.show()


def overlay_detections(image: np.ndarray,
                       online_buoys: np.ndarray,
                       online_port: np.ndarray,
                       online_star: np.ndarray,
                       offline_buoys: np.ndarray,
                       offline_port: np.ndarray,
                       offline_star: np.ndarray,
                       title: str | None = None):
    """
    buoy_detections is [data index, time, range index]

    :return:
    """
    # Plotting parameters
    circ_rad = 10
    circ_thick = 2
    rope_color = np.array([0, 255, 0], dtype=np.uint8)
    buoy_color = np.array([0, 0, 255], dtype=np.uint8)

    # Buoy
    if online_buoys:
        online_buoys_indices = online_buoys[:, [0, 2]].astype(int)
    if offline_buoys:
        offline_buoys_indices = offline_buoys[:, [0, 2]].astype(int)

    # Ropes
    # combine port ans starboard detections
    online_ropes_combined = safe_vstack(online_port, online_star)
    offline_ropes_combined = safe_vstack(offline_port, offline_star)

    if online_ropes_combined:
        online_ropes = online_ropes_combined[:, [0, 2]].astype(int)

    if offline_ropes_combined:
        offline_ropes = offline_ropes_combined[: [0, 2]].astype(int)

    # Raw image converted to RGBd
    img_color = np.dstack((image, image, image))

    # Combined
    img_combined = np.copy(img_color)
    if online_ropes_combined:
        img_combined[online_ropes[:, 0], online_ropes[:, 1]] = rope_color

    if offline_ropes_combined:
        img_combined[offline_ropes[:, 0], offline_ropes[:, 1]] = rope_color
    # img_combined[self.post_rope > 0] = rope_color

    # buoys
    # online
    if online_buoys:
        for center in online_buoys_indices:
            # Color is reversed because cv assumes bgr??
            color = buoy_color[::-1]  # (color[0], color[1], color[2])
            color_tup = (255, 0, 0)
            cv.circle(img_combined, (center[1], center[0]),
                      radius=circ_rad, color=color_tup, thickness=circ_thick)

    # offline
    if offline_buoys:
        for center in offline_buoys_indices:
            # Color is reversed because cv assumes bgr??
            color = buoy_color[::-1]  # (color[0], color[1], color[2])
            color_tup = (255, 0, 0)
            cv.circle(img_combined, (center[1], center[0]),
                      radius=circ_rad, color=color_tup, thickness=circ_thick)

    detect_fig, (ax1, ax2) = plt.subplots(1, 2)
    if title is None:
        detect_fig.suptitle(f'Detection overlay')
    else:
        detect_fig.suptitle(f'{title} Detection overlay')

    ax1.title.set_text('Original')
    ax1.imshow(img_color)

    ax2.title.set_text('Combined detections')
    ax2.imshow(img_combined)
    plt.gcf().set_dpi(300)
    plt.show()


if __name__ == '__main__':
    """
    testing of online processing
    """
    # === Data selection/import ===
    data_set_length = 1490
    sss_file_name = f'/home/julian/sss_data/sss_data_{data_set_length}.jpg'
    time_file_name = f'/home/julian/sss_data/sss_seqs_{data_set_length}.csv'
    output_path = '/home/julian/detector_results/image'

    sss_data = cv.imread(sss_file_name, cv.IMREAD_GRAYSCALE)
    sss_times = np.genfromtxt(time_file_name, delimiter=',')
    data_length = sss_data.shape[0]

    # === Parameters ===
    # Truncate data parameters
    start_ind = 0
    end_ind = 0

    # Online window size parameter
    window_size = 30  # min 25
    step_size = 5  #

    # Other processing parameters
    max_range_ing = 175
    time_threshold = 0.01

    # === Perform truncation, operates on the complete data set ===
    # Check supplied indices
    if start_ind <= 0 or start_ind > data_length:
        start_ind = 0

    if end_ind <= 0 or end_ind > data_length:
        end_ind = data_length

    # Truncate
    sss_data = sss_data[start_ind:end_ind, :]
    sss_times = sss_times[start_ind: end_ind]

    # Update size
    data_length = sss_data.shape[0]

    # output
    online_buoys = None
    online_port = None
    online_star = None

    offline_buoys = None
    offline_port = None
    offline_star = None

    # offline
    offline_analysis = process_sss(sss_data=sss_data,
                                   sss_times=sss_times,
                                   output_path=output_path,
                                   start_ind=0, end_ind=0,
                                   max_range_ind=max_range_ing)

    offline_analysis.perform_algae_farm_detection()

    offline_buoys = offline_analysis.final_bouys
    offline_port = offline_analysis.final_ropes_port
    offline_star = offline_analysis.final_ropes_star

    overlay_detections_simple(image=sss_data,
                              buoys=offline_buoys,
                              port=offline_port,
                              star=offline_star,
                              title="Offline")

    # online
    # Loop over data
    for i in range(0, data_length, step_size):
        if i + window_size > data_length:
            break

        start_ind = i
        end_ind = start_ind + window_size

        current_data = sss_data[start_ind:end_ind, :]
        current_times = sss_times[start_ind:end_ind]

        # process complete window
        print(f'Start Index: {start_ind}  - End Index: {end_ind}')

        online_analysis = process_sss(sss_data=current_data,
                                      sss_times=current_times,
                                      output_path=output_path,
                                      start_ind=0, end_ind=0,
                                      max_range_ind=max_range_ing)

        online_analysis.perform_algae_farm_detection()


        # ===== Updates =====
        # Buoy
        if online_analysis.final_bouys is not None:
            # apply offset
            offset_online_buoys = online_analysis.final_bouys.copy()
            # Format: [[data index, data time, range index]]
            offset_online_buoys[:, 0] += i
            if online_buoys is None:
                online_buoys = offset_online_buoys
            else:
                online_buoys = concat_arrays_with_time_threshold(online_buoys,
                                                                 offset_online_buoys,
                                                                 time_threshold)

        if online_analysis.final_ropes_port is not None:
            offset_ropes_port = online_analysis.final_ropes_port.copy()
            # Format: [[data index, data time, range index]]
            offset_ropes_port[:, 0] += i
            if online_port is None:
                online_port = offset_ropes_port
            else:
                online_port = concat_arrays_with_time_threshold(online_port,
                                                                offset_ropes_port,
                                                                time_threshold)

        if online_analysis.final_ropes_star is not None:
            offset_ropes_star = online_analysis.final_ropes_star.copy()
            # Format: [[data index, data time, range index]]
            offset_ropes_star[:, 0] += i
            if online_star is None:
                online_star = offset_ropes_star
            else:
                online_star = concat_arrays_with_time_threshold(online_star,
                                                                offset_ropes_star,
                                                                time_threshold)

    overlay_detections_simple(image=sss_data,
                              buoys=online_buoys,
                              port=online_port,
                              star=online_star,
                              title="Online")
