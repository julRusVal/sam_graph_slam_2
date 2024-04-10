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

from sam_graph_slam_2.detectors.sss_image_detector import process_sss

if __name__ == '__main__':
    # Parameters
    # process_sam_sss = False
    # ===== Data preprocessing and manual detections =====
    show_flipping = False
    show_manual_detection = False
    perform_flipping = False  # fix real data

    # ===== Gradient method =====
    perform_grad_method = False
    grad_show_intermediate = True
    grad_show = True
    grad_med_size = 7
    grad_gauss_size = 5
    grad_grad_size = 5

    # ===== Canny edge detector =====
    perform_custom_canny = True
    show_custom_canny = True
    canny_med_size = 3  # 7  # rope
    canny_gauss_size = 7  # rope
    canny_sobel_size = 5  # rope
    canny_l_thresh = 175  # rope
    canny_h_thresh = 225  # rope
    # ===== Template matching =====
    buoy_med_size = 7  # buoy
    buoy_gauss_size = 7  # buoy
    buoy_sobel_size = 5  # buoy
    show_buoy_grad = True
    do_blob = False  # buoy
    do_template = True  # buoy
    show_template = False  # buoy
    show_template_raw = False  # buoy
    template_size = 21  # buoy
    template_sigma = 2  # buoy
    template_feature_width = 10  # buoy
    template_l_threshold = 1000  # buoy
    template_h_threshold = 3000  # buoy

    # ===== Standard canny detector =====
    perform_standard_canny = False
    standard_canny_show = False
    standard_canny_med_size = 7
    standard_canny_l_thresh = 175
    standard_canny_h_thresh = 225

    # ===== CPD method =====
    perform_cpd = False
    cpd_show = False
    cpd_max_depth = 100
    cpd_ratio = 1.1  # 1.55 was default
    cpd_med_size = 0

    # ===== Combined detector output =====
    perform_combined_detector = False

    # ===== Post ====
    perform_post = True
    ringing_max_count = 2
    ringing_show = False
    limiting_min = 30
    limiting_max = 100
    limiting_show = False
    inds_med_size = 45  # 45 worked well
    show_final_post = True

    buoy_center_size_threshold = 5
    buoy_center_exclusion_zone = 25

    rope_exclusion_size = 25

    show_final_inds_port = True

    start_time = time.time()

    # Data is produced by the util_sss_saver_node.py
    # TODO rename as seq IDs are no longer used, now using
    # The data is labeled by its length: sss_data_{length}.jpg and sss_seq_{length}.csv
    # lengths: bag file
    # 1450: unity_algae, standard, 0.3, sss noise
    # 1490: unity_algae_1, reduced, 0.05 sss noise

    data_set_length = 1490
    sss_file_name = f'/home/julian/sss_data/sss_data_{data_set_length}.jpg'
    time_file_name = f'/home/julian/sss_data/sss_seqs_{data_set_length}.csv'
    output_path = '/home/julian/detector_results/image'

    sss_data = cv.imread(sss_file_name, cv.IMREAD_GRAYSCALE)
    sss_times = np.genfromtxt(time_file_name, delimiter=',')

    start_ind = 0  # 2000  # 3400  # 3400  # 6300
    end_ind = 0  # 7608# 6000  # 5700  # 4600  # 7200
    max_range_ing = 175

    # !!! These all apply to the real algae data set !!!
    # detections = [[7106, 1092], [6456, 1064],
    #               [5570, 956], [4894, 943],
    #               [4176, 956], [3506, 924],
    #               [2356, 911], [1753, 949],
    #               [1037, 941], [384, 943]]

    detections = [[7096, 907], [6452, 937],
                  [5570, 956], [4894, 943],
                  [4176, 956], [3506, 924],
                  [2356, 911], [1753, 949],
                  [1037, 941], [384, 943]]

    flipped_regions = [[5828, -1]]

    # %% Process SAM SSS
    sss_analysis = process_sss(sss_data=sss_data,
                               sss_times= sss_times,
                               output_path=output_path,
                               start_ind=start_ind, end_ind=end_ind,
                               max_range_ind=max_range_ing,
                               cpd_max_depth=cpd_max_depth, cpd_ratio=cpd_ratio,
                               flipping_regions=flipped_regions,
                               flip_original=perform_flipping)

    # if show_flipping:
    #     sss_analysis.flip_data(flipped_sections=flipped_regions)

    if show_manual_detection:
        sss_analysis.mark_manual_detections(detections)

    if perform_grad_method:
        sss_analysis.filter_median(grad_med_size, show=grad_show_intermediate)
        sss_analysis.filter_gaussian(grad_gauss_size, show=grad_show_intermediate)
        grad_output = sss_analysis.gradient_cross_track(grad_grad_size, show=grad_show_intermediate)
        grad_method_results = sss_analysis.filter_threshold(threshold=200, show=grad_show)

    else:
        grad_method_results = None

    if perform_custom_canny:
        canny_custom, custom_dx, custom_dx_neg = sss_analysis.canny_custom(canny_med_size,
                                                                           canny_gauss_size,
                                                                           canny_sobel_size,
                                                                           canny_l_thresh, canny_h_thresh,
                                                                           show=show_custom_canny)
        # sss_analysis.show_thresholds(custom_dx, 100, 1000, 'Custom Dx Positive')
        # sss_analysis.show_thresholds(custom_dx_neg, 100, 1000, 'Custom Dx Negative')

        # ==== find the first and second rising edges
        # sss_analysis.find_rising_edges(canny_custom, 150, 2, True)

        # ===== New =====
        if do_blob:
            # Create blob detector
            params = cv.SimpleBlobDetector_Params()

            # Set parameters
            params.minThreshold = 140
            params.maxThreshold = 500
            params.filterByArea = True
            params.minArea = 15
            params.filterByCircularity = False
            params.minCircularity = 0.7

            # Create detector with parameters
            detector = cv.SimpleBlobDetector_create(params)

            # Detect blobs
            keypoints = detector.detect(custom_dx)

            # Draw blobs on the image
            image_with_keypoints = cv.drawKeypoints(custom_dx, keypoints, np.array([]), (0, 0, 255),
                                                    cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Plot the original image and the image with keypoints
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(custom_dx)
            axs[0].set_title('Original Image')
            axs[0].axis('off')

            axs[1].imshow(cv.cvtColor(image_with_keypoints, cv.COLOR_BGR2RGB))
            axs[1].set_title('Image with Keypoints')
            axs[1].axis('off')

            plt.tight_layout()
            plt.gcf().set_dpi(300)
            plt.show()

        if do_template:
            buoy_dx, rope_dx_neg = sss_analysis.down_range_gradient(m_size=buoy_med_size,
                                                                    g_size=buoy_gauss_size,
                                                                    s_size=buoy_sobel_size,
                                                                    show=show_buoy_grad)
            matching_image = np.copy(buoy_dx).astype(np.float32)
            if template_size % 2 == 0:
                template_size += 1
            template = sss_analysis.construct_template_kernel(template_size, template_sigma, template_feature_width)
            template = template.astype(np.float32)
            template_result = cv2.matchTemplate(matching_image, template, cv2.TM_CCOEFF)

            # matchTemplate results will need padding
            # W' = W - w + 1, where W': final width, W: initial width, w: template width
            # Above holds for the height as well
            pad_size = template_size // 2
            template_result = np.pad(template_result, pad_size)

            if show_template:
                sss_analysis.show_thresholds(template_result,
                                             template_l_threshold,
                                             template_h_threshold,
                                             'Grad Template',
                                             reference_data=not show_template_raw)

                plt.imshow(template)
        else:
            template_result = None

    else:
        canny_custom = None
        template_result = None

    if perform_standard_canny:
        standard_canny = sss_analysis.canny_standard(m_size=standard_canny_med_size,
                                                     l_threshold=standard_canny_l_thresh,
                                                     h_threshold=standard_canny_h_thresh,
                                                     show=standard_canny_show)
    else:
        standard_canny = None

    if perform_cpd:
        sss_analysis.set_working_to_original()
        sss_analysis.filter_median(cpd_med_size, show=cpd_show)
        sss_analysis.cpd_perform_detection(0)
        sss_analysis.cpd_perform_detection(1)
        if cpd_show:
            sss_analysis.cpd_plot_detections()

    if perform_combined_detector:
        sss_analysis.show_detections(grad_results=grad_method_results,
                                     canny_results=canny_custom)

    pre_end_time = time.time()
    post_end_time = None

    while perform_post:
        # check if the needed pre-processing has been saved and
        if canny_custom is None:
            print("Post failed: canny_custom not found!")
            break

        if template_result is None:
            print("Post failed: tem not found!")
            break

        # Thresholding of 'raw' detections
        # ROPE: Thresholding is applied during canny
        # Buoy: template_results requires thresholding
        template_result_threshold = np.zeros_like(template_result, np.uint8)
        template_result_threshold[template_result >= template_h_threshold] = 255

        sss_analysis.post_initialize(rope_detections=canny_custom,
                                     buoy_detections=template_result_threshold,
                                     use_port=True,
                                     use_starboard=True)

        # Process buoy detections
        sss_analysis.post_find_buoy_centers(min_region_size=buoy_center_size_threshold,  # 5
                                            exclusion_zone=buoy_center_exclusion_zone)

        # Work in progress,
        # sss_analysis.post_find_buoy_offsets(window_size=55, plot=True)

        # Process rope detections
        """
        Rope detection is carried out in multiple steps
        - Ringing removal
        - Range limiting

        """
        # Remove ringing
        # Useful to perform before the limiting the range
        sss_analysis.post_remove_ringing_rope(max_count=ringing_max_count,
                                              show_results=ringing_show)

        # Enforce max detection range
        sss_analysis.post_limit_range(min_index=limiting_min,
                                      max_index=limiting_max,
                                      show_results=limiting_show)

        sss_analysis.post_exclude_rope_in_buoy_area(radius=rope_exclusion_size, show_results=False)

        post_port_detection_inds, post_star_detection_inds = sss_analysis.find_rising_edges(data=sss_analysis.post_rope,
                                                                                            threshold=0,
                                                                                            max_count=2,
                                                                                            show=False,
                                                                                            save_output=False)

        port_inter_raw, port_inter_med = sss_analysis.post_interleave_detection_inds(
            channel_data=post_port_detection_inds,
            channel_id='port',
            med_filter_kernel=inds_med_size,
            show_results=False)

        star_inter_raw, star_inter_med = sss_analysis.post_interleave_detection_inds(
            channel_data=post_star_detection_inds,
            channel_id='star',
            med_filter_kernel=inds_med_size,
            show_results=False)
        # sss_analysis.post_de_interleave(interleaved_med)

        sss_analysis.post_interleaved_to_2d(interleaved_port=port_inter_med,
                                            interleaved_star=star_inter_med)

        if show_final_inds_port:
            sss_analysis.post_plot_inds(channel_id='port')

        # sss_analysis.post_find_buoy_offsets(window_size=55, plot=True)

        sss_analysis.post_final_buoys(plot=False)
        sss_analysis.post_final_ropes(plot=False)

        post_end_time = time.time()

        if show_final_post:
            sss_analysis.post_overlay_detections()
            sss_analysis.post_overlay_detections_pretty()

        # Generate

        break

    # Show timings
    pre_time = pre_end_time - start_time

    if post_end_time is not None:
        post_time = post_end_time - pre_end_time
        complete_time = post_end_time - start_time

    else:
        post_time = 0
        complete_time = pre_time

    size = sss_analysis.end_ind - sss_analysis.start_ind - 1
    print(f"Processed count : {size}")
    print(f"Pre-processing time: {pre_time}")
    print(f"Post-processing time: {post_time}")
    print(f"Complete time: {complete_time}")
