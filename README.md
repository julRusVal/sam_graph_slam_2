DO SOME Duck-you-mend-thing
# Overview

# Pipeline Scenario
## pipeline_map_publisher.py
Responsible for publishing the map, as markers.

## pipeline_point_cloud_detector.py 
Pipeline detector based on mbes. Very basic, uses Hough circles to determine the center of the pipe

## pipeline_gt_dr_publisher.py
Generates DR odometry. This should be phased out once there is a working ROS 2 DR.
### TODOs
- (FIX) prevent z values from going above the waters surface

# Algae Farm Scenario
## algae_map_publisher.py: Responsible for publishing the map, as markers. The current map uses a list of names that
correspond to the frames of algae farm elements, Buoys and ropes, of the Unity sim environment
### TODOs
- (FIX) Markers are being reused and modified for the different MarkerArrays, 'copying' them is not properly decoupling
- (FEATURE) define buoys by lat/lon coords

# algae_sss_detector.py
Image processing based method for detecting ropes and buoys from sss returns.
Subscription topics:
- SSS, as defined by SIDESCAN_TOPIC in sam_msgs
- Depth, as defined by DEPTH_TOPIC in dead_reckoning_msgs
- Line depth, as defined by MAP_LINE_DEPTH_TOPIC in sam_graph_slam_msgs

# Utilities
- util_sss_saver_node.py: Utility for saving the sss returns

- util_image_saver_node.py: (In progress) Utility for saving camera imagery. This needs to be ported to ROS 2
  and adapted to the new environment (Nothing in particular).

# Packages
## detectors
(This content should be moved to its onw package at some point)

## helpers
- general_helpers: geometry and basic file saving stuff
- gtsam_helpers: mostly functions to construct gtsam objects
- gtsam_ros_helpers: conversions between ros and gtsam objects
- pointcloud2_conversions: this might be depreciated, CHECK!
- ros_helpers: mostly functions for handling ROS time messages and objects 

## slam_packages