DO SOME Duck-you-mend-thing



# Pipeline Scenario
- pipeline_map_publisher.py: Responsible for publishing the map, as markers.

- pipeline_point_cloud_detector.py: Pipeline detector based on mbes. Very basic, uses Hough circles to determine the 
  center of the pipe

- pipeline_gt_dr_publisher.py: Generates DR odometry. This should be phased out once there is a working ROS 2 DR.

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