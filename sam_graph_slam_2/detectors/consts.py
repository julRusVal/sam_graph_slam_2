from enum import Enum

"""
This is copied from my sam_slam_branch of smarc_perception/sss_object_detection

For now this is just for convenience and because I'm still a little unclear on the ROS2 python imports
"""


class Side(Enum):
    """The side-scan sonar ping side (port or starboard)"""
    PORT = 0
    STARBOARD = 1


class ObjectID(Enum):
    """
    ObjectID for object detection

    These used by the detector messages to identify the type of object detected.
    Class Ids are provided as strings of the names.
    """
    NADIR = 0
    ROPE = 1
    BUOY = 2
    PIPE = 3
