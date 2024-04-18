#!/usr/bin/env python3

import ast

import rclpy
from rclpy.node import Node

# ROS messages
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


def change_marker_depth(marker: Marker, depth: float):
    """
    Changes the marker z value to the specified depth
    :param marker: Marker from visualization_msgs.msg
    :param depth: new z value, float
    :return:
    """
    modified_marker = marker
    modified_marker.pose.position.z = depth
    return modified_marker


def change_marker_color(marker: Marker, color: [float, float, float]):
    """
    Changes the marker color to the specified value
    :param marker: Marker from visualization_msgs.msg
    :param color: [r, g, b] given in float values
    :return:
    """

    modified_marker = marker
    modified_marker.color.r = color[0]
    modified_marker.color.g = color[1]
    modified_marker.color.b = color[2]
    return modified_marker


class PipelineMapPublisher(Node):
    """
    Publishes the map as markers

    pipeline_end_coords: list[list[float, float]], x and y coords of pipeline markers.
    Corresponds to buoys, thought there is no detection of point features in this scenario, in that they define the
    boundries of the pipeline.
    """

    def __init__(self,
                 robot_name=None,  # Robot name
                 map_frame=None,  # default frame, used to
                 simulated_data=None,  # adjust if the data is real or simulated
                 pipeline_end_coords=None,
                 pipeline_depth=None,
                 pipeline_lines=None,
                 pipeline_colors=None):

        super().__init__("pipeline_map_publisher")
        self.get_logger().info("Created Pipeline Map Publisher")

        # Declare parameters
        self.declare_node_parameters()

        # Set values
        if robot_name is None:
            self.robot_name = self.get_parameter("robot_name").value
        else:
            self.robot_name = robot_name

        if map_frame is None:
            self.map_frame = self.get_parameter("map_frame").value
        else:
            self.map_frame = map_frame

        # Data type
        if simulated_data is None:
            self.simulated_data = self.get_parameter("simulated_data").value
        else:
            self.simulated_data = simulated_data

        # Debug - value appear incorrect, parameter problem?
        self.get_logger().info(f"simulated_data: {self.simulated_data}")

        if self.simulated_data:
            self.data_type = 'sim'
        else:
            self.data_type = 'real'

        # Map coordinates
        if pipeline_end_coords is None:
            # Parameter stored as a string corresponding to list[list[float,float]]
            self.end_coords = ast.literal_eval(self.get_parameter("pipeline_end_coords").value)
        else:
            self.end_coords = pipeline_end_coords

        if pipeline_depth is None:
            self.depth = self.get_parameter("pipeline_depth").value
        else:
            self.depth = pipeline_depth

        if pipeline_lines is None:
            # Parameter stored as a string corresponding to list[list[int, int]]
            self.lines = ast.literal_eval(self.get_parameter("pipeline_lines").value)
        else:
            self.lines = pipeline_lines

        if pipeline_colors is None:
            # Parameter stored as a string corresponding to list[list[float,float,float]
            self.colors = ast.literal_eval(self.get_parameter("pipeline_colors").value)
        else:
            self.colors = pipeline_colors

        if len(self.lines) != len(self.colors):
            self.colors = [[1.0, 1.0, 0.0] for i in range(len(pipeline_lines))]

        # Verboseness
        self.verbose_map_publish = self.get_parameter("verbose_map_publish").value

        self.cnt = 0
        self.buoy_positions_utm = []
        self.buoy_markers = None
        self.rope_inner_markers = None
        self.rope_outer_markers = None
        self.rope_lines = None  # Is this needed for anything??
        self.marker_rate = 1.0
        self.outer_marker_scale = 5.0
        self.inner_marker_scale = 2.0
        self.buoy_marker_list = None

        # Visualization options
        self.n_buoys_per_rope = 8
        self.rope_markers_on_z_plane = True  # Modifies rope markers to z=0, useful for 2d estimation
        self.publisher_period = 1.0

        self.construct_buoy_markers()

        # Define topic names and create publishers
        # TODO better name
        self.marker_topic = f'/{self.robot_name}/{self.data_type}/marked_positions'
        self.rope_outer_marker_topic = f'/{self.robot_name}/{self.data_type}/rope_outer_marker'
        self.rope_marker_topic = f'/{self.robot_name}/{self.data_type}/marked_rope'
        self.rope_lines_topic = f'/{self.robot_name}/{self.data_type}/marked_rope_lines'

        self.get_logger().info(f"Point features topic: {self.marker_topic}")
        self.get_logger().info(f"Line features topic: {self.rope_outer_marker_topic}")

        # used to define buoys
        # These buoys are used by the graph to initialize the buoy and rope priors
        # Important to be accurate
        self.marker_pub = self.create_publisher(msg_type=MarkerArray, topic=self.marker_topic, qos_profile=10)

        # Used to define the extents of ropes
        self.rope_outer_marker_pub = self.create_publisher(msg_type=MarkerArray,
                                                           topic=self.rope_outer_marker_topic, qos_profile=10)
        # used to visualize the ropes
        self.rope_marker_pub = self.create_publisher(msg_type=MarkerArray,
                                                     topic=self.rope_marker_topic, qos_profile=10)

        # Visualizes the rope as a line -> I think it might be kind of ugly
        self.rope_lines_pub = self.create_publisher(msg_type=MarkerArray,
                                                    topic=self.rope_lines_topic, qos_profile=10)

        self.publisher_timer = self.create_timer(timer_period_sec=self.publisher_period,
                                                 callback=self.publish_markers)

    def declare_node_parameters(self):
        """
        Declare the relevant parameters for the map node

        This will set the default values to the initial pipeline scenario
        :return:
        """

        self.get_logger().info("Declaring Parameters with initial pipeline scenario")

        self.declare_parameter("robot_name", "sam")
        self.declare_parameter("map_frame", "odom")
        self.declare_parameter("simulated_data", True)

        self.declare_parameter("pipeline_end_coords",
                               "[[-260, -829],"
                               "[-263, -930],"
                               "[-402, -1081],"
                               "[-403, -1178]]")

        self.declare_parameter("pipeline_depth", -85.0)
        self.declare_parameter("pipeline_lines", "[[0, 1],[1, 2],[2, 3]]")

        self.declare_parameter("pipeline_colors", "[[1.0, 1.0, 0.0],"
                                                  "[0.0, 1.0, 1.0],"
                                                  "[1.0, 0.0, 1.0]]")

        self.declare_parameter("verbose_map_publish", False)

    def construct_buoy_markers(self):
        if len(self.end_coords) == 0:
            return

        # ===== Buoys ====

        self.buoy_marker_list = []  # Used later for the forming the lines
        self.buoy_markers = MarkerArray()
        for i, end_coord in enumerate(self.end_coords):
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = i
            marker.pose.position.x = float(end_coord[0])
            marker.pose.position.y = float(end_coord[1])
            marker.pose.position.z = float(self.depth)
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(self.outer_marker_scale)
            marker.scale.y = float(self.outer_marker_scale)
            marker.scale.z = float(self.outer_marker_scale)
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            self.buoy_markers.markers.append(marker)

            self.buoy_marker_list.append(marker)  # Record the markers for later use as the rope end

        # ===== Ropes =====
        if self.lines is not None and len(self.lines) > 0:
            self.rope_inner_markers = MarkerArray()  # intermediate buoys on each rope
            self.rope_outer_markers = MarkerArray()  # end buoys of each rope

            self.rope_lines = MarkerArray()

            for rope_ind, rope in enumerate(self.lines):
                # Create points of the line
                point_start = Point()
                point_start.x = float(self.end_coords[rope[0]][0])
                point_start.y = float(self.end_coords[rope[0]][1])
                point_start.z = float(self.depth)

                point_end = Point()
                point_end.x = float(self.end_coords[rope[1]][0])
                point_end.y = float(self.end_coords[rope[1]][1])
                point_end.z = float(self.depth)

                # Rope as a line
                # === Construct rope line msg ===
                rope_line_marker = Marker()
                rope_line_marker.header.frame_id = self.map_frame
                rope_line_marker.type = Marker.LINE_STRIP
                rope_line_marker.action = Marker.ADD
                rope_line_marker.id = int(self.n_buoys_per_rope + 1) * rope_ind
                rope_line_marker.scale.x = 1.0  # Line width
                rope_line_marker.color.r = 1.0  # Line color (red)
                rope_line_marker.color.a = 1.0  # Line transparency (opaque)
                rope_line_marker.points = [point_start, point_end]
                rope_line_marker.pose.orientation.w = 1.0

                self.rope_lines.markers.append(rope_line_marker)

                # Rope as a line of buoys

                # markers representing the ends of the ropes
                start_marker = change_marker_color(self.buoy_marker_list[rope[0]], self.colors[rope_ind])
                end_marker = change_marker_color(self.buoy_marker_list[rope[1]], self.colors[rope_ind])

                if self.rope_markers_on_z_plane:
                    start_marker = change_marker_depth(start_marker, 0.0)
                    end_marker = change_marker_depth(end_marker, 0.0)

                self.rope_outer_markers.markers.append(start_marker)
                self.rope_outer_markers.markers.append(end_marker)

                # Intermediate rope markers
                rope_buoys = self.calculate_rope_buoys(point_start, point_end)

                for buoy_ind, rope_buoy in enumerate(rope_buoys):
                    marker_id = int(self.n_buoys_per_rope + 1) * rope_ind + buoy_ind + 1
                    marker = Marker()
                    marker.header.frame_id = self.map_frame
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.id = marker_id
                    marker.lifetime.sec = 0

                    marker.pose.position.x = rope_buoy.x
                    marker.pose.position.y = rope_buoy.y
                    if self.rope_markers_on_z_plane:
                        marker.pose.position.z = 0.0
                    else:
                        marker.pose.position.z = self.depth

                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0
                    marker.pose.orientation.w = 1.0

                    marker.scale.x = self.inner_marker_scale
                    marker.scale.y = self.inner_marker_scale
                    marker.scale.z = self.inner_marker_scale

                    marker.color.r = self.colors[rope_ind][0]
                    marker.color.g = self.colors[rope_ind][1]
                    marker.color.b = self.colors[rope_ind][2]
                    marker.color.a = 1.0

                    # # Create the line segment points
                    # line_points = [point_start, point_end]
                    #
                    # # Set the line points
                    # marker.points = line_points

                    # Append the marker to the MarkerArray
                    self.rope_inner_markers.markers.append(marker)
                    self.get_logger().info(f'|MAP_MARKER| Rope {rope_ind} - {buoy_ind} added')

    def publish_markers(self):
        # if None in [self.buoy_markers, self.rope_inner_markers]:
        #     self.construct_buoy_markers()
        if self.verbose_map_publish:
            self.get_logger().info(f"Map published")

        if self.buoy_markers is not None:
            self.marker_pub.publish(self.buoy_markers)

        if self.rope_inner_markers is not None:
            self.rope_marker_pub.publish(self.rope_inner_markers)
            self.rope_lines_pub.publish(self.rope_lines)

        if self.rope_outer_markers is not None:
            self.rope_outer_marker_pub.publish(self.rope_outer_markers)

    def calculate_rope_buoys(self, start_point, end_point):
        positions = []

        # Calculate the step size between each position
        step_size = 1.0 / (self.n_buoys_per_rope + 1)

        # Calculate the vector difference between start and end points
        delta = Point()
        delta.x = end_point.x - start_point.x
        delta.y = end_point.y - start_point.y

        # Calculate the intermediate positions
        for i in range(1, self.n_buoys_per_rope + 1):
            position = Point()

            # Calculate the position based on the step size and delta
            position.x = start_point.x + i * step_size * delta.x
            position.y = start_point.y + i * step_size * delta.y
            position.z = 0.0

            positions.append(position)

        return positions


def main(args=None):
    rclpy.init(args=args)
    pipeline_map_publisher = PipelineMapPublisher()
    try:
        rclpy.spin(pipeline_map_publisher)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    print("Update")
    print("Parameters are defined by the class")
    # rospy.init_node('pipeline_markers_nodes', anonymous=False)
    # rate = rospy.Rate(1)
    #
    # robot_name = rospy.get_param('robot_name', 'sam')
    # frame_name = rospy.get_param('frame', 'map')
    # simulated_data = rospy.get_param('simulated_data', False)
    #
    # # Map Defaults
    # default_end_coords = [[-260, -829],
    #                       [-263, -930],
    #                       [-402, -1081],
    #                       [-403, -1178]]
    #
    # default_depth = -85
    #
    # default_lines = [[0, 1],  # was called self.ropes
    #                  [1, 2],
    #                  [2, 3]]
    #
    # # Load Map from params if possible
    # pipeline_end_coords = ast.literal_eval(rospy.get_param("pipeline_end_coords", "[]"))
    # if len(pipeline_end_coords) == 0:
    #     print("Pipeline marker node: Using default coordinates")
    #     pipeline_end_coords = default_end_coords
    #
    # pipeline_depth = rospy.get_param("pipeline_depth", default=1)
    # if pipeline_depth > 0:
    #     print("Pipeline marker node: Using default depth")
    #     pipeline_depth = default_depth
    #
    # pipeline_lines = ast.literal_eval(rospy.get_param("pipeline_lines", "[]"))
    # if len(pipeline_lines) == 0:
    #     print("Pipeline marker node: Using default lines")
    #     pipeline_lines = default_lines
    #
    # pipeline_colors = ast.literal_eval(rospy.get_param("pipeline_colors", "[]"))
    #
    # pipeline_marker_server = publish_pipeline_markers(robot_name=robot_name, map_frame=frame_name,
    #                                                   simulated_data=simulated_data,
    #                                                   pipeline_end_coords=pipeline_end_coords,
    #                                                   pipeline_depth=pipeline_depth,
    #                                                   pipeline_lines=pipeline_lines,
    #                                                   pipeline_colors=pipeline_colors)
    #
    # if None in [pipeline_marker_server.buoy_markers, pipeline_marker_server.rope_inner_markers]:
    #     pipeline_marker_server.construct_buoy_markers()
    #
    # pipeline_marker_server.publish_markers()
    #
    # rospy.spin()
