#!/usr/bin/env python3

import ast

import rclpy
from rclpy import time
from rclpy.node import Node

import tf2_ros

from std_msgs.msg import Float32
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from sam_graph_slam_2_msgs.msg import Topics as GraphSlamTopics

"""
Node to publish markers of the given frame
"""


def create_marker_copy(original_marker: Marker) -> Marker:
    """
        Copies an original marker and modifies its ID.
        """
    # Create a new marker object by copying the original marker
    new_marker = Marker()
    new_marker.header = original_marker.header
    new_marker.ns = original_marker.ns
    new_marker.type = original_marker.type
    new_marker.action = original_marker.action
    new_marker.pose = original_marker.pose
    new_marker.scale = original_marker.scale
    new_marker.color = original_marker.color
    new_marker.lifetime = original_marker.lifetime
    new_marker.frame_locked = original_marker.frame_locked
    new_marker.points = list(original_marker.points)
    new_marker.colors = list(original_marker.colors)
    new_marker.text = original_marker.text
    new_marker.mesh_resource = original_marker.mesh_resource
    new_marker.mesh_use_embedded_materials = original_marker.mesh_use_embedded_materials

    return new_marker


def change_marker_depth(marker: Marker, depth: float):
    """
    Changes the marker z value to the specified depth
    :param marker: Marker from visualization_msgs.msg
    :param depth: new z value, float
    :return:
    """
    modified_marker = create_marker_copy(marker)
    modified_marker.pose.position.z = depth
    return modified_marker


def change_marker_id(marker: Marker, id: int):
    """
    Changes the marker id, int. This is intended to prevent id conflicts
    :param marker: Marker from visualization_msgs.msg
    :param id: new id, int
    :return:
    """

    modified_marker = create_marker_copy(marker)
    modified_marker.id = id
    return modified_marker


def change_marker_color(marker: Marker, color: [float, float, float]):
    """
    Changes the marker color to the specified value
    :param marker: Marker from visualization_msgs.msg
    :param color: [r, g, b] given in float values
    :return:
    """

    modified_marker = create_marker_copy(marker)
    modified_marker.color.r = color[0]
    modified_marker.color.g = color[1]
    modified_marker.color.b = color[2]
    return modified_marker


class AlgaeMapPublisher(Node):
    def __init__(self,
                 robot_name: str | None = None,
                 map_frame: str | None = None,
                 simulated_data: bool | None = None,
                 buoy_frames: list[str] | None = None,
                 rope_frames: list[str] | None = None,
                 line_ends: list[list[int]] | None = None):
        """

        :param buoy_frames:
        :param rope_frames:
        :param line_ends:
        """
        super().__init__('Algae_map_publisher', namespace=robot_name)
        self.get_logger().info("Created Algae Map Publisher")

        # ===== Parameters =====
        # === Declare parameters ===
        self.declare_node_parameters()

        # === Robot name ==
        if robot_name is None:
            self.robot_name = self.get_parameter('robot_name').value
        else:
            self.robot_name = robot_name

        # === Map parameters ===
        if map_frame is None:
            self.map_frame = self.get_parameter("map_frame").value
        else:
            self.map_frame = map_frame

        if buoy_frames is None:
            self.buoy_frames = ast.literal_eval(self.get_parameter("buoy_frames").value)
        else:
            self.buoy_frames = buoy_frames

        if rope_frames is None:
            self.rope_frames = ast.literal_eval(self.get_parameter("rope_frames").value)
        else:
            self.rope_frames = rope_frames

        if line_ends is None:
            self.line_ends = ast.literal_eval(self.get_parameter("line_ends").value)
        else:
            self.line_ends = line_ends

        # === Other parameters ===
        self.cnt = 0
        self.buoy_positions_map = []
        self.valid_positions = False
        self.depth = 0.0
        self.valid_depth = False  # would also like to pass as arg
        self.buoy_markers = None
        self.rope_inner_markers = None
        self.rope_outer_markers = None
        self.rope_lines = None  # Is this needed for anything??
        self.marker_rate = 1.0
        self.outer_marker_scale = .50
        self.inner_marker_scale = .25
        self.valid_markers = False
        self.buoy_marker_list = None

        # Visualization options
        self.n_buoys_per_rope = 1
        self.rope_markers_at_depth = False  # Modifies rope markers to z=0, useful for 2d estimation
        self.publisher_period = 1.0

        if self.line_ends is not None:
            self.line_colors = [[1.0, 1.0, 0.0] for i in range(len(self.line_ends))]

        self.buoy_color = [0.0, 1.0, 1.0]

        # Verboseness options
        self.verbose_map_publisher = self.get_parameter("verbose_map_publisher").value
        self.get_logger().info(f"verbose_map_publisher: {self.verbose_map_publisher}")

        # ===== Construct =====
        # Determine buoy positions from the frame names -> make corresponding markers
        # determine rope depth from frame names, publish as a float?

        # ===== Ouputs =====
        # === Topics ===
        # Define topic names and create publishers
        # TODO better name
        # self.marker_topic = f'/{self.robot_name}/{self.data_type}/marked_positions'
        # self.rope_outer_marker_topic = f'/{self.robot_name}/{self.data_type}/rope_outer_marker'
        # self.rope_marker_topic = f'/{self.robot_name}/{self.data_type}/marked_rope'
        # self.rope_lines_topic = f'/{self.robot_name}/{self.data_type}/marked_rope_lines'

        # Topics used for both visualization and for defining the map
        self.point_feature_topic = GraphSlamTopics.MAP_POINT_FEATURE_TOPIC
        self.line_feature_topic = GraphSlamTopics.MAP_LINE_FEATURE_TOPIC
        self.line_feature_depth_topic = GraphSlamTopics.MAP_LINE_DEPTH_TOPIC

        # These topics are used only for visualizations
        self.rope_marker_topic = GraphSlamTopics.MAP_MARKED_LINE_SPHERES_TOPIC
        self.rope_lines_topic = GraphSlamTopics.MAP_MARKED_LINE_LINES_TOPIC

        self.get_logger().info(f"Point features topic: {self.point_feature_topic}")
        self.get_logger().info(f"Line features topic: {self.line_feature_topic}")
        # === Publishers ===
        # = Map Topics =
        # Defines point features (buoys)
        # Intended for consumption by the graph slam node to define the map
        self.point_feature_pub = self.create_publisher(msg_type=MarkerArray, topic=self.point_feature_topic,
                                                       qos_profile=10)

        # Defines the end points of line features (ropes or pipes)
        # Intended for consumption by the graph slam node to define the map
        self.line_feature_pub = self.create_publisher(msg_type=MarkerArray, topic=self.line_feature_topic,
                                                      qos_profile=10)

        # Defines the depth of the line features
        # Intended for consumption by the sss detector node
        self.line_depth_pub = self.create_publisher(msg_type=Float32, topic=self.line_feature_depth_topic,
                                                    qos_profile=10)

        # = Visualization topics =
        # Visualization of the ropes
        self.rope_marker_pub = self.create_publisher(msg_type=MarkerArray, topic=self.rope_marker_topic,
                                                     qos_profile=10)

        # Visualization of the rope as a line -> I think it might be kind of ugly
        self.rope_lines_pub = self.create_publisher(msg_type=MarkerArray, topic=self.rope_lines_topic,
                                                    qos_profile=10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.publisher_timer = self.create_timer(self.publisher_period, self.publish_markers)

    def declare_node_parameters(self):
        """
        declare the parameters of the node.
        :return:
        """
        # TODO
        # The current defaults work with the Algae Fam TFed prefab
        default_robot_name = 'sam0'
        self.declare_parameter('robot_name', default_robot_name)

        # Lists are passed as string because some types of lists, nested lists, are not passed as parameters well
        # List of buoy frame names
        default_buoy_frames = ("['Algae0_a0_buoy_Sphere_yellow (2)', 'Algae0_a0_buoy_Sphere_yellow (6)', "
                               "'Algae0_a0_buoy_Sphere_yellow (5)', 'Algae0_a0_buoy_Sphere_yellow (3)']")

        # List of rope frame name, used to find the depth of the horizontal lines
        default_rope_frames = "['Algae0_Cylinder (1)']"

        # List of rope ends with respect to the indices of the buoys given in buoy_frames
        default_line_ends = ("[[0,1],[2,3],"  # main ropes
                             "[0,2],[1,3]]")  # cross ropes

        self.declare_parameter("map_frame", "odom")
        self.declare_parameter('buoy_frames', default_buoy_frames)
        self.declare_parameter("rope_frames", default_rope_frames)
        self.declare_parameter("line_ends", default_line_ends)

        self.declare_parameter("verbose_map_publisher", False)

    def find_rope_depth(self):
        """
        calculate the average depth of the ropes, depths will be reported as a positive value
        :return:
        """

        if self.valid_depth:
            return

        rope_depth_sum = 0
        rope_depth_count = 0
        for index, frame in enumerate(self.rope_frames):
            try:
                trans = self.tf_buffer.lookup_transform(self.map_frame, frame, rclpy.time.Time())
                rope_depth_sum += abs(trans.transform.translation.z)
                rope_depth_count += 1
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.get_logger().info(f'Could not transform {frame} to map')
                continue

        if rope_depth_count > 0:
            self.depth = rope_depth_sum / rope_depth_count
            self.get_logger().info(f"Depth: {self.depth}")
            self.valid_depth = True
        else:
            self.depth = 0
            self.valid_depth = False

    def find_buoy_map_positions(self):
        """
        Determine the map positions of the buoys given their tf frame name.
        Once determined, this does not need to be updated.

        NOTE: Will not update!
        :return:
        """
        if self.valid_positions:
            return

        buoy_positions = []
        for index, frame in enumerate(self.buoy_frames):
            try:
                trans = self.tf_buffer.lookup_transform(self.map_frame, frame, rclpy.time.Time())
                buoy_positions.append([trans.transform.translation.x,
                                       trans.transform.translation.y,
                                       trans.transform.translation.z])
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.get_logger().info(f'Could not transform {frame} to map')
                return

        # Success!
        self.valid_positions = True
        self.buoy_positions_map = buoy_positions

    def calculate_rope_buoys(self, start_point: Point, end_point: Point) -> list[Point]:
        """
        Find the positions of intermediate buoys
        :param start_point: x,y coords of the start
        :param end_point: x,y coords of the end
        :return:
        """
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

    def construct_buoy_markers(self):
        # Check for valid positions
        # Only publish if all map transforms are found
        if not self.valid_positions:
            self.find_buoy_map_positions()

        if not self.valid_depth:
            self.find_rope_depth()
            return

        # ===== Buoys ====
        self.buoy_marker_list = []  # Used later for the forming the lines
        self.buoy_markers = MarkerArray()
        for i, end_coord in enumerate(self.buoy_positions_map):
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = i
            marker.pose.position.x = float(end_coord[0])
            marker.pose.position.y = float(end_coord[1])
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(self.outer_marker_scale)
            marker.scale.y = float(self.outer_marker_scale)
            marker.scale.z = float(self.outer_marker_scale)
            marker.color.r = self.buoy_color[0]
            marker.color.g = self.buoy_color[1]
            marker.color.b = self.buoy_color[2]
            marker.color.a = 1.0

            self.buoy_markers.markers.append(marker)

            self.buoy_marker_list.append(marker)  # Record the markers for later use as the rope end

        # ===== Ropes =====
        if self.line_ends is not None and len(self.line_ends) > 0:
            self.rope_inner_markers = MarkerArray()  # intermediate buoys on each rope
            self.rope_outer_markers = MarkerArray()  # end buoys of each rope

            self.rope_lines = MarkerArray()  # this should plot the line as a line in rviz

            for line_ind, line in enumerate(self.line_ends):
                # Create points of the line
                point_start = Point()
                point_start.x = float(self.buoy_positions_map[line[0]][0])
                point_start.y = float(self.buoy_positions_map[line[0]][1])
                point_start.z = float(self.depth)

                point_end = Point()
                point_end.x = float(self.buoy_positions_map[line[1]][0])
                point_end.y = float(self.buoy_positions_map[line[1]][1])
                point_end.z = float(self.depth)

                # Rope as a line
                # === Construct rope line msg ===
                rope_line_marker = Marker()
                marker_id = int(self.n_buoys_per_rope + 1) * line_ind + len(self.buoy_positions_map)
                rope_line_marker.header.frame_id = self.map_frame
                rope_line_marker.type = Marker.LINE_STRIP
                rope_line_marker.action = Marker.ADD
                rope_line_marker.id = marker_id
                rope_line_marker.scale.x = 1.0  # Line width
                rope_line_marker.color.r = 1.0  # Line color (red)
                rope_line_marker.color.a = 1.0  # Line transparency (opaque)
                rope_line_marker.points = [point_start, point_end]
                rope_line_marker.pose.orientation.w = 1.0

                # print(f"Outer: {marker_id} ({line_ind})")
                self.rope_lines.markers.append(rope_line_marker)

                # Rope as a line of buoys
                # markers representing the ends of the ropes
                start_marker = change_marker_color(self.buoy_marker_list[line[0]], self.line_colors[line_ind])
                end_marker = change_marker_color(self.buoy_marker_list[line[1]], self.line_colors[line_ind])

                if self.rope_markers_at_depth:
                    start_marker = change_marker_depth(start_marker, -self.depth)
                    end_marker = change_marker_depth(end_marker, -self.depth)

                # ADD the end markers
                # The IDs are changed to prevent ID
                start_id = len(self.rope_outer_markers.markers)
                end_id = start_id + 1
                self.rope_outer_markers.markers.append(change_marker_id(start_marker, start_id))
                self.rope_outer_markers.markers.append(change_marker_id(end_marker, end_id))

                # Intermediate rope markers
                rope_buoys = self.calculate_rope_buoys(point_start, point_end)

                for buoy_ind, rope_buoy in enumerate(rope_buoys):
                    marker_id = int(self.n_buoys_per_rope + 1) * line_ind + buoy_ind + 1 + len(self.buoy_positions_map)
                    marker = Marker()
                    marker.header.frame_id = self.map_frame
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.id = marker_id
                    marker.lifetime.sec = 0

                    marker.pose.position.x = rope_buoy.x
                    marker.pose.position.y = rope_buoy.y
                    if self.rope_markers_at_depth:
                        marker.pose.position.z = float(-self.depth)
                    else:
                        marker.pose.position.z = 0.0

                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0
                    marker.pose.orientation.w = 1.0

                    marker.scale.x = self.inner_marker_scale
                    marker.scale.y = self.inner_marker_scale
                    marker.scale.z = self.inner_marker_scale

                    marker.color.r = self.line_colors[line_ind][0]
                    marker.color.g = self.line_colors[line_ind][1]
                    marker.color.b = self.line_colors[line_ind][2]
                    marker.color.a = 1.0

                    # # Create the line segment points
                    # line_points = [point_start, point_end]
                    #
                    # # Set the line points
                    # marker.points = line_points

                    # Append the marker to the MarkerArray
                    self.rope_inner_markers.markers.append(marker)
                    # print(f"Inner: {int(self.n_buoys_per_rope + 1) * line_ind + buoy_ind + 1}"
                    #       f"({line_ind}) ({buoy_ind})")
                    self.get_logger().info(f'|MAP_MARKER| Rope {line_ind} - {buoy_ind} added')

        self.valid_markers = True

    def publish_markers(self):
        if self.valid_markers and self.valid_depth:
            if self.verbose_map_publisher:
                self.get_logger().info(f"Map published")

            # Publish only if
            if self.buoy_markers is not None:
                self.point_feature_pub.publish(self.buoy_markers)

            if self.rope_outer_markers is not None:
                self.line_feature_pub.publish(self.rope_outer_markers)

            # Publish depth
            depth_msg = Float32()
            depth_msg.data = self.depth
            self.line_depth_pub.publish(depth_msg)

            if self.rope_inner_markers is not None:
                self.rope_marker_pub.publish(self.rope_inner_markers)
                self.rope_lines_pub.publish(self.rope_lines)

        elif not self.valid_markers:
            if self.verbose_map_publisher:
                self.get_logger().info(f"Invalid markers")
            self.construct_buoy_markers()  # this will try to find positions and depths
            # if None in [self.buoy_markers, self.rope_inner_markers]:  # old
            #     self.construct_buoy_markers()  # old

        elif not self.valid_depth:
            if self.verbose_map_publisher:
                self.get_logger().info(f"Invalid depths")
            self.find_rope_depth()

        else:
            self.get_logger().info("Is there a problem with the map publisher?")


def main(args=None):
    rclpy.init(args=args)
    # frame_names = ['Algae0_a0_buoy_Sphere_yellow (2)', 'Algae0_a0_buoy_Sphere_yellow (6)',
    #                'Algae0_a0_buoy_Sphere_yellow (5)', 'Algae0_a0_buoy_Sphere_yellow (3)']  # List your frames here
    # frame_names = ['Algae0']
    node = AlgaeMapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
