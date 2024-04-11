import rclpy
from rclpy import time
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tf2_ros
import tf2_geometry_msgs

class FrameMarkerPublisher(Node):
    def __init__(self, frame_names):
        super().__init__('frame_marker_publisher')
        self.frame_names = frame_names
        self.marker_pub = self.create_publisher(MarkerArray, 'frame_markers', 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0, self.publish_markers)

    def declare_node_params(self):
        """
        declare the parameters of the node.
        :return:
        """
        # TODO
        pass

    def publish_markers(self):
        marker_array = MarkerArray()
        for index, frame in enumerate(self.frame_names):
            try:
                trans = self.tf_buffer.lookup_transform('map', frame, rclpy.time.Time())
                self.get_logger().info(f'Success! transformed {frame} to map')
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.get_logger().info(f'Could not transform {frame} to map')
                continue

            marker = Marker()
            marker.header.frame_id = 'map'
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = trans.transform.translation.x
            marker.pose.position.y = trans.transform.translation.y
            marker.pose.position.z = trans.transform.translation.z
            marker.id = index

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    frame_names = ['Algae0_a0_buoy_Sphere_yellow (2)', 'Algae0_a0_buoy_Sphere_yellow (6)',
                   'Algae0_a0_buoy_Sphere_yellow (5)', 'Algae0_a0_buoy_Sphere_yellow (3)']  # List your frames here
    # frame_names = ['Algae0']
    node = FrameMarkerPublisher(frame_names)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
