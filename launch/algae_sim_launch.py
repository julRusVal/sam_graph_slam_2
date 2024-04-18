import os
import launch
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():
    param_config = os.path.join(
        get_package_share_directory('sam_graph_slam_2'),
        'config',
        'algae_config.yaml'
    )

    rviz_config = os.path.join(
        get_package_share_directory('sam_graph_slam_2'),
        'rviz',
        'rviz2_pipeline.rviz'
    )

    robot_name = "sam0"

    print("Launching pipeline")
    print(f"Rviz config: {rviz_config}")
    print(f"Parameter config: {param_config}")

    return launch.LaunchDescription([

        # Replace with the real DR once ported
        # NOTE: node renamed
        Node(
            package='sam_graph_slam_2',
            executable='pipeline_dr_gt_publisher_node',
            namespace=robot_name,
            name='algae_dr_gt_publisher_node',
            output='screen',
            parameters=[param_config]
        ),
        # Depth
        Node(
            package="sam_dead_reckoning",
            executable="depth_node",
            namespace=robot_name,
            name="depth_node",
            output="screen",
            parameters=[{
                "robot_name": robot_name,
                "simulation": True
            }]
        ),
        Node(
            package="sam_graph_slam_2",
            executable="algae_map_publisher_node",
            namespace=robot_name,
            name="algae_map_publisher_node",
            output="screen",
            parameters=[param_config]
        ),
        Node(
            package="sam_graph_slam_2",
            executable="algae_sss_detector",
            namespace=robot_name,
            name="algae_sss_detector",
            output="screen",
            parameters=[param_config]
        )


    ])
