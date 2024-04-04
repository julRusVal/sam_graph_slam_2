import os
import launch
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


# TODO figure out how to specify the file paths as wrt the package
def generate_launch_description():
    param_config = os.path.join(
        get_package_share_directory('sam_graph_slam_2'),
        'config',
        'pipeline_config.yaml'
    )

    rviz_config = os.path.join(
        get_package_share_directory('sam_graph_slam_2'),
        'rviz',
        'rviz2_pipeline.rviz'
    )

    boring_way = '/home/julian/colcon_ws/src/sam_graph_slam_2/rviz/rviz2_pipeline.rviz'

    print("Launching pipeline")
    print(f"Rviz config: {rviz_config}")
    print(f"Parameter config: {param_config}")

    return launch.LaunchDescription([

        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', '/home/julian/bag_files/pipeline_sim'],
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config]
        ),
        Node(
            package='sam_graph_slam_2',
            executable='pipeline_dr_gt_publisher_node',
            name='pipeline_dr_gt_publisher_node',
            output='screen',
            parameters=[param_config]
        ),
        Node(
            package='sam_graph_slam_2',
            executable='pipeline_detector_node',
            name='pipeline_detector_node',
            output='screen',
            parameters=[param_config]
        )

    ])
