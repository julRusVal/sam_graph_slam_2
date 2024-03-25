import launch
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


# TODO figure out how to specify the file paths as wrt the package
def generate_launch_description():
    return launch.LaunchDescription([
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', '/home/julian/bag_files/pipeline_sim'],
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', '/home/julian/colcon_ws/src/sam_graph_slam_2/rviz/rviz2_pipeline.rviz']
        ),
        Node(
            package='sam_graph_slam_2',
            executable='pipeline_dr_gt_publisher_node',
            name='pipeline_dr_gt_publisher_node',
            output='screen'
        )

    ])
