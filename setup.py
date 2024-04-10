from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'sam_graph_slam_2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='julian',
    maintainer_email='julian@todo.todo',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pipeline_dr_gt_publisher_node = sam_graph_slam_2.pipeline_slam_gt_dr_publisher:main',
            'pipeline_detector_node = sam_graph_slam_2.pipeline_point_cloud_detector:main',
            'sam_slam_node = sam_graph_slam_2.sam_slam_node:main',
            'pipeline_map_node = sam_graph_slam_2.pipeline_map_publisher:main'
        ],
    },
)
