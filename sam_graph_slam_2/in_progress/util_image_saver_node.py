"""

"""

# class sam_image_saver:
#     def __init__(self, camera_down_top_name, camera_left_top_name, camera_right_top_name, buoy_top_name,
#                  file_path=None):
#         # ===== Set topic names ====== file paths for output =====
#         # Down
#         self.cam_down_image_topic = camera_down_top_name + '/image_color'
#         self.cam_down_info_topic = camera_down_top_name + '/camera_info'
#         # Left
#         self.cam_left_image_topic = camera_left_top_name + '/image_color'
#         self.cam_left_info_topic = camera_left_top_name + '/camera_info'
#         # Right
#         self.cam_right_image_topic = camera_right_top_name + '/image_color'
#         self.cam_right_info_topic = camera_right_top_name + '/camera_info'
#         # Buoys
#         self.buoy_topic = buoy_top_name
#
#         # ===== File paths for output =====
#         self.output_path = file_path
#         if self.output_path is None or not isinstance(file_path, str):
#             file_path_prefix = ''
#         else:
#             file_path_prefix = self.output_path + '/'
#
#         # Down
#         self.down_info_file_path = file_path_prefix + 'down_info.csv'
#         self.down_gt_file_path = file_path_prefix + 'down_gt.csv'
#         # Left
#         self.left_info_file_path = file_path_prefix + 'left_info.csv'
#         self.left_times_file_path = file_path_prefix + 'left_times.csv'
#         self.left_gt_file_path = file_path_prefix + 'left_gt.csv'
#         # Right
#         self.right_info_file_path = file_path_prefix + 'right_info.csv'
#         self.right_gt_file_path = file_path_prefix + 'right_gt.csv'
#         # Buoy
#         self.buoy_info_file_path = file_path_prefix + 'buoy_info.csv'
#
#         # ===== Frame and tf stuff =====
#         self.map_frame = 'map'
#         self.gt_frame_id = 'gt/sam/base_link'
#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
#         self.gt_topic = '/sam/sim/odom'
#
#         # ==== Local data =====
#         # Camera ground_truth and information
#         self.down_gt = []
#         self.down_info = []
#         self.down_times = []
#         self.left_gt = []
#         self.left_info = []
#         self.left_times = []
#         self.right_gt = []
#         self.right_info = []
#         self.left_times = []
#         self.gt_poses_from_topic = []
#         self.buoys = []
#
#         # ===== Image processing =====
#         # self.bridge = CvBridge()  # Not had problems
#
#         # ===== States =====
#         self.last_time = self.get_clock().now()
#         self.tf_ready = False
#         self.gt_updated = False
#         self.buoys_received = False
#         self.image_received = False
#         self.data_written = False
#
#         # ===== Subscriptions =====
#         # Down camera
#         self.cam_down_image_subscriber = rospy.Subscriber(self.cam_down_image_topic,
#                                                           Image,
#                                                           self.image_callback,
#                                                           'down')
#
#         self.cam_down_info_subscriber = rospy.Subscriber(self.cam_down_info_topic,
#                                                          CameraInfo,
#                                                          self.info_callback,
#                                                          'down')
#
#         # Left camera
#         self.cam_left_image_subscriber = rospy.Subscriber(self.cam_left_image_topic,
#                                                           Image,
#                                                           self.image_callback,
#                                                           'left')
#
#         self.cam_left_info_subscriber = rospy.Subscriber(self.cam_left_info_topic,
#                                                          CameraInfo,
#                                                          self.info_callback,
#                                                          'left')
#
#         # Right camera
#         self.cam_right_image_subscriber = rospy.Subscriber(self.cam_right_image_topic,
#                                                            Image,
#                                                            self.image_callback,
#                                                            'right')
#
#         self.cam_right_info_subscriber = rospy.Subscriber(self.cam_right_info_topic,
#                                                           CameraInfo,
#                                                           self.info_callback,
#                                                           'right')
#
#         # Ground truth
#         self.gt_subscriber = rospy.Subscriber(self.gt_topic,
#                                               Odometry,
#                                               self.gt_callback)
#
#         # Buoys
#         self.buoy_subscriber = rospy.Subscriber(self.buoy_topic,
#                                                 MarkerArray,
#                                                 self.buoy_callback)
#
#         # ===== Timers =====
#         self.time_check = rospy.Timer(rospy.Duration(1),
#                                       self.time_check_callback)
#
#     # ===== Callbacks =====
#     def old_down_image_callback(self, msg, camera_id):
#
#         print(camera_id)
#         # Record gt
#         current, _ = self.get_gt_trans_in_map()
#         current.append(msg.header.seq)
#         self.down_gt.append(current)
#
#         # TODO get cvbridge working
#         # Convert to cv format
#         # try:
#         #     # Convert your ROS Image message to OpenCV2
#         #     cv2_img = self.bridge.imgmsg_to_cv2(msg)  # "bgr8"
#         # except CvBridgeError:
#         #     print('CvBridge Error')
#         # else:
#         #     # Save your OpenCV2 image as a jpeg
#         #     # time = msg.header.stamp  # cast as string to use in name
#         #     if self.output_path is None or not isinstance(self.output_path, str):
#         #         save_path = f'{msg.header.seq}.jpg'
#         #     else:
#         #         save_path = self.output_path + f'/d:{msg.header.seq}.jpg'
#         #     cv2.imwrite(save_path, cv2_img)
#
#         # Convert with home-brewed conversion
#         # https://answers.ros.org/question/350904/cv_bridge-throws-boost-import-error-in-python-3-and-ros-melodic/
#         cv2_img = self.imgmsg_to_cv2(msg)
#
#         # Write to 'disk'
#         if self.output_path is None or not isinstance(self.output_path, str):
#             save_path = f'd_{msg.header.seq}.jpg'
#         else:
#             save_path = self.output_path + f'/down/d_{msg.header.seq}.jpg'
#         cv2.imwrite(save_path, cv2_img)
#
#         # Update state and timer
#         self.image_received = True
#         self.last_time = self.get_clock().now()
#
#         return
#
#     def info_callback(self, msg, camera_id):
#         if camera_id == 'down':
#             if len(self.down_info) == 0:
#                 self.down_info.append(msg.K)
#                 self.down_info.append(msg.P)
#                 self.down_info.append([msg.width, msg.height])
#         elif camera_id == 'left':
#             if len(self.left_info) == 0:
#                 self.left_info.append(msg.K)
#                 self.left_info.append(msg.P)
#                 self.left_info.append([msg.width, msg.height])
#         elif camera_id == 'right':
#             if len(self.right_info) == 0:
#                 self.right_info.append(msg.K)
#                 self.right_info.append(msg.P)
#                 self.right_info.append([msg.width, msg.height])
#         else:
#             print('Unknown camera_id passed to info callback')
#
#     def image_callback(self, msg, camera_id):
#         """
#         Based on down_image_callback
#         """
#
#         now_stamp = self.get_clock().now()
#         msg_stamp = msg.header.stamp
#
#         # record gt
#         current_id = msg.header.seq
#
#         # ===== Pick one method =====
#         # Method 1 - frame transform
#         # current_pose, pose_stamp = self.get_gt_trans_in_map(gt_time=msg.header.stamp)
#         # pose_time = pose_stamp.to_sec()
#         # current.append(current_id)
#
#         # Method 2 - subscription method of gt
#         current_pose_and_time = self.gt_poses_from_topic[-1]
#         current = current_pose_and_time[0:-1]
#         pose_time = current_pose_and_time[-1]
#         current.append(current_id)
#
#         # Record data
#         if camera_id == 'down':
#             # Record the ground truth and times
#             self.left_gt.append(current)
#             self.left_times.append([now_stamp.to_sec(),
#                                     msg_stamp.to_sec(),
#                                     pose_time])
#         elif camera_id == 'left':
#             # Record the ground truth and times
#             self.left_gt.append(current)
#             self.left_times.append([now_stamp.to_sec(),
#                                     msg_stamp.to_sec(),
#                                     pose_time])
#         elif camera_id == 'right':
#             # Record the ground truth and times
#             self.left_gt.append(current)
#             self.left_times.append([now_stamp.to_sec(),
#                                     msg_stamp.to_sec(),
#                                     pose_time])
#         else:
#             print('Unknown camera_id passed to image callback')
#             return
#
#         # Display call back info
#         print(f'image callback - {camera_id}: {current_id}')
#         print(current)
#
#         # Convert to cv2 format
#         cv2_img = imgmsg_to_cv2(msg)
#
#         # Write to 'disk'
#         if self.output_path is None or not isinstance(self.output_path, str):
#             save_path = f'{camera_id}_{current_id}.jpg'
#         else:
#             save_path = self.output_path + f'/{camera_id}/{current_id}.jpg'
#         cv2.imwrite(save_path, cv2_img)
#
#         # Update state and timer
#         self.image_received = True
#         self.last_time = self.get_clock().now()
#
#         return
#
#     def gt_callback(self, msg):
#         """
#         Call back for the ground truth subscription, msg is of type nav_msgs/Odometry.
#         The data is saved in a list w/ format [x, y, z, q_w, q_x, q_y, q_z].
#         Note the position of q_w, this is for compatibility with gtsam and matlab
#         """
#         transformed_pose = self.transform_pose(msg.pose, from_frame=msg.header.frame_id, to_frame=self.map_frame,
#                                                req_transform_time=None)
#
#         gt_position = transformed_pose.pose.position
#         gt_quaternion = transformed_pose.pose.orientation
#         gt_time = transformed_pose.header.stamp.to_sec()
#
#         self.gt_poses_from_topic.append([gt_position.x, gt_position.y, gt_position.z,
#                                          gt_quaternion.w, gt_quaternion.x, gt_quaternion.y, gt_quaternion.z,
#                                          gt_time])
#
#         self.gt_updated = True
#
#     def buoy_callback(self, msg):
#         if not self.buoys_received:
#             for marker in msg.markers:
#                 self.buoys.append([marker.pose.position.x,
#                                    marker.pose.position.y,
#                                    marker.pose.position.z])
#
#             self.buoys_received = True
#
#     # Timer
#     def time_check_callback(self, event):
#         if not self.image_received:
#             return
#         delta_t = (self.get_clock().now() - self.last_time)
#         if delta_t.to_sec() >= 5 and not self.data_written:
#             print('Data written')
#             self.write_data()
#             self.data_written = True
#
#         return
#
#     # ===== Transforms =====
#     def transform_pose(self, pose, from_frame, to_frame, req_transform_time=None):
#         trans = self.wait_for_transform(from_frame=from_frame,
#                                         to_frame=to_frame,
#                                         req_transform_time=req_transform_time)
#
#         pose_transformed = tf2_geometry_msgs.do_transform_pose(pose, trans)
#
#         return pose_transformed
#
#     def wait_for_transform(self, from_frame, to_frame, req_transform_time=None):
#         """Wait for transform from from_frame to to_frame"""
#         trans = None
#
#         if isinstance(req_transform_time, Time):
#             transform_time = req_transform_time
#         else:
#             transform_time = rospy.Time()
#
#         while trans is None:
#             try:
#                 trans = self.tf_buffer.lookup_transform(to_frame,
#                                                         from_frame,
#                                                         transform_time,
#                                                         rospy.Duration(1))
#
#             except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
#                     tf2_ros.ExtrapolationException) as error:
#                 print('Failed to transform. Error: {}'.format(error))
#
#         return trans
#
#     def get_gt_trans_in_map(self, gt_time=None):
#         """
#         Finds pose of the ground truth.
#         First,the transform between the map and the ground truth frame.
#         Second, the transform is applied to a null_pose located at the origin.
#         Modifying the orientation of this pose might be need to prevent later
#         processing on the ground truth
#         Returns [ x, y, z, q_w, q_x, q_y, q_z]
#         """
#
#         trans = self.wait_for_transform(from_frame=self.gt_frame_id,
#                                         to_frame=self.map_frame,
#                                         req_transform_time=gt_time)
#
#         null_pose = PoseStamped()
#         null_pose.pose.orientation.w = 1.0
#         pose_in_map = tf2_geometry_msgs.do_transform_pose(null_pose, trans)
#
#         pose_stamp = pose_in_map.header.stamp
#
#         pose_list = [pose_in_map.pose.position.x,
#                      pose_in_map.pose.position.y,
#                      pose_in_map.pose.position.z,
#                      pose_in_map.pose.orientation.w,
#                      pose_in_map.pose.orientation.x,
#                      pose_in_map.pose.orientation.y,
#                      pose_in_map.pose.orientation.z]
#
#         return pose_list, pose_stamp
#
#     # ===== Utilities =====
#     def write_data(self):
#         """
#         Save all the relevant data
#         """
#         # Down
#         write_array_to_csv(self.down_info_file_path, self.down_info)
#         write_array_to_csv(self.down_gt_file_path, self.down_gt)
#         # Left
#         write_array_to_csv(self.left_info_file_path, self.left_info)
#         write_array_to_csv(self.left_times_file_path, self.left_times)
#         write_array_to_csv(self.left_gt_file_path, self.left_gt)
#         # Right
#         write_array_to_csv(self.right_info_file_path, self.right_info)
#         write_array_to_csv(self.right_gt_file_path, self.right_gt)
#         # Buoy
#         write_array_to_csv(self.buoy_info_file_path, self.buoys)
#
#         return