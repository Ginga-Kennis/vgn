#!/usr/bin/env python

import cv_bridge
import numpy as np
import rospy
import sensor_msgs.msg
import time

from vgn import vis
from vgn.utils import ros_utils
from vgn.utils.transform import Rotation, Transform
from vgn.perception import *

# tag lies on the table in the center of the workspace
T_base_tag = Transform(Rotation.identity(), [0.1, 0.5, 0.1]) # identity transformation

class Ur5eGraspController(object):
    def __init__(self):
        self.size = rospy.get_param("/ur5e_grasp/size")
        self.base_frame_id = rospy.get_param("/ur5e_grasp/base_frame_id")
        self.tf_tree = ros_utils.TransformTree()
        self.define_workspace()
        self.tsdf_server = TSDFServer()

    def define_workspace(self):
        z_offset = -0.06
        t_tag_task = np.r_[[-0.5 * self.size, -0.5 * self.size, z_offset]]
        T_tag_task = Transform(Rotation.identity(), t_tag_task)
        self.T_base_task = T_base_tag * T_tag_task

        # robot base to task frame
        self.tf_tree.broadcast_static(self.T_base_task, self.base_frame_id, "task")
        rospy.sleep(1.0)  # wait for the TF to be broadcasted
    
    def run(self):
        time.sleep(10.0)
        vis.clear()
        vis.draw_workspace(self.size)
        # move to initial position
        tsdf, pc = self.acquire_tsdf()
        vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
        vis.draw_points(np.asarray(pc.points))
    
    def acquire_tsdf(self):
        self.tsdf_server.reset()
        self.tsdf_server.integrate = True
        print("integrate tsdf")
        # move to positions to get depth image
        time.sleep(1.0)

        self.tsdf_server.integrate = False
        tsdf = self.tsdf_server.low_res_tsdf
        pc = self.tsdf_server.high_res_tsdf.get_cloud()

        return tsdf, pc

class TSDFServer(object):
    def __init__(self):
        # LOAD PARAMETER
        self.cam_frame_id = rospy.get_param("/ur5e_grasp/cam/frame_id")
        self.cam_topic_name = rospy.get_param("/ur5e_grasp/cam/topic_name")
        self.intrinsic = CameraIntrinsic.from_dict(rospy.get_param("/ur5e_grasp/cam/intrinsic"))
        self.size = 0.3 #空間のサイズ? 30cm*30cm*30cm?

        self.cv_bridge = cv_bridge.CvBridge()
        self.tf_tree = ros_utils.TransformTree()
        self.integrate = False
        rospy.Subscriber(self.cam_topic_name, sensor_msgs.msg.Image, self.sensor_cb)

    def reset(self):
        self.low_res_tsdf = TSDFVolume(self.size, 40)
        self.high_res_tsdf = TSDFVolume(self.size, 120)

    def sensor_cb(self, msg):
        if not self.integrate:
            return
        
        img = self.cv_bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001
        T_cam_task = self.tf_tree.lookup(
            self.cam_frame_id, "task", msg.header.stamp, rospy.Duration(0.1)
        )
        # task_frameからcamera_frameまでの変換を表している


        self.low_res_tsdf.integrate(img, self.intrinsic, T_cam_task)
        self.high_res_tsdf.integrate(img, self.intrinsic, T_cam_task)





def main():
    rospy.init_node("ur5e_grasp")
    ur5e_grasp = Ur5eGraspController()

    while True:
        ur5e_grasp.run()


if __name__ == "__main__":
    main()
