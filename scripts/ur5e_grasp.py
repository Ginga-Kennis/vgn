#!/usr/bin/env python
import argparse
from pathlib import Path

import cv_bridge
import numpy as np
import geometry_msgs.msg
import rospy
import sensor_msgs.msg
import time

from vgn import vis
from vgn.experiments.clutter_removal import State
from vgn.detection import VGN
from vgn.utils import ros_utils
from vgn.utils.transform import Rotation, Transform
from vgn.perception import *
from vgn.utils.ur5e_control import Ur5eCommander
from vgn.utils.gripper_control import GripperController

# tag lies on the table in the center of the workspace
T_base_tag = Transform(Rotation.identity(), [0.1, 0.5, 0.1]) # identity transformation

class Ur5eGraspController(object):
    def __init__(self):
        self.size = rospy.get_param("/ur5e_grasp/size")
        self.finger_depth = rospy.get_param("/ur5e_grasp/finger_depth")
        self.base_frame_id = rospy.get_param("/ur5e_grasp/base_frame_id")
        self.scan_joints = rospy.get_param("/ur5e_grasp/scan_joints")
        self.T_tool0_tcp = Transform.from_dict(rospy.get_param("/ur5e_grasp/T_tool0_tcp"))
        self.T_tcp_tool0 = self.T_tool0_tcp.inverse()

        self.tf_tree = ros_utils.TransformTree()
        self.tsdf_server = TSDFServer()
        self.ur5e_commander = Ur5eCommander()
        # self.gripper_controller = GripperController()
        self.plan_grasps = VGN(args.model, rviz=True)

        self.define_workspace()
        self.create_planning_scene()

    

    def define_workspace(self):
        z_offset = -0.06
        t_tag_task = np.r_[[-0.5 * self.size, -0.5 * self.size, z_offset]]
        T_tag_task = Transform(Rotation.identity(), t_tag_task)
        self.T_base_task = T_base_tag * T_tag_task

        # robot base to task frame
        self.tf_tree.broadcast_static(self.T_base_task, self.base_frame_id, "task")
        rospy.sleep(1.0)  # wait for the TF to be broadcasted

    def create_planning_scene(self):
        # collision box for table
        table = geometry_msgs.msg.PoseStamped()
        table.header.frame_id = self.base_frame_id
        table.pose = ros_utils.to_pose_msg(T_base_tag)
        table.pose.position.z -= 0.03
        self.ur5e_commander.scene.add_box("table",table,size=(0.6, 0.6, 0.02))
    
    def run(self):
        time.sleep(10.0)
        vis.clear()
        vis.draw_workspace(self.size)
        self.ur5e_commander.goto_home()

        tsdf, pc = self.acquire_tsdf()
        vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
        vis.draw_points(np.asarray(pc.points))
        rospy.loginfo("Reconstructed scene")

        state = State(tsdf, pc)
        grasps, scores, planning_time = self.plan_grasps(state)
        vis.draw_grasps(grasps, scores, self.finger_depth)
        rospy.loginfo("Planned grasps")


        if len(grasps) == 0:
            rospy.loginfo("No grasps detected")
            return
        
        grasp, score = self.select_grasp(grasps, scores)
        vis.draw_grasp(grasp, score, self.finger_depth)
        rospy.loginfo("Selected grasp")

        self.ur5e_commander.goto_home()
        label = self.execute_grasp(grasp)
        rospy.loginfo("Grasp execution")
    
    def acquire_tsdf(self):
        self.ur5e_commander.goto_joints(self.scan_joints[0])

        self.tsdf_server.reset()
        self.tsdf_server.integrate = True

        for joint_target in self.scan_joints[1:]:
            self.ur5e_commander.goto_joints(joint_target)

        self.tsdf_server.integrate = False
        tsdf = self.tsdf_server.low_res_tsdf
        pc = self.tsdf_server.high_res_tsdf.get_cloud()

        return tsdf, pc
    
    def select_grasp(self, grasps, scores):
        heights = np.empty(len(grasps))
        for i,grasp in enumerate(grasps):
            heights[i] = grasp.pose.translation[2]
        idx = np.argmax(heights)
        grasp, score = grasps[idx], scores[idx]

        # make sure camera is pointing forward
        rot = grasp.pose.rotation
        axis = rot.as_matrix()[:, 0]
        if axis[0] < 0:
            rospy.loginfo("FLIPED GRASP")
            grasp.pose.rotation = rot * Rotation.from_euler("z", np.pi)

        return grasp, score
    
    def execute_grasp(self, grasp):
        T_task_grasp = grasp.pose
        T_base_grasp = self.T_base_task * T_task_grasp
        

        self.ur5e_commander.goto_pose(T_base_grasp * self.T_tcp_tool0)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    main()
