import collections
import numpy as np
import cv2
import math
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

from vgn.detection import VGN
from vgn.grasp import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform
from vgn.sfs import VoxelSpace

State = collections.namedtuple("State", ["tsdf", "pc"])
    
def to_matrix(q,t):
    r = np.array(R.from_quat(q).as_matrix())
    t = np.array(t)
    matrix = np.hstack((r,t.reshape(-1,1)))
    matrix = np.vstack((matrix,np.array([0,0,0,1])))
    
    return matrix

def from_matrix(matrix):
    q = R.from_matrix(matrix[:3,:3]).as_quat().tolist()
    t = matrix[:3,3].tolist()
    return q+t

def get_segimage(image,n):
    # rgb → gray
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 反転 
    image = cv2.bitwise_not(image)

    # save segmented image
    seg_image_name = f"./seg_image/view{n}.png"
    cv2.imwrite(seg_image_name,image)

    # 2値化
    threshold = 240
    ret, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return image

def calc_distance(pose1,pose2):
    p1 = pose1[4:]
    p2 = pose2[4:]
    q1 = Quaternion(np.array(pose1[:4]))
    q2 = Quaternion(np.array(pose2[:4]))
    q_dist = Quaternion.absolute_distance(q1,q2)
    p_dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
    return p_dist + q_dist



       
class Env:
    def __init__(self):
        self.sim = ClutterRemovalSim(scene="packed", object_set="packed/train", gui=False)
        self.vgn = VGN(model_path=Path("data/models/vgn_conv.pth"),rviz=False)
        self.voxel_space = VoxelSpace([0.0,0.3],[0.0,0.3],[0.0,0.3],[0.003,0.003,0.003],np.array([[540, 0.0, 320],[0.0, 540, 240],[0.0, 0.0, 1.0]]),0.05)
        
        self.distance_threshold = 0.1
        
    def reset(self,num_objects):
        self.num_objects = num_objects # number of objects in simulation
        self.sim.reset(self.num_objects)
        self.voxel_space.reset()

        self.num_steps = 0
        self.curr_pose = [0.9689932, 0, 0, -0.2470875, 0.15, -0.15, 0.6]  # initial pose ([0.15, -0.15, 0.6]) →　([0.15, 0.15, 0.05])
        self.sfs(self.curr_pose[:4],self.curr_pose[4:],0)

        self.done = False

        self.goal_pose = self.get_goalpose() # calculate from VGN
        if self.goal_pose == False:
            print("********* NO GRASP DETECTED *************")
            return False
        else:
            self.goal_pose = from_matrix(self.goal_pose.as_matrix())
            self.distance = calc_distance(self.curr_pose,self.goal_pose)
            self.num_points = self.voxel_space.num_points
            return 
        
                  
    def step(self,action):
        self.num_steps += 1
        self.curr_pose = action
        self.sfs(self.curr_pose[:4],self.curr_pose[4:],self.num_steps)

        distance = calc_distance(self.curr_pose,self.goal_pose)
        num_points = self.voxel_space.num_points
        reward = self.calc_reward(distance,num_points)
        
        if distance < self.distance_threshold:
            self.done = True

        return reward ,self.done
    
    def last_step(self):
        self.num_steps += 1
        self.curr_pose = self.goal_pose
        self.sfs(self.curr_pose[:4],self.curr_pose[4:],self.num_steps)
        self.voxel_space.visualize_pcd()

        distance = calc_distance(self.curr_pose,self.goal_pose)
        num_points = self.voxel_space.num_points
        
        reward = self.calc_reward(distance,num_points)
        if distance < self.distance_threshold:
            self.done = True

        return reward ,self.done
    

    def get_goalpose(self):
        """
        Get goal pose from initital scene using VGN
        """
        tsdf, pc, _ = self.sim.acquire_tsdf(n=6, N=None)
        
        if pc.is_empty():
            self.reset(self.num_objects)
            return False
        
        # plan grasps
        state = State(tsdf, pc)
        grasps, _ , _ = self.vgn(state)
        
        if len(grasps) == 0:
            return False

        grasp = grasps[0]
        
        T_world_grasp = grasp.pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp
        
        return T_world_pregrasp
    
    def sfs(self,q,t,n):
        rgb_image, _, _ = self.sim.camera2.get_image(q,t)
        # save rgb image
        image_name = f"./image/view{n}.png"
        cv2.imwrite(image_name,rgb_image)

        # convert rgb → segmented image
        seg_image = get_segimage(rgb_image,n)
        self.voxel_space.sfs(seg_image,to_matrix(q, t))

    def calc_reward(self,distance,num_points):
        rw1 = distance / self.distance
        rw2 = num_objets / self.num_points
        rw = rw1 + rw2

        self.distance = distance
        self.num_points = num_points
        return rw


    
    
        
        
        
        
if __name__ == "__main__":
    env = Env()
    
    for _ in range(100):
        print("-------------------------------")
        num_objets = 3
        state = env.reset(num_objets)
        print(env.step([0.682982, 0.6830478, -0.1830045, -0.1830045, 0.45, 0.15, 0.51961524]))
        print(env.step([0.2499775, 0.9330252, -0.2499775, -0.0669813, 0.3, 0.41, 0.51961524]))
        print(env.step([-0.2499999999983246, 0.933012701893705, -0.2499999999983246, 0.06698729809959345, 6.40987562e-17, 0.409807621, 0.519615242]))
        print(env.step([-0.6830127018975046, 0.6830127018975047, -0.18301270187249408, 0.18301270187249408, -0.15, 0.15, 0.519615242]))
        print(env.step([0.933012701893705, -0.2499999999983246, 0.06698729809959346, -0.2499999999983246, -1.04160479e-16, -0.109807621, 0.519615242]))
        print(env.step([0.9330127018661368, 0.2500000000294135, -0.06698729825151725, -0.2500000000294135, 0.3, -0.10980762, 0.51961524]))
        print(env.last_step()                                                                                                                          )
                       