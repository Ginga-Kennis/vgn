import collections
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from vgn.detection import VGN
from vgn.grasp import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform
from vgn.sfs import VoxelSpace



MAX_CONSECUTIVE_FAILURES = 2

State = collections.namedtuple("State", ["tsdf", "pc"])

def get_distance(curr_pose,goal_pose):
        return 1
    
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

       
class Env:
    def __init__(self):
        self.sim = ClutterRemovalSim(scene="packed", object_set="packed/test", gui=True)
        self.VGN = VGN(model_path=Path("data/models/vgn_conv.pth"),rviz=False)
        self.voxel_space = VoxelSpace([0.0,0.3],[0.0,0.3],[0.0,0.3],[0.0075,0.0075,0.0075],np.array([[540, 0.0, 320],[0.0, 540, 240],[0.0, 0.0, 1.0]]),0.05)
        
        self.num_objects = 1
        
        # max steps per scene(オブジェクトの数で変わる？)
        self.max_steps = 50
        # distance threshold to be done
        self.dist_threshold = 1
        
        
    def reset(self,num_objects):
        self.num_objects = num_objects
        self.sim.reset(self.num_objects)
        self.num_steps = 0
        # initial pose ([0.15, -0.15, 0.6]) →　([0.15, 0.15, 0.05])
        self.curr_pose = [0.9689932, 0, 0, -0.2470875, 0.15, -0.15, 0.6]
        self.goal_pose = from_matrix(self.get_goalpose().as_matrix()) # calculate from VGN
        
        # distance of self.curr_pose - self.goal_pose
        self.distance = get_distance(self.curr_pose,self.goal_pose)
        
        self.done = False
        self.truncated = False
        
        state = [self.curr_pose,self.goal_pose]
        
        _, seg_image = self.get_image(self.curr_pose[:4], self.curr_pose[4:],1)
        self.voxel_space.sfs(seg_image,to_matrix(self.curr_pose[:4], self.curr_pose[4:]))
        _, seg_image = self.get_image(self.goal_pose[:4], self.goal_pose[4:],2)
        self.voxel_space.sfs(seg_image,to_matrix(self.goal_pose[:4], self.goal_pose[4:]))
                
    
        return state
        
    def get_goalpose(self):
        """
        Get goal pose from initital scene using VGN
        """
        tsdf, pc, _ = self.sim.acquire_tsdf(n=6, N=None)
        
        if pc.is_empty():
            self.reset(self.num_objects)
        
        # plan grasps
        state = State(tsdf, pc)
        grasps, _ , _ = self.VGN(state)
        
        if len(grasps) == 0:
            self.reset(self.num_objects)
        
        grasp = grasps[0]
        
        T_world_grasp = grasp.pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp
        
        return T_world_pregrasp
                  
    def step(self,action):
        self.num_steps += 1
        self.curr_pose = action
        
        prev_distance = self.distance
        self.distance = get_distance(self.curr_pose,self.goal_pose)
        
        state = [self.curr_pose, self.goal_pose]
        
        if self.distance < self.dist_threshold:
            self.done = True  # ゴールする or 物体に当たったら終わり？
        if self.num_steps > self.max_steps:
            self.truncated = True   
            
        reward = self.calc_reward(prev_distance,self.distance,self.done)
        
        return state, reward, self.done, self.truncated
    
    def calc_reward(self,prev_distance,curr_distance,done):
        pass
    
        
    # def get_image(self,extrinsic,n):
    #     # extrinsic = Transform.look_at(np.array([0.15, 0.0, 0.6]), np.array([0.15, 0.15, 0.05]) , np.array([0.0, 0.0, 1.0]) )
    #     # start_point = Transform(Rotation.from_quat([ 0.6830121, 0.6830121, -0.183015, 0.183015 ]), np.array([-0.15,-0.13,0.675]))
    #     return self.sim.render(extrinsic=extrinsic, image_number=n)
    
    def get_image(self,q,t,n):
        rgb_image, depth_image, seg_image = self.sim.camera2.get_image(q,t)
        image_name = f"./image/view{n}.png"
        seg_image_name = f"./seg_image/view{n}.png"
        cv2.imwrite(image_name,rgb_image)
        cv2.imwrite(seg_image_name,seg_image)
        return rgb_image, seg_image
        
        
        
        
if __name__ == "__main__":
    env = Env()
    
    for _ in range(10):
        num_objets = np.random.randint(1,5)
        env.reset(num_objets)