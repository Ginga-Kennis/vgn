import collections
import numpy as np
from pathlib import Path

from vgn.detection import VGN
from vgn.grasp import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform
from vgn.sfs import VoxelSpace

MAX_CONSECUTIVE_FAILURES = 2

State = collections.namedtuple("State", ["tsdf", "pc"])
          
class Env:
    def __init__(self):
        self.sim = ClutterRemovalSim(scene="packed", object_set="packed/train", gui=True)
        self.VGN = VGN(model_path=Path("data/models/vgn_conv.pth"),rviz=False)
        self.voxel_space = VoxelSpace(x_range=[0.0,0.3], y_range=[0.0,0.3], z_range=[0.0,0.3], 
                                      voxel_size=[0.75,0.75,0.75],
                                      K=self.sim.camera.intrinsic,focal_length=0.00188)
        
        # initialize
        self.num_steps = 0
        self.curr_pose = np.array([[0.15, 0.0, 0.6]])  # initial camera extrinsic
            
            
    """
    action : extrinsic of camera
    """        
    def step(self,action):
        self.num_steps += 1
        self.curr_ = action
        
        # get image & SfS
        seg_image = self.get_image(extrinsic=self.curr_pose,n=self.num_steps)
        self.voxel_space.sfs(image=seg_image,extrinsic=self.curr_pose)
        
        state = [self.voxel_space, self.curr_pose, self.goal_pos]
        reward = self.calc_reward(state)
        terminated = False  # ゴールする or 物体に当たったら終わり？
        truncated = False   # self.num_steps > MAX_STEPS
        
        return state, reward, terminated, truncated 
        
    
    def reset(self,num_objects):
        self.sim.reset(num_objects)
        
        self.goal_pose = self.get_goalpose()
        
        # get initial image
        seg_image = self.get_image(self.curr_pose,1)
        self.voxel_space.sfs(image=seg_image,extrinsic=self.curr_pose)
        
        state = [self.voxel_space, self.curr_pose, self.goal]
        
        return state
        
    def get_goalpose(self):
        # scan scene
        tsdf, pc, _ = self.sim.acquire_tsdf(n=6, N=None)
        
        if pc.is_empty():
            self.reset()
        
        # plan grasps
        state = State(tsdf, pc)
        grasps, _ , _ = self.VGN(state)
        
        if len(grasps) == 0:
            self.reset()
        
        return grasps[0]
        
    def get_image(self,extrinsic,n):
        # extrinsic = Transform.look_at(curr_pose, target_pos, np.array([0.0, 0.0, 1.0]) )
        # start_point = Transform(Rotation.from_quat([ 0.6830121, 0.6830121, -0.183015, 0.183015 ]), np.array([-0.15,-0.13,0.675]))
        return self.sim.render(extrinsic=extrinsic, image_number=n)
        
        
if __name__ == "__main__":
    env = Env()
    
    for _ in range(1):
        env.reset(2)