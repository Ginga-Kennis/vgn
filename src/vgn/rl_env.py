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

def get_distance(curr_pose,goal_pose):
        return 1


          
class Env:
    def __init__(self):
        self.sim = ClutterRemovalSim(scene="packed", object_set="packed/train", gui=True)
        self.VGN = VGN(model_path=Path("data/models/vgn_conv.pth"),rviz=False)
        
        # max steps per scene(オブジェクトの数で変わる？)
        self.max_steps = 50
        # distance threshold to be done
        self.dist_threshold = 1
        
    def reset(self,num_objects):
        self.sim.reset(num_objects)
        self.num_steps = 0
        self.curr_pose = np.array([[ 1.,         -0.        ,  0.        , -0.15      ],
                                   [ 0.,         -0.96476382, -0.26311741,  0.15787044],
                                   [ 0.,          0.26311741, -0.96476382,  0.57885829],
                                   [ 0.,          0.        ,  0.        ,  1.        ]])  

        self.goal_pose = self.get_goalpose().pose.as_matrix() # calculate from VGN
        
        # distance of self.curr_pose - self.goal_pose
        self.distance = get_distance(self.curr_pose,self.goal_pose)
        
        self.done = False
        self.truncated = False
        
        state = [self.curr_pose,self.goal_pose]
        return state
        
    def get_goalpose(self):
        """
        Get goal pose from initital scene using VGN
        """
        tsdf, pc, _ = self.sim.acquire_tsdf(n=6, N=None)
        
        if pc.is_empty():
            self.reset()
        
        # plan grasps
        state = State(tsdf, pc)
        grasps, _ , _ = self.VGN(state)
        
        if len(grasps) == 0:
            self.reset()
        
        return grasps[0]
                  
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
        
        
if __name__ == "__main__":
    env = Env()
    
    for _ in range(5):
        num_objets = np.random.randint(1,5)
        env.reset(num_objets)