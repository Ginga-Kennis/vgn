import collections
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from pathlib import Path

from vgn.detection import VGN
from vgn.grasp import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform
from vgn.sfs import VoxelSpace
from vgn.rl_utils import *

State = collections.namedtuple("State", ["tsdf", "pc"])

# ENVIRONMENT PARAMS
NUM_OBJECTS = 2
INITIAL_POSE = [1, 0, 0, 0, 0.15, 0.15, 0.6]
MAX_STEPS = 30
GOAL_THRESHOLD = 0.03
CAMERA_RADIUS = 0.1
ACTION_QUAT_SACLE = 0.05
ACTION_TRANS_SACLE = 0.05
PREGRASP_X = 0.0
PREGRASP_Z = -0.15

# VOXEL SPACE PARAMS
X_RANGE = [0.0,0.3]
Y_RANGE = [0.0,0.3]
Z_RANGE = [0.0,0.3]
VOXEL_SIZE = [0.0075,0.0075,0.0075]
K = [[540, 0.0, 320],[0.0, 540, 240],[0.0, 0.0, 1.0]]
NEAR = 0.05
TABLE_HEIGHT = 0.05




class Env(gym.Env):
    def __init__(self):
        # Initialize params
        self.max_steps = MAX_STEPS
        self.goal_threshold = GOAL_THRESHOLD 
        self.r = CAMERA_RADIUS   
    
        # Initialize (Simulation, VoxelSpace,VGN)
        self.sim = ClutterRemovalSim(scene="packed", object_set="packed/train", gui=False)
        self.voxel_space = VoxelSpace(X_RANGE,Y_RANGE,Z_RANGE,VOXEL_SIZE,K,NEAR,TABLE_HEIGHT)
        self.vgn = VGN(model_path=Path("data/models/vgn_conv.pth"),rviz=False)

        self.observation_space = spaces.Dict(
            {
                "s3d" : spaces.Box(low=0, high=255,shape=(40,40,40), dtype=np.uint8),
                "pose" : spaces.Box(low=-2, high=2,shape=(14,),dtype=np.float32)
            }
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)


    def _get_obs(self):
        pose = np.concatenate((self.curr_pose,self.goal_pose), axis=0)
        return {"s3d" : self.s3d.astype(np.uint8), "pose" : pose.astype(np.float32)}
    
    def _get_curr_pose(self,action):
        action[:4] *= ACTION_QUAT_SACLE     # scale quat
        action[4:] *= ACTION_TRANS_SACLE     # scale trans
        self.curr_pose = self.curr_pose + action
        self.curr_pose[:4] /= np.linalg.norm(self.curr_pose[:4])
    
    def _get_info(self):
        return {"distance" : self.curr_distance, "num_points" : self.curr_num_points, "collision" : self.collision}
        
        
    def reset(self,seed=None, options=None):
        super().reset(seed=seed)
        # reset num_steps
        self.num_steps = 0

        # 1 : Reset (Simulation, VoxelSpace)
        self.sim.reset(NUM_OBJECTS)
        self.voxel_space.reset()

        # 2 : get goal pose
        self.goal_pose = self.get_goalpose()
        if self.goal_pose == False:
            print("********* NO GRASP DETECTED *************")
            return False
        else:
            self.goal_pose = np.array(from_matrix(self.goal_pose.as_matrix()))

        # 3 : set curr_pose to Initial pose
        self.curr_pose = np.array(INITIAL_POSE)


        # 4 : SfS
        self.sfs(self.curr_pose[:4],self.curr_pose[4:],self.num_steps)

        # 5 : set params
        self.prev_num_points = 0
        self.curr_num_points = self.voxel_space.num_points
        self.prev_distance = 0
        self.curr_distance = calc_distance(self.curr_pose,self.goal_pose)
        self.pointcloud = self.voxel_space.pointcloud
        self.s3d = self.voxel_space.s3d
        self.collision = False
        self.goal = False
        self.done = False
        self.truncated = False

        # 6 : get state & info
        state = self._get_obs()
        info = self._get_info()
        print(f"RESET : {info}")
        return state, info

        
                  
    def step(self,action):
        # 1 : increment num_steps
        self.num_steps += 1

        # 2 : set current_pose (curr_pose + action)
        self._get_curr_pose(action)

        # 3 : SfS
        self.sfs(self.curr_pose[:4],self.curr_pose[4:],self.num_steps)

        # 4 : set parameters
        self.prev_num_points = self.curr_num_points
        self.curr_num_points = self.voxel_space.num_points
        self.prev_distance = self.curr_distance
        self.curr_distance = calc_distance(self.curr_pose,self.goal_pose)
        self.pointcloud = self.voxel_space.pointcloud
        self.s3d = self.voxel_space.s3d
        self.collision = check_collision(self.pointcloud,self.curr_num_points,self.curr_pose[4:],self.r)
        self.goal = self.curr_distance <= self.goal_threshold
        self.done = self.collision or self.goal
        self.truncated = self.num_steps > self.max_steps
        # visualize_pcd(self.pointcloud)

        # 5 : calculate reward
        reward = self.calc_reward()

        state = self._get_obs()
        info = self._get_info()
        print(f"STEP : {info}")
        return state, reward, self.done, self.truncated, info
    
    

    def get_goalpose(self):
        """
        Get goal pose from initital scene using VGN
        """
        tsdf, pc, _ = self.sim.acquire_tsdf(n=6, N=None)
        
        if pc.is_empty():
            return False
        
        # plan grasps
        state = State(tsdf, pc)
        grasps, _ , _ = self.vgn(state)
        
        if len(grasps) == 0:
            return False

        grasp = grasps[0]
        
        T_world_grasp = grasp.pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [PREGRASP_X, 0.00, PREGRASP_Z])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp
        
        return T_world_pregrasp
    
    def sfs(self,q,t,n):
        rgb_image, _, _ = self.sim.camera2.get_image(q,t)
        seg_image = get_segimage(rgb_image,n,save_image=True)
        self.voxel_space.sfs(seg_image,to_matrix(q, t))

    def calc_reward(self):
        # GOAL : reward = 10
        if self.goal == True:
            rw = 5
        
        # COLLISION : reward = -10
        elif self.collision == True:
            rw = -5

        # NORMAL : 
        else:
            rw = ((self.prev_distance - self.curr_distance) / self.prev_distance) + ((self.prev_num_points - self.curr_num_points) / self.prev_num_points)
        return rw


    
if __name__ == "__main__":
    env = Env()
    check_env(env)
