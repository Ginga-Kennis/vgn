import collections
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from pathlib import Path
from pyquaternion import Quaternion

from vgn.detection import VGN
from vgn.grasp import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform

from sfs import VoxelSpace
from utils import *

State = collections.namedtuple("State", ["tsdf", "pc"])

# ENVIRONMENT PARAMS
NUM_OBJECTS = 1
INITIAL_POSE = [-3.1415, 0, 0, 0.15, 0.15, 0.6]  # Top down view
MAX_STEPS = 15
INIT_GOAL_THRESHOLD = 0.1
END_GOAL_THRESHOLD = 0.01
COLLISION_RADIUS = 0.05    # 5cm
ACTION_ORI_SACLE = 0.262  # 15度
ACTION_TRANS_SACLE = 0.05 # 5cm
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
        self.goal_threshold = INIT_GOAL_THRESHOLD 
        self.r = COLLISION_RADIUS 

         
    
        # Initialize (Simulation, VoxelSpace,VGN)
        self.sim = ClutterRemovalSim(scene="packed", object_set="rl", gui=True)
        self.voxel_space = VoxelSpace(X_RANGE,Y_RANGE,Z_RANGE,VOXEL_SIZE,K,NEAR,TABLE_HEIGHT)
        self.vgn = VGN(model_path=Path("data/models/vgn_conv.pth"),rviz=False)

        self.observation_space = spaces.Dict(
            {
                "s3d" : spaces.Box(low=0, high=255,shape=(1,40,40,40), dtype=np.uint8),
                "pose" : spaces.Box(low=-5, high=5,shape=(12,),dtype=np.float32)
            }
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)


    def _get_obs(self):
        pose = np.concatenate((self.curr_pose.eulerxyz,self.goal_pose.eulerxyz), axis=0)
        return {"s3d" : self.s3d.astype(np.uint8), "pose" : pose.astype(np.float32)}
    
    def _get_curr_pose(self,action):
        action = np.clip(action,-1.0,1.0)
        action[:3] *= ACTION_ORI_SACLE
        action[3:] *= ACTION_TRANS_SACLE

        self.curr_pose.update(action)
    
    def _get_info(self):
        return {"p_dist" : self.pos_distance, "o_dist" : self.quat_distance, "distance" : self.distance , "threshold" : self.goal_threshold,"num_points" : self.curr_num_points ,"collision" : self.collision, "goal" : self.goal}
        
        
    def reset(self,seed=None, options=None):
        super().reset(seed=seed)
        # reset num_steps
        self.num_steps = 0
        if self.goal_threshold > END_GOAL_THRESHOLD:
            self.goal_threshold += (END_GOAL_THRESHOLD - INIT_GOAL_THRESHOLD) / 30000

        # reset till find valid grasp pose
        while True:
            # 1 : Reset (Simulation, VoxelSpace)
            num_objects = np.random.randint(1,5)
            self.sim.reset(num_objects)
            self.voxel_space.reset()

            # 2 : get goal pose
            goal_pose = self.get_goalpose()

            if goal_pose != False:
                goal_pose = np.array(from_matrix(goal_pose.as_matrix()))
                self.goal_pose = Pose(goal_pose[:4],goal_pose[4:])
                break



        # 3 : set curr_pose to Initial pose
        self.curr_pose = Pose(np.array(INITIAL_POSE)[:3],np.array(INITIAL_POSE)[3:])


        # 4 : SfS
        self.sfs(self.curr_pose.quatxyz[:4],self.curr_pose.quatxyz[4:],self.num_steps)

        # 5 : set params
        self.init_num_points = self.voxel_space.num_points
        self.curr_num_points = self.voxel_space.num_points
        self.prev_pos_distance = 0
        self.pos_distance,self.quat_distance, self.distance = calc_distance(self.curr_pose.quatxyz,self.goal_pose.quatxyz)
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
        self.sfs(self.curr_pose.quatxyz[:4],self.curr_pose.quatxyz[4:],self.num_steps)

        # 4 : set parameters
        self.curr_num_points = self.voxel_space.num_points
        self.pos_distance,self.quat_distance, self.distance = calc_distance(self.curr_pose.quatxyz,self.goal_pose.quatxyz)
        self.pointcloud = self.voxel_space.pointcloud
        self.s3d = self.voxel_space.s3d
        self.collision = check_collision(self.pointcloud,self.curr_num_points,self.curr_pose.p,self.r)
        self.goal = self.distance <= self.goal_threshold
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
        seg_image = get_segimage(rgb_image,n,save_image=False)
        self.voxel_space.sfs(seg_image,to_matrix(q, t))

    def calc_reward(self):
        if self.goal == True:
            rw = 1
        elif self.truncated == True:
            rw = -0.4
        elif self.collision == True:
            rw = -0.4
        else:
            rw = -self.distance - (self.curr_num_points/self.init_num_points)*0.5
        print(rw)
        return rw



    
if __name__ == "__main__":
    env = Env()
    check_env(env)
