import rospy
import collections
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from pathlib import Path

from vgn import vis
from vgn.detection import VGN
from vgn.grasp import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform

from sfs import VoxelSpace
from utils import *
from visualizer import CamposeVisualizer

State = collections.namedtuple("State", ["tsdf", "pc"])

# ENVIRONMENT PARAMS
INITIAL_POSE = [0.973249, 0, 0, -0.2297529, 0.15, -0.15, 0.6]  # Top down view
MAX_STEPS = 7
GOAL_THRESHOLD = 0.03
# COLLISION_RADIUS = 0.05    # 5cm
ACTION_TRANS_SACLE = 0.05 # 5cm
PREGRASP_X = 0.0
PREGRASP_Z = -0.15
RVIZ = False
ALPHA = 0.5

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
         
        # Initialize (Simulation, VoxelSpace,VGN)
        self.sim = ClutterRemovalSim(scene="packed", object_set="rl", gui=True)
        self.voxel_space = VoxelSpace(X_RANGE,Y_RANGE,Z_RANGE,VOXEL_SIZE,K,NEAR,TABLE_HEIGHT)
        self.vgn = VGN(model_path=Path("data/models/vgn_conv.pth"),rviz=False)

        # gym environment definition
        self.observation_space = spaces.Dict(
            {
                "s3d" : spaces.Box(low=0, high=255,shape=(1,40,40,40), dtype=np.uint8),
                "pose" : spaces.Box(low=-30, high=30,shape=(14,),dtype=np.float32)
            }
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # rviz visuzlizer
        if RVIZ == True:
            rospy.init_node("sim_grasp", anonymous=True)
            self.camposevisualizer = CamposeVisualizer()


    def _get_obs(self):
        pose = np.concatenate((self.curr_pose,self.goal_pose), axis=0)
        return {"s3d" : self.s3d.astype(np.uint8), "pose" : pose.astype(np.float32)}
    
    def _get_curr_pose(self,action):
        action = np.clip(action,-1,1)
    
        # unit quaternion
        action[:4] /= np.linalg.norm(action[:4])

        # update quaternion
        q = Quaternion(self.curr_pose[:4])
        q_ = Quaternion(action[:4])
        q = (q_ * q).normalised

        self.curr_pose[0] = q[0]
        self.curr_pose[1] = q[1]
        self.curr_pose[2] = q[2]
        self.curr_pose[3] = q[3]

        # update translation
        action[4:] *= ACTION_TRANS_SACLE
        self.curr_pose[4:] += action[4:]


    def _get_info(self):
        return {"p_dist" : self.curr_pos_distance, "o_dist" : self.curr_quat_distance, "threshold" : self.goal_threshold,"num_points" : self.curr_num_points ,"goal" : self.done}
        
        
    def reset(self,seed=None, options=None):
        super().reset(seed=seed)
        # reset num_steps
        self.num_steps = 0

        # reset till find valid grasp pose
        while True:
            # 1 : Reset (Simulation, VoxelSpace)
            num_objects = np.random.randint(1,5)
            self.sim.reset(num_objects)
            self.voxel_space.reset()
            if RVIZ == True:
                vis.clear()
                vis.draw_workspace(self.sim.size)


            # 2 : get goal pose
            goal_pose = self.get_goalpose()

            if goal_pose != False:
                self.goal_pose = np.array(from_matrix(goal_pose.as_matrix()))
                break
        
        if RVIZ == True:
            self.camposevisualizer.reset()



        # 3 : set curr_pose to Initial pose
        self.curr_pose = np.array(INITIAL_POSE)
        if RVIZ == True:
            self.camposevisualizer.publish(self.curr_pose)


        # 4 : SfS
        self.sfs(self.curr_pose[:4],self.curr_pose[4:],self.num_steps)

        # 5 : set params
        self.init_num_points = self.voxel_space.num_points
        self.curr_num_points = self.voxel_space.num_points
        self.pointcloud = self.voxel_space.pointcloud
        self.s3d = self.voxel_space.s3d

        self.init_pos_distance, self.init_quat_distance = calc_distance(self.curr_pose,self.goal_pose)
        self.curr_pos_distance, self.curr_quat_distance = calc_distance(self.curr_pose,self.goal_pose)

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
        if RVIZ == True:
            self.camposevisualizer.publish(self.curr_pose)

        # 3 : SfS
        self.sfs(self.curr_pose[:4],self.curr_pose[4:],self.num_steps)

        # 4 : set parameters
        self.curr_num_points = self.voxel_space.num_points
        self.pointcloud = self.voxel_space.pointcloud
        self.s3d = self.voxel_space.s3d

        self.curr_pos_distance, self.curr_quat_distance = calc_distance(self.curr_pose,self.goal_pose)

        self.done = (self.curr_pos_distance <= self.goal_threshold) and (self.curr_quat_distance <= self.goal_threshold)
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
        grasps, scores , _ = self.vgn(state)
        
        if len(grasps) == 0:
            return False

        grasp,score = grasps[0],scores[0]
        
        
        T_world_grasp = grasp.pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [PREGRASP_X, 0.00, PREGRASP_Z])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        if RVIZ == True:
            vis.draw_tsdf(tsdf.get_grid().squeeze(), tsdf.voxel_size)
            vis.draw_points(np.asarray(pc.points))
            vis.draw_grasp(grasp, score, self.sim.gripper.finger_depth)
        
        # self.sim.execute_grasp(grasp,allow_contact=True)
        return T_world_pregrasp
    
    def sfs(self,q,t,n):
        rgb_image, _, _ = self.sim.camera2.get_image(q,t)
        seg_image = get_segimage(rgb_image,n,save_image=False)
        self.voxel_space.sfs(seg_image,to_matrix(q, t))

    def calc_reward(self):
        rw = (self.init_pos_distance - self.curr_pos_distance) / self.init_pos_distance + (self.init_quat_distance - self.curr_quat_distance) / self.init_quat_distance + ALPHA*(self.init_num_points - self.curr_num_points) / self.init_num_points
        print(rw)
        return rw



    
if __name__ == "__main__":
    # print((Transform.look_at(np.array([0.15,-0.15,0.6]),np.array([0.15,0.15,0.0]),np.array([0.0, 0.0, 1.0]))).inverse().as_matrix())
    env = Env()
    check_env(env)
