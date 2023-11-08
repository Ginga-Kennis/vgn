import collections
import numpy as np
import cv2
import math
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

from vgn.detection import VGN
from vgn.grasp import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform
from vgn.sfs import VoxelSpace
    
def to_matrix(q,t):
    """
    - input -
    q : Quaternion
    t : Translation

    - output -
    matrix : Affine transformation matrix
    """
    r = np.array(R.from_quat(q).as_matrix())
    t = np.array(t)
    matrix = np.hstack((r,t.reshape(-1,1)))
    matrix = np.vstack((matrix,np.array([0,0,0,1])))
    return matrix

def from_matrix(matrix):
    """
    - input -
    matrix : Affine transformation matrix

    - output -
    q + t  : Quaternion + Translation
    """
    q = R.from_matrix(matrix[:3,:3]).as_quat().tolist()
    t = matrix[:3,3].tolist()
    return q+t

def get_segimage(image,n,save_image=True):
    """
    - input -
    image      : Rgb image(3 channel)
    n          : Number of steps (save as view{n}.png)
    save_image : save segmented image if True

    -output-
    segmented_image : Segmented image(1 channel) 
    """
    # rgb → gray
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 反転 
    gray_image = cv2.bitwise_not(gray_image)

    # 2値化
    threshold = 240
    ret, segmented_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    # save gray image
    if save_image == True:
        seg_image_name = f"./seg_image/view{n}.png"
        cv2.imwrite(seg_image_name,gray_image)

    return segmented_image

def visualize_pcd(pointcloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    
    # visualize
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(pcd)
    viewer.run()
    viewer.destroy_window()

def calc_distance(pose1,pose2):
    """
    - input -
    pose1 : pose1
    pose2 : pose2

    - output -
    p_dist + q_dist : distance between pose1 & pose2
    """
    p1 = pose1[4:]
    p2 = pose2[4:]
    q1 = Quaternion(np.array(pose1[:4]))
    q2 = Quaternion(np.array(pose2[:4]))
    q_dist = Quaternion.absolute_distance(q1,q2)
    p_dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

    return p_dist + q_dist

def check_collision(pointcloud,num_points,pose,r):
    """
    - input -
    pointcloud : [x,y,z] * num_points
    pose       : center [x,y,z]
    r          : radius

    - output -
    collision : True/False
    """
    pose_array = np.tile(np.array(pose),(num_points,1))
    distance = np.linalg.norm(pointcloud - pose_array, axis=1)
    ind = np.where(distance < r)
    # True if collision with pointcloud or ground
    if len(ind[0]) != 0 or pose[2] - 0.05 < r:
        return True
    return False


State = collections.namedtuple("State", ["tsdf", "pc"])
INITIAL_POSE = [0.968993181842469, 0.0, 0.0, -0.24708746132252002, 0.15, -0.15, 0.6]

class Env:
    def __init__(self):
        # Initialize params
        self.max_steps = 20
        self.goal_threshold = 0.03 
        self.r = 0.1   # 半径10cm
    
        # Initialize (Simulation, VoxelSpace,VGN)
        self.sim = ClutterRemovalSim(scene="packed", object_set="packed/train", gui=True)
        self.voxel_space = VoxelSpace([0.0,0.3],[0.0,0.3],[0.0,0.3],[0.003,0.003,0.003],np.array([[540, 0.0, 320],[0.0, 540, 240],[0.0, 0.0, 1.0]]),0.05,0.05)
        self.vgn = VGN(model_path=Path("data/models/vgn_conv.pth"),rviz=False)
        
        
    def reset(self,num_objects):
        # 1 : Reset (Simulation, VoxelSpace)
        self.num_objects = num_objects
        self.sim.reset(self.num_objects)
        self.voxel_space.reset()

        # 3 : set curr_pose to Initial pose
        self.curr_pose = INITIAL_POSE

        # 2 : get goal pose
        self.goal_pose = self.get_goalpose()
        if self.goal_pose == False:
            print("********* NO GRASP DETECTED *************")
            return False
        else:
            self.goal_pose = from_matrix(self.goal_pose.as_matrix())


        # 3 : initialize num_steps
        self.num_steps = 0

        # 4 : SfS
        self.sfs(self.curr_pose[:4],self.curr_pose[4:],self.num_steps)

        # 4 : set params
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

        state = [self.s3d, self.curr_pose, self.goal_pose]

        return state

        
                  
    def step(self,action):
        # 1 : increment num_steps
        self.num_steps += 1

        # 2 : set current_pose to action
        self.curr_pose = action

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

        # 5 : calculate reward
        reward = self.calc_reward()

        state = [self.s3d, self.curr_pose, self.goal_pose]

        return state, reward, self.done, self.truncated
    

    # Temporary
    def last_step(self):
        # 1 : increment num_steps
        self.num_steps += 1

        # 2 : set current_pose to action
        self.curr_pose = self.goal_pose

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

        # 5 : calculate reward
        reward = self.calc_reward()
        visualize_pcd(self.pointcloud)

        state = [self.s3d, self.curr_pose, self.goal_pose]

        return state, reward, self.done, self.truncated
    

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
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.00, -0.15])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp
        
        return T_world_pregrasp
    
    def sfs(self,q,t,n):
        rgb_image, _, _ = self.sim.camera2.get_image(q,t)
        seg_image = get_segimage(rgb_image,n,save_image=True)
        self.voxel_space.sfs(seg_image,to_matrix(q, t))

    def calc_reward(self):
        # GOAL : reward = 10
        if self.goal == True:
            rw = 10
        
        # COLLISION : reward = -10
        elif self.collision == True:
            rw = -10

        # NORMAL : 
        else:
            rw = ((self.prev_distance - self.curr_distance) / self.prev_distance) + (abs(self.curr_num_points - self.prev_num_points) / self.prev_num_points)

        return rw


    
if __name__ == "__main__":
    env = Env()
    
    for _ in range(1000):
        print("-------------------------------")
        num_objets = np.random.randint(1,5)
        state = env.reset(1)
        if state == False:
            continue
        state, reward, done,truncated = env.step([0.682982, 0.6830478, -0.1830045, -0.1830045, 0.45, 0.15, 0.51961524])
        print(reward,done)
        state, reward, done,truncated = env.step([0.2499775, 0.9330252, -0.2499775, -0.0669813, 0.3, 0.41, 0.51961524])
        print(reward,done)
        state, reward, done,truncated = env.step([-0.2499999999983246, 0.933012701893705, -0.2499999999983246, 0.06698729809959345, 6.40987562e-17, 0.409807621, 0.519615242])
        print(reward,done)
        state, reward, done,truncated = env.step([-0.6830127018975046, 0.6830127018975047, -0.18301270187249408, 0.18301270187249408, -0.15, 0.15, 0.519615242])
        print(reward,done)
        state, reward, done,truncated = env.step([0.933012701893705, -0.2499999999983246, 0.06698729809959346, -0.2499999999983246, -1.04160479e-16, -0.109807621, 0.519615242])
        print(reward,done)
        state, reward, done,truncated = env.step([0.9330127018661368, 0.2500000000294135, -0.06698729825151725, -0.2500000000294135, 0.3, -0.10980762, 0.51961524])                                                                          
        print(reward,done)
        state, reward, done,truncated = env.last_step()   
        print(reward,done)
