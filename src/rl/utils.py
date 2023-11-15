import math
import cv2
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

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
        if n == 0:
            seg_image_name = f"./seg_image/initial.png"
        else:
            seg_image_name = f"./seg_image/view{n}.png"
        cv2.imwrite(seg_image_name,segmented_image)

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
    distance = 0.5 * p_dist + 0.5 * q_dist
    return p_dist, q_dist, distance 

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