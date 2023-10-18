import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import mcubes
import open3d as o3d

class VoxelSpace:
    def __init__(self, xlim, ylim, zlim, voxel_size,K):
        # number of voxel in each axis
        self.voxel_number = [np.abs(xlim[1] - xlim[0]) / voxel_size[0], np.abs(ylim[1] - ylim[0]) / voxel_size[1], np.abs(zlim[1] - zlim[0]) / voxel_size[2]]
        
        # total number of voxels
        self.total_number = np.prod(self.voxel_number).astype(int)

        # (total_number, 4)
        # The first three values are the x-y-z-coordinates of the voxel, the fourth value is the occupancy
        self.voxel = np.ones((self.total_number, 4))

        l = 0
        for x in range(int(self.voxel_number[0])):
            for y in range(int(self.voxel_number[1])):
                for z in range(int(self.voxel_number[2])):
                    self.voxel[l] = [x * voxel_size[0], y * voxel_size[1], z * voxel_size[2], 1] 
                    l += 1

        self.voxel = self.voxel.T

        # camera intrinsic
        self.K = K

    def sfs(self, image, extrinsic):
        height, width, image, silhouette = self.preprocess_image(image)

        #perspective projection matrix
        p_matrix = self.calc_p_matrix(extrinsic)
        print(p_matrix)

        
        # projection to the image plane (points2D = (u,v,1) * 41^3)
        points2D = np.matmul(p_matrix, self.voxel)
        print(points2D)
        points2D = np.floor(points2D / points2D[2, :]).astype(np.int32) # 3行目を1に揃える
        print(points2D)
        points2D[np.where(points2D < 0)] = 0  # check for negative image coordinate

        
        # check for out-of-bounds (height) coordinate
        # ind1 = np.where(points2D[0, :] >= width) # check for u value bigger than width
        # points2D[:,ind1] = 0
        # ind2 = np.where(points2D[1, :] >= height)  # check for v value bigger than width
        # points2D[:,ind2] = 0

        x_good = np.logical_and(points2D[0,:] >= 0, points2D[0,:] < width)
        y_good = np.logical_and(points2D[1,:] >= 0, points2D[1,:] < height)
        indices = np.where(np.logical_and(x_good,y_good))



        # print(np.shape(silhouette))
        # accumulate the value of each voxel in the current image
        
        self.voxel[:,3] += silhouette.T[points2D.T[:,0], points2D.T[:,1]]

        
        self.voxel3D = np.zeros(self.voxels_number_act)
        
        l = 0
        z1 = 0
        for z in range(self.voxels_number_act[2]):
            x1 = 0
            for x in range(self.voxels_number_act[0]):
                y1 = 0
                for y in range(self.voxels_number_act[1]):
                    self.voxel3D[y1,x1,z1] = self.voxel[l,3]
                    l += 1
                    y1 += 1
                x1 += 1
            z1 += 1

        print(np.unique(self.voxel3D)) 
        # print(self.voxel3D)

        error_amount = 5
        maxv = np.max(self.voxel3D[:, 3])
        iso_value = maxv-np.round(((maxv)/100)*error_amount)-0.5

        vertices, triangles = mcubes.marching_cubes(self.voxel3D,iso_value)
        print(vertices,triangles)
        mcubes.export_mesh(vertices, triangles, "pointcloud.dae","pointcloud")


    def calc_p_matrix(self,extrinsic):
        print(self.K)
        print(extrinsic)
        return np.matmul(self.K,extrinsic)
    
    def preprocess_image(self,image):
        height, width = np.shape(image)
        image[np.where(image != 0)] = 1
        silhouette = image > 0
        return height, width, image, silhouette

def preprocess_image(image):
        height, width = np.shape(image)
        image[np.where(image != 0)] = 1
        silhouette = image > 0
        return height, width, image, silhouette


if __name__ == "__main__":
    xlim = [0,0.3]
    ylim = [0,0.3]
    zlim = [0,0.3]
    voxel_size = [0.0075,0.0075,0.0075]
    K = np.array([[891.318115234375, 0.0, 628.9073486328125],
                  [0.0, 891.318115234375, 362.3601989746094],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    
    voxel = VoxelSpace(xlim,ylim,zlim,voxel_size,K)
    
    # image = cv2.imread("seg_image/view1.png",0)
    # extrinsic = np.array([[0.0, 1.0, 0.0, -0.15], 
    #                       [0.8660254, 0.0, -0.5, -0.12990381], 
    #                       [-0.5, 0.0, -0.8660254, 0.675]], dtype=np.float32)
    
    # voxel.sfs(image,extrinsic)

    image = cv2.imread("seg_image/view2.png",0)
    extrinsic = np.array([[-8.66025404e-01, 5.00000000e-01, -1.38777878e-17, 5.49038106e-02],
                          [4.33012702e-01, 7.50000000e-01, -5.00000000e-01, -1.77451905e-01],
                          [-2.50000000e-01, -4.33012702e-01, -8.66025404e-01, 7.02451905e-01]], dtype=np.float32)
    
    voxel.sfs(image,extrinsic)

    image = cv2.imread("seg_image/view3.png",0)
    extrinsic = np.array([[-8.66025404e-01, -5.00000000e-01, -1.38777878e-17,  2.04903811e-01],
                          [-4.33012702e-01,  7.50000000e-01, -5.00000000e-01, -4.75480947e-02],
                          [ 2.50000000e-01, -4.33012702e-01, -8.66025404e-01,  6.27451905e-01]], dtype=np.float32)
    
    voxel.sfs(image,extrinsic)

    image = cv2.imread("seg_image/view4.png",0)
    extrinsic = np.array([[-1.11022302e-16, -1.00000000e+00, -5.55111512e-17, 1.50000000e-01],
                          [-8.66025404e-01, 1.11022302e-16, -5.00000000e-01, 1.29903811e-01],
                          [5.00000000e-01, -5.55111512e-17, -8.66025404e-01, 5.25000000e-01]], dtype=np.float32)
    
    voxel.sfs(image,extrinsic)

    image = cv2.imread("seg_image/view5.png",0)
    extrinsic = np.array([[8.66025404e-01, -5.00000000e-01, -2.77555756e-17, -5.49038106e-02],
                          [-4.33012702e-01, -7.50000000e-01, -5.00000000e-01, 1.77451905e-01],
                          [2.50000000e-01, 4.33012702e-01, -8.66025404e-01, 4.97548095e-01]], dtype=np.float32)
    
    voxel.sfs(image,extrinsic)

    image = cv2.imread("seg_image/view6.png",0)
    extrinsic = np.array([[8.66025404e-01, 5.00000000e-01, 1.38777878e-17, -2.04903811e-01],
                          [4.33012702e-01, -7.50000000e-01, -5.00000000e-01, 4.75480947e-02],
                          [-2.50000000e-01, 4.33012702e-01, -8.66025404e-01, 5.72548095e-01]], dtype=np.float32)
    
    voxel.sfs(image,extrinsic)

