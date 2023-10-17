import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

class VoxelSpace:
    def __init__(self, xlim, ylim, zlim, voxel_size,K):
        # number of voxel in each axis
        self.voxels_number = [np.abs(xlim[1] - xlim[0]) / voxel_size[0], np.abs(ylim[1] - ylim[0]) / voxel_size[1], np.abs(zlim[1] - zlim[0]) / voxel_size[2]]
        self.voxels_number_act = np.array(self.voxels_number).astype(int) + 1
        
        # total number of voxels
        self.total_number = np.prod(self.voxels_number_act).astype(int)

        # (total_number, 4)
        # The first three values are the x-y-z-coordinates of the voxel, the fourth value is the occupancy
        self.voxel = np.ones((self.total_number, 4))

        self.sx = xlim[0]
        self.ex = xlim[1]
        self.sy = ylim[0]
        self.ey = ylim[1]
        self.sz = zlim[0]
        self.ez = zlim[1]

        l = 0
        for z in np.linspace(self.sz, self.ez, self.voxels_number_act[2]):
            for x in np.linspace(self.sx, self.ex, self.voxels_number_act[0]):
                for y in np.linspace(self.sy, self.ey, self.voxels_number_act[1]):
                    self.voxel[l] = [x, y, z, 1] 
                    l += 1

        self.object_points3D = np.copy(self.voxel).T
        self.voxel[:,3] = 0

        # camera intrinsic
        self.K = K

    def sfs(self, image, extrinsic):
        height, width, image, silhouette = self.preprocess_image(image)
        print(height)
        print(width)

        #perspective projection matrix
        p_matrix = self.calc_p_matrix(extrinsic)


        # projection to the image plane
        points2D = np.matmul(p_matrix, self.object_points3D)
        points2D = np.floor(points2D / points2D[2, :]).astype(np.int32)
        print(points2D)
        print(np.where(points2D < 0))
        points2D[np.where(points2D < 0)] = 0  # check for negative image coordinate

        
        ind1 = np.where(points2D[0, :] >= width) # check for out-of-bounds (width) coordinate
        points2D[:,ind1] = 0
        ind2 = np.where(points2D[1, :] >= height)  # check for out-of-bounds (height) coordinate
        points2D[:,ind2] = 0

        # accumulate the value of each voxel in the current image
        self.voxel[:,3] += silhouette.T[points2D.T[:,0], points2D.T[:,1]]

    def calc_p_matrix(self,extrinsic):
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
    xlim = [0.3,0]
    ylim = [0.3,0]
    zlim = [0.3,0]
    voxel_size = [0.0075,0.0075,0.0075]
    K = np.array([[891.318115234375, 0.0, 628.9073486328125],
                  [0.0, 891.318115234375, 362.3601989746094],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    
    voxel = VoxelSpace(xlim,ylim,zlim,voxel_size,K)
    
    image = cv2.imread("seg_image/view1.png",0)
    extrinsic = np.array([[0.0, 1.0, 0.0, -0.15], 
                          [0.8660254, 0.0, -0.5, -0.12990381], 
                          [-0.5, 0.0, -0.8660254, 0.675]], dtype=np.float32)
    
    voxel.sfs(image,extrinsic)

