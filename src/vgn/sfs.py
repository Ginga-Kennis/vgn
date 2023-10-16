import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import cv2

class VoxelSpace:
    def __init__(self, x_lim, y_lim, z_lim, voxel_size):
        # number of voxel in each axis
        self.voxel_number = [np.abs(x_lim[1] - x_lim[0]) / voxel_size,np.abs(y_lim[1] - y_lim[0]) / voxel_size, np.abs(z_lim[1] - z_lim[0]) / voxel_size]
        
        # total number of voxels
        self.total_number = np.prod(self.voxel_number).astype(int)

        # (total_number, 4)
        # The first three values are the x-y-z-coordinates of the voxel, the fourth value is the occupancy
        self.voxel = np.ones((self.total_number, 4))

        self.voxel3Dx = np.meshgrid(np.linspace(x_lim[0], x_lim[1], self.voxel_number[0].astype(int)))
        self.voxel3Dy = np.meshgrid(np.linspace(y_lim[0], y_lim[1], self.voxel_number[1].astype(int)))
        self.voxel3Dz = np.meshgrid(np.linspace(z_lim[0], z_lim[1], self.voxel_number[2].astype(int)))

        l = 0
        for z in np.linspace(z_lim[0], z_lim[1], self.voxel_number[2].astype(int)):
            for x in np.linspace(x_lim[0], x_lim[1], self.voxel_number[0].astype(int)):
                for y in np.linspace(y_lim[0], y_lim[1], self.voxel_number[1].astype(int)):
                    self.voxel[l] = [x, y, z, 1] 
                    l += 1


                


        


    #     # camera properties
    #     self.focal_length = focal_length
    #     self.K = np.ndarray([self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0])

    # def sfs(self, image, extrinsic):
    #     height, width = np.shape(image)
    #     p_matrix = self.calc_p_matrix(extrinsic)

    #     image_plane = np.matmul(p_matrix,self.voxel)



    # def calc_p_matrix(self,extrinsic):
    #     return np.matmul(self.K,extrinsic)


if __name__ == "__main__":
    voxel = VoxelSpace([0,30],[0,30],[0,30],0.75)
    print(np.shape(voxel.voxel))


