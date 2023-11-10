import numpy as np


class VoxelSpace:
    def __init__(self, x_range, y_range, z_range, voxel_size,K,near,table_height):
        # size of each voxel
        self.voxel_size = voxel_size

        # number of voxel in each axis
        self.num_voxels = [np.abs(x_range[1] - x_range[0]) / self.voxel_size[0], np.abs(y_range[1] - y_range[0]) / self.voxel_size[1], np.abs(z_range[1] - z_range[0]) / self.voxel_size[2]]
        
        # total number of voxels
        self.total_number = np.prod(self.num_voxels).astype(int)
        
        # (x,y,z,projected,inside or outside silhouette,occupancy)
        self.voxels = np.ones((self.total_number, 6))

        l = 0
        for z in range(1,int(self.num_voxels[2])+1):
            for y in range(1,int(self.num_voxels[1])+1):
                for x in range(1,int(self.num_voxels[0])+1):
                    self.voxels[l] = [self.voxel_size[0] * (x - 0.5),self.voxel_size[1] * (y - 0.5),self.voxel_size[2] * (z - 0.5), 0.0, 2.0, 0.0] 
                    l += 1
        
        # (Xw,Yw,Zw,1)
        self.points3D_world = np.copy(self.voxels[:,:4]).T
        self.points3D_world[3,:] = 1
        
        # camera intrinsic
        self.K = np.array(K)
        self.near = near

        self.table_height = table_height
        

    def sfs(self, image, extrinsic):
        height, width, image, silhouette = self.preprocess_image(image)

        # perspective projection matrix
        extrinsic = np.linalg.inv(extrinsic) 
        p_matrix = self.calc_p_matrix(extrinsic[0:3,:])

        # projection to the image plane (points2D = (u,v,1) * self.total_number)
        points2D = np.matmul(p_matrix, self.points3D_world)
        points2D = np.floor(points2D / points2D[2, :]).astype(np.int32)

        # (Xc,Yc,Zc,1)
        points3D_camera = np.matmul(extrinsic,self.points3D_world)

        # check for points less than focal length
        ind1 = np.where((points3D_camera[2,:] < self.near))
        # check for (u < 0, width < u) and (v < 0, height < v)
        ind2 = np.where(((points2D[0, :] < 0) | (points2D[0, :] >= width) | (points2D[1, :] < 0) | (points2D[1, :] >= height))) 
        points2D[:,ind2] = 0  # just for error handling

        # Not projected
        ind = np.unique(np.concatenate((ind1[0],ind2[0])))

        # 2 : not projected
        # 1 : inside silhouette
        # 0 : outside silhouette
        self.voxels[:,4] = silhouette[points2D.T[:,1], points2D.T[:,0]].astype(int)
        self.voxels[ind,4] = 2

        # CASE1 : never projected(self.voxels[:,3] == 0.0) & inside silhouette(self.voxels[:,4] == 1.0))
        case1 = np.where((self.voxels[:,3] == 0.0) & (self.voxels[:,4] == 1.0))
        self.voxels[case1[0],5] = 1.0

        # CASE2 : outside silhouette(self.voxels[:,4] == 0.0)) & occupied(self.voxels[:,4] == 1.0)
        case2 = np.where((self.voxels[:,4] == 0.0) & (self.voxels[:,5] == 1.0))
        self.voxels[case2[0],5] = 0.0

        self.remove_table(self.table_height)

        # update num_projected
        projected = np.ones((self.total_number, 1))
        projected[ind,0] = 0.0
        self.voxels[:,3] += projected[:,0]

        # reset inside or outside silhouette
        self.voxels[:,4] = 2

    def reset(self):
        self.voxels[:,3] = 0
        self.voxels[:,4] = 2
        self.voxels[:,5] = 0

            

    def remove_table(self,table_height):
        self.voxels[np.where(self.voxels[:,2] < table_height)[0],5] = 0

    def calc_p_matrix(self,extrinsic):
        return np.matmul(self.K,extrinsic)
    
    def preprocess_image(self,image):
        height, width = np.shape(image)
        image[np.where(image != 0)] = 1
        silhouette = image > 0
        return height, width, image, silhouette
    
    @property
    def pointcloud(self):
        ind = np.where(self.voxels[:,5] == 1.0)
        pcd = self.voxels[ind[0],0:3]
        return pcd
    
    @property
    def num_points(self):
        return np.shape(self.pointcloud)[0]
    
    @property
    def s3d(self):
        voxel_space = self.voxels[:,5]
        return voxel_space.reshape(1,int(self.num_voxels[0]),int(self.num_voxels[1]),int(self.num_voxels[2]))




        
    


    


