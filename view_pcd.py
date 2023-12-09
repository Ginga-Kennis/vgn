import open3d as o3d

if __name__ == "__main__":
    path = "experiment/round1/view0.pcd"
    pcd = o3d.io.read_point_cloud(path)

    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(pcd)
    viewer.run()
    viewer.destroy_window()
