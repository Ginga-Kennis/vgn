import open3d as o3d
import numpy as np

def rotate_left_view(vis):
    """Viewを左に回転する"""
    ctr = vis.get_view_control()
    ctr.rotate(10.0, 0.0)
    return False

def rotate_right_view(vis):
    """Viewを右に回転する"""
    ctr = vis.get_view_control()
    ctr.rotate(-10.0, 0.0)
    return False

def pan_up_view(vis):
    """カメラを上に向ける"""
    ctr = vis.get_view_control()
    ctr.camera_local_rotate(0.0, -10.0)
    return False

def pan_down_view(vis):
    """カメラを下に向ける"""
    ctr = vis.get_view_control()
    ctr.camera_local_rotate(0.0, 10.0)
    return False

def scale_up_view(vis):
    """ズームアウトする"""
    ctr = vis.get_view_control()
    ctr.scale(1)
    return False

def scale_down_view(vis):
    """ズームインする"""
    ctr = vis.get_view_control()
    ctr.scale(-1)
    return False

def translate_left_view(vis):
    """カメラを左に移動する"""
    ctr = vis.get_view_control()
    ctr.translate(10.0, 0.0)
    return False

def translate_right_view(vis):
    """カメラを右に移動する"""
    ctr = vis.get_view_control()
    ctr.translate(-10.0, 0.0)
    return False

def translate_up_view(vis):
    """カメラを上に移動する"""
    ctr = vis.get_view_control()
    ctr.translate(0.0, 10.0)
    return False

def translate_down_view(vis):
    """カメラを下に移動する"""
    ctr = vis.get_view_control()
    ctr.translate(0.0, -10.0)
    return False

if __name__ == "__main__":
    path = "experiment/round5/view0.pcd"
    pcd = o3d.io.read_point_cloud(path)

    # set point cloud clor
    color = np.array([0.0, 0.0, 1.0])
    pcd.paint_uniform_color(color)

    # bounding box
    bound = pcd.get_axis_aligned_bounding_box()
    bound.color = (0, 0, 0)
    bound.max_bound = np.array([0.3, 0.3, 0.3])
    bound.min_bound = np.array([0.0, 0.0, 0.05])

    key_callback = {}
    key_callback[ord("F")] = rotate_left_view
    key_callback[ord("H")] = rotate_right_view
    key_callback[ord("T")] = pan_up_view
    key_callback[ord("G")] = pan_down_view
    key_callback[ord("Z")] = scale_up_view
    key_callback[ord("X")] = scale_down_view
    key_callback[ord("A")] = translate_left_view
    key_callback[ord("D")] = translate_right_view
    key_callback[ord("W")] = translate_up_view
    key_callback[ord("S")] = translate_down_view


    # vis = o3d.visualization.draw_geometries_with_key_callbacks([pcd, bound],key_callback)
    vis = o3d.visualization.draw_geometries_with_key_callbacks([pcd, bound], key_callback)
    
