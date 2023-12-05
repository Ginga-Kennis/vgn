import rospy
import numpy as np
from visualization_msgs.msg import *
from vgn.perception import CameraIntrinsic
from vgn.utils.transform import Transform,Rotation

RED = [1.0, 0.0, 0.0]
BLUE = [0, 0.6, 1.0]
GREY = [0.9, 0.9, 0.9]

class CamposeVisualizer:
    def __init__(self):
        # 固有値（不変）
        self.base_frame = "task"
        self.scale = [0.002, 0.0, 0.0]
        self.color = BLUE
        self.intrinsic = CameraIntrinsic(540,540,320,240,640,480)
        self.near = 0.0
        self.far = 0.02
        

        self.create_campose_publisher()
        self.reset()

    def create_campose_publisher(self,topic="camera_pose"):
        self.campose_pub = rospy.Publisher(topic,MarkerArray,queue_size=1)

    def publish(self, pose):
        self.id += 1
        pose_msg = Transform(Rotation.from_quat([pose[0],pose[1],pose[2],pose[3]]),pose[4:])
        marker = create_view_marker(self.base_frame, pose_msg,self.scale,self.color,self.intrinsic,self.near,self.far,ns="",id=self.id)
        self.markers.append(marker) 
        self.campose_pub.publish(MarkerArray(self.markers))

    def reset(self):
        self.markers = []
        self.id = 0

        # pose = np.array([0,0,0,1,0,0,100])
        # pose_msg = Transform(Rotation.from_quat([pose[0],pose[1],pose[2],pose[3]]),pose[4:])
        # marker = create_view_marker(self.base_frame, pose_msg,self.scale,self.color,self.intrinsic,self.near,self.far,ns="",id=self.id)
        self.campose_pub.publish(self.markers)


def create_view_marker(frame, pose, scale, color, intrinsic, near, far, ns="", id=0):
    marker = create_marker(Marker.LINE_LIST, frame, pose, scale, color, ns, id)
    x_n = near * intrinsic.width / (2.0 * intrinsic.fx)
    y_n = near * intrinsic.height / (2.0 * intrinsic.fy)
    z_n = near
    x_f = far * intrinsic.width / (2.0 * intrinsic.fx)
    y_f = far * intrinsic.height / (2.0 * intrinsic.fy)
    z_f = far
    points = [
        [x_n, y_n, z_n],
        [-x_n, y_n, z_n],
        [-x_n, y_n, z_n],
        [-x_n, -y_n, z_n],
        [-x_n, -y_n, z_n],
        [x_n, -y_n, z_n],
        [x_n, -y_n, z_n],
        [x_n, y_n, z_n],
        [x_f, y_f, z_f],
        [-x_f, y_f, z_f],
        [-x_f, y_f, z_f],
        [-x_f, -y_f, z_f],
        [-x_f, -y_f, z_f],
        [x_f, -y_f, z_f],
        [x_f, -y_f, z_f],
        [x_f, y_f, z_f],
        [x_n, y_n, z_n],
        [x_f, y_f, z_f],
        [-x_n, y_n, z_n],
        [-x_f, y_f, z_f],
        [-x_n, -y_n, z_n],
        [-x_f, -y_f, z_f],
        [x_n, -y_n, z_n],
        [x_f, -y_f, z_f],
    ]
    marker.points = [to_point_msg(p) for p in points]
    return marker

def create_marker(type, frame, pose, scale=None, color=None, ns="", id=0):
    if scale is None:
        scale = [1, 1, 1]
    elif np.isscalar(scale):
        scale = [scale, scale, scale]
    if color is None:
        color = (1, 1, 1)
    msg = Marker()
    msg.header.frame_id = frame
    msg.header.stamp = rospy.Time()
    msg.ns = ns
    msg.id = id
    msg.type = type
    msg.action = Marker.ADD
    msg.pose = to_pose_msg(pose)
    msg.scale = to_vector3_msg(scale)
    msg.color = to_color_msg(color)
    return msg

def to_pose_msg(transform):
    msg = geometry_msgs.msg.Pose()
    msg.position = to_point_msg(transform.translation)
    msg.orientation = to_quat_msg(transform.rotation)
    return msg

def to_point_msg(point):
    msg = geometry_msgs.msg.Point()
    msg.x = point[0]
    msg.y = point[1]
    msg.z = point[2]
    return msg

def to_quat_msg(orientation):
    quat = orientation.as_quat()
    msg = geometry_msgs.msg.Quaternion()
    msg.x = quat[0]
    msg.y = quat[1]
    msg.z = quat[2]
    msg.w = quat[3]
    return msg

def to_vector3_msg(vector3):
    msg = geometry_msgs.msg.Vector3()
    msg.x = vector3[0]
    msg.y = vector3[1]
    msg.z = vector3[2]
    return msg

def to_color_msg(color):
    msg = std_msgs.msg.ColorRGBA()
    msg.r = color[0]
    msg.g = color[1]
    msg.b = color[2]
    msg.a = color[3] if len(color) == 4 else 1.0
    return msg

    