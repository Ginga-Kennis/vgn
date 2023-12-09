import rospy
import geometry_msgs.msg
import control_msgs.msg
import actionlib
import moveit_commander


def to_pose_msg(transform):
    """Convert a `Transform` object to a Pose message."""
    msg = geometry_msgs.msg.Pose()
    msg.position = to_point_msg(transform.translation)
    msg.orientation = to_quat_msg(transform.rotation)
    return msg

def to_point_msg(position):
    """Convert numpy array to a Point message."""
    msg = geometry_msgs.msg.Point()
    msg.x = position[0]
    msg.y = position[1]
    msg.z = position[2]
    return msg

def to_quat_msg(orientation):
    """Convert a `Rotation` object to a Quaternion message."""
    quat = orientation.as_quat()
    msg = geometry_msgs.msg.Quaternion()
    msg.x = quat[0]
    msg.y = quat[1]
    msg.z = quat[2]
    msg.w = quat[3]
    return msg


class UR5eCommander(object):
    def __init__(self):
        self.name = "ur5e"
        self.connect_to_move_group()
        
    def connect_to_move_group(self):
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander(self.name)
        self.move_group.set_planning_time(10)
        
    def goto_joints(self,joints,velocity_scaling=0.1,acceleration_scaling=0.1):
        self.move_group.set_max_velocity_scaling_factor(velocity_scaling)
        self.move_group.set_max_acceleration_scaling_factor(acceleration_scaling)
        self.move_group.set_joint_value_target(joints)
        plan = self.move_group.plan()[1]
        
        user_input = input("EXECUTE PLAN [y/n] : ")
        if user_input == "y":
            success = self.move_group.execute(plan, wait=True)
        else:
            print("ABORTED PLAN")
            success = False

        self.move_group.stop()
        return success
    
    def goto_pose(self,pose,velocity_scaling=0.1,acceleration_scaling=0.1):
        pose_msg = to_pose_msg(pose)
        self.move_group.set_max_velocity_scaling_factor(velocity_scaling)
        self.move_group.set_max_acceleration_scaling_factor(acceleration_scaling)
        self.move_group.set_pose_target(pose_msg)
        plan = self.move_group.plan()[1]

        user_input = input("EXECUTE PLAN [y/n] : ")
        if user_input == "y":
            success = self.move_group.execute(plan, wait=True)
        else:
            print("ABORTED PLAN")
            success = False
        
        self.move_group.clear_pose_targets()
        return success
    
    def goto_initial_pose(self):
        self.goto_joints([1.5708, -0.785398, -1.22173, -2.0944, 1.5708, 0.0])
    
class GripperController(object):
    def __init__(self):
        # Start the ROS node
        rospy.init_node('gripper_controller')
        
        # Create an action client
        self.client = actionlib.SimpleActionClient(
            '/gripper_controller/gripper_cmd',  # namespace of the action topics
            control_msgs.msg.GripperCommandAction # action type
        )
        
        # Wait until the action server has been started and is listening for goals
        self.client.wait_for_server()


    def gripper_control(self,width):
        # Create a goal to send (to the action server)
        goal = control_msgs.msg.GripperCommandGoal()
        goal.command.position = width   # From 0.0 to 0.8
        goal.command.max_effort = -1.0  # Do not limit the effort
        self.client.send_goal(goal)

        self.client.wait_for_result()
        return self.client.get_result()
    
if __name__ == "__main__":
    ur5e_controller = UR5eCommander()
    gripper_controller = GripperController()
    
    ur5e_controller.goto_joints([1.5708,-1.5708,-1.0472,-2.0944,1.5708,0])
    gripper_controller.gripper_control(0.8)
