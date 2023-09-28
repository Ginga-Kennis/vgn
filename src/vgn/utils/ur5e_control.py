import moveit_commander
import rospy

from vgn.utils import ros_utils


class Ur5eCommander(object):
    def __init__(self):
        self.name = "ur5e"
        self.connect_to_move_group()

    def connect_to_move_group(self):
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander(self.name)

    def goto_home(self):
        self.goto_joints([1.5708,-1.5708,0,-1.5708,0,0])

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
        pose_msg = ros_utils.to_pose_msg(pose)
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
