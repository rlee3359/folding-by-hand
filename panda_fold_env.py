import rospy
import actionlib
import numpy as np, sys, math
from sklearn.linear_model import LinearRegression
import torch
from math import fmod

from pyquaternion import Quaternion
import time
import cv2

import pyrealsense2 as rs

from tf.transformations import quaternion_from_euler, euler_from_quaternion
from std_srvs.srv import Empty
from armer_msgs.msg import MoveToPoseAction, MoveToPoseGoal, MoveToNamedPoseActionGoal, ManipulatorState, MoveToNamedPoseAction, MoveToNamedPoseGoal, MoveToJointPoseAction, MoveToJointPoseGoal, JointVelocity
from franka_gripper.msg import GraspActionGoal, MoveActionGoal, GraspAction, MoveAction, GraspGoal, MoveGoal
from franka_msgs.srv import SetCartesianImpedance, SetFullCollisionBehavior, SetFullCollisionBehaviorRequest
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, TwistStamped



crop_keypoints = np.array([
  [0,0],
  [300, 0],
  [0,300],
  [300, 300],
  ])

rgb_keypoints = np.array([
  [261,80],
  [566,77],
  [260, 380],
  [566, 383]
  ])

ir_keypoints = np.array([
  [298,102],
  [523,100],
  [298, 325],
  [522, 325]
  ])


class PandaFoldEnv:
    def __init__(self):
        self.calib_reg = torch.load("calib.reg")
        rospy.init_node("FabricEnv")
        self.pose_state = None
        self.pos_state = None
        # self.wrist_angle_state = None
        self.setup_ros()
        self.disable_compliance()
        self.setup_camera()
        self.W = 300
        self.reset()

    def setup_ros(self):
        self.vel_pub = rospy.Publisher('/arm/cartesian/velocity', TwistStamped, queue_size= 1)
        self.named_pos_pub = rospy.Publisher('/arm/joint/named/goal', MoveToNamedPoseActionGoal, queue_size = 1)
        self.gripper_pub = rospy.Publisher('/franka_gripper/grasp/goal', GraspActionGoal, queue_size = 1)
        self.gripper_open_pub = rospy.Publisher('/franka_gripper/move/goal', MoveActionGoal, queue_size = 1)

        self.joint_vel_pub = rospy.Publisher('/arm/joint/velocity', JointVelocity, queue_size=1)

        self.state_sub = rospy.Subscriber('/arm/state', ManipulatorState, self.state_callback)

        self.grasp_cli = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
        self.grasp_cli.wait_for_server()
        self.open_cli = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
        self.open_cli.wait_for_server()
        self.pose_cli = actionlib.SimpleActionClient('/arm/cartesian/pose', MoveToPoseAction)
        self.pose_cli.wait_for_server()
        self.named_pose_cli = actionlib.SimpleActionClient('/arm/joint/named', MoveToNamedPoseAction)
        self.named_pose_cli.wait_for_server()
        self.joint_pose_cli = actionlib.SimpleActionClient('/arm/joint/pose', MoveToJointPoseAction)
        self.joint_pose_cli.wait_for_server()

        self.recover = rospy.ServiceProxy('/arm/recover', Empty)
        self.controller_switcher = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController) 
        self.collision_proxy = rospy.ServiceProxy('/franka_control/set_full_collision_behavior', SetFullCollisionBehavior) 
        self.impedance_proxy = rospy.ServiceProxy('/franka_control/set_cartesian_impedance', SetCartesianImpedance)
        
        req = SetFullCollisionBehaviorRequest(
            lower_torque_thresholds_acceleration= [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],  # [Nm]
            upper_torque_thresholds_acceleration= [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],  # [Nm]
            lower_torque_thresholds_nominal= [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],  # [Nm]
            upper_torque_thresholds_nominal= [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],  # [Nm]
            lower_force_thresholds_acceleration= [20.0, 20.0, 12.0, 25.0, 25.0, 25.0],  # [N, N, N, Nm, Nm, Nm]
            upper_force_thresholds_acceleration= [20.0, 20.0, 20.0, 25.0, 25.0, 25.0],  # [N, N, N, Nm, Nm, Nm]
            lower_force_thresholds_nominal= [20.0, 20.0, 5.0, 25.0, 25.0, 25.0],  # [N, N, N, Nm, Nm, Nm]
            upper_force_thresholds_nominal= [20.0, 20.0, 20.0, 25.0, 25.0, 25.0] # [N, N, N, Nm, Nm, Nm]
            )

        self.collision_proxy(req)


    def setup_camera(self):
        print("Setting Up Camera...")
        self.pipeline = rs.pipeline()

        # Configure streams
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)
        self.queue = rs.frame_queue(1)

        # Start streaming
        profile = self.pipeline.start(config, self.queue)
        # align_to = rs.stream.depth
        # self.align = rs.align(align_to)

        depth_sensor = profile.get_device().first_depth_sensor()
        preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
        for i in range(int(preset_range.max)+1):
            visual_preset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
            print(f"{i}: {visual_preset}")
            if visual_preset == "Short Range":
                print("Found Preset")
                depth_sensor.set_option(rs.option.visual_preset, i)

        print("Camera Connected.")

    def goto_named(self, name, speed=0.5):
        goal = MoveToNamedPoseGoal()
        goal.pose_name = name
        self.named_pose_cli.send_goal(goal)
        self.named_pose_cli.wait_for_result()

    def reset(self):
        self.home()
        # self.reset_cloth()


        self.open_gripper()
        return self.get_obs()

    def home(self):
        # home_pose = [0.4, 0.0, 0.64, -1, 0, 0, 0]
        # self.go_to_pose(home_pose)
        print("Homing Robot")
        # self.goto_named("fabric_off_right", 0.8)
        home = [1.5768947245300844,
             -0.2151543058637987,
              0.021943047403766396, 
             -3.011487409640698,
             -0.023157180035619358,
              2.81021538168234, 
              0.808336588765184]
        self.go_to_joint_pose(home, speed=0.5)

    def set_impedance(self, impedance):
        self.impedance_proxy.call(impedance)
    
    def start_controller(self):
        pass
        # active_controller = ['joint_group_velocity_controller']
        # args = [active_controller, [''], 0, False, 0.0] 
        # self.controller_switcher.call(*args)

    def stop_controller(self):
        pass
        # active_controller = ['joint_group_velocity_controller']
        # args = [[''], active_controller, 0, False, 0.0] 
        # self.controller_switcher.call(*args)


    def enable_compliance(self):
        pass
        # self.stop_controller()
        # self.set_impedance([1000.0, 1000.0, 100.0, 10.0, 10.0, 10.0])
        # self.start_controller()


    def disable_compliance(self):
        pass
        # self.stop_controller()
        # self.set_impedance([3000.0, 3000.0, 3000.0, 300.0, 300.0, 300.0])
        # self.start_controller()


    def go_to_pose(self, pose, speed=1.0):
        target = PoseStamped()
        target.header.frame_id='panda_link0'
        target.pose.position.x = pose[0]
        target.pose.position.y = pose[1]
        target.pose.position.z = pose[2]

        target.pose.orientation.x = pose[3]
        target.pose.orientation.y = pose[4]
        target.pose.orientation.z = pose[5]
        target.pose.orientation.w = pose[6]

        goal = MoveToPoseGoal()
        goal.pose_stamped=target
        self.pose_cli.send_goal(goal)
        self.pose_cli.wait_for_result()


    def go_to_joint_pose(self, q, speed=1.0):
        goal = MoveToJointPoseGoal()
        goal.joints = q
        goal.speed = speed
        self.joint_pose_cli.send_goal(goal)
        self.joint_pose_cli.wait_for_result()


    def close_gripper(self):
        goal = GraspGoal()
        goal.speed =1.0 
        goal.force = 0.5
        goal.epsilon.inner = 0.0
        goal.epsilon.outer = 0.0
        goal.width = 0.0
        self.grasp_cli.send_goal(goal)
        self.grasp_cli.wait_for_result()

    def open_gripper(self, width=0.03):
        goal = MoveGoal()
        goal.speed = 1.0
        goal.width = width
        self.open_cli.send_goal(goal)
        self.open_cli.wait_for_result()


    def set_angle_joint(self, target):
        speed = 1
        K = speed * 4
        diff = self.angle_diff(target)
        rate = rospy.Rate(100)
        while abs(diff) > 0.005:
            vel = K* diff#/abs(diff)
            vel_msg = JointVelocity()
            vel_msg.joints = [0,0,0,0,0,0,-vel]
            self.joint_vel_pub.publish(vel_msg)
            diff = self.angle_diff(target)
            rate.sleep()

        for i in range(10):
            stop_msg = JointVelocity()
            stop_msg.joints = [0,0,0,0,0,0,0]
            self.joint_vel_pub.publish(stop_msg)
            rate.sleep()


    def angle_diff(self, target):
        diff = target - self.wrist_angle_state
        if diff > np.pi/2:
            diff -= np.pi
        elif diff < -np.pi/2:
            diff += np.pi
        return diff

    def wrap_to_pi(self, angle):
        if angle > np.pi:
            angle = angle - np.pi
        elif angle < 0:
            angle = angle + np.pi
        return angle


    def drop_center(self):
        x,y = env.pixel_to_pos((150,150))
        # self.goto_named("fabric_centered", speed=0.5)
        drop_pose = [x, y, 0.45] + [-0.707,-0.707,0.0, 0.0]
        self.go_to_pose(drop_pose, 0.5)
        self.go_to_pose(drop_pose, 0.5)
        self.open_gripper(0.05)
        

    def move_pick_to_nonzero(self, pos, mask):
        pos = np.array(pos)
        inds = np.argwhere(mask)
        # reverse xy of inds
        # inds = inds[:, [1, 0]]
        dists = np.linalg.norm(inds - pos, axis=1)
        return inds[np.argmin(dists)]


    def perform_fold(self, pick, place):
        _, depth, _, mask = self.get_image()
        if not mask[int(pick[0]), int(pick[1])]:
            pick = self.move_pick_to_nonzero(pick, mask)
            print("New Pick", pick)

        # Convert to world coordinates
        x,y = self.pixel_to_pos(pick)
        ex,ey = self.pixel_to_pos(place)

        inds = np.transpose(np.nonzero(np.logical_not(mask)))
        dist = np.abs(pick - inds)
        dist = np.sum(dist, axis=1)
        min_dist = np.min(dist)

        if min_dist < 10 or mask[int(place[0]), int(place[1])]:
            print("Fold action")
            self.open_gripper(0.03)
        else:
            self.open_gripper(0.007)

        depth = depth * mask

        table_to_cam = 0.83
        mean_point_meters = np.mean(depth[depth>0])*0.00025
        highest_point_meters = np.min(depth[depth>0])*0.00025
        max_height_of_cloth = table_to_cam - highest_point_meters
        mean_height_of_cloth = table_to_cam - mean_point_meters

        # Get deltas
        dx = ex - x
        dy = ey - y

        # Get angle
        angle = self.wrap_to_pi(np.arctan2(dy, dx) - np.pi/2)

        table_height = 0.02
        pre_fold_height = 0.01 + max_height_of_cloth
        fold_height = 0.03 + mean_height_of_cloth

        # self.goto_named("fabric_centered", speed=0.5)

        # while True:
        #         print(quaternion_from_euler(-3.14,0.0,angle))
        #         print(euler_from_quaternion([-0.707,-0.707,0.0, 0.0]))
        #         print(self.pose_state)
        #         input()

        
        quat = list(quaternion_from_euler(-3.14,0.0,angle))
        pre_grasp = [x, y, table_height + pre_fold_height] + [-0.707,-0.707,0.0, 0.0]
        self.go_to_pose(pre_grasp, 0.6)
        self.recover()
        
        self.set_angle_joint(angle)
        self.recover()

        # grasp_pose = [x, y, grasp_height, -1, 0, 0, 0]
        self.descend_to_contact()
        self.recover()
        self.close_gripper()
        self.recover()

        self.ascend(table_height+fold_height, 0.5)
        self.recover()

        self.go_to_pos([ex,ey, table_height+fold_height], 0.7)
        self.recover()
        self.open_gripper(0.04)
        self.ascend(table_height+pre_fold_height+0.04, 0.5)
        # up_pose = [ex, ey, table_height+pre_fold_height+0.04] + [-0.707,-0.707,0.0, 0.0]
        # self.go_to_pose(up_pose, 0.6)
        self.recover()
        # self.set_angle_joint(0)
        self.recover()

        # self.goto_named("fabric_centered", speed=0.5)
        # self.goto_named("fabric_off_right", speed=0.5)
        self.home()
        self.home()
        self.home()

        self.recover()

    def go_to_pose_pid(self, target, speed):
        K = 0.22 * speed
        lim = 0.2
        diff = target - self.pose_state
        dist = np.linalg.norm(diff)
        rate = rospy.Rate(100)
        while dist > 0.005:
            diff = diff/dist
            # if dist < 0.05: 
            #     diff = diff
            # else:
            #     diff = norm_diff

            vel = K*diff
            # vel = np.clip(vel, -lim,lim)
            vel_msg = TwistStamped()
            vel_msg.twist.linear.x = vel[0]
            vel_msg.twist.linear.y = vel[1]
            vel_msg.twist.linear.z = vel[2]
            vel_msg.twist.angular.z = vel[3]
            self.vel_pub.publish(vel_msg)
            diff = target - self.pose_state
            dist = np.linalg.norm(diff)
            rate.sleep()

        vel_msg = TwistStamped()
        self.vel_pub.publish(vel_msg)


    def descend_to_contact(self):
        rate = rospy.Rate(100)
        vel_msg = TwistStamped()
        vel_msg.twist.linear.z = -0.09
        while not any(self.full_state.cartesian_contact):
            self.vel_pub.publish(vel_msg)
            rate.sleep()

        vel_msg = TwistStamped()
        self.vel_pub.publish(vel_msg)

    def ascend(self, height, speed):
        speed = np.clip(speed, 0, 1)
        K = 0.1 * speed

        diff = height - self.pos_state[2]
        dist = np.linalg.norm(diff)

        rate = rospy.Rate(100)
        while dist > 0.005:
            diff = diff/dist

            vel = K*diff
            vel_msg = TwistStamped()
            vel_msg.twist.linear.z = vel
            self.vel_pub.publish(vel_msg)
            diff = height - self.pos_state[2]
            dist = np.linalg.norm(diff)
            rate.sleep()

        vel_msg = TwistStamped()
        self.vel_pub.publish(vel_msg)


    def go_to_pos_pid(self, target, speed):
        speed = np.clip(speed, 0, 1)
        K = 1.5 * speed

        diff = target - self.pos_state
        dist = np.linalg.norm(diff)

        rate = rospy.Rate(100)
        while dist > 0.005:
            # diff = diff/dist
            vel = K*diff
            vel_msg = TwistStamped()
            vel_msg.twist.linear.x = vel[0]
            vel_msg.twist.linear.y = vel[1]
            vel_msg.twist.linear.z = vel[2]
            self.vel_pub.publish(vel_msg)
            diff = target - self.pos_state
            dist = np.linalg.norm(diff)
            rate.sleep()

        vel_msg = TwistStamped()
        self.vel_pub.publish(vel_msg)


    def go_to_pos(self, target, speed):
        speed = np.clip(speed, 0, 1)
        K = 2 * speed

        diff = target - self.pos_state
        dist = np.linalg.norm(diff)

        rate = rospy.Rate(100)
        while dist > 0.005:
            # diff = diff/dist
            vel = K*diff
            vel_msg = TwistStamped()
            vel_msg.twist.linear.x = vel[0]
            vel_msg.twist.linear.y = vel[1]
            vel_msg.twist.linear.z = vel[2]
            self.vel_pub.publish(vel_msg)
            diff = target - self.pos_state
            dist = np.linalg.norm(diff)
            rate.sleep()

        vel_msg = TwistStamped()
        self.vel_pub.publish(vel_msg)

    def state_callback(self, msg):
        pose_state = msg.ee_pose.pose
        self.full_state = msg
        self.pos_state = np.array([pose_state.position.x,
                                   pose_state.position.y,
                                   pose_state.position.z])

        quat = Quaternion(pose_state.orientation.w,
                          pose_state.orientation.x,
                          pose_state.orientation.y,
                          pose_state.orientation.z)


        euler = quat.yaw_pitch_roll
        self.wrist_angle_state = -euler[0]

        self.pose_state = np.hstack([self.pos_state, np.array(euler)])


    def get_image(self):
        frames = self.queue.wait_for_frame().as_frameset()
        depth = np.asanyarray(frames.get_depth_frame().get_data())
        color = np.asanyarray(frames.get_color_frame().get_data())
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        ir    = np.asanyarray(frames.get_infrared_frame().get_data())

        ir_h,_ = cv2.findHomography(ir_keypoints, crop_keypoints)
        rgb_h,_ = cv2.findHomography(rgb_keypoints, crop_keypoints)

        color = cv2.warpPerspective(color, rgb_h, (self.W,self.W))
        ir = cv2.warpPerspective(ir, ir_h, (self.W,self.W))
        depth = cv2.warpPerspective(depth, ir_h, (self.W,self.W))

        color = cv2.rotate(color, cv2.ROTATE_90_CLOCKWISE)
        ir = cv2.rotate(ir, cv2.ROTATE_90_CLOCKWISE)
        depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)


        mask = ir > 20
        # depth[mask] = 0
        color[mask<1] = (0,0,0)

        return color, depth, ir, mask

    def get_obs(self):
        color, depth, ir, mask = self.get_image()

        color = cv2.resize(color, (64,64))
        depth = cv2.resize(depth, (64,64))
        # cv2.resize(mask, (200,200))
        return color, depth

    
    def step(self, pick, place):
        pick = pick[::-1]
        pick = pick * self.W/64
        place = place[::-1]
        place = place * self.W/64

        x,y = self.pixel_to_pos(pick)
        ex,ey = self.pixel_to_pos(place)

        self.perform_fold(x,y,ex,ey)
        return self.get_obs(), None, None, None


    def pixel_to_pos(self, pixel):
        x = pixel[1]
        y = pixel[0]
        # 0.5, -0.09
        X = [[x,y]]
        real = self.calib_reg.predict(X)[0]
        return tuple(real)

if __name__ == '__main__':
    env = PandaFoldEnv()
    env.home()

    node_rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        color, _, ir, _ = env.get_image()

        indices = np.transpose(np.nonzero(color))
        ind = indices[np.random.randint(indices.shape[0])]
        end = np.random.randint(0,300,2)

        color = cv2.arrowedLine(color, (ind[1], ind[0]), (end[1], end[0]), (0,250,100), 2)
        color = cv2.circle(color, (ind[1], ind[0]), 5, (0,0,250), 2)
        color = cv2.circle(color, (end[1], end[0]), 5, (250,0,0), 2)
        cv2.destroyAllWindows()
        cv2.imshow("color", color)
        cv2.waitKey(1)

        env.perform_fold(ind, end)
        node_rate.sleep()
