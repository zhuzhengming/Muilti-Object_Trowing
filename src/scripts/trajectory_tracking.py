import sys
sys.path.append("../")
import time
import os
import rospy
import copy
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from std_msgs.msg import Int64
import threading
import pdb
from pathlib import Path
from sys import path
from ruckig import InputParameter, Ruckig, Trajectory
import kinematics.allegro_hand_sym as allegro
from datetime import datetime
from trajectory_generator import TrajectoryGenerator

SIMULATION = True
DEBUG = True

class ThrowingController:
    def __init__(self, path_prefix='../', box_position=None):
        self.box_position = box_position if box_position is not None else np.zeros(3)
        self.path_prefix = path_prefix
        self._allegro_init()
        self._iiwa_init()
        self._control_init()
        self._general_init()
        self.start = False
        time.sleep(1.0)


    def _general_init(self):
        hedgehog_path = '../hedgehog_revised'
        brt_path = '../brt_data'
        xml_path = '../description/iiwa7_allegro_throwing.xml'

        self.q_min = np.array([-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239, -2.96705972839,
                          -2.09439510239, -3.05432619099])
        self.q_max = np.array([2.96705972839, 2.09439510239, 2.96705972839, 2.09439510239, 2.96705972839,
                          2.09439510239, 3.05432619099])

        self.trajectoryGenerator = TrajectoryGenerator(self.q_max, self.q_min,
                                                       hedgehog_path, brt_path,
                                                       self.box_position, xml_path)

        # initialize ROS subscriber/publisher
        self.fsm_state_pub = rospy.Publisher('fsm_state', String, queue_size=1)
        self.target_state_pub = rospy.Publisher('/computed_torque_controller/target_state', JointState,
                                                queue_size=10)  # for debug
        self.optitrack_sub = rospy.Subscriber(
                            '/vrpn_client_node/cube_z/pose_from_iiwa_7_base',
                            PoseStamped,
                            self._optitrack_callback,
                            queue_size=100
                        )

        self.box_pos_sub = rospy.Subscriber(
            '/vrpn_client_node/box_1/pose_from_iiwa_7_base',
            PoseStamped,
            self._box_pos_callback,
            queue_size=100
        )

        self.obj_cur = []
        self.obj_trajectory = []
        self.stamp = []
        self.real_pos = []
        self.real_vel = []
        self.real_eff = []
        self.target_pos = []
        self.target_vel = []
        self.target_eff = []
        self.pos_error_sum = np.zeros(7)
        self.vel_error_sum = np.zeros(7)

        # get parameters and initialization
        self.ERROR_THRESHOLD = rospy.get_param('/ERROR_THRESHOLD')  # Threshold to switch from homing to throwing state
        self.GRIPPER_DELAY = rospy.get_param('/GRIPPER_DELAY')

        self.time_throw = np.inf  # Planned time of throwing
        self.fsm_state = "IDLE"


    def _allegro_init(self):
        # allegro controller
        self.hand_home_pose = np.array(rospy.get_param('/hand_home_pose'))
        self.envelop_pose = np.array(rospy.get_param('/exp_envelop_pose'))
        self.release_pose_A = np.array(rospy.get_param('/release_pose_A'))
        self.release_pose_B = np.array(rospy.get_param('/release_pose_B'))
        self.joint_cmd_pub = rospy.Publisher('/allegroHand_0/joint_cmd', JointState, queue_size=10)
        rospy.Subscriber('/allegroHand_0/joint_states', JointState, self._hand_joint_states_callback)
        self.hand = allegro.Robot(right_hand=False, path_prefix=self.path_prefix)  # load the left hand kinematics
        self.fingertip_sites = ['index_site', 'middle_site', 'ring_site',
                                'thumb_site']
        self._qh = np.zeros(16)
        self.hand_joint_cmd = JointState()
        self.hand_joint_cmd.name = ['joint_0', 'joint_1', 'joint_2', 'joint_3',
                                    # index finger: ad/abduction, extensions
                                    'joint_4', 'joint_5', 'joint_6', 'joint_7',  # middle finger
                                    'joint_8', 'joint_9', 'joint_10', 'joint_11',  # ring finger
                                    'joint_12', 'joint_13', 'joint_14', 'joint_15']  # thumb
        self.hand_joint_cmd.position = []  # 0-3: index, 4-7: middle, 8-11: ring, 12-15: thumb

    def _iiwa_init(self):
        # iiwa controller
        self.command_pub = rospy.Publisher('/iiwa_impedance_joint', JointState, queue_size=10)
        rospy.Subscriber('/iiwa/joint_states', JointState, self.joint_states_callback, queue_size=10)
        rospy.Subscriber('/throw_node/throw_state', Int64, self.scheduler_callback)

        # Initialize robot state, to be updated from robot
        self.robot_state = JointState()
        self.robot_state.position = [0.0 for _ in range(7)]
        self.robot_state.velocity = [0.0 for _ in range(7)]
        self.robot_state.effort = [0.0 for _ in range(7)]

        # Initialize target state, to be updated from planner
        self.target_state = JointState()

        # Ruckig margins for throwing
        self.MARGIN_VELOCITY = rospy.get_param('/MARGIN_VELOCITY')
        self.MARGIN_ACCELERATION = rospy.get_param('/MARGIN_ACCELERATION')
        self.MARGIN_JERK = rospy.get_param('/MARGIN_JERK')

        # constraints of iiwa 7
        self.max_velocity = np.array(rospy.get_param('/max_velocity'))
        self.max_acceleration = np.array(rospy.get_param('/max_acceleration'))
        self.max_jerk = np.array(rospy.get_param('/max_jerk'))


    def _control_init(self):
        self.dt = 5e-3

        # qs for the initial state
        self.qs = np.array([0.4217-2.0, 0.5498-0.4, 0.1635, -0.7926, -0.0098, 0.6, 1.2881])
        self.qs_dot = np.zeros(7)
        self.qs_dotdot = np.zeros(7)

        # qd for the throwing state
        qd_offset = [0.0, 0.0, 0.2, 0.2, -0.1, 0.2, 0.2]
        qd_dot_offset = np.zeros(7)

        self.qd = self.qs + qd_offset
        self.qd_dot = qd_dot_offset
        self.qd_dotdot = np.zeros(7)

    def save_tracking_data_to_npy(self):

        save_dir = '../output/data/ee_tracking/throw_tracking_batch'
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f'throwing_{timestamp}.npy')

        data_to_save = {
            'stamp': self.stamp,
            'real_pos': self.real_pos.copy(),
            'real_vel': self.real_vel.copy(),
            'real_eff': self.real_eff.copy(),
            'target_pos': self.target_pos.copy(),
            'target_vel': self.target_vel.copy(),
            'target_eff': self.target_eff.copy(),
            'obj_trajectory': self.obj_trajectory.copy()
        }

        def async_save():
            np.save(filename, data_to_save)
            print(f"Data saved to {filename}")

            self.stamp.clear()
            self.real_pos.clear()
            self.real_vel.clear()
            self.real_eff.clear()
            self.target_pos.clear()
            self.target_vel.clear()
            self.target_eff.clear()
            self.obj_trajectory.clear()

        threading.Thread(target=async_save).start()

    def run_multi_throwing(self, boxes_pos, max_run_time=60.0):
        start_time = rospy.get_time()
        dT = self.dt
        rate = rospy.Rate(1.0 / dT)
        (final_trajectory,
         best_throw_config_pair,
         intermediate_time) = self.multi_match_configuration(boxes_pos)
        qA = best_throw_config_pair[:7]
        qA_dot = best_throw_config_pair[7:14]
        qA_ddot = np.zeros(7)
        qB = best_throw_config_pair[14:21]
        qB_dot = best_throw_config_pair[21:28]
        qB_ddot = np.zeros(7)

        if SIMULATION:
            throwing_traj_1 = self.get_traj_from_ruckig(self.qs, self.qs_dot, np.zeros(7),
                                                      qA, qA_dot, qA_ddot,
                                                      margin_velocity=self.MARGIN_VELOCITY,
                                                      margin_acceleration=self.MARGIN_ACCELERATION)
            throwing_traj_2 = self.get_traj_from_ruckig(qA, qA_dot, qA_ddot,
                                                        qB, qB_dot, qB_ddot,
                                                        margin_velocity=self.MARGIN_VELOCITY,
                                                        margin_acceleration=self.MARGIN_ACCELERATION)

            trajectory_back = self.get_traj_from_ruckig(qB, qB_dot, np.zeros(7),
                                                        self.qs, self.qs_dot, self.qs_dotdot,
                                                        margin_velocity=self.MARGIN_VELOCITY * 0.5,
                                                        margin_acceleration=self.MARGIN_ACCELERATION * 0.5)

            traj_back = self.trajectoryGenerator.process_trajectory(trajectory_back)

            self.trajectoryGenerator.robot._set_joints(self.qs, render=True)

            intermediate_time, traj_throw = self.trajectoryGenerator.concatenate_trajectories(
                throwing_traj_1, throwing_traj_1
            )

            self.trajectoryGenerator.throw_simulation_mujoco(intermediate_time=intermediate_time,
                                                             ref_sequence=traj_throw)

            self.trajectoryGenerator.throw_simulation_mujoco(ref_sequence=traj_back)
            return
        else:
            self._send_hand_position(self.envelop_pose)
        while not rospy.is_shutdown():
            if (rospy.get_time() - start_time) > max_run_time:
                rospy.logwarn("run() time limit exceeded, saving data and breaking...")
                break
            # Publish state and fsm for debug
            self.fsm_state_pub.publish(self.fsm_state)
            # update robot state
            q_cur = np.array(self.robot_state.position)
            q_cur_dot = np.array(self.robot_state.velocity)
            q_cur_effort = np.array(self.robot_state.effort)

            if self.fsm_state == "IDLE":
                if DEBUG and not self.start:
                    pdb.set_trace(header="Press C to start homing...")
                    self.start = True
                    print("HOMING...")

                self.homing_traj = self.get_traj_from_ruckig(q_cur, q_cur_dot, np.zeros(7),
                                                             self.qs, self.qs_dot, self.qs_dotdot,
                                                             margin_velocity=self.MARGIN_VELOCITY * 0.2,
                                                             margin_acceleration=self.MARGIN_ACCELERATION *0.2)

                if self.homing_traj is None:
                    rospy.logerr("Trajectory is None")

                # update state
                self.fsm_state = "HOMING"
                self.time_start_homing = rospy.get_time()

            elif self.fsm_state == "HOMING":
                # Activate integrator term when close to target
                error_position = np.array(q_cur) - np.array(self.qs)
                if np.linalg.norm(error_position) < self.ERROR_THRESHOLD:
                    # Jump to next state
                    self.fsm_state = "IDLE_THROWING"
                    time.sleep(8)
                    if DEBUG:
                        print("IDLE_THROWING")
                        # pdb.set_trace(header="Press C to see the throwing trajectory...")
                    self.throw_time_A = self.scheduler_callback(Int64(1), qA, qA_dot, qA_ddot)

                time_now = rospy.get_time()
                ref = self.homing_traj.at_time(time_now - self.time_start_homing)
                self.target_state.header.stamp = time_now
                self.target_state.position = ref[0]
                self.target_state.velocity = ref[1]
                self.target_state.effort = ref[2]

                self.target_state_pub.publish(self.target_state)
                self.command_pub.publish(self.convert_command_to_ROS(time_now, ref[0], ref[1], ref[2]))

            elif self.fsm_state == "IDLE_THROWING":
                continue

            elif self.fsm_state == "THROWING":
                time_now = rospy.get_time()
                throwing_time = time_now - self.time_start_throwing
                release_A_time = self.throw_time_A - self.GRIPPER_DELAY
                release_B_time = self.time_throw - self.GRIPPER_DELAY

                if throwing_time > release_A_time:
                    self._send_hand_position(self.release_pose_A)
                elif throwing_time > release_B_time:
                    self._send_hand_position(self.release_pose_B)

                if time_now - self.time_start_throwing > self.throw_time_A - dT:
                    self.fsm_state = "SLOWING"
                    self.trajectory_back = self.get_traj_from_ruckig(q_cur, q_cur_dot, np.zeros(7),
                                                                     self.qs, self.qs_dot, self.qs_dotdot,
                                                                     margin_velocity=self.MARGIN_VELOCITY * 0.2,
                                                                     margin_acceleration=self.MARGIN_ACCELERATION * 0.2)
                    if self.trajectory_back is None:
                        rospy.logerr("Trajectory is None")
                    if DEBUG:
                        print("SLOWING")
                        # pdb.set_trace(header="Press C to see the slowing trajectory...")
                    self.time_start_slowing = time_now

                    if throwing_time <= release_A_time:
                        ref = self.throwing_traj.at_time(throwing_time)
                        self.target_state.header.stamp = time_now
                        self.target_state.position = ref[0]
                        self.target_state.velocity = ref[1]
                        self.target_state.effort = ref[2]
                    elif throwing_time <= release_B_time and throwing_time > release_A_time:
                        ref = self.trajectory_back.at_time(throwing_time - release_A_time)
                        self.target_state.header.stamp = time_now
                        self.target_state.position = ref[0]
                        self.target_state.velocity = ref[1]
                        self.target_state.effort = ref[2]

                    self.target_state_pub.publish(self.target_state)
                    self.command_pub.publish(self.convert_command_to_ROS(time_now, ref[0], ref[1], ref[2]))

            elif self.fsm_state == "SLOWING":
                time_now = rospy.get_time()
                if time_now - self.time_start_slowing > self.trajectory_back.duration - dT:
                    break

                ref = self.trajectory_back.at_time(time_now - self.time_start_slowing)
                self.target_state.header.stamp = time_now
                self.target_state.position = ref[0]
                self.target_state.velocity = ref[1]
                self.target_state.effort = ref[2]

                self.target_state_pub.publish(self.target_state)
                self.command_pub.publish(self.convert_command_to_ROS(time_now, ref[0], ref[1], ref[2]))

            rate.sleep()


    def run(self, max_run_time=60.0,
            save_data=True):
        # ------------ Control Loop ------------ #
        start_time = rospy.get_time()
        dT = self.dt
        rate = rospy.Rate(1.0 / dT)

        qd, qd_dot, qd_dotdot = self.match_configuration(posture='posture1')

        if SIMULATION:
            self.simulation_mujoco(qd, qd_dot, qd_dotdot)
            return
        else:
            self._send_hand_position(self.envelop_pose)

        while not rospy.is_shutdown():
            if (rospy.get_time() - start_time) > max_run_time:
                rospy.logwarn("run() time limit exceeded, saving data and breaking...")
                if save_data:
                    self.save_tracking_data_to_npy() # save data if stuck
                break

            # Publish state and fsm for debug
            self.fsm_state_pub.publish(self.fsm_state)
            # update robot state
            q_cur = np.array(self.robot_state.position)
            q_cur_dot = np.array(self.robot_state.velocity)
            q_cur_effort = np.array(self.robot_state.effort)


            if self.fsm_state == "IDLE":
                if DEBUG and not self.start:
                    pdb.set_trace(header="Press C to start homing...")
                    self.start = True
                    print("HOMING...")

                self.homing_traj = self.get_traj_from_ruckig(q_cur, q_cur_dot, np.zeros(7),
                                                             self.qs, self.qs_dot, self.qs_dotdot,
                                                             margin_velocity=self.MARGIN_VELOCITY * 0.2,
                                                             margin_acceleration=self.MARGIN_ACCELERATION *0.2)
                if self.homing_traj is None:
                    rospy.logerr("Trajectory is None")

                # print("IDLING: initial state", self.q0, "homing trajectory duration", self.homing_traj.duration)

                # update state
                self.fsm_state = "HOMING"
                self.time_start_homing = rospy.get_time()


            elif self.fsm_state == "HOMING":

                # Activate integrator term when close to target
                error_position = np.array(q_cur) - np.array(self.qs)
                if np.linalg.norm(error_position) < self.ERROR_THRESHOLD:
                    # Jump to next state
                    self.fsm_state = "IDLE_THROWING"
                    time.sleep(8)
                    if DEBUG:
                        print("IDLE_THROWING")
                        # pdb.set_trace(header="Press C to see the throwing trajectory...")
                    _ = self.scheduler_callback(Int64(1), qd, qd_dot, qd_dotdot)

                time_now = rospy.get_time()
                ref = self.homing_traj.at_time(time_now - self.time_start_homing)
                self.target_state.header.stamp = time_now
                self.target_state.position = ref[0]
                self.target_state.velocity = ref[1]
                self.target_state.effort = ref[2]

                self.target_state_pub.publish(self.target_state)
                self.command_pub.publish(self.convert_command_to_ROS(time_now, ref[0], ref[1], ref[2]))

            elif self.fsm_state == "IDLE_THROWING":
                continue

            elif self.fsm_state == "THROWING":
                # execute throwing trajectory
                time_now = rospy.get_time()
                throwing_time = time_now - self.time_start_throwing
                release_time = self.throwing_traj.duration - self.GRIPPER_DELAY
                brake_time = self.throwing_traj.duration - dT

                # release gripper
                if throwing_time > release_time:
                    threading.Thread(target=self.deactivate_gripper).start()

                # if throwing_time > brake_time:
                #     self.fsm_state = "RELEASE_BRAKE"
                #
                #     self.brake_traj = self.get_traj_from_ruckig(
                #         q_cur, q_cur_dot, np.zeros(7),
                #         q_cur, np.zeros(7), np.zeros(7),
                #         margin_velocity=self.MARGIN_VELOCITY * 0.5 ,
                #         margin_acceleration=self.MARGIN_ACCELERATION * 0.3
                #     )
                #     self.time_release = time_now
                #     self.brake_start_position = q_cur

                if time_now - self.time_start_throwing > self.throwing_traj.duration - dT:
                    self.fsm_state = "SLOWING"

                    self.trajectory_back = self.get_traj_from_ruckig(q_cur, q_cur_dot, np.zeros(7),
                                                                     self.qs, self.qs_dot, self.qs_dotdot,
                                                                margin_velocity=self.MARGIN_VELOCITY * 0.2,
                                                                margin_acceleration=self.MARGIN_ACCELERATION * 0.2)
                    if self.trajectory_back is None:
                        rospy.logerr("Trajectory is None")

                    if DEBUG:
                        print("SLOWING")
                        # pdb.set_trace(header="Press C to see the slowing trajectory...")
                    self.time_start_slowing = time_now

                ref = self.throwing_traj.at_time(throwing_time)
                self.target_state.header.stamp = time_now
                self.target_state.position = ref[0]
                self.target_state.velocity = ref[1]
                self.target_state.effort = ref[2]

                self.target_state_pub.publish(self.target_state)
                self.command_pub.publish(self.convert_command_to_ROS(time_now, ref[0], ref[1], ref[2]))

                # record error
                ratio = 1.0
                if(self.time_throw - throwing_time < ratio * self.time_throw):

                    self.stamp.append(time_now)
                    self.real_pos.append(q_cur)
                    self.real_vel.append(q_cur_dot)
                    self.real_eff.append(q_cur_effort)
                    self.target_pos.append(self.target_state.position)
                    self.target_vel.append(self.target_state.velocity)
                    self.target_eff.append(self.target_state.effort)

            elif self.fsm_state == "RELEASE_BRAKE":
                time_now = rospy.get_time()
                ref = self.brake_traj.at_time(time_now - self.time_release)
                self.target_state.header.stamp = time_now
                self.target_state.position = ref[0]
                self.target_state.velocity = ref[1]
                self.target_state.effort = ref[2]

                self.target_state_pub.publish(self.target_state)
                ref_pos = self.brake_start_position * 0.2 + ref[0] * 0.8
                ref_vel = ref[1] * 0.7
                self.command_pub.publish(self.convert_command_to_ROS(time_now, ref_pos, ref_vel, ref[2]))

                if np.linalg.norm(self.robot_state.velocity) < 0.1:
                    self.fsm_state = "SLOWING"
                    self.trajectory_back = self.get_traj_from_ruckig(q_cur, q_cur_dot, np.zeros(7),
                                                                     self.qs, self.qs_dot, self.qs_dotdot,
                                                                     margin_velocity=self.MARGIN_VELOCITY * 0.2,
                                                                     margin_acceleration=self.MARGIN_ACCELERATION * 0.2)
                    if self.trajectory_back is None:
                        rospy.logerr("Trajectory is None")

                    if DEBUG:
                        print("SLOWING")
                        # pdb.set_trace(header="Press C to see the slowing trajectory...")
                    self.time_start_slowing = time_now

            elif self.fsm_state == "SLOWING":
                time_now = rospy.get_time()
                # record fly trajectory data
                self.obj_trajectory.append(self.obj_cur)

                if time_now - self.time_start_slowing > self.trajectory_back.duration - dT:
                    if save_data:
                        self.save_tracking_data_to_npy()
                    break

                ref = self.trajectory_back.at_time(time_now - self.time_start_slowing)
                self.target_state.header.stamp = time_now
                self.target_state.position = ref[0]
                self.target_state.velocity = ref[1]
                self.target_state.effort = ref[2]


                self.target_state_pub.publish(self.target_state)
                self.command_pub.publish(self.convert_command_to_ROS(time_now, ref[0], ref[1], ref[2]))

            rate.sleep()

    def match_configuration(self, posture=None):
        base0 = np.array(self.box_position)[:2]
        print("box_position", self.box_position)
        q_candidates, phi_candidates, x_candidates = self.trajectoryGenerator.brt_robot_data_matching(
                                                    posture=posture,
                                                    box_pos=self.box_position)
        q_candidates, phi_candidates, x_candidates = self.trajectoryGenerator.filter_candidates(q_candidates,
                                                                                                phi_candidates,
                                                                                                x_candidates)
        if len(q_candidates) == 0:
            print("No result found")
            return 0

        trajs, throw_configs = self.trajectoryGenerator.generate_throw_config(
            q_candidates,
            phi_candidates,
            x_candidates,
            base0,
            qs=self.qs,
            qs_dot=self.qs_dot,
            posture=posture,
            simulation=False)

        if len(trajs) == 0:
            print("No trajectory found")
            return 0

        # select the minimum-time trajectory
        traj_durations = [traj.duration for traj in trajs]
        selected_idx = np.argmin(traj_durations)
        traj_throw = trajs[selected_idx]
        throw_config_full = throw_configs[selected_idx]

        qs = throw_config_full[0]
        qs_dot = throw_config_full[3]
        qs_dotdot = np.zeros_like(qs)

        return qs, qs_dot, qs_dotdot

    ## ---- ROS conversion and callbacks functions ---- ##
    def convert_command_to_ROS(self, time_now, qd, qd_dot, qd_dotdot):
        command = JointState()
        command.header.stamp = rospy.Time.from_sec(time_now)
        command.name = ['iiwa_joint1', 'iiwa_joint2', 'iiwa_joint3', 'iiwa_joint4', 'iiwa_joint5', 'iiwa_joint6',
                        'iiwa_joint7']
        command.position =  qd
        command.velocity = qd_dot
        command.effort = qd_dotdot

        return command

    def joint_states_callback(self, state):
        self.robot_state.header = copy.deepcopy(state.header)
        self.robot_state.position = copy.deepcopy(state.position)
        self.robot_state.velocity = copy.deepcopy(state.velocity)
        self.robot_state.effort = copy.deepcopy(state.effort)

    def _hand_joint_states_callback(self, data):
        self._qh = np.copy(np.array(data.position))

    def _send_hand_position(self, joints: np.ndarray) -> None:
        self.hand_joint_cmd.position = list(joints)
        self.joint_cmd_pub.publish(self.hand_joint_cmd)

    def _optitrack_callback(self, msg):
        self.obj_cur = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

    def _box_pos_callback(self, msg):
        BOX_HEIGHT = 0.1
        self.box_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        self.box_position[2] -= BOX_HEIGHT

    def scheduler_callback(self, msg, qA, qA_dot, qA_dotdot,
                           qB=None, qB_dot=None, qB_dotdot=None):
        # print("scheduler msg", msg)
        if self.fsm_state == "IDLE_THROWING" and msg.data == 1:
            # compute new trajectory to throw from current position
            q_cur = np.array(self.robot_state.position)
            q_cur_dot = np.array(self.robot_state.velocity)

            self.throwing_traj = self.get_traj_from_ruckig(q_cur, q_cur_dot, np.zeros(7),
                                                           qA, qA_dot, qA_dotdot,
                                                      margin_velocity=self.MARGIN_VELOCITY,
                                                      margin_acceleration=self.MARGIN_ACCELERATION)
            if qB is not None:
                self.throwing_traj_extend = self.get_traj_from_ruckig(qA, qA_dot, np.zeros(7),
                                                               qB, qB_dot, qB_dotdot,
                                                               margin_velocity=self.MARGIN_VELOCITY,
                                                               margin_acceleration=self.MARGIN_ACCELERATION)
            else:
                self.throwing_traj_extend = None
            if self.throwing_traj is None:
                rospy.logerr("Trajectory is None")

            self.time_start_throwing = rospy.get_time()
            self.time_throw = self.throwing_traj.duration + self.throwing_traj_extend.duration\
                if self.throwing_traj_extend is not None else self.throwing_traj.duration

            self.fsm_state = "THROWING"
            # print("Throwing...")
        return self.throwing_traj.duration

    def deactivate_gripper(self):
        self._send_hand_position(self.hand_home_pose)

    # Path to the build directory including a file similar to 'ruckig.cpython-37m-x86_64-linux-gnu'.
    build_path = Path(__file__).parent.absolute().parent / 'build'
    path.insert(0, str(build_path))

    def get_traj_from_ruckig(self, q0, q0_dot, q0_dotdot,
                             qd, qd_dot, qd_dotdot,
                             margin_velocity=1.0, margin_acceleration=0.7,
                             margin_jerk=None):

        if margin_jerk is None:
            margin_jerk = self.MARGIN_JERK

        inp = InputParameter(len(q0))
        inp.current_position = q0
        inp.current_velocity = q0_dot
        inp.current_acceleration = q0_dotdot

        inp.target_position = qd
        inp.target_velocity = qd_dot
        inp.target_acceleration = qd_dotdot

        inp.max_velocity = self.max_velocity * margin_velocity
        inp.max_acceleration = self.max_acceleration * margin_acceleration
        inp.max_jerk = self.max_jerk * margin_jerk

        otg = Ruckig(len(q0))
        trajectory = Trajectory(len(q0))
        _ = otg.calculate(inp, trajectory)

        return trajectory

    def simulation_mujoco(self, qd, qd_dot, qd_dotdot):
        throwing_traj = self.get_traj_from_ruckig(self.qs, self.qs_dot, np.zeros(7),
                                                       qd, qd_dot, qd_dotdot,
                                                       margin_velocity=self.MARGIN_VELOCITY,
                                                       margin_acceleration=self.MARGIN_ACCELERATION)

        trajectory_back = self.get_traj_from_ruckig(qd, qd_dot, np.zeros(7),
                                                         self.qs, self.qs_dot, self.qs_dotdot,
                                                         margin_velocity=self.MARGIN_VELOCITY * 0.5,
                                                         margin_acceleration=self.MARGIN_ACCELERATION * 0.5)

        self.trajectoryGenerator.robot._set_joints(self.qs, render=True)
        intermediate_time, traj_throw_back = self.trajectoryGenerator.concatenate_trajectories(
            throwing_traj, trajectory_back
        )

        self.trajectoryGenerator.throw_simulation_mujoco(intermediate_time=intermediate_time, ref_sequence=traj_throw_back)

    def multi_match_configuration(self, boxes_pos):
        _, best_throw_config_pair, _ = (
            self.trajectoryGenerator.multi_waypoint_solve(boxes_pos, animate=False, full_search=True))

        if best_throw_config_pair is None:
            return None

        desire_q_A = best_throw_config_pair[0][0]
        desire_q_A_dot = best_throw_config_pair[0][3]
        desire_q_B = best_throw_config_pair[1][0]
        desire_q_B_dot = best_throw_config_pair[1][3]

        return np.array([desire_q_A, desire_q_A_dot, desire_q_B, desire_q_B_dot])


if __name__ == '__main__':
    rospy.init_node("throwing_controller", anonymous=True)
    box_position = [1.3, 0.07, -0.1586]
    box1 = np.array([1.25, 0.35, -0.1])
    box2 = np.array([0.4, 1.3, -0.1])
    multi_box_positions = np.array([box1, box2])
    throwing_controller = ThrowingController(box_position=box_position)
    for nTry in range(100):
        print("test number", nTry + 1)

        throwing_controller.fsm_state = "IDLE"
        # throwing_controller.run(save_data=False)
        throwing_controller.run_multi_throwing(multi_box_positions)

        time.sleep(3)

        if rospy.is_shutdown():
            break