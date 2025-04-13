import sys
sys.path.append("../")
import time
import os
import rospy
import copy
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

from std_msgs.msg import String
from sensor_msgs.msg import JointState
from std_msgs.msg import Int64
import threading

from pathlib import Path
from sys import path
from ruckig import InputParameter, Ruckig, Trajectory
from utils.mujoco_interface import Robot
import kinematics.allegro_hand_sym as allegro
from trajectory_generator import TrajectoryGenerator

import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt

# global variables
SIMULATION = True  # Set to True to run the simulation before commanding the real robot
REAL_ROBOT_STATE = False  # Set to True to use the real robot state to start the simulation

## ---- ROS conversion and callbacks functions ---- ##
class Throwing_controller:
    def __init__(self, simulator='mujoco', test_id=None, path_prefix='../', box_position=None):
        rospy.init_node("throwing_controller", anonymous=True)
        hedgehog_path = '../hedgehog_revised'
        brt_path = '../brt_data'
        xml_path = '../description/iiwa7_allegro_ycb.xml'
        self.box_position = box_position if box_position is not None else 0

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

        # allegro controller
        self.hand_home_pose = np.array(rospy.get_param('/hand_home_pose'))
        self.envelop_pose = np.array(rospy.get_param('/envelop_pose'))
        self.joint_cmd_pub = rospy.Publisher('/allegroHand_0/joint_cmd', JointState, queue_size=10)
        rospy.Subscriber('/allegroHand_0/joint_states', JointState, self._hand_joint_states_callback)
        self.hand = allegro.Robot(right_hand=False, path_prefix=path_prefix)  # load the left hand kinematics
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
        self.simulator = simulator

        # Ruckig margins for throwing
        self.MARGIN_VELOCITY = rospy.get_param('/MARGIN_VELOCITY')
        self.MARGIN_ACCELERATION = rospy.get_param('/MARGIN_ACCELERATION')
        self.MARGIN_JERK = rospy.get_param('/MARGIN_JERK')

        # constraints of iiwa 7
        self.max_velocity = np.array(rospy.get_param('/max_velocity'))
        self.max_acceleration = np.array(rospy.get_param('/max_acceleration'))
        self.max_jerk = np.array(rospy.get_param('/max_jerk'))

        # q0 is the home state and control period
        if SIMULATION:
            self.q0 = np.array([-0.32032486, 0.02707055, -0.22881525, -1.42611918, 1.38608943, 0.5596685, -1.34659665 + np.pi])
            self.dt = 5e-3
        else:
            self.dt = 5e-3
            robot_state = rospy.wait_for_message("/iiwa/joint_states", JointState)
            self.q0 = np.array(robot_state.position)

        # qs for the initial state
        self.qs = np.array([0.4217, 0.8498, 0.1635, -1.0926, -0.0098, 0.6, 1.2881])
        self.qs_dot = np.zeros(7)
        self.qs_dotdot = np.zeros(7)

        # qd for the throwing state
        qd_offset = np.zeros(7)
        qd_dot_offset = np.zeros(7)

        self.test_id = test_id
        if self.test_id is not None:
            qd_offset[self.test_id] = -0.3
            qd_dot_offset[self.test_id] = -self.max_velocity[self.test_id] * self.MARGIN_VELOCITY
            if self.test_id == 3:
                qd_offset[self.test_id] = -qd_offset[self.test_id]
                qd_dot_offset[self.test_id] = -qd_dot_offset[self.test_id]

        # qd_offset = np.array([-0.3, 0.0, -0.2, 0.0, -0.3, 0.0, 0.0])
        # qd_dot_offset = np.array([-0.4, 0.0, -0.2, 0.0, -0.3, 0.0, 0.0])
        self.qd = self.qs + qd_offset
        self.qd_dot = qd_dot_offset
        self.qd_dotdot = np.zeros(7)

        # compute the nominal throwing and slowing trajectory
        self.trajectory = self.get_traj_from_ruckig(self.qs, self.qs_dot, self.qs_dotdot,
                                                    self.qd, self.qd_dot, self.qd_dotdot,
                                                    margin_velocity=self.MARGIN_VELOCITY,
                                                    margin_acceleration=self.MARGIN_ACCELERATION)

        self.trajectory_back = self.get_traj_from_ruckig(self.qd, self.qd_dot, self.qd_dotdot,
                                                         self.qs, self.qs_dot, self.qs_dotdot,
                                                         margin_velocity=self.MARGIN_VELOCITY,
                                                         margin_acceleration=self.MARGIN_ACCELERATION * 0.5)

        if self.trajectory is None or self.trajectory_back is None:
            rospy.logerr("Trajectory is None")

        self.traj_time = self.trajectory.duration
        self.traj_back_time = self.trajectory_back.duration

        if simulator == 'mujoco' and SIMULATION == True:
            obj_name = ''
            model = mujoco.MjModel.from_xml_path(xml_path)
            data = mujoco.MjData(model)
            mujoco.mj_step(model, data)

            self.view = viewer.launch_passive(model, data)

            obj_names = ['banana', 'bottle', 'chip_can', 'soft_scrub', 'sugar_box']
            num = 0
            obj = obj_names[num]
            self.r = Robot(model, data, self.view, auto_sync=True, obj_names=obj_names)

        # Run simulation once for visualization
        if SIMULATION:
            self.run_simulation_mujoco()

        self.stamp = []
        self.real_pos = []
        self.real_vel = []
        self.real_eff = []
        self.target_pos = []
        self.target_vel = []
        self.target_eff = []
        self.pos_error_sum = np.zeros(7)
        self.vel_error_sum = np.zeros(7)

        while True:
            if SIMULATION:
                self.robot_state.position = np.array(self.r.q)
                self.robot_state.velocity = np.array(self.r.dq)

            print("Got robot state")
            break

        # get parameters and initialization
        self.ERROR_THRESHOLD = rospy.get_param('/ERROR_THRESHOLD')  # Threshold to switch from homing to throwing state
        self.GRIPPER_DELAY = rospy.get_param('/GRIPPER_DELAY')

        self.time_throw = np.inf  # Planned time of throwing
        self.fsm_state = "IDLE"

        self.nIter_time = 0.0
        time.sleep(1.0)

    def run_simulation_mujoco(self):

        if REAL_ROBOT_STATE:
            # get initial robot state from ROS message
            robot_state = rospy.wait_for_message("/iiwa/joint_states", JointState)
            print("initial robot state")

            q0 = np.array(robot_state.position)
            q0_dot = np.array(robot_state.velocity)
            self.qs[0] = q0[0]
            self.qs_dot[0] = q0_dot[0]
            # compute the nominal throwing and slowing trajectory
            self.trajectory = self.get_traj_from_ruckig(self.qs, self.qs_dot, self.qs_dotdot,
                                                   self.qd, self.qd_dot, self.qd_dotdot,
                                                   margin_velocity=self.MARGIN_VELOCITY,
                                                   margin_acceleration=self.MARGIN_ACCELERATION)
            self.trajectory_back = self.get_traj_from_ruckig(self.qd, self.qd_dot, self.qd_dotdot,
                                                        self.qs, self.qs_dot, self.qs_dotdot,
                                                   margin_velocity=self.MARGIN_VELOCITY,
                                                   margin_acceleration=self.MARGIN_ACCELERATION * 0.5)

            if self.trajectory is None or self.trajectory_back is None:
                rospy.logerr("Trajectory is None")

            self.traj_time = self.trajectory.duration
            self.traj_back_time = self.traj_time.duration
        else:
            # get initial robot state from Mujoco
            q0 = np.copy(self.r.q)
            q0_dot = np.copy(self.r.dq)

        qs = np.copy(self.r.q)
        qs[0] += 0.2
        qs_dot = np.copy(self.r.dq)

        # reset the robot to initial configuration
        self.r.iiwa_hand_go(q=q0, d_pose=q0_dot, qh=np.zeros(16))
        # pdb.set_trace(header="PDB PAUSE: Press C to start simulation...")

        # generate the trajectory to go to qs
        trajectory_to_qs = self.get_traj_from_ruckig(q0, q0_dot, np.zeros(7),
                                                     qs, qs_dot, self.qs_dotdot,
                                                     margin_velocity=0.2, margin_acceleration=0.1)

        if trajectory_to_qs is None:
            rospy.logerr("Trajectory is None")

        flag = True
        delta_t = 0.002

        # execute homing trajectory
        tt =0
        while True:
            ref = trajectory_to_qs.at_time(tt)
            self.r.iiwa_hand_go(q=ref[0], d_pose=ref[1], qh=np.zeros(16))

            time.sleep(delta_t)
            tt += delta_t
            if tt > trajectory_to_qs.duration:
                break

        tt = 0
        while True:
            if flag:
                ref_full = self.trajectory.at_time(tt)
                ref = [ref_full[i][:7] for i in range(3)]

                self.r.iiwa_hand_go(q=ref[0], d_pose=ref[1], qh=np.zeros(16))
            else:
                # slow down the robot
                ref_full = self.trajectory_back.at_time(tt - self.traj_time)
                ref = [ref_full[i][:7] for i in range(3)]

                self.r.iiwa_hand_go(q=ref[0], d_pose=ref[1], qh=np.zeros(16))

            tt += delta_t
            if tt > self.trajectory.duration and tt <= self.trajectory.duration + self.trajectory_back.duration:
                flag = False
            time.sleep(delta_t)
            if tt > 6.0:
                break
        # self.view.close()

    def save_tracking_data_to_npy(self):

        filename = '../output/data/throwing.npy'
        # Save data to npy files
        # self.pos_error_sum[self.test_id] += np.sum(np.abs(
        #     np.array(self.target_pos)[:,self.test_id] - np.array(self.real_pos)[:,self.test_id]
        # ))
        #
        # self.vel_error_sum[self.test_id] += np.sum(np.abs(
        #     np.array(self.target_vel)[:,self.test_id] - np.array(self.real_vel)[:,self.test_id]
        # ))

        np.save(filename, {'stamp': self.stamp,
                           'real_pos': self.real_pos,
                           'real_vel': self.real_vel,
                           'real_eff': self.real_eff,
                           'target_pos': self.target_pos,
                           'target_vel': self.target_vel,
                           'target_eff': self.target_eff})
        print("Tracking data saved to npy files.")


    def run(self, max_run_time=30.0, render=True):
        # ------------ Control Loop ------------ #
        start_time = time.time()
        dT = self.dt
        rate = rospy.Rate(1.0 / dT)

        if not SIMULATION:
            qs, qs_dot, qs_dotdot = self.match_configuration()
        else:
            qs, qs_dot, qs_dotdot = self.qs, self.qs_dot, self.qs_dotdot

        while not rospy.is_shutdown():
            if (time.time() - start_time) > max_run_time:
                rospy.logwarn("run() time limit exceeded, saving data and breaking...")
                self.save_tracking_data_to_npy() # save data if stuck
                break

            # Publish state and fsm for debug
            self.fsm_state_pub.publish(self.fsm_state)
            # update robot state
            if SIMULATION:
                q_cur = np.array(self.r.q)
                q_cur_dot = np.array(self.r.dq)
                q_cur_effort = np.zeros(7)
            else:
                q_cur = np.array(self.robot_state.position)
                q_cur_dot = np.array(self.robot_state.velocity)
                q_cur_effort = np.array(self.robot_state.effort)


            if self.fsm_state == "IDLE":
                # pdb.set_trace(header="Press C to start homing...")
                # print("HOMING...")

                self.homing_traj = self.get_traj_from_ruckig(q_cur, q_cur_dot, np.zeros(7),
                                                             qs, qs_dot, qs_dotdot,
                                                             margin_velocity=self.MARGIN_VELOCITY,
                                                             margin_acceleration=self.MARGIN_ACCELERATION)
                if self.homing_traj is None:
                    rospy.logerr("Trajectory is None")

                # print("IDLING: initial state", self.q0, "homing trajectory duration", self.homing_traj.duration)

                # update state
                self.fsm_state = "HOMING"
                self.time_start_homing = rospy.get_time()


            elif self.fsm_state == "HOMING":

                # Activate integrator term when close to target
                error_position = np.array(q_cur) - np.array(self.qs)
                if SIMULATION:
                    self.ERROR_THRESHOLD *= 2
                if np.linalg.norm(error_position) < self.ERROR_THRESHOLD:
                    # Jump to next state
                    self.fsm_state = "IDLE_THROWING"
                    time.sleep(0.5)
                    # print("IDLE_THROWING")
                    # pdb.set_trace(header="Press C to see the throwing trajectory...")
                    self.scheduler_callback(Int64(1))

                time_now = rospy.get_time()
                ref = self.homing_traj.at_time(time_now - self.time_start_homing)
                self.target_state.header.stamp = time_now
                self.target_state.position = ref[0]
                self.target_state.velocity = ref[1]
                self.target_state.effort = ref[2]

                if SIMULATION:
                    self.r.iiwa_hand_go(q=self.target_state.position,
                                        d_pose=self.target_state.velocity,
                                        qh=np.zeros(16), render=render)
                else:
                    self.target_state_pub.publish(self.target_state)
                    self.command_pub.publish(self.convert_command_to_ROS(time_now, ref[0], ref[1], ref[2]))

            elif self.fsm_state == "IDLE_THROWING":
                continue

            elif self.fsm_state == "THROWING":
                # execute throwing trajectory
                time_now = rospy.get_time()
                # release gripper
                if time_now - self.time_start_throwing > self.throwing_traj.duration - self.GRIPPER_DELAY:
                    threading.Thread(target=self.deactivate_gripper).start()

                if time_now - self.time_start_throwing > self.throwing_traj.duration - dT:
                    self.fsm_state = "SLOWING"

                    self.trajectory_back = self.get_traj_from_ruckig(q_cur, q_cur_dot, self.qd_dotdot,
                                                                     self.qs, self.qs_dot, self.qs_dotdot,
                                                                margin_velocity=self.MARGIN_VELOCITY * 0.5,
                                                                margin_acceleration=self.MARGIN_ACCELERATION * 0.5)
                    if self.trajectory_back is None:
                        rospy.logerr("Trajectory is None")

                    # print("SLOWING")
                    # pdb.set_trace(header="Press C to see the slowing trajectory...")
                    self.time_start_slowing = time_now

                ref = self.throwing_traj.at_time(time_now - self.time_start_throwing)
                self.target_state.header.stamp = time_now
                self.target_state.position = ref[0]
                self.target_state.velocity = ref[1]
                self.target_state.effort = ref[2]

                if SIMULATION:
                    self.r.iiwa_hand_go(q=self.target_state.position,
                                        d_pose=self.target_state.velocity,
                                        qh=np.zeros(16), render=render)
                else:
                    self.target_state_pub.publish(self.target_state)
                    self.command_pub.publish(self.convert_command_to_ROS(time_now, ref[0], ref[1], ref[2]))

                # record error
                if(self.time_throw - (time_now - self.time_start_throwing) < 1 * self.time_throw):

                    self.stamp.append(time_now)
                    self.real_pos.append(q_cur)
                    self.real_vel.append(q_cur_dot)
                    self.real_eff.append(q_cur_effort)
                    self.target_pos.append(self.target_state.position)
                    self.target_vel.append(self.target_state.velocity)
                    self.target_eff.append(self.target_state.effort)

            elif self.fsm_state == "SLOWING":
                time_now = rospy.get_time()

                if time_now - self.time_start_slowing > self.trajectory_back.duration - dT:
                    self.save_tracking_data_to_npy()
                    break

                ref = self.trajectory_back.at_time(time_now - self.time_start_slowing)
                self.target_state.header.stamp = time_now
                self.target_state.position = ref[0]
                self.target_state.velocity = ref[1]
                self.target_state.effort = ref[2]

                if SIMULATION:
                    self.r.iiwa_hand_go(q=self.target_state.position,
                                        d_pose=self.target_state.velocity,
                                        qh=np.zeros(16), render=render)
                else:
                    self.target_state_pub.publish(self.target_state)
                    self.command_pub.publish(self.convert_command_to_ROS(time_now, ref[0], ref[1], ref[2]))

            rate.sleep()

    def match_configuration(self, posture=None):
        base0 = -self.box_position[:2]
        q_candidates, phi_candidates, x_candidates = (
            self.trajectoryGenerator.brt_robot_data_matching(posture=posture))
        if len(q_candidates) == 0:
            print("No result found")
            return 0

        trajs, throw_configs = self.trajectoryGenerator.generate_throw_config(
            q_candidates, phi_candidates, x_candidates, base0)

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

    def scheduler_callback(self, msg):
        # print("scheduler msg", msg)
        if self.fsm_state == "IDLE_THROWING" and msg.data == 1:
            # compute new trajectory to throw from current position
            if SIMULATION:
                q_cur = self.r.q
                q_cur_dot = self.r.dq
            else:
                q_cur = np.array(self.robot_state.position)
                q_cur_dot = np.array(self.robot_state.velocity)

            self.throwing_traj = self.get_traj_from_ruckig(q_cur, q_cur_dot, np.zeros(7),
                                                           self.qd, self.qd_dot, self.qd_dotdot,
                                                      margin_velocity=self.MARGIN_VELOCITY,
                                                      margin_acceleration=self.MARGIN_ACCELERATION)
            if self.throwing_traj is None:
                rospy.logerr("Trajectory is None")

            self.time_start_throwing = rospy.get_time()
            self.time_throw = self.throwing_traj.duration
            self.fsm_state = "THROWING"
            # print("Throwing...")

    def deactivate_gripper(self):
        if SIMULATION:
            self.r.hand_move_torque(qh=self.r.home_pose, move_hand_only=True)
        else:
            self.r.move_to_joints(self.hand_home_pose, vel=[0.2, 8.0])


    # Path to the build directory including a file similar to 'ruckig.cpython-37m-x86_64-linux-gnu'.
    build_path = Path(__file__).parent.absolute().parent / 'build'
    path.insert(0, str(build_path))

    def get_traj_from_ruckig(self, q0, q0_dot, q0_dotdot,
                             qd, qd_dot, qd_dotdot,
                             margin_velocity=1.0, margin_acceleration=0.7,
                             margin_jerk=None):

        """
            Generates a smooth trajectory using the Ruckig algorithm for the given joint states.

            Parameters:
            - q0: Initial joint positions
            - q0_dot: Initial joint velocities
            - q0_dotdot: Initial joint accelerations
            - qd: Target joint positions
            - qd_dot: Target joint velocities
            - qd_dotdot: Target joint accelerations
            - margin_velocity: Velocity margin
            - margin_acceleration: Acceleration margin
            - margin_jerk: Jerk margin

            Returns:
            - trajectory: The generated trajectory object
            """

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


#######################################################################################################################
# Test the parameters of KD controller
    def batch_test_kp_kd(self, kp_min=10, kp_max=1000,
                         kd_min=5, kd_max=500,
                         num_point=10, N_runs=5,
                         SAVE_FILE=True):
        if self.test_id == 3:
            kp_max = 800
            kd_max = 400
        elif self.test_id == 4 or self.test_id == 5:
            kp_max = 500
            kd_max = 250
        elif self.test_id == 6:
            kp_max = 300
            kd_max = 200

        self.view.close() # close mujoco viewer

        kp_candidates = np.linspace(kp_min, kp_max, num_point)
        kd_candidates = np.linspace(kd_min, kd_max, num_point)

        pos_error = np.zeros((num_point, num_point))
        vel_error = np.zeros((num_point, num_point))

        for i, kp_val in enumerate(kp_candidates):
            for j, kd_val in enumerate(kd_candidates):
                self.r._joint_kp[self.test_id] = kp_val
                self.r._joint_kd[self.test_id] = kd_val

                self.pos_error_sum[self.test_id] = 0
                self.vel_error_sum[self.test_id] = 0

                for n in range(N_runs):
                    self.stamp.clear()
                    self.real_pos.clear()
                    self.real_vel.clear()
                    self.real_eff.clear()
                    self.target_pos.clear()
                    self.target_vel.clear()
                    self.target_eff.clear()

                    self.fsm_state = "IDLE"
                    self.run(render=False)
                    time.sleep(1)
                    if rospy.is_shutdown():
                        break

                pos_error[i, j] = self.pos_error_sum[self.test_id] / N_runs
                vel_error[i, j] = self.vel_error_sum[self.test_id] / N_runs
                print(f"joint {self.test_id:d}, processing: {(i*num_point + (j+1))/(num_point*num_point)*100:.2f}%")

        if SAVE_FILE:
            data_to_save = {
                'kp_candidates': kp_candidates,
                'kd_candidates': kd_candidates,
                'pos_error': pos_error,
                'vel_error': vel_error
            }

            filepath = '../output/data/test_kp_kd'
            os.makedirs(filepath, exist_ok=True)
            save_file = os.path.join(filepath, f"kp_kd_joint{self.test_id}.npy")

            np.save(save_file, data_to_save)

        else:
            self.plot_tracking_data(kp_candidates, kd_candidates, pos_error, vel_error)



    def plot_tracking_data(self, kp_candidate, kd_candidate, pos_error_sum, vel_error_sum):
        kp_mesh, kd_mesh = np.meshgrid(kp_candidate, kd_candidate)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection='3d')

        surface1 = ax1.plot_surface(kp_mesh, kd_mesh, pos_error_sum, edgecolor='none')
        ax1.set_xlabel('Kp')
        ax1.set_ylabel('Kd')
        ax1.set_zlabel('Position Tracking Error')

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        surface2 = ax2.plot_surface(kp_mesh, kd_mesh, vel_error_sum, edgecolor='none')
        ax2.set_xlabel('Kp')
        ax2.set_ylabel('Kd')
        ax2.set_zlabel('Velocity Tracking Error')

        plt.show()

if __name__ == '__main__':

    simulation_mode = 'mujoco'
    # for test_id in range(7):
    #     throwing_controller = Throwing_controller(simulator=simulation_mode, test_id=test_id)
    #     throwing_controller.batch_test_kp_kd()
    #     throwing_controller.view.close()
    #     print("joint finished:", test_id)

    throwing_controller = Throwing_controller(simulator=simulation_mode,
                                              test_id=6)
    for nTry in range(100):
        print("test number", nTry + 1)

        throwing_controller.fsm_state = "IDLE"
        throwing_controller.run()

        time.sleep(1)

        # Stop controller when ROS is stopped
        if rospy.is_shutdown():
            break