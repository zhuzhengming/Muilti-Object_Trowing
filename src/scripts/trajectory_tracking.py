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


class ThrowingController:
    def __init__(self, path_prefix='../', box_position=None):
        self.box_position = box_position if box_position is not None else 0
        self.path_prefix = path_prefix
        self._allegro_init()
        self._iiwa_init()
        self._control_init()
        self._general_init()
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
                                                       self.box_position, xml_path, model_exist=True)

        # initialize ROS subscriber/publisher
        self.fsm_state_pub = rospy.Publisher('fsm_state', String, queue_size=1)
        self.target_state_pub = rospy.Publisher('/computed_torque_controller/target_state', JointState,
                                                queue_size=10)  # for debug

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
        self.envelop_pose = np.array([-0.14, 1.78, 1.20, 1.45, -0.32, 1.71, 1.37, 0.85, -0.51, 1.77, 1.41, 0.55, 0.81, 0.55, 0.17, 1.35])
        # self.envelop_pose = np.array(rospy.get_param('/envelop_pose'))
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

        self._send_hand_position(self.envelop_pose)

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
        # self.max_acceleration = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
        # self.max_jerk = np.array([7500, 3750, 5000, 6250, 7500, 10000, 10000])


    def _control_init(self):
        self.dt = 5e-3
        robot_state = rospy.wait_for_message("/iiwa/joint_states", JointState)
        self.q0 = np.array(robot_state.position)

        # qs for the initial state
        self.qs = np.array([0.4217, 0.8498, 0.1635, -1.0926, -0.0098, 0.6, 1.2881])
        self.qs_dot = np.zeros(7)
        self.qs_dotdot = np.zeros(7)

        # qd for the throwing state
        qd_offset = [0.0, 0.0, 0.2, 0.2, -0.1, 0.2, 0.2]
        qd_dot_offset = np.zeros(7)

        self.qd = self.qs + qd_offset
        self.qd_dot = qd_dot_offset
        self.qd_dotdot = np.zeros(7)

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


    def run(self, max_run_time=30.0):
        # ------------ Control Loop ------------ #
        start_time = time.time()
        dT = self.dt
        rate = rospy.Rate(1.0 / dT)

        qd, qd_dot, qd_dotdot = self.match_configuration(posture='posture1')


        while not rospy.is_shutdown():
            if (time.time() - start_time) > max_run_time:
                rospy.logwarn("run() time limit exceeded, saving data and breaking...")
                self.save_tracking_data_to_npy() # save data if stuck
                break

            # Publish state and fsm for debug
            self.fsm_state_pub.publish(self.fsm_state)
            # update robot state
            q_cur = np.array(self.robot_state.position)
            q_cur_dot = np.array(self.robot_state.velocity)
            q_cur_effort = np.array(self.robot_state.effort)


            if self.fsm_state == "IDLE":
                # pdb.set_trace(header="Press C to start homing...")
                # print("HOMING...")

                self.homing_traj = self.get_traj_from_ruckig(q_cur, q_cur_dot, np.zeros(7),
                                                             self.qs, self.qs_dot, self.qs_dotdot,
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
                if np.linalg.norm(error_position) < self.ERROR_THRESHOLD:
                    # Jump to next state
                    self.fsm_state = "IDLE_THROWING"
                    time.sleep(0.5)
                    # print("IDLE_THROWING")
                    # pdb.set_trace(header="Press C to see the throwing trajectory...")
                    self.scheduler_callback(Int64(1), qd, qd_dot, qd_dotdot)

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


                self.target_state_pub.publish(self.target_state)
                self.command_pub.publish(self.convert_command_to_ROS(time_now, ref[0], ref[1], ref[2]))

            rate.sleep()

    def match_configuration(self, posture=None):
        base0 = -np.array(self.box_position)[:2]
        q_candidates, phi_candidates, x_candidates = (
            self.trajectoryGenerator.brt_robot_data_matching(posture=posture))
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

    def scheduler_callback(self, msg, qd, qd_dot, qd_dotdot):
        # print("scheduler msg", msg)
        if self.fsm_state == "IDLE_THROWING" and msg.data == 1:
            # compute new trajectory to throw from current position
            q_cur = np.array(self.robot_state.position)
            q_cur_dot = np.array(self.robot_state.velocity)

            self.throwing_traj = self.get_traj_from_ruckig(q_cur, q_cur_dot, np.zeros(7),
                                                           qd, qd_dot, qd_dotdot,
                                                      margin_velocity=self.MARGIN_VELOCITY,
                                                      margin_acceleration=self.MARGIN_ACCELERATION)
            if self.throwing_traj is None:
                rospy.logerr("Trajectory is None")

            self.time_start_throwing = rospy.get_time()
            self.time_throw = self.throwing_traj.duration
            self.fsm_state = "THROWING"
            # print("Throwing...")

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

if __name__ == '__main__':
    rospy.init_node("throwing_controller", anonymous=True)
    box_position = [0.2, -1.1, 0.3]
    throwing_controller = ThrowingController(box_position=box_position)
    for nTry in range(100):
        print("test number", nTry + 1)

        throwing_controller.fsm_state = "IDLE"
        throwing_controller.run()

        time.sleep(1)

        if rospy.is_shutdown():
            break