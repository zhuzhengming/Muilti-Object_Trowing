"""
control interface for iiwa and allegro hand
"""
import sys
sys.path.append("../")
import time

import rospy
import numpy as np
from functools import partial
import matplotlib
import tools.rotations as rot
import kinematics.allegro_hand_sym as allegro
# from iiwa_tools.srv import GetIK, GetFK
from trac_ik_python.trac_ik import IK
from urdf_parser_py.urdf import URDF  # need to install it under py3
import kinematics.kdl_parser as kdl_parser
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

import PyKDL as kdl
# https://bitbucket.org/traclabs/trac_ik/src/master/trac_ik_python/, install it by `pip install -e .` under the cond env


from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension

import copy
import signal
import sys


class Robot():
    @staticmethod
    def clean_up(signum, frame):
        if not rospy.is_shutdown():
            rospy.signal_shutdown("User requested shutdown")
        sys.exit()

    def __init__(self, optitrack_frame_names=None, position_control=True, calibration=False, camera=False,
                 camera_object_name=None, path_prefix='../'):

        rospy.init_node('iiwa_allegro_controller', anonymous=True)

        self.optitrack_frame_names = optitrack_frame_names

        # recalibration should be done if the relative pose between the marker and iiwa base changes
        # self.iiwa_base2m = np.loadtxt(path_prefix +
        #                               'description/config/iwwa_link_0_2_iiwabase7_calibration.txt')  # iiwa_link_0 to iiwa_base7
        # self.allegro_base2_ee = np.loadtxt(path_prefix +
        #                                    'description/config/allegro_left_base_2_iiwa_link_ee.txt')  # allegro_base to iiwa_link_ee
        self.base2world = None
        self.base2world_b_ = True

        # subscriber all object poses from the Optitrack system
        if self.optitrack_frame_names is not None:
            self._x_obj = {}

            for marker in self.optitrack_frame_names:
                rospy.Subscriber('/vrpn_client_node/' + marker + '/pose', PoseStamped,
                                 partial(self.object_pose_callback, name=marker,
                                         obj=marker in self.optitrack_frame_names))

        # For hand initialization
        self.joint_cmd_pub = rospy.Publisher('/allegroHand_0/joint_cmd', JointState, queue_size=10)
        rospy.Subscriber('/allegroHand_0/joint_states', JointState, self._hand_joint_states_callback)
        self.hand = allegro.Robot(right_hand=False, path_prefix=path_prefix)  # load the left hand kinematics
        self.fingertip_sites = ['index_site', 'middle_site', 'ring_site',
                                'thumb_site']  # These site points are the fingertip (center of semisphere) positions
        self._qh = np.zeros(16) # joints of hand
        self._q = np.zeros(7) # joints of iiwa
        self._dq = np.zeros(7) # velocity of iiwa's joints
        self._effort = np.zeros(7) # effort of iiwa
        self.hand_joint_cmd = JointState()
        self.hand_joint_cmd.name = ['joint_0', 'joint_1', 'joint_2', 'joint_3',
                                    # index finger: ad/abduction, extensions
                                    'joint_4', 'joint_5', 'joint_6', 'joint_7',  # middle finger
                                    'joint_8', 'joint_9', 'joint_10', 'joint_11',  # ring finger
                                    'joint_12', 'joint_13', 'joint_14', 'joint_15']  # thumb
        self.hand_joint_cmd.position = []  # 0-3: index, 4-7: middle, 8-11: ring, 12-15: thumb
        self.hand_joint_cmd.velocity = []
        self.hand_joint_cmd.effort = []
        self.hand_bounds = np.array([])

        # For iiwa initialization
        self.iiwa_bounds = np.array([[-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239, -2.96705972839,
                                      -2.09439510239, -3.05432619099],
                                     [2.96705972839, 2.09439510239, 2.96705972839, 2.09439510239, 2.96705972839,
                                      2.09439510239, 3.05432619099]])
        self._iiwa_js_sub = rospy.Subscriber("/iiwa/joint_states", JointState, self._iiwa_joint_state_cb)

        if position_control:
            self.control_mode = "position"
            self._iiwa_position_pub = rospy.Publisher("/iiwa/PositionController/command", Float64MultiArray,
                                                      queue_size=10)
            self._sending_torque = False
        else:
            self.control_mode = "torque"
            self._sending_torque = False
            self._torque_cmd = np.zeros(7)
            self._iiwa_torque_pub = rospy.Publisher("/iiwa/TorqueController/command", Float64MultiArray,
                                                    queue_size=10)
            # parameters of PID
            self._joint_kp = np.array(rospy.get_param('/PD/joint_kp'))
            self._joint_kd = np.array(rospy.get_param('/PD/joint_kd'))
            self.q_cmd = None
            self.x_cmd = None

            # For torque control in Cartesian space, send to another sricpt not iiwa driver
            self.iiwa_cmd_pub_pose = rospy.Publisher('/iiwa_impedance_pose', PoseStamped, queue_size=10)

            # For torque control in Joint space, send to another sricpt not iiwa driver
            self.iiwa_cmd_pub_joint = rospy.Publisher('/iiwa_impedance_joint', JointState,
                                                      queue_size=10)
        # self.fk_service = '/iiwa/iiwa_fk_server'
        # self.get_fk = rospy.ServiceProxy(self.fk_service, GetFK)

        # pre-defined home position of hand and iiwa
        self.freq = rospy.get_param('/freq')# max frequency of iiwa
        self.dt = 1. / self.freq
        self._iiwa_home = np.array(rospy.get_param('_iiwa_home'))
        self._iiwa_home_pose = np.array(rospy.get_param('_iiwa_home_pose'))
        self._hand_home = np.zeros(16)
        self._hand_home[12] = 0.7

        # for iiwa base calibration
        # if calibration:
        #     self.marker_name = ['iiwa_base7', 'iiwa_ee_m'] # for calibration, always in world frame

        # iiwa ik
        self.iiwa_start_link = "iiwa_link_0"
        self.iiwa_end_link = "iiwa_link_ee"

        # From iiwa driver package
        self._urdf_str = rospy.get_param('/robot_description')

        # relax_ik
        self._ik_solver = IK(self.iiwa_start_link, self.iiwa_end_link, solve_type="distance", timeout=0.005, epsilon=5e-4)
        lower_bound, upper_bound = self._ik_solver.get_joint_limits()
        self._ik_solver.set_joint_limits(lower_bound, upper_bound)

        # self._iiwa_urdf = URDF.from_xml_string(self._urdf_str)
        self._iiwa_urdf = URDF.from_xml_file(path_prefix + 'description/iiwa_description/urdf/iiwa7_lasa.urdf')

        self._iiwa_urdf_tree = kdl_parser.kdl_tree_from_urdf_model(self._iiwa_urdf)
        self._iiwa_urdf_chain = self._iiwa_urdf_tree.getChain(self.iiwa_start_link, self.iiwa_end_link)
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self._iiwa_urdf_chain)

        # self.fk_solver.
        self.jac_calc = kdl.ChainJntToJacSolver(self._iiwa_urdf_chain)
        # kdl.JntSpaceInertiaMatrix

        time.sleep(1)
        signal.signal(signal.SIGINT, Robot.clean_up)
        # rospy.spin()


    # update the position of objects optitrack detected
    def object_pose_callback(self, data: PoseStamped, name='iiwa_base7', obj=False):
        current_object_pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        current_object_quat = np.array(
            [data.pose.orientation.w, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z])
        tmp = np.concatenate([current_object_pos, current_object_quat])  # in Optitrack frame (world)
        self._x_obj[name] = tmp
        # todo, do the calibration
        if self.base2world is not None:
            # update the pose of objects to be manipulated, in base frame (iiwa_link_0)
            self._x_obj[name] = rot.pose_mul(rot.pose_inv(self.base2world), tmp)
        else:
            self._x_obj[name] = tmp  # in Optitrack frame
            if name == 'iiwa_base7' and self.base2world_b_:  # update it only once, and do not move the hand base anymore.
                self.base2world = rot.pose_mul(self._x_obj['iiwa_base7'], self.iiwa_base2m)
                self.base2world_b_ = False

    # update current joint angle
    def _hand_joint_states_callback(self, data):
        self._qh = np.copy(np.array(data.position))

    # update position, velicity, effort of iiwa
    def _iiwa_joint_state_cb(self, data):
        self._q = np.copy(np.array(data.position))
        self._dq = np.copy(np.array(data.velocity))
        self._effort = np.copy(np.array(data.effort))

    def _send_hand_position(self, joints: np.ndarray) -> None:
        self.hand_joint_cmd.position = list(joints)
        self.joint_cmd_pub.publish(self.hand_joint_cmd)

    def _send_iiwa_torque(self, torques: np.ndarray) -> None:

        iiwa_torque_cmd = Float64MultiArray()

        # Fill out layout information
        layout = MultiArrayLayout()
        layout.dim.append(MultiArrayDimension())
        layout.data_offset = 0
        layout.dim[0].size = len(torques)
        layout.dim[0].stride = 1
        layout.dim[0].label = "joints"
        iiwa_torque_cmd.layout = layout

        # Fill torques
        iiwa_torque_cmd.data = torques

        self._iiwa_torque_pub.publish(iiwa_torque_cmd)

    def kinesthetic_teaching(self):
        assert self.control_mode == 'torque'
        while True and not rospy.is_shutdown():
            self._send_iiwa_torque(np.zeros(7))
            time.sleep(self.dt)

    def _send_iiwa_position(self, joints: np.ndarray) -> None:

        iiwa_position_cmd = Float64MultiArray()

        # Fill out layout information
        layout = MultiArrayLayout()
        layout.dim.append(MultiArrayDimension())
        layout.data_offset = 0
        layout.dim[0].size = len(joints)
        layout.dim[0].stride = 1
        layout.dim[0].label = "joints"
        iiwa_position_cmd.layout = layout

        # Fill joints
        iiwa_position_cmd.data = joints

        self._iiwa_position_pub.publish(iiwa_position_cmd)

    def iiwa_go_home(self):
        if self.control_mode == 'position':
            self.move_to_joints(self._iiwa_home, vel=[0.1, 1])
        else:
            self._sending_torque = True
            # self._iiwa_joint_control(self._iiwa_home)
            self.iiwa_cartesion_impedance_control(self._iiwa_home_pose)
            # self.q_cmd = copy.deepcopy(self.q)
            self.x_cmd = copy.deepcopy(self._iiwa_home_pose)
            self._sending_torque = False
            print("Finish going home")

    def iiwa_cmd(self, x):
        p = PoseStamped()
        p.header.frame_id = 'iiwa_link_0'
        p.header.stamp = rospy.Time.now()

        p.pose.position.x = x[0]
        p.pose.position.y = x[1]
        p.pose.position.z = x[2]
        p.pose.orientation.w = x[3]
        p.pose.orientation.x = x[4]
        p.pose.orientation.y = x[5]
        p.pose.orientation.z = x[6]

        self.iiwa_cmd_pub_pose.publish(p)

    def hand_go_home(self):
        self.move_to_joints(self._hand_home)


    def iiwa_cartesion_impedance_control(self, xd, vel=0.05):
        self._sending_torque = True
        # interpolation
        xd_list = self.motion_generation(xd, vel=vel, cartesian=True)
        for i in range(xd_list.shape[0]):
            # self._iiwa_impedance(xd_list[i, :])
            # publish to controller_utils2 and keep sending command
            self.iiwa_cmd(xd_list[i, :])
            time.sleep(self.dt)

        self._sending_torque = False

    def _iiwa_impedance(self, pose: np.ndarray, d_pose=None):
        if d_pose is None:
            d_pose = np.zeros(6)

        kp = np.array([300, 40.])
        kd = np.sqrt(kp) * 2

        # Fx based on errors of position and velocity
        Fx = kp[0] * (pose[:3] - self.x[:3]) + kd[0] * (d_pose[:3] - self.dx[:3])


        # Fr based on errors of oriantation
        q = self.x[3:]  # [w x y z]
        qd = pose[3:]
        if qd[0] < 0:
            qd = -qd
        # d_theta = (quaternion.from_float_array(qd) * (quaternion.from_float_array(q)).conjugate()).log() * 2
        # d_theta = quaternion.as_float_array(d_theta)[1:]
        axis, angle = rot.quat2axisangle(rot.quat_mul(qd, rot.quat_conjugate(q)))
        d_theta = np.array(axis) * angle
        # if np.linalg.norm(d_theta) >1:
        #     print(d_theta)
        Fr = kp[1] * d_theta + kd[1] * (d_pose[3:] - self.dx[3:6])

        F = np.concatenate([Fx, Fr])
        J = self.J # Jacobian matrix
        # impedance_acc_des0 = J.T.dot(np.linalg.solve(J.dot(J.T) + 1e-10 * np.eye(6), F))
        impedance_acc_des1 = J.T @ F

        # Add stiffness and damping in the null space of the Jacobian
        nominal_qpos = np.zeros(7)
        null_space_damping = 0.1
        null_space_stiffness = 10
        projection_matrix = J.T.dot(np.linalg.solve(J.dot(J.T) + 1e-10 * np.eye(6), J))
        projection_matrix = np.eye(projection_matrix.shape[0]) - projection_matrix
        null_space_control = -null_space_damping * self.dq
        null_space_control += -null_space_stiffness * (
                self.q - nominal_qpos)
        tau_null = projection_matrix.dot(null_space_control)
        tau_null_c = np.clip(tau_null, -5, 5)  # set the torque limit for null space control  todo, normalize it
        impedance_acc_des = impedance_acc_des1 + tau_null_c

        # self.send_torque(impedance_acc_des + self.C)
        self._send_iiwa_torque(impedance_acc_des)

    def _send_iiwa_torque(self, torques: np.ndarray) -> None:

        iiwa_torque_cmd = Float64MultiArray()

        # Fill out layout information
        layout = MultiArrayLayout()
        layout.dim.append(MultiArrayDimension())
        layout.data_offset = 0
        layout.dim[0].size = len(torques)
        layout.dim[0].stride = 1
        layout.dim[0].label = "joints"
        iiwa_torque_cmd.layout = layout

        # Fill torques
        iiwa_torque_cmd.data = torques

        self._iiwa_torque_pub.publish(iiwa_torque_cmd)

    # joint space PID control
    # gravity complementation
    # tau = (kp * error_q + kd * error_dq) + J^{t} F_{ext}

    def _iiwa_joint_space_impedance(self, q_target, d_qd=None):
        """
        directly sending torque
        :param q_target:
        :return:
        """

        qacc_max = 10.0
        if d_qd is None:
            d_qd = np.zeros(7)
        while not rospy.is_shutdown():
            error_q = q_target - self.q
            error_dq = d_qd - self.dq

            print(np.max(np.abs(error_q)))
            if np.max(np.abs(error_q)) < 1e-2 and np.max(np.abs(error_dq)) < 1e-2:
                break

            qacc_des = self._joint_kp * error_q + self._joint_kd * error_dq
            qacc_des = np.clip(qacc_des, -qacc_max, qacc_max)

            self._send_iiwa_torque(qacc_des)

    # joint space PD control
    def _iiwa_joint_control(self, q_target, qd_target=None, vel=0.05, interpolate=True):
        """
        joint space control by linear interpolation
        :param q_target:
        :param vel:
        :return:
        """
        if interpolate:
            error = self.q - q_target
            t = np.max(np.abs(error)) / vel
            NTIME = int(t / self.dt)
            print("Linear interpolation by", NTIME, "joints")
            q_list = np.linspace(self.q, q_target, NTIME)
            qd_list = np.linspace(self.dq, qd_target, NTIME)
            for i in range(NTIME):
                # self._iiwa_joint_space_impedance(q_list[i, :])

                # send to torque_service.py
                pub_msg = JointState()
                pub_msg.position = q_list[i, :].tolist()
                pub_msg.velocity = qd_list[i, :].tolist()
                self.iiwa_cmd_pub_joint.publish(pub_msg)
        else:
            pub_msg = JointState()
            pub_msg.position = q_target
            pub_msg.velocity = qd_target
            self.iiwa_cmd_pub_joint.publish(pub_msg)
        # self.q_cmd = q_list[-1, :]


    def move_to_joints(self, joints: np.ndarray, vel=[0.2, 1],
                       fix_fingers=None, run=True, last_joint=None):
        """
        linear interpolation in joint space
        :param joints:
        :param vel:
        :return:
        """
        if fix_fingers is None:
            fix_fingers = [None, None, None, None]
        n = len(joints)
        assert n in [7, 16, 23]
        if n == 7:
            error = self.q - joints
            vel = vel[0]
        elif n == 16:
            error = self.qh - joints
            vel = vel[1]
        else:
            error = self.q_all - joints
            vel = vel[0]
        t = np.max(np.abs(error)) / vel
        NTIME = int(t / self.dt)

        print("Linear interpolation by", NTIME, "joints")
        if len(joints) == 7:
            if not run:
                # interpolate from last joints but do not run
                q_list = np.linspace(last_joint, joints, NTIME)
                return q_list
            else:
                # interpolate from current position
                q_list = np.linspace(self.q, joints, NTIME)
                for i in range(NTIME):
                    self._send_iiwa_position(q_list[i, :])
                    time.sleep(self.dt)

        elif len(joints) == 16:
            if not run:
                # interpolate from last joints but do not run
                q_list = np.linspace(last_joint, joints, NTIME)
                return q_list
            else:
                q_list = np.linspace(self.qh, joints, NTIME)
                for i in range(NTIME):
                    q_d = q_list[i, :]
                    for j, q_fix in enumerate(fix_fingers):
                        if q_fix is not None:
                            breakpoint()
                            q_d[j * 4: j * 4 + 4] = q_fix
                    self._send_hand_position(q_d)
                    time.sleep(self.dt)
        else:
            if not run:
                q_list = np.linspace(last_joint, joints, NTIME)
                return q_list
            else:
                q_list = np.linspace(self.q_all, joints, NTIME)
                for i in range(NTIME):
                    self._send_iiwa_position(q_list[i, :7])
                    self._send_hand_position(q_list[i, 7:])
                    time.sleep(self.dt)
        print('Trajectory has been executed.')

        # while True:
        #
        #     error = self.q - joints
        #    # c = c / (np.abs(error) + 1e-6)  # for each joint should take different c?
        #
        #     dq_desired = - c * error
        #     q_desired = self.q + dq_desired * self.dt
        #     self._send_iiwa_position(q_desired)
        #     time.sleep(self.dt)
        #     if np.linalg.norm(error) < 1e-4:
        #         break

    def trac_ik_solver(self, target_pose: np.ndarray, seed=None):
        # Initial status of the joints as seed.
        if seed is None:
            seed = self.q

        next_joint_positions = self._ik_solver.get_ik(seed,
                                                      target_pose[0],
                                                      target_pose[1],
                                                      target_pose[2],  # X, Y, Z
                                                      target_pose[4],
                                                      target_pose[5],
                                                      target_pose[6],
                                                      target_pose[3])  # QX, QY, QZ, QW
        if next_joint_positions is None:
            print('IK failed')  # very easy to fail...
            sys.exit(1)
        q = np.array(next_joint_positions)
        for i in range(7):
            assert q[i] > self.iiwa_bounds[0, i] and q[i] < self.iiwa_bounds[1, i]

        return np.array(next_joint_positions)

    def move_to_target_cartesian_pose(self, target_pose: np.ndarray,
                                      run=True, last_joint=None):
        """

        :param target_pose: [x,y,z,qw,qx,qy,qz]
        :return:
        """
        if not run:
            # IK from last joint configuration
            desired_joints = self.trac_ik_solver(target_pose, seed=last_joint)
            assert len(desired_joints) == 7
            return self.move_to_joints(desired_joints,vel=[0.2, 1], run=run, last_joint=last_joint)
        else:
            desired_joints = self.trac_ik_solver(target_pose)
            assert len(desired_joints) == 7
            self.move_to_joints(desired_joints, vel=[0.2, 1])

    # Input: Cartesian space: a set of poses : (n,7) array, n: num of viapoints. [position, quaternion]
    # Output: Cartesian space or Joint space by IK
    def motion_generation(self, poses, vel=0.05, intepolation='linear', cartesian=False):

        if len(poses.shape) == 1:
            poses = poses.reshape(1, -1)
        poses = np.concatenate([self.x.reshape(1, -1), poses], axis=0)  # add current points

        keypoints_num = poses.shape[0]

        path_length = 0
        for i in range(keypoints_num - 1):
            path_length += np.linalg.norm(poses[i, :3] - poses[i + 1, :3])
        path_time = path_length / vel

        joint_seed = self.q
        joint_list = []
        x_list = []
        for i in range(keypoints_num - 1):
            path_i = np.linalg.norm(poses[i, :3] - poses[i + 1, :3])
            sample_num = int(path_i / vel / self.dt + 1)
            # if sample_num < 400:
            #     sample_num = 400

            if intepolation == 'linear':
                pos = np.concatenate((np.linspace(poses[i, 0], poses[i + 1, 0], num=sample_num).reshape(-1, 1),
                                      np.linspace(poses[i, 1], poses[i + 1, 1], num=sample_num).reshape(-1, 1),
                                      np.linspace(poses[i, 2], poses[i + 1, 2], num=sample_num).reshape(-1, 1)),
                                     axis=1)  # print
            ori = rot.slerp(poses[i, 3:], poses[i + 1, 3:],
                            np.array(range(sample_num)) / (sample_num - 1))
            x = np.concatenate([pos, ori], axis=1)
            x_list.append(x)

            # output is in joint space
            if not cartesian:
                for j in range(sample_num):
                    target_x = np.concatenate((pos[j, :], ori[j, :]))
                    desired_joints = self.trac_ik_solver(target_x, seed=joint_seed)
                    joint_list.append(desired_joints)
                    joint_seed = desired_joints

        if cartesian:
            return np.vstack(np.vstack(x_list))
        else:
            return np.vstack(joint_list)

    def forward_kine(self, q, quat=True, return_jac=True):
        """
        forward kinematics for all fingers
        :param quat: return quaternion or rotation matrix
        :param q: numpy array  (16,) or (8,)
        :return: x:  pose and jacobain
        """
        assert len(q) == 7

        q_ = kdl_parser.joint_to_kdl_jnt_array(q)
        end_frame = kdl.Frame()
        self.fk_solver.JntToCart(q_, end_frame) # end_frame is the output in Cartesian space

        x = np.array([end_frame.p[0], end_frame.p[1], end_frame.p[2]])
        if quat:
            qua = kdl.Rotation(end_frame.M).GetQuaternion()  # Notice that the quaternion is [x y z w]
            qua = np.array([qua[3], qua[0], qua[1], qua[2]])  # [w, x, y, z]
            pose = np.concatenate([x, qua])
        else:
            R = np.array([[end_frame.M[0, 0], end_frame.M[0, 1], end_frame.M[0, 2]],
                          [end_frame.M[1, 0], end_frame.M[1, 1], end_frame.M[1, 2]],
                          [end_frame.M[2, 0], end_frame.M[2, 1], end_frame.M[2, 2]]])
            T = np.concatenate([R, x.reshape(-1, 1)], axis=1)
            T = np.concatenate([T, np.array([[0, 0, 0, 1]])], axis=0)

        if return_jac:
            jac = kdl.Jacobian(7)
            self.jac_calc.JntToJac(q_, jac)
            jac_array = kdl_parser.kdl_matrix_to_mat(jac)

        if return_jac:
            return pose, jac_array
        else:
            return pose

    # impedance control
    def iiwa_joint_space_impedance_PD(self, q: np.ndarray):
        """
             \tau = M(q) ( \ddq + kp e + kd \dot{e} ) + Coriolis + gravity
        :param q: the direct goal joint position
        :return:
        """
        error_q = q - self.q
        error_dq = 0 - self.dq
        kp = np.array([800, 800, 800, 800, 330, 160, 130.])
        kd = np.array([800 / 15., 800 / 15., 800 / 15., 800 / 15., 30, 19, 15.])

        # M, C?
        tau = self.M @ (kp * error_q + kd * error_dq) + self.C
        # tau = (kp * error_q + kd * error_dq) + J^{t} F_{ext}
        return tau

    def full_joint_space_control(self, q, qh=None):

        if qh is not None:
            q = np.concatenate([q, qh])

        tau = self.iiwa_joint_space_impedance_PD(q[:7])
        # tau_hand = self.hand_move_torque(q[7:23], kh_scale=[0.2, 0.2, 0.2, 0.2])
        # self.send_torque(np.concatenate([tau, tau_hand]))


    def iiwa_joint_space_reaching(self, q, qh=None, vel=[1., 1], coupling=False):
        """
         reaching motion in joint space, with a linear interpolation
        :param coupling: if coupling the motion for iiwa and hand? if not, the iiwa will reach the goal and then the hand moves
        :param q: goal point for iiwa
        :param qh:  for hand
        :param vel: joint velocity for iiwa and hand
        :return:
        """

        error = np.max(np.abs(q - self.q))
        nums = int(error / vel[0] / 0.002)
        q_list = np.linspace(self.q, q, num=nums)
        if qh is None or not coupling:
            qh0 = np.copy(self.qh)
            qh_list = np.vstack([qh0]*nums)
        else:
            qh_list = np.linspace(self.qh, qh, num=nums)

        for i in range(nums):
            self.full_joint_space_control(q_list[i, :], qh_list[i, :])
            # time.sleep(0.002)

        if not coupling:  # move the arm first, then move the hand
            assert qh is not None
            error_h = np.max(np.abs(qh - self.qh))
            nums_h = int(error_h / vel[1] / 0.002)
            qh_list = np.linspace(self.qh, qh, num=nums_h)
            for i in range(nums_h):
                self.full_joint_space_control(q, qh_list[i, :])  # iiwa keeps static, move the hand only

    # test function
    def sin_test_joint_space(self, i=2, a=0.1):
        t0 = time.time()
        q0 = copy.deepcopy(self.q)
        self._sending_torque = True
        while 1:
            t = time.time() - t0
            qd = np.copy(q0)
            qd[i] += a * np.sin(2 * np.pi * 0.2 * t)
            self._iiwa_joint_control(qd)
            time.sleep(self.dt)
            if t > 10:
                break

        self._sending_torque = False
        print("Finish test.")

    def iiwa_step_test(self, i=6, a=0.1, exe_time=10):
        """
        :return:
        """

        v_record = []
        v_actual_record = []
        qd_record = []
        q_actual_record = []
        v_error_record = []
        error_record = []

        t0 = time.time()
        q0 = np.copy(self.q)
        qd = np.copy(self.q)
        smoothing_duration = 0.3
        while 1:
            # generate sin wave
            t = time.time() - t0
            qd_dot = np.zeros_like(self.q)

            if t < smoothing_duration:
                qd[i] = q0[i] + a * np.sin(2 * np.pi * 0.2 * t)
                qd_dot[i] = a * 2 * np.pi * 0.2 * np.cos(2 * np.pi * 0.2 * t)
                self._iiwa_joint_control(qd, qd_dot, vel=0.2)
                time.sleep(self.dt)
            else:
                qd[i] = q0[i] + a * np.sin(2 * np.pi * 0.2 * t)
                qd_dot[i] = a * 2 * np.pi * 0.2 * np.cos(2 * np.pi * 0.2 * t)
                self._iiwa_joint_control(qd, qd_dot, vel=0.05, interpolate=False)
                time.sleep(self.dt)

            error = qd[i] - self.q[i]
            error_percent = (error / a) * 100
            v_error = qd_dot[i] - self.dq[i]
            v_error_percent = (v_error / (a * 2 * np.pi * 0.2 )) * 100


            # position tracking
            qd_record.append([t, qd[i]])
            q_actual_record.append([t, self.q[i]])
            error_record.append([t, error_percent])

            # velocity tracking
            v_record.append([t, qd_dot[i]])
            v_actual_record.append([t, self.dq[i]])
            v_error_record.append([t, v_error_percent])

            if t > exe_time:
                break

        timestamp = np.array([entry[0] for entry in q_actual_record])
        actual_position = np.array([entry[1] for entry in q_actual_record])
        target_position = np.array([entry[1] for entry in qd_record])
        error_percent = np.array([entry[1] for entry in error_record])

        actual_velocity = np.array([entry[1] for entry in v_actual_record])
        target_velocity = np.array([entry[1] for entry in v_record])
        error_velocity_percent = np.array([entry[1] for entry in v_error_record])

        # write into file
        filename = '../output/data/sin_test_{}.npy'.format(i)
        np.save(filename, {'timestamp': timestamp,
                                'actual_position': actual_position,
                                'target_position': target_position,
                                'error_position_percent': error_percent,
                                'actual_velocity': actual_velocity,
                                'target_velocity': target_velocity,
                                'error_velocity_percent': error_velocity_percent})

        # fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        #
        # axs[0].plot(timestamp, position, label='Actual Position (q)', linestyle='-')
        # axs[0].set_xlabel('Timestamp')
        # axs[0].set_ylabel('Position')
        # axs[0].set_title('Trajectory of Joint {}'.format(i))
        # axs[0].grid(True)
        #
        # axs[1].plot(timestamp, error_percent, label='Position Error', linestyle='-')
        # axs[1].set_xlabel('Time (s)')
        # axs[1].set_ylabel('Error')
        # axs[1].set_title('Position Error for Joint {}'.format(i))
        # axs[1].grid(True)
        #
        # plt.tight_layout()
        #
        # plt.show()

    @property
    def xh(self) -> list:
        # fingertip (center of semisphere) positions
        return self.hand.forward_kine(self.qh)  # end position of fingertip

    @property
    def x_obj(self) -> dict:
        # get object poses in hand base frame
        return copy.deepcopy(self._x_obj) # position of object

    @property
    def q_all(self):
        return np.concatenate([self.q, self.qh]) # position of iiwa and hand: 7+16=23 dim

    @property
    def q(self):
        return self._q  # iiwa postion: 7 dim

    @property
    def dq(self):
        return self._dq # iiwa velocity: 7 dim

    @property
    def effort(self):
        return self._effort # iiwa effort: 7 dim

    @property
    def qh(self):
        return self._qh # hand position 4*4=16 dim

    @property
    def x(self):
        return self.forward_kine(self.q, return_jac=False) #end position of iiwa

    @property
    def J(self):
        x, jac = self.forward_kine(self.q, return_jac=True) # Jacobian of iiwa
        return jac

    @property
    def Jh(self):
        return self.hand.get_jac(self.qh) # Jacobian of hand

    @property
    def dx(self):
        """
            Cartesian velocities of the end-effector frame
            Compute site end-effector Jacobian
        :return: (6, )
        """
        dx = self.J @ self.dq #end velocity of iiwa
        return dx.flatten()

    @property
    def hand_base2iiwa(self):

        return rot.pose_mul(self.x, self.allegro_base2_ee)



if __name__ == "__main__":
    r = Robot(camera=False, optitrack_frame_names=['iiwa_base7', 'realsense_m'],
              camera_object_name=['cross_part', 'bottle'], position_control=False)

    for i in range(7):
        r.iiwa_step_test(i=i, a=0.2, exe_time=10)
        print("over")
        time.sleep(5)
