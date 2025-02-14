"""
Control tools for the iiwa and allegro hand.

basic functions:
 - read all states from MuJoCo simulator and store them as properties (see the end part of this file)
 - send torque commands to the robotic system
 - run one step simulation in MuJoCo
 - reset/reposition all robots/objects

Controllers:
 - impedance controller in Cartesian space for iiwa
 - impedance controller in joint space for allegro hand
 - A coupled DS to reach an attractor for both iiwa and hand


Notes:
 - All quatersions are in (w x y z) order
 - Fingers of hand are in ('index', 'middle', 'ring', 'thumb') order
"""
import time

import numpy as np
import mujoco
from mujoco import viewer
import tools.rotations as rot
import quaternion

import fileinput
import re
import kinematics.allegro_hand_sym as allegro
import matplotlib.pyplot as plt


class Robot:
    def __init__(self, m: mujoco._structs.MjModel, d: mujoco._structs.MjModel, view, obj_names=[], auto_sync=True,
                 q0=None):
        self.m = m
        self.d = d
        self.view = view
        self.auto_sync = auto_sync
        self.fingers = ['index', 'middle', 'ring', 'thumb']  # the order of fingers, which is the same as the .xml file
        if len(obj_names):
            self.obj_names = obj_names  # Note that the order of objects must be the same as the .xml
            self.obj_id = {i: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, i) for i in self.obj_names}
            print("Order of objects:", self.obj_names)
            # for i in self.obj_names:
            #     self.m.body(i).mass = 0.001  # set mass for each object

        if q0 is None:
            self.q0 = np.array(
                [-0.32032486, 0.02707055, -0.22881525, -1.42611918, 1.38608943, 0.5596685, -1.34659665 + np.pi])

        self.modify_joint(self.q0)  # set the initial joint positions
        self.step()  # run one step of simulation
        self.view.sync()  # sync the viewer so that the GUI can show the current state of robots
        self.viewer_setup()  # setup camera perspective of the GUI, you can adjust it in GUI by mouse and print self.view.cam to get the current one you like

        # hand kinematics
        self.hand = allegro.Robot(right_hand=False)  # load the left hand kinematics
        self.fingertip_sites = ['index_site', 'middle_site', 'ring_site',
                                'thumb_site']  # These site points are the fingertip (center of semisphere) positions

    def step(self):
        mujoco.mj_step(self.m, self.d)  # run one-step dynamics simulation

    def viewer_setup(self):
        """
        setup camera angles and distances
        These data is generated from a notebook, change the view direction you wish and print the view.cam to get desired ones
        :return:
        """
        self.view.cam.trackbodyid = 0  # id of the body to track ()
        # self.viewer.cam.distance = self.sim.model.stat.extent * 0.05  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.view.cam.distance = 0.6993678113883466  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.view.cam.lookat[0] = 0.55856114  # x,y,z offset from the object (works if trackbodyid=-1)
        self.view.cam.lookat[1] = 0.00967048
        self.view.cam.lookat[2] = 1.20266637
        self.view.cam.elevation = -21.105028822285007  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.view.cam.azimuth = 94.61867426942274  # camera rotation around the camera's vertical axis

    def send_torque(self, torque):
        """
        control joints by torque and send to mujoco, then run a step.
        input the joint control torque
        Here I imply that the joints with id from 0 to n are the actuators.
        so, please put robot xml before the object xml.
        todo, use joint_name2id to avoid this problem
        :param torque:  (n, ) numpy array
        :return:
        """

        self.d.ctrl[:len(torque)] = torque  # apply the control torque
        self.step()
        if self.auto_sync:
            self.view.sync()
        # self.virenderer.update_scene(data)

    def moveto_attractor(self, xd, qh, dt=0.002, couple_hand=True, scaling=1, u_add=None):
        """
        move to the position attractor by the coupled DS
        :param couple_hand: If the hand motion and iwwa pose are coupled
        :param xd: (7,) position and quaternion[x, q]
        :param qh: (16,)  joint angles
        :return:
        """
        dx, w, d_theta = self.coupled_DS(xd[:3], xd[3:], qh, couple_hand=couple_hand, scaling=scaling)
        # print(dx, w, d_theta)
        d_pose = np.concatenate([dx, w])
        # integral
        xd_, qh_ = self.integral_from_vel(dx, w, d_theta, dt)

        # print(xd_[:3] - xd[:3])
        # print(rot.ori_dis(xd_[3:], xd[3:]) * 180/np.pi)
        self.iiwa_hand_go(xd_, qh_, d_pose=None, kh_scale=[0.2, 0.2, 0.2, 0.2], u_add=u_add)

    def coupled_DS(self, xd, qd, theta_d, couple_hand=True, scaling=1):
        """
            Given the attractor, calculate the velocity command
        :param xd:  (3,) position of end-effector
        :param qd:   (4,) quaternion of end-effector
        :param theta_d: (16, ) joint angles of hand
        :param scaling:  change the speed factor of the DS
        :return: dx, w, \dot{theta} : velocities (3,), (3,), (16,)
        """
        x, q = self.x[:3], self.x[3:]
        theta = self.qh
        if q[0] < 0:
            q = -q
        if qd[0] < 0:
            qd = -qd
        q = quaternion.from_float_array(q)
        qd = quaternion.from_float_array(qd)

        # parameters for DS
        a = 10 * scaling
        b = 100 * scaling
        c = 70 * scaling
        lambda_1 = 10
        lambda_2 = 50
        lambda_2 = 150

        # position DS
        a = a / (np.linalg.norm(x - xd) + 0.002)
        dx = -1 * a * (x - xd)

        # orientation DS
        dis = np.linalg.norm(x - xd)
        if dis < 0.005:
            dis = 0
        t = np.exp(- lambda_1 * dis)
        qd_ = quaternion.slerp(q, qd, 0, 1, t)
        eps = 1 / 180. * np.pi
        b = b / (quaternion.rotation_intrinsic_distance(q, qd_) + eps)
        w = - b * (q * qd_.conj()).log().vec

        # hand DS
        theta_d_ = theta + (theta_d - theta) * np.exp(- lambda_2 * dis)
        # print(theta_d_)
        if couple_hand is False:
            theta_d_ = theta_d
        c = c / (np.abs(theta - theta_d_) + eps)  # for each joint should take different c?

        d_theta = - c * (theta - theta_d_)

        return dx, w, d_theta

    def integral_from_vel(self, dx, w, d_theta, dt):
        """

        :param dx: (3,), ee vel
        :param w:  (3, )ee angular vel
        :param d_theta: (16, ) allegro hand joint vel
        :param dt: time interval
        :return: [x, q], qh,   pose of ee, desired joints of hand
        """
        x = self.x[:3] + dx * dt
        q = quaternion.from_rotation_vector(w * dt) * quaternion.from_float_array(self.x[3:])
        q = quaternion.as_float_array(q)

        qh = self.qh + d_theta * dt

        return np.concatenate([x, q]), qh

    def iiwa_hand_go(self, pose, qh, d_pose=None, dqh=None, u_add=None, kh_scale=None):
        """
        Give the desired pose of ee and joint positions of hand, using the Cartesian space impedance controller for iiwa
         and joint-space impedance controller for hand to calculate the desired joint torque and send it to MuJoCo
        :param pose: (7,), desired pose of ee
        :param qh: (16, ), desired positions of joints for hand
        :param d_pose: (6,), vel of ee
        :param dqh: (16,), vel of joints for hand
        :param u_add: (16, ), only for adding additional torques for grasping
        :param kh_scale: (4,), scaling factor of kp and kd for the joint-space impedance controller for hand
        :return:
        """

        iiwa_torque = self.iiwa_impedance_control(pose, d_pose=d_pose)
        hand_torque = self.hand_move_torque(qh=qh, dqh=dqh, u_add=u_add, kh_scale=kh_scale)
        u = np.concatenate([iiwa_torque, hand_torque])
        self.send_torque(u)

    def iiwa_impedance_control(self, pose, d_pose=None):
        """
        Give the desired pose of ee, using the Cartesian space impedance controller for iiwa to calculate torque
        :param pose: (7,), desired pose of ee
        :param d_pose: (6,), vel of ee
        :return:  (7,), computed torque for iiwa
        """
        if d_pose is None:
            d_pose = np.zeros(6)
        kp = np.array([300, 200.])
        kd = np.sqrt(kp) * 2 * 2.
        # kd = np.sqrt(kp) * 1
        pos_error = pose[:3] - self.x[:3]
        vel_error = d_pose[:3] - self.dx[:3]
        Fx = kp[0] * (pose[:3] - self.x[:3]) + kd[0] * (d_pose[:3] - self.dx[:3])
        q = self.x[3:]  # [w x y z]
        qd = pose[3:]
        # d_theta = (quaternion.from_float_array(qd) * (quaternion.from_float_array(q)).conjugate()).log() * 2
        # d_theta = quaternion.as_float_array(d_theta)[1:]
        axis, angle = rot.quat2axisangle(rot.quat_mul(qd, rot.quat_conjugate(q)))
        d_theta = np.array(axis) * angle
        Fr = kp[1] * d_theta + kd[1] * (d_pose[3:] - self.dx[3:6])
        F = np.concatenate([Fx, Fr])
        J = self.J
        impedance_acc_des0 = J.T.dot(np.linalg.solve(J.dot(J.T) + 1e-10 * np.eye(6), F))
        impedance_acc_des1 = J.T @ F

        # Add stiffness and damping in the null space of the Jacobian, to make the joints close to zeros
        nominal_qpos = np.zeros(7)
        null_space_damping = 0.1 * 10
        null_space_stiffness = 10 * 5
        projection_matrix = J.T.dot(np.linalg.solve(J.dot(J.T) + 1e-10 * np.eye(6), J))
        projection_matrix = np.eye(projection_matrix.shape[0]) - projection_matrix
        null_space_control = -null_space_damping * self.dq
        null_space_control += -null_space_stiffness * (
                self.q - nominal_qpos)
        tau_null = projection_matrix.dot(null_space_control)
        impedance_acc_des = impedance_acc_des1 + tau_null

        # self.send_torque(impedance_acc_des + self.C)
        return impedance_acc_des + self.C

    def hand_move_torque(self, qh=None, dqh=None, u_add=None, kh_scale=None):
        """
        impedance control for the allegro hand
        :param qh: (16, ), desired positions of joints for hand
        :param dqh:
        :param u_add:
        :param kh_scale:
        :return: (16,), computed torque for hand
        """
        if qh is None:
            qh = np.zeros(16)
            qh[12] = 0.5
        if dqh is None:
            dqh = np.zeros(16)
        if u_add is None:
            u_add = np.zeros(16)
        if kh_scale is None:
            kh_scale = [1, 1, 1, 1.]

        error_q = qh - self.qh
        error_dq = dqh - self.dqh
        u = np.zeros(16)

        kp = np.ones(16) * 0.4
        kp = 0.4 * np.concatenate(
            [np.ones(4) * kh_scale[0], np.ones(4) * kh_scale[1], np.ones(4) * kh_scale[2], np.ones(4) * kh_scale[3]])
        # kd = 2 * np.sqrt(kp) * 0.01
        kd = 2 * np.sqrt(kp) * 0.01
        # kd = 0.01
        # kd = np.ones(16) * 1
        qacc_des = kp * error_q + kd * error_dq + self.C_[7:] + u_add

        # print('vel', self.dqh[:4])
        # print('pos_error', error_q[:4])
        # print('control torque:', qacc_des[12:])

        # u = np.concatenate([np.zeros(7), qacc_des])
        # self.send_torque(u)
        return qacc_des

    def pinch_grasp_force(self, F, pairs=None, force_direction=None):
        """
        applying additional forces for pinch grasp. This is a feedforward controller for adding contact forces

        :param pairs: which two fingers are used for pinch grasp, 0-3 for index, middle, ring, thumb
        :param F: scalar or list, force altitude
        :param force_direction:
        :return: (16,), additional torque in joints
        """
        if force_direction is None:
            if pairs is None:
                pairs = [[0, 3]]  # this means that the pinch grasp is built by the index and thumb fingertips

            if type(F) == float:
                F = len(pairs) * [F]
            else:
                assert len(F) == len(pairs)

        poses = self.hand.forward_kine(self.qh)  # the poses and jacs for all fingertips
        jacs = self.hand.get_jac(self.qh)
        tau_add = np.zeros(16)
        if force_direction is None:
            for k, pair in enumerate(pairs):
                center_pos = (poses[pair[0]][:3] + poses[pair[1]][
                                                   :3]) / 2  # use the center point as the direction to apply addition grasp force
                F_k = F[k]
                for i in pair:
                    direction = center_pos - poses[i][:3]
                    F_fingertip = F_k * direction / np.linalg.norm(direction)
                    tau = jacs[i][:3, :].T @ F_fingertip
                    tau_add[i * 4:i * 4 + 4] = tau
        else:
            for k, direction in enumerate(force_direction):
                if len(direction):
                    F_fingertip = F[k] * np.array(direction) / np.linalg.norm(direction)
                    tau = jacs[k][:3, :].T @ F_fingertip
                    tau_add[k * 4:k * 4 + 4] = tau

        return tau_add

    def modify_joint(self, joints: np.ndarray) -> None:
        """
        :param joints: (7,) or (16,) or (23,), modify joints for iiwa or/and allegro hand
        :return:
        """
        assert len(joints) in [7, 16, 23]
        if len(joints) == 7:
            self.d.qpos[:7] = joints
            self.d.qpos[:7] = joints
        elif len(joints) == 16:
            self.d.qpos[7:23] = joints
        else:
            self.d.qpos[:23] = joints

    def modify_obj_pose(self, obj_name: str, pose: np.ndarray) -> None:
        """

        :param obj_name: the name of the object, from self.obj_names
        :param pose:   (7,) or (3,), the pose/position command
        :return: None
        """
        start_index = self.obj_names.index(obj_name) * 7 + 7 + 16
        len_pose = len(pose)
        assert len_pose in [3, 7]

        self.d.qpos[start_index: start_index + len_pose] = pose

    def iiwa_joint_space_PD(self, q: np.ndarray):
        """
             \tau = M(q) ( \ddq + kp e + kd \dot{e} ) + Coriolis + gravity
        :param q: the direct goal joint position
        :return:
        """
        error_q = q - self.q
        error_dq = 0 - self.dq
        kp = np.array([800, 800, 800, 800, 330, 160, 130.])
        # kd = np.array([80, 100, 80, 80, 10, 1, 1.])
        # kd = np.zeros(7)
        kd = np.array([800 / 15., 800 / 15., 800 / 15., 800 / 15., 30, 19, 15.])

        tau = self.M @ (kp * error_q + kd * error_dq) + self.C
        # tau = (kp * error_q + kd * error_dq) + self.C
        # breakpoint()
        # self.send_torque(tau)
        return tau

    def full_joint_space_control(self, q, qh=None):

        if qh is not None:
            q = np.concatenate([q, qh])

        tau = self.iiwa_joint_space_PD(q[:7])
        tau_hand = self.hand_move_torque(q[7:23], kh_scale=[0.2, 0.2, 0.2, 0.2])
        self.send_torque(np.concatenate([tau, tau_hand]))

    def run(self):
        while 1:
            self.send_torque(r.C_)
            time.sleep(0.002)

    def iiwa_step_test(self, i=6, dq=0.2):
        """

        :param i: index of iiwa joints, to determine the kp, kd for each joint
        :param dq: offset for the step response
        :return:
        """

        q0 = self.q
        q1 = np.copy(q0)
        qh0 = np.copy(self.qh)
        q1[i] += dq
        q_record = []
        t0 = time.time()
        first = True
        while 1:
            t1 = time.time() - t0

            self.full_joint_space_control(q1, qh0)
            # time.sleep(0.002)
            error = q1[i] - self.q[i]
            if np.abs(error) < dq * 0.01 and first:
                print(t1, error)  # show the rise time
                first = False
            q_record.append(np.array([t1, error]))
            if t1 > 10:
                break

        q_record = np.vstack(q_record)

        # plot the figure of step response, try to avoid overshooting
        plt.plot(q_record[:, 0], q_record[:, 1])
        plt.xlabel('Time (s)')
        plt.ylabel('Error (rad)')
        plt.title('iiwa joint ' + str(i))
        plt.xlim([0, np.max(q_record[:, 0])])
        plt.ylim([None, np.max(q_record[:, 1])])
        plt.show()

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

        if not coupling and qh is not None:  # move the arm first, then move the hand
            error_h = np.max(np.abs(qh - self.qh))
            nums_h = int(error_h / vel[1] / 0.002)
            qh_list = np.linspace(self.qh, qh, num=nums_h)
            for i in range(nums_h):
                self.full_joint_space_control(q, qh_list[i, :])  # iiwa keeps static, move the hand only

        # print(q - r.q)
        # print(qh - r.qh)

    def iiwa_joint_space_test(self, i=0, t=10):

        t0 = time.time()
        f = 0.1
        q0 = self.q
        qh0 = self.qh
        error_mean = []
        while time.time() - t0 < t:
            t_now = time.time() - t0
            q1 = np.copy(q0)
            q1[i] += 0.2 * np.sin(2 * np.pi * f * t_now)
            self.full_joint_space_control(q1, qh0)
            time.sleep(0.002)
            print(q1[i] - self.q[i])
            error_mean.append(q1[i] - self.q[i])

        error_mean = np.abs(error_mean)
        print("mean error", np.mean(error_mean), np.std(error_mean))

    @property
    def q(self):
        """
        iiwa joint angles
        :return: (7, ), numpy array
        """
        return self.d.qpos[:7]  # noting that the order of joints is based on the order in *.xml file

    @property
    def qh(self):
        """
        hand angles: index - middle - ring - thumb
        :return:  (16, )
        """
        return self.d.qpos[7:23]

    @property
    def q_all(self):
        """
        iiwa joint angles and allegro hand angles
        :return: (23, )
        """
        return self.d.qpos[:23]

    @property
    def dq(self):
        """
        iiwa joint velocities
        :return: (7, )
        """
        return self.d.qvel[:7]

    @property
    def dqh(self):
        """
        hand angular velocities: index - middle - ring - thumb
        :return:  (16, )
        """
        return self.d.qvel[7:23]

    @property
    def dq_all(self):
        """
        iiwa and allegro joint velocities
        :return: (23, )
        """
        return self.d.qvel[:23]

    @property
    def ddq(self):
        """
        iiwa joint acc
        :return: (7, )
        """
        return self.d.qacc[:7]

    @property
    def ddqh(self):
        """
        hand angular acc: index - middle - ring - thumb
        :return:  (16, )
        """
        return self.d.qacc[7:23]

    @property
    def ddq_all(self):
        """
        iiwa and allegro joint acc
        :return: (23, )
        """
        return self.d.qacc[:23]

    @property
    def xh(self):
        """
        hand fingertip poses: index - middle - ring - thumb
        All quaternions are in [w, x, y, z] order
        :return: (4, 7)
        """
        poses = []
        for i in self.fingers:
            site_name = i + '_site'
            xpos = self.d.site(site_name).xpos
            xquat = rot.mat2quat(self.d.site(site_name).xmat.reshape(3, 3))
            poses.append(np.concatenate([xpos, xquat]))
        return np.vstack(poses)

    @property
    def x(self):
        """
        Cartesian position and orientation (quat) of the end-effector frame, from site
        return: (7, )
        """
        xpos = self.d.site('ee_site').xpos
        xquat = rot.mat2quat(self.d.site('ee_site').xmat.reshape(3, 3))
        return np.concatenate([xpos, xquat])

    @property
    def kuka_base(self):
        xpos = self.d.site('kuka_base_site').xpos
        xquat = rot.mat2quat(self.d.site('kuka_base_site').xmat.reshape(3, 3))
        return np.concatenate([xpos, xquat])


    @property
    def p(self):
        """
        :return: transformation matrix (4, 4) of the end-effector of iiwa
        """
        pos = self.d.site('ee_site').xpos.reshape(-1, 1)
        R = self.d.site('ee_site').xmat.reshape(3, 3)
        return np.concatenate([np.concatenate([R, pos], axis=1), np.array([[0., 0, 0, 1]])])

    @property
    def x_obj(self):
        """
        :return: [(7,),...] objects poses by list, 
         // computed by mj_fwdPosition/mj_kinematics
        https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html?highlight=xipos#mjdata
        """  # print(joint_id)
        poses = []
        for i in self.obj_names:
            poses.append(np.concatenate([self.d.body(i).xpos, self.d.body(i).xquat]))
        return poses

    @property
    def x_obj_dict(self):
        """
        :return: [(7,),...] objects poses by list, 
         // computed by mj_fwdPosition/mj_kinematics
        https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html?highlight=xipos#mjdata
        """  # print(joint_id)
        poses = {}
        for i in self.obj_names:
            poses[i] = (np.concatenate([self.d.body(i).xpos, self.d.body(i).xquat]))
        return poses

    @property
    def dx_obj(self):
        """
        object velocities
        :return: [(6,),...]
        """
        velocities = []
        for i in self.obj_names:
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY,
                                     self.obj_id[i], vel, 0)  # 1 for local, 0 for world, rot:linear
            velocities.append(np.concatenate([vel[3:], vel[:3]]))

        return velocities

    @property
    def J(self):
        """
            Compute site end-effector Jacobian
        :return: (6, 7)
        """
        J_shape = (3, self.m.nv)
        jacp = np.zeros(J_shape)
        jacr = np.zeros(J_shape)
        mujoco.mj_jacSite(self.m, self.d, jacp, jacr, 0)
        return np.vstack((jacp[:3, :7], jacr[:3, :7]))

    @property
    def dx(self):
        """
            Cartesian velocities of the end-effector frame
            Compute site end-effector Jacobian
        :return: (6, )
        """
        dx = self.J @ self.dq
        return dx.flatten()

    @property
    def M(self):
        """
        get inertia matrix for iiwa in joint space
        :return:
        """
        M = np.zeros([self.m.nv, self.m.nv])
        # M2 = np.zeros([self.m.nv, 1])
        mujoco.mj_fullM(self.m, M, self.d.qM)

        return M[:7, :7]

    @property
    def C(self):
        """
        for iiwa, C(qpos,qvel), Coriolis, computed by mj_fwdVelocity/mj_rne (without acceleration)
        :return: (7, )
        """
        return self.d.qfrc_bias[:7]

    @property
    def C_(self):
        """
        for all, C(qpos,qvel), Coriolis, computed by mj_fwdVelocity/mj_rne (without acceleration)
        :return: (nv, )
        """
        return self.d.qfrc_bias[:23]


def replace_line_in_xml(file_path, search_pattern, replace_line):
    try:
        # Read the XML file
        with fileinput.FileInput(file_path, inplace=True) as file:
            for line in file:
                # Replace lines that match the search pattern
                if re.search(search_pattern, line):
                    line = replace_line + '\n'
                print(line, end='')

        print("Replacement completed successfully.")
    except FileNotFoundError:
        print("File not found.")


if __name__ == "__main__":
    xml_path = 'description/iiwa7_allegro_ycb.xml'
    obj_name = ''
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    # viewer.launch(model, data)
    view = viewer.launch_passive(model, data)
    # in notebook, we need ro run view.sync() manually, and set auto_sync=False

    obj_names = ['banana', 'bottle', 'chip_can', 'soft_scrub', 'sugar_box']
    num = 0
    obj = obj_names[num]
    r = Robot(model, data, view, auto_sync=True, obj_names=obj_names)

    q0 = np.array(
        [-0.32032434, 0.02706913, -0.22881953, -1.42621454, 1.3862661, 0.55966738, 1.79477984 - np.pi * 3 / 2])
    r.d.qpos[:7] = q0
    r.step()
    view.sync()
    x0 = r.x
    qh0 = r.qh
    for i in range(100):
        r.iiwa_hand_go(x0, qh0)

    # r.iiwa_joint_space_test(i=0, t=10)
    qh = np.zeros(16)
    qh[12] = 1
    r.iiwa_joint_space_reaching(np.zeros(7), qh=qh, coupling=False)
    # r.run()
    # r.iiwa_step_test()
