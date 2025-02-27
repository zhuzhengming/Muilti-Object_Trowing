import sys
sys.path.append("../")
import numpy as np
import mujoco
from mujoco import viewer
import tools.rotations as rot
import quaternion
import kinematics.allegro_hand_sym as allegro

class Robot:
    def __init__(self, m: mujoco._structs.MjModel,
                 d: mujoco._structs.MjModel,
                 view, obj_names=[], auto_sync=True,
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

        if q0 is None:
            self.q0 = np.array(
                [-0.32032486, 0.02707055, -0.22881525, -1.42611918, 1.38608943, 0.5596685, -1.34659665 + np.pi])

        self.modify_joint(self.q0)  # set the initial joint positions
        self.step()  # run one step of simulation
        self.view.sync()  # sync the viewer so that the GUI can show the current state of robots
        self.viewer_setup()  # setup camera perspective of the GUI, you can adjust it in GUI by mouse and print self.view.cam to get the current one you like

        # hand kinematics
        self.hand = allegro.Robot(right_hand=False, path_prefix='../')  # load the left hand kinematics
        self.fingertip_sites = ['index_site', 'middle_site', 'ring_site',
                                'thumb_site']  # These site points are the fingertip (center of semisphere) positions

        self._joint_kp = np.array([800, 800, 400, 400, 200, 150, 100])
        self._joint_kd = np.array([200, 200, 100, 100, 30, 30, 10])
        self.max_torque = np.array([80, 80, 80, 80, 80, 80, 80])
        self.tau_end = 19.62 + 50

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
        print(torque)
        self.step()
        if self.auto_sync:
            self.view.sync()

    def iiwa_joint_impedance(self, q_target, d_qd=None):
        """
        directly sending torque
        :param q_target:
        :return:
        """
        if d_qd is None:
            d_qd = np.zeros(7)

        error_q = q_target - self.q
        error_dq = d_qd - self.dq

        qacc_des = self._joint_kp * error_q + self._joint_kd * error_dq
        qacc_des = np.clip(qacc_des, -self.max_torque, self.max_torque)

        # gravity compensation
        gravity_torque = np.array([0, 0, self.tau_end, 0, 0, 0])
        tau_torque_joint = np.dot(self.J.T, gravity_torque)
        qacc_des += tau_torque_joint

        return qacc_des

    def iiwa_hand_go(self, q, qh, d_pose=None, dqh=None, u_add=None, kh_scale=None):
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

        iiwa_torque = self.iiwa_joint_impedance(q, d_qd=d_pose)
        hand_torque = self.hand_move_torque(qh=qh, dqh=dqh, u_add=u_add, kh_scale=kh_scale)
        u = np.concatenate([iiwa_torque, hand_torque])
        self.send_torque(u)

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

    def full_joint_space_control(self, q, qh=None):

        if qh is not None:
            q = np.concatenate([q, qh])

        tau = self.iiwa_joint_impedance(q[:7])
        tau_hand = self.hand_move_torque(q[7:23], kh_scale=[0.2, 0.2, 0.2, 0.2])
        self.send_torque(np.concatenate([tau, tau_hand]))

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