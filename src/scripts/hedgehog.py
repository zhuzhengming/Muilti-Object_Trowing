"""
Algorithm to get velocity hedgehog_data
Input:
* robot -- robot model(forward kinematic and differential forward kinematics)
Output:
* max_z_phi_gamma -- Max velocity in z, phi, gamma cell, d
* q_z_phi_gamma -- The configuration to archive the max velocity
Data:
* q_min, q_max -- robot joint limit
* q_dot_min, q_dot_max -- robot joint velocity limit
* Delta_q -- joint grid size
* D, Z, Phi, Gamma -- velocity hedgehog_data grids
"""
import sys
sys.path.append("../")
import time
import cvxpy as cp
import numpy as np
import mujoco
from mujoco import viewer
from tqdm import tqdm
import rospy


class VelocityHedgehog:
    def __init__(self, q_min, q_max, q_dot_min, q_dot_max, robot_path, train_mode=False):
        self.q_min = q_min
        self.q_max = q_max
        self.q_dot_min = q_dot_min
        self.q_dot_max = q_dot_max
        self.model = mujoco.MjModel.from_xml_path(robot_path)
        self.data = mujoco.MjData(self.model)

        if train_mode is False:
            self.view = viewer.launch_passive(self.model, self.data)
            self.q0 = np.array([-0.32032486, 0.02707055, -0.22881525, -1.42611918, 1.38608943, 0.5596685, -1.34659665 + np.pi])
            self.hand_home_pose = np.array(rospy.get_param('/hand_home_pose'))
            self.envelop_pose = np.array(rospy.get_param('/envelop_pose'))
            self._set_joints(self.q0.tolist(), render=True)
            self.viewer_setup()

    def viewer_setup(self):
        """
        setup camera angles and distances
        These data is generated from a notebook, change the view direction you wish and print the view.cam to get desired ones
        :return:
        """
        self.view.cam.trackbodyid = 0  # id of the body to track ()
        # self.viewer.cam.distance = self.sim.model.stat.extent * 0.05  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.view.cam.distance = 0.6993678113883466 * 4  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.view.cam.lookat[0] = 0.55856114  # x,y,z offset from the object (works if trackbodyid=-1)
        self.view.cam.lookat[1] = 0.00967048
        self.view.cam.lookat[2] = 1.20266637
        self.view.cam.elevation = -21.105028822285007  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.view.cam.azimuth = 94.61867426942274  # camera rotation around the camera's vertical axis

    def _set_joints(self, q: list, q_dot: list=None, render=False):
        self.data.qpos[:7] = q
        if q_dot is not None:
            self.data.qvel[:7] = q_dot
        mujoco.mj_forward(self.model, self.data)
        if render:
            mujoco.mj_step(self.model, self.data)
            self.view.sync()

    def _set_hand_joints(self, qh: list, qh_dot: list=None, render=False):
        self.data.qpos[7:23] = qh
        if qh_dot is not None:
            self.data.qvel[7:23] = qh_dot
        if render:
            mujoco.mj_forward(self.model, self.data)
            self.view.sync()

    def _set_object_position(self, id, x, vel=None):
        jnt_adr = self.model.body_jntadr[id]
        if jnt_adr != -1:
            self.data.qpos[jnt_adr:jnt_adr + 3] = x
            if vel is not None:
                dof_adr = self.model.body_dofadr[id]
                self.data.qvel[dof_adr:dof_adr + 3] = vel[:3]
        else:
            self.data.xpos[id] = x
        mujoco.mj_step(self.model, self.data)
        self.view.sync()

    def forward(self, q: list, render=False) -> (np.ndarray, np.ndarray):
        self._set_joints(q, render=render)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        # ee_site id is 0
        site_id = self.model.site("ee_site").id
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        J = np.vstack((jacp, jacr))[:, :7]
        AE = self.x2base

        return AE, J

    def print_simulator_info(self):
        print("===== Body Information =====")
        for i in range(self.model.nbody):
            body_name = self.model.body(i).name
            body_id = i
            body_jntadr = self.model.body_jntadr[i]
            print(f"Body {i}: Name = {body_name}, ID = {body_id}, Joint Address = {body_jntadr}")
        print("\n")

        print("===== Joint Information =====")
        for i in range(self.model.njnt):
            joint_name = self.model.jnt(i).name
            joint_id = i
            joint_type = self.model.jnt_type[i]
            joint_qposadr = self.model.jnt_qposadr[i]
            joint_dofadr = self.model.jnt_dofadr[i]
            print(f"Joint {i}: Name = {joint_name}, ID = {joint_id}, Type = {joint_type}, "
                  f"Qpos Address = {joint_qposadr}, DOF Address = {joint_dofadr}")
        print("\n")

        print("===== Site Information =====")
        for i in range(self.model.nsite):
            site_name = self.model.site(i).name
            site_id = i
            site_bodyid = self.model.site_bodyid[i]
            print(f"Site {i}: Name = {site_name}, ID = {site_id}, Body ID = {site_bodyid}")
        print("\n")


    @property
    def dq(self):
        """
        iiwa joint velocities
        :return: (7, )
        """
        return self.data.qvel[:7]

    @property
    def J(self):
        """
            Compute site end-effector Jacobian
        :return: (6, 7)
        """
        J_shape = (3, self.model.nv)
        jacp = np.zeros(J_shape)
        jacr = np.zeros(J_shape)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, 0)
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
    def x2base(self):
        """
        Cartesian position of end_effector to kuka_base
        """
        ee_global_pos = self.data.site("ee_site").xpos.copy()
        kuka_base_pos = self.data.body("kuka_base").xpos.copy()

        return ee_global_pos - kuka_base_pos

def computeMesh(q_min, q_max, delta_q):
    qs = [np.arange(qn, qa, delta_q) for qn, qa in zip(q_min, q_max)]
    return np.array(np.meshgrid(*qs)).transpose().reshape(-1, q_max.shape[0])

def filter(Q, VelocityHedgehog: VelocityHedgehog,
           singularity_thres, ZList, DISList,
           Z_TOLERANCE=0.01, DIS_TOLERANCE=0.01):
    """
        Inputs:
            Q (np.ndarray or list): Joint configurations, shape (N, D).
            robot (Robot): Robot model with `forward(q)` method returning (AE, J).
            singularity_thres (float): Threshold for filtering singular configurations.
            ZList (np.ndarray or list): Height intervals for grouping, shape (M,).
            DISList (np.ndarray or list): XY-plane distance intervals for grouping, shape (K,).
            Z_TOLERANCE (float): Tolerance for height grouping. Default 0.01.
            DIS_TOLERANCE (float): Tolerance for distance grouping. Default 0.01.

        Returns:
            Qzd (list): Grouped joint configurations, shape (M, K).
            aezd (list): Grouped end-effector positions, shape (M, K).
            jzd (list): Grouped Jacobain matrices, shape (M, K).
        """


    qlist = []
    AElist = []
    Jlist = []
    with tqdm(total=len(Q), desc="Filtering configurations", unit="config") as pbar:
        for q in Q:
            # do not planning joint 0 and joint 6
            q = [0.0] + q.tolist() + [0.0]
            AE, J = VelocityHedgehog.forward(q)
            u, s, vh = np.linalg.svd(J)
            if np.min(s) < singularity_thres:
                pbar.update(1)
                continue
            qlist.append(q)
            AElist.append(AE)
            Jlist.append(J)
            pbar.update(1)

        zlen = ZList.shape[0]
        pad_zs = np.r_[-np.inf, ZList]
        pad_dis = np.r_[-np.inf, DISList]
        Qzd = [[[] for j in range(DISList.shape[0])] for i in range(zlen)]
        aezd = [[[] for j in range(DISList.shape[0])] for i in range(zlen)]
        jzd = [[[] for j in range(DISList.shape[0])] for i in range(zlen)]
        num = 0

    with tqdm(total=len(qlist), desc="Grouping configurations", unit="config") as pbar:
        for i, q in enumerate(qlist):
            zi = np.argmax(abs(pad_zs - AElist[i][2]) < Z_TOLERANCE) # belong to which height
            di = np.argmax(abs(pad_dis - np.linalg.norm(AElist[i][:2])) < DIS_TOLERANCE) # belong to which distance
            if zi != 0 and di != 0:
                Qzd[zi - 1][di - 1].append(q)
                aezd[zi - 1][di - 1].append(AElist[i])
                jzd[zi - 1][di - 1].append(Jlist[i])
                num += 1
            pbar.update(1)

    # print("total num:", num)
    return Qzd, aezd, jzd

def LP(phi, gamma, Jinv, fracyx, qdmin, qdmax):
    """
        max s that:
        - q_dot_min <= inv(J(q))*v <= q_dot_max
        - v = s*    [ cos(gamma) * cos(AE_y / AE_x + phi)   ]
                    [ cos(gamma) * sin(AE_y / AE_x + phi)   ]
                    [ sin(gamma)                            ]
        :param phi:
        :param gamma:
        :param q:
        :param robot:
        :return:
        """

    s = cp.Variable(1)
    v = cp.Variable(3)
    objective = cp.Maximize(s)
    constraints = [qdmin <= Jinv @ v, Jinv @ v <= qdmax,
                   v == s * np.array(
                       [np.cos(gamma) * np.cos(fracyx + phi),
                        np.cos(gamma) * np.sin(fracyx + phi),
                        np.sin(gamma)])]

    prob = cp.Problem(objective, constraints)

    result = prob.solve(warm_start=True)
    return s.value[0]


def main(VelocityHedgehog: VelocityHedgehog, delta_q, Dis, Z, Phi, Gamma,
         svthres=0.1, z_tolerance=0.01, dis_tolerance=0.01):

    num_joints = VelocityHedgehog.q_min.shape[0]
    total_combinations = len(Z) * len(Dis) * len(Phi) * len(Gamma)

    # Build robot dataset
    # The joint0 and the last joint don't contribute to the pose
    # Because joint0 need to adjust the alpha
    q_candidates = computeMesh(VelocityHedgehog.q_min[1:6], VelocityHedgehog.q_max[1:6], delta_q)

    # Filter out q with small singular value
    # Group by(Q_f , Z , D)
    Qzd, aezd, Jzd = filter(Q=q_candidates, VelocityHedgehog=VelocityHedgehog,
                            singularity_thres=svthres, ZList=Z, DISList=Dis,
                            Z_TOLERANCE=z_tolerance, DIS_TOLERANCE=dis_tolerance
                            )

    # initialize velocity hedgehog_data
    num_z = Z.shape[0]
    num_dis = Dis.shape[0]
    num_phi = Phi.shape[0]
    num_gamma = Gamma.shape[0]

    vel_max = np.zeros((num_z, num_dis, num_phi, num_gamma))
    argmax_q = np.zeros((num_z, num_dis, num_phi, num_gamma, num_joints)) # the configuration with max_velocity
    q_ae = np.zeros((num_z, num_dis, num_phi, num_gamma, 3))

    # Build velocity hedgehog_data
    with tqdm(total=total_combinations, desc="Overall Progress", unit="comb") as pbar:
        for i in range(Z.shape[0]):
            for j in range(Dis.shape[0]):
                # for every z and distance
                qzd = Qzd[i][j]
                if len(qzd) == 0:
                    # filtered q without result
                    pbar.update(len(Phi) * len(Gamma))
                    continue

                # print("height: {:.2f}, DIS: {:.2f}, Num(q): {}".format(Z[i], Dis[j], len(qzd)))

                # for processing visualization
                current_z = Z[i]
                current_dis = Dis[j]
                group_info = f"Z={current_z:.2f}, Dis={current_dis:.2f}"

                vels = np.zeros((len(qzd), num_phi, num_gamma))
                # check all the joint configuration in the specific z and d
                # solve specific max velocity using LP
                for k, q in enumerate(qzd):
                    AE = aezd[i][j][k]
                    J = Jzd[i][j][k]
                    # fracyx = AE[1] / AE[0]
                    fracyx = np.arctan2(AE[1], AE[0])
                    Jinv = np.linalg.pinv(J[:3, :]) # pseudo inverse of Jacobian
                    qdmin, qdmax = VelocityHedgehog.q_dot_min, VelocityHedgehog.q_dot_max

                    # fix z, dis, for every gamma and phi
                    vels[k, :, :] = np.array([[LP(phi, gamma, Jinv, fracyx, qdmin, qdmax) for gamma in Gamma] for phi in Phi])

                    pbar.update(1)
                    pbar.set_postfix_str(group_info)

                # get the only one maximum velocity, and relative configuration, AE position
                # for every (z,d,gamma,phi)
                vel_max[i,j,:,:] = np.max(vels, axis=0)
                argmax_q[i,j,:,:] = np.array(qzd)[np.argmax(vels, axis=0), :]
                q_ae[i,j,:,:] = np.array(aezd[i][j])[np.argmax(vels, axis=0), :]

    return vel_max, argmax_q, q_ae

if __name__ == '__main__':

    prefix = '../hedgehog_data/'
    robot_path = '../description/iiwa7_allegro_throwing.xml'

    q_min = np.array([-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239, -2.96705972839,
                                      -2.09439510239, -3.05432619099])
    q_max = -q_min

    q_dot_max = np.array([1.71, 1.74, 1.745, 2.269, 2.443, 3.142, 3.142])
    q_dot_min = -q_dot_max

    # for iiwa configuration
    delta_z = 0.05
    delta_dis = 0.05
    delta_phi = np.pi / 12
    delta_gamma = np.pi / 36
    gamma_offset = np.pi / 9
    delta_q = 0.3

    Z = np.arange(0, 1.2, delta_z)
    Dis = np.arange(0, 1.1, delta_dis) # remove the length of joint0
    Phi = np.arange(-np.pi / 2, np.pi / 2, delta_phi)
    Gamma = np.arange(gamma_offset, np.pi / 2 - gamma_offset, delta_gamma)

    Robot = VelocityHedgehog(q_min, q_max, q_dot_min, q_dot_max, robot_path, train_mode=True)
    vel_max, argmax_q, q_ae = main(Robot, delta_q, Dis, Z, Phi, Gamma)

    # save file
    np.save(prefix + 'robot_zs.npy', Z)
    np.save(prefix + 'robot_diss.npy', Dis)
    np.save(prefix + 'robot_phis.npy', Phi)
    np.save(prefix + 'robot_gammas.npy', Gamma)

    # np.save(prefix+'argmax_q.npy', argmax_q)
    # np.save(prefix+'q_idx.npy', q_ae)
    np.save(prefix+'z_dis_phi_gamma_vel_max.npy', vel_max)

    # hash construct
    # construct an id for quick search
    print("Constructing q_idx")
    num_z, num_dis, num_phi, num_gamma = len(Z), len(Dis), len(Phi), len(Gamma)
    qs = []
    aes = []
    qid_iter = 0
    q_idxs = np.zeros((num_z, num_dis, num_phi, num_gamma))
    for i in range(num_z):
        for j in range(num_dis):
            for k in range(num_phi):
                for l in range(num_gamma):
                    q = argmax_q[i, j, k, l, :]
                    ae = q_ae[i, j, k, l, :]
                    exist = False
                    for d, qi in enumerate(qs):
                        if np.allclose(qi, q):
                            qid = d
                            exist = True
                            break
                    if not exist:
                        qid = qid_iter
                        qid_iter += 1
                        qs.append(q)
                        aes.append(ae)
                    # [z, dis, phi, gamma] -> q_id
                    # q_id -> q, ae
                    q_idxs[i, j, k, l] = qid
    np.save(prefix + 'z_dis_phi_gamma_vel_max_q_idxs', q_idxs)
    np.save(prefix + 'q_idx_qs', np.array(qs))
    np.save(prefix + 'q_idx_ae', np.array(aes))
    print("done")

