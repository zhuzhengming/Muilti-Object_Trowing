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


class VelocityHedgehog:
    def __init__(self, q_min, q_max, q_dot_min, q_dot_max, robot_path):
        self.q_min = q_min
        self.q_max = q_max
        self.q_dot_min = q_dot_min
        self.q_dot_max = q_dot_max
        self.model = mujoco.MjModel.from_xml_path(robot_path)
        self.data = mujoco.MjData(self.model)

        self.view = viewer.launch_passive(self.model, self.data)

        self.q0 = np.array([-0.32032486, 0.02707055, -0.22881525, -1.42611918, 1.38608943, 0.5596685, -1.34659665 + np.pi])
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
        self.view.cam.distance = 0.6993678113883466  # how much you "zoom in", model.stat.extent is the max limits of the arena
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

    def forward(self, q: list) -> (np.ndarray, np.ndarray):
        self._set_joints(q)
        J_shape = (3, self.model.nv)
        jacp = np.zeros(J_shape)
        jacr = np.zeros(J_shape)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, 0)
        J = np.vstack((jacp[:3, :7], jacr[:3, :7]))
        AE = self.data.body("allegro_base").xpos.copy()

        return AE, J

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
            jzd (list): Grouped Jacobian matrices, shape (M, K).
        """


    qlist = []
    AElist = []
    Jlist = []
    with tqdm(total=len(Q), desc="Filtering configurations", unit="config") as pbar:
        for q in Q:
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

    print("total num:", num)
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

    print("Filtering done!")

    # initialize velocity hedgehog_data
    num_z = Z.shape[0]
    num_dis = Dis.shape[0]
    num_phi = Phi.shape[0]
    num_gamma = Gamma.shape[0]
    vel_max = np.zeros((num_z, num_dis, num_phi, num_gamma))
    argmax_q = np.zeros((num_z, num_dis, num_phi, num_gamma, num_joints))
    q_ae = np.zeros((num_z, num_dis, num_phi, num_gamma, 3))

    # Build velocity hedgehog_data
    with tqdm(total=total_combinations, desc="Overall Progress", unit="comb") as pbar:
        for i in range(Z.shape[0]):
            for j in range(Dis.shape[0]):
                qzd = Qzd[i][j]
                if len(qzd) == 0:
                    # filtered q
                    pbar.update(len(Phi) * len(Gamma))
                    continue

                print("height: {:.2f}, AE: {:.2f}, Num(q): {}".format(Z[i], Dis[j], len(qzd)))
                current_z = Z[i]
                current_dis = Dis[j]
                group_info = f"Z={current_z:.2f}, Dis={current_dis:.2f}"

                start = time.time()
                vels = np.zeros((len(qzd), num_phi, num_gamma))
                # check all the q in the same z and d
                # solve specific max velocity using LP
                for k, q in enumerate(qzd):
                    AE = aezd[i][j][k]
                    J = Jzd[i][j][k]
                    fracyx = AE[1] / AE[0]
                    Jinv = np.linalg.pinv(J[:3, :]) # pseudo inverse of Jacobian
                    qdmin, qdmax = VelocityHedgehog.q_dot_min, VelocityHedgehog.q_dot_max

                    vels[k, :, :] = np.array([[LP(phi, gamma, Jinv, fracyx, qdmin, qdmax) for gamma in Gamma] for phi in Phi])

                    pbar.update(1)
                    pbar.set_postfix_str(group_info)

                # get the maximum velocity wrt z,d,gamma,phi
                vel_max[i,j,:,:] = np.max(vels, axis=0)
                argmax_q[i,j,:,:] = np.array(qzd)[np.argmax(vels, axis=0), :]
                q_ae[i,j,:,:] = np.array(aezd[i][j])[np.argmax(vels, axis=0), :]

                timecost = time.time() - start
                print("use {0:.2f}s for {1} q, {2:.3f} s per q".format(timecost, len(qzd), timecost / len(qzd)))
    return vel_max, argmax_q, q_ae

if __name__ == '__main__':

    prefix = '../hedgehog_data/'
    robot_path = '../description/iiwa7_allegro_ycb.xml'

    q_min = np.array([-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239, -2.96705972839,
                                      -2.09439510239, -3.05432619099])
    q_max = np.array([2.96705972839, 2.09439510239, 2.96705972839, 2.09439510239, 2.96705972839,
                                      2.09439510239, 3.05432619099])
    q_dot_min = -np.array([1.71, 0.5, 1.745, 1.6, 2.443, 3.142, 3.142])
    q_dot_max = np.array([1.71, 0.5, 1.745, 1.6, 2.443, 3.142, 3.142])

    # for iiwa configuration
    Z = np.arange(0, 1.2, 0.05)
    Dis = np.arange(0, 1.0, 0.05) # remove the length of joint0
    Phi = np.arange(-np.pi / 2, np.pi / 2, np.pi / 12)
    gamma_offset = np.pi / 9
    Gamma = np.arange(gamma_offset, np.pi / 2 - gamma_offset, np.pi / 36 )

    np.save(prefix+'robot_zs.npy', Z)
    np.save(prefix+'robot_diss.npy', Dis)
    np.save(prefix+'robot_phis.npy', Phi)
    np.save(prefix+'robot_gammas.npy', Gamma)

    delta_q = 0.3

    Robot = VelocityHedgehog(q_min, q_max, q_dot_min, q_dot_max, robot_path)
    vel_max, argmax_q, q_ae = main(Robot, delta_q, Dis, Z, Phi, Gamma)
    np.save(prefix+'argmax_q.npy', argmax_q)
    np.save(prefix+'q_idx.npy', q_ae)
    np.save(prefix+'z_dis_phi_gamma_vel_max.npy', vel_max)

    # hash construct
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
                    ae = argmax_q[i, j, k, l, :]
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
                    q_idxs[i, j, k, l] = qid
    np.save(prefix + 'z_dis_phi_gamma_vel_max_q_idxs', q_idxs)
    np.save(prefix + 'q_idx_qs', np.array(qs))
    np.save(prefix + 'q_idx_ae', np.array(aes))
    print("done")

