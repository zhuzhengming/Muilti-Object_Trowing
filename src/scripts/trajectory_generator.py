""""
generate throwing trajectory by querying BRT and hedgehog
input: box_position
output: target[q, q_dot]
"""

from hedgehog import VelocityHedgehog
import argparse
import time
import math
import numpy as np
from pathlib import Path
from sys import path
from ruckig import InputParameter, Ruckig, Trajectory, Result
import rospy
import mujoco


class TrajectoryGenerator:
    def __init__(self, q_ul, q_ll, hedgehog_path, brt_path, box_position, robot_path):
        self.q_ul = q_ul
        self.q_ll = q_ll
        self.hedgehog_path = hedgehog_path
        self.brt_path = brt_path
        self.gravity = -9.81
        self.GAMMA_TOLERANCE = 0.2/180.0*np.pi
        self.Z_TOLERANCE = 0.01
        self.box_position = box_position

        # mujoco similator
        if robot_path is None:
            self.robot_path = '../description/iiwa7_allegro_ycb.xml'
        q_min = np.array([-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239, -2.96705972839,
                          -2.09439510239, -3.05432619099])
        q_max = np.array([2.96705972839, 2.09439510239, 2.96705972839, 2.09439510239, 2.96705972839,
                          2.09439510239, 3.05432619099])
        q_dot_min = -np.array([1.71, 0.5, 1.745, 1.6, 2.443, 3.142, 3.142])
        q_dot_max = np.array([1.71, 0.5, 1.745, 1.6, 2.443, 3.142, 3.142])

        self.robot = VelocityHedgehog(q_min, q_max, q_dot_min, q_dot_max, robot_path)

        self.max_velocity = np.array(rospy.get_param('/max_velocity'))
        self.max_acceleration = np.array(rospy.get_param('/max_acceleration'))
        self.max_jerk = np.array(rospy.get_param('/max_jerk'))
        self.MARGIN_VELOCITY = rospy.get_param('/MARGIN_VELOCITY')
        self.MARGIN_ACCELERATION = rospy.get_param('/MARGIN_ACCELERATION')
        self.MARGIN_JERK = rospy.get_param('/MARGIN_JERK')

        self.load_data()

    def load_data(self):
        # load hedgehog data
        self.robot_zs = np.load(self.hedgehog_path + '/robot_zs.npy')
        self.robot_dis = np.load(self.hedgehog_path + '/robot_dis.npy')
        self.robot_phis = np.load(self.hedgehog_path + '/robot_phis.npy')
        self.robot_gamma = np.load(self.hedgehog_path + '/robot_gamma.npy')
        self.num_gammas = len(self.robot_gamma)

        self.mesh = np.load(self.hedgehog_path + '/q_idx_qs.npy')
        self.robot_phi_gamma_velos_naive = np.load(self.hedgehog_path + '/z_dis_phi_gamma_vel_max.npy')
        self.robot_phi_gamma_q_idxs_naive = np.load(self.hedgehog_path + '/z_dis_phi_gamma_vel_max_q_idxs.npy')
        self.ae = np.load(self.hedgehog_path + '/q_idx_ae.npy')

        self.zs_step = 0.05
        assert self.num_gammas == self.robot_phi_gamma_q_idxs_naive.shape[3]

        # load brt data
        self.brt_tensor = np.load(self.brt_path + '/brt_tensor.npy')
        self.brt_zs = np.load(self.brt_path + '/brt_zs.npy')



    def brt_robot_data_matching(self, thres_v=0.1, thres_dis=0.01, thres_phi=0.04):
        """
        original point is the base of robot
        Given target position, find out initial guesses of (q, phi, x)
        :param box_position:
        :param thres_dis:
        :param thres_v:
        :return: candidates of q, phi, x
        """
        start = time.time()
        z_target_to_base = self.box_position[-1]
        DIS = self.box_position[:2]

        # align the z idx
        num_robot_zs = self.robot_zs.shape[0]
        num_brt_zs = self.brt_zs.shape[0]
        brt_z_min, brt_z_max = np.min(self.brt_zs), np.max(self.brt_zs)
        if z_target_to_base + brt_z_min > min(self.robot_zs):
            rzs_idx_start = round((z_target_to_base + brt_z_min) / self.zs_step)
            bzs_idx_start = 0
        else:
            rzs_idx_start = 0
            bzs_idx_start = -round((z_target_to_base + brt_z_min) / self.zs_step)
        if z_target_to_base + brt_z_max > max(self.robot_zs):
            rzs_idx_end = num_robot_zs - 1
            bzs_idx_end = num_brt_zs - 1 - round((z_target_to_base + brt_z_max - max(self.robot_zs)) / self.zs_step)
        else:
            rzs_idx_end = num_robot_zs - 1 + round((z_target_to_base + brt_z_max - max(self.robot_zs)) / self.zs_step)
            bzs_idx_end = num_brt_zs - 1
        assert bzs_idx_end - bzs_idx_start == rzs_idx_end - rzs_idx_start, \
            "bzs: {0}, {1}; rzs: {2}, {3}".format(bzs_idx_start, bzs_idx_end, rzs_idx_start, rzs_idx_end)
        z_num = bzs_idx_end - bzs_idx_start + 1
        if z_num == 0 or rzs_idx_end <= 0:
            return [], [], []

        # BRT-Tensor = {z, dis(length=1), phi(length=1), gamma, idx} -> brt state,
        self.brt_tensor = self.brt_tensor[bzs_idx_start:bzs_idx_end + 1, ...]

        # Fixed-base limitation
        # Robot tensor = {z, dis, phi, gamma} -> max_v
        robot_tensor_v = np.expand_dims(self.robot_phi_gamma_velos_naive[rzs_idx_start: rzs_idx_end + 1, ...], axis=4)

        # Filter
        # 1.distance
        b = np.linalg.norm(DIS)
        robot_tensor_v = robot_tensor_v[:, np.where(self.robot_dis < b)[0], ...]

        # 2 calculate desired r
        cos_phi = np.cos(self.robot_phis)
        d_cosphi = self.robot_dis[self.robot_dis < b, np.newaxis] @ cos_phi[np.newaxis, :]
        r = np.sqrt(b**2 - self.robot_dis[self.robot_dis < b, None]**2 + d_cosphi**2) - d_cosphi
        r_tensor = r[None, :, :, None, None]
        mask_r = abs(-self.brt_tensor[:, :, :, :, :, 0] - r_tensor) < thres_dis

        # choose these brt data which are close to r wrt thres_v
        validate = np.argwhere((robot_tensor_v - thres_v - self.brt_tensor[:, :, :, :, :, 4] > 0)  # velocity satisfy
                               * mask_r
                               )

        q_indices = np.copy(validate[:, :4])
        q_indices[:, 0] += rzs_idx_start
        qids = self.robot_phi_gamma_q_idxs_naive[tuple(q_indices.T)].astype(int)
        q_candidates = self.mesh[qids, :]
        q_ae = self.ae[qids]
        phi_candidates = self.robot_phis[validate[:, 2]]
        x_candidates = self.brt_tensor[:, 0, 0, :, :, :][tuple(np.r_['-1', validate[:, :1], validate[:, 3:5]].T)][:, :4]
        error_index = np.nonzero(np.sum(np.isnan(x_candidates), axis=1))
        if error_index[0].shape[0] > 0:
            print("--error!!!")
            a = input("input to continue")
            q_candidates = np.delete(q_candidates, error_index, axis=0)
            q_ae = np.delete(q_ae, error_index, axis=0)
            phi_candidates = np.delete(phi_candidates, error_index, axis=0)
            x_candidates = np.delete(x_candidates, error_index, axis=0)

        # calculate alpha
        beta = np.arctan2(DIS[1], DIS[0])
        dis = np.linalg.norm(q_ae[:, :2], axis=1)
        alpha = -np.arccos(np.clip((dis - x_candidates[:, 0] * np.cos(phi_candidates)) / b, -1, 1)) * np.sign(
        phi_candidates) + beta
        AE_alpha = np.arctan2(q_ae[:, 1], q_ae[:, 0])

        # use joint 1 to control alpha
        q_candidates[:, 0] += alpha - AE_alpha
        q_candidates[q_candidates[:, 0] > np.pi, 0] -= 2 * np.pi
        q_candidates[q_candidates[:, 0] < -np.pi, 0] += 2 * np.pi

        return q_candidates, phi_candidates, x_candidates

    def get_full_throwing_config(self, robot, q, phi, x):
        """
        Return full throwing configurations
        :input param robot description, q, phi, x
        :return:
        """
        r = x[0]
        z = x[1]
        r_dot = x[2]
        z_dot = x[3]

        # kinemetic forward
        AE, J = self.robot.forward(q)

        throwing_angle = np.arctan2(AE[1], AE[0]) + phi
        EB_dir = np.array([np.cos(throwing_angle), np.sin(throwing_angle)])

        J_xyz = J[:3, :]
        J_xyz_pinv = np.linalg.pinv(J_xyz)




    def generate_throw_config(self, q_candidates, phi_candidates, x_candidates, base0):
        n_candidates = q_candidates.shape[0]

        # get full throwing configuration and trajectories
        traj_durations =[]
        trajs = []
        throw_configs = []

        # record bad trajectories
        num_outlimit, num_hit, num_ruckigerr, num_small_deviation = 0, 0, 0, 0

        for i in range(n_candidates):
            candidate_idx = i

            # 1.check joint0 limitation
            q0 = q_candidates[candidate_idx][0]
            if q0 > self.q_ul[0] or q0 < self.q_ll[0]:
                num_outlimit += 1
                continue

            throw_config_full =

    def get_traj_from_ruckig(self, q0, q0_dot,
                             qd, qd_dot,base0, based,
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

        zeros2 = np.zeros(2)

        input_length = len(q0) + len(zeros2)
        inp = InputParameter(input_length)
        inp.current_position = np.concatenate((q0, base0))
        inp.current_velocity = np.concatenate((q0_dot, zeros2))
        inp.current_acceleration = np.zeros(9)

        inp.target_position = np.concatenate((qd, based))
        inp.target_velocity = np.concatenate((qd_dot, zeros2))
        inp.target_acceleration = np.zeros(9)

        inp.max_velocity = np.array([self.max_velocity * margin_velocity, 2.0, 2.0])
        inp.max_acceleration = np.array([self.max_acceleration * margin_acceleration, 4.0, 4.0])
        inp.max_jerk = np.array([self.max_jerk * margin_jerk, 500, 500])

        otg = Ruckig(input_length)
        trajectory = Trajectory(input_length)
        _ = otg.calculate(inp, trajectory)

        return trajectory










    def solve(self, box_position):
        base0 = -box_position[:2]
        q_candidates, phi_candidates, x_candidates =


