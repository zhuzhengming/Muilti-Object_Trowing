""""
generate throwing trajectory by querying BRT and hedgehog
input: box_position
output: target[q, q_dot]
"""
import sys
from lxml import etree

sys.path.append("../")
from hedgehog import VelocityHedgehog
import time
import math
import numpy as np
from ruckig import InputParameter, Ruckig, Trajectory, Result
import rospy
import mujoco
import glfw


class TrajectoryGenerator:
    def __init__(self, q_ul, q_ll, hedgehog_path, brt_path, box_position, robot_path):
        rospy.init_node("trajectory_generator", anonymous=True)
        self.q_ul = q_ul
        self.q_ll = q_ll
        self.q0 = np.array([-0.32032486, 0.02707055, -0.22881525, -1.42611918, 1.38608943, 0.5596685, 0])
        self.q0_dot = np.zeros(7)
        self.hedgehog_path = hedgehog_path
        self.brt_path = brt_path
        self.gravity = -9.81
        self.GAMMA_TOLERANCE = 0.2/180.0*np.pi
        self.Z_TOLERANCE = 0.01
        self.box_position = box_position

        # mujoco similator
        if robot_path is None:
            self.robot_path = '../description/iiwa7_allegro_throwing.xml'

        q_dot_max = np.array([1.71, 1.74, 1.745, 2.269, 2.443, 3.142, 3.142])
        q_dot_min = -q_dot_max

        self.robot = VelocityHedgehog(self.q_ll, self.q_ul, q_dot_min, q_dot_max, robot_path)

        # self.max_velocity = np.array([1.71, 1.74, 1.745, 2.269, 2.443, 3.142, 3.142])
        self.max_acceleration = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
        self.max_jerk = np.array([7500, 3750, 5000, 6250, 7500, 10000, 10000])
        self.max_velocity = np.array(rospy.get_param('/max_velocity'))
        # self.max_acceleration = np.array(rospy.get_param('/max_acceleration'))
        # self.max_jerk = np.array(rospy.get_param('/max_jerk'))
        self.MARGIN_VELOCITY = rospy.get_param('/MARGIN_VELOCITY')
        self.MARGIN_ACCELERATION = rospy.get_param('/MARGIN_ACCELERATION')
        self.MARGIN_JERK = rospy.get_param('/MARGIN_JERK')

        self.load_data()

    def load_data(self):
        # load hedgehog data
        self.robot_zs = np.load(self.hedgehog_path + '/robot_zs.npy')
        self.robot_dis = np.load(self.hedgehog_path + '/robot_diss.npy')
        self.robot_phis = np.load(self.hedgehog_path + '/robot_phis.npy')
        self.robot_gamma = np.load(self.hedgehog_path + '/robot_gammas.npy')
        self.zs_step = 0.05

        self.robot_phi_gamma_velos_naive_all = np.load(self.hedgehog_path + '/z_dis_phi_gamma_vel_max.npy')

        # different hedgehog data for different posture
        # posture1
        self.p1_robot_phi_gamma_velos_naive = self.robot_phi_gamma_velos_naive_all[0]
        self.p1_robot_phi_gamma_q_idxs_naive = np.load(self.hedgehog_path + '/q_idxs_posture1.npy')
        self.p1_ae = np.load(self.hedgehog_path + '/q_idx_ae_posture1.npy')
        self.p1_mesh = np.load(self.hedgehog_path + '/q_idx_qs_posture1.npy')

        # posture2
        self.p2_robot_phi_gamma_velos_naive = self.robot_phi_gamma_velos_naive_all[1]
        self.p2_robot_phi_gamma_q_idxs_naive = np.load(self.hedgehog_path + '/q_idxs_posture2.npy')
        self.p2_ae = np.load(self.hedgehog_path + '/q_idx_ae_posture2.npy')
        self.p2_mesh = np.load(self.hedgehog_path + '/q_idx_qs_posture2.npy')

        # load brt data
        self.brt_tensor = np.load(self.brt_path + '/brt_tensor.npy')
        self.brt_zs = np.load(self.brt_path + '/brt_zs.npy')


    def brt_robot_data_matching(self, posture, thres_v=0.1, thres_dis=0.01, thres_phi=0.04, thres_r=0.1, box_pos=None):
        """
        original point is the base of robot
        Given target position, find out initial guesses of (q, phi, x)
        :param box_position:
        :param thres_dis:
        :param thres_v:
        :return: candidates of q, phi, x
        """
        if posture == "posture1":
            print(posture)
            ae = self.p1_ae
            mesh = self.p1_mesh
            robot_phi_gamma_velos_naive = self.p1_robot_phi_gamma_velos_naive
            robot_phi_gamma_q_idxs_naive = self.p1_robot_phi_gamma_q_idxs_naive
        elif posture == "posture2":
            print(posture)
            ae = self.p2_ae
            mesh = self.p2_mesh
            robot_phi_gamma_velos_naive = self.p2_robot_phi_gamma_velos_naive
            robot_phi_gamma_q_idxs_naive = self.p2_robot_phi_gamma_q_idxs_naive

        if box_pos is None:
            z_target_to_base = self.box_position[-1]
            AB = self.box_position[:2]
        else:
            z_target_to_base = box_pos[-1]
            AB = box_pos[:2]

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

        # BRT-Tensor = {z, dis, phi, gamma, layers} -> [r, z, r_dot, z_dot]
        self.brt_tensor = self.brt_tensor[bzs_idx_start:bzs_idx_end + 1, ...]

        # Fixed-base limitation
        # Robot tensor = [z, dis, phi, gamma, layers] -> [r, z, r_dot, z_dot, max_v]
        robot_tensor_v = np.expand_dims(robot_phi_gamma_velos_naive[rzs_idx_start: rzs_idx_end + 1, ...], axis=4)

        # Selection
        # 1.distance < b
        b = np.linalg.norm(AB) # from target position
        robot_tensor_v = robot_tensor_v[:, np.where(self.robot_dis < b)[0], ...]

        # 2 calculate desired r
        # given [dis, phi, target_position] -> [r, z, r_dot, z_dot] -> [r, gamma]
        cos_phi = np.cos(self.robot_phis)
        sin_phi = np.sin(self.robot_phis)
        d_cosphi = self.robot_dis[self.robot_dis < b, np.newaxis] @ cos_phi[np.newaxis, :]
        d_sinphi = self.robot_dis[self.robot_dis < b, np.newaxis] @ sin_phi[np.newaxis, :]
        r = np.sqrt(b ** 2 - d_sinphi ** 2) - d_cosphi
        r_tensor = r[None, :, :, None, None] #[None, dis, phi, None, None]
        mask_r = abs(-self.brt_tensor[:, :, :, :, :, 0] - r_tensor) < thres_dis

        # 3.choose these brt data which are close to r wrt thres_v
        validate = np.argwhere((robot_tensor_v -
                                thres_v - self.brt_tensor[:, :, :, :, :, 4] > 0)  # velocity satisfy
                               * mask_r)

        q_indices = np.copy(validate[:, :4])
        q_indices[:, 0] += rzs_idx_start
        if np.any(q_indices >= robot_phi_gamma_q_idxs_naive.shape[0]):
            return [], [], []

        qids = robot_phi_gamma_q_idxs_naive[tuple(q_indices.T)].astype(int)
        q_candidates = mesh[qids, :]
        q_ae = ae[qids]
        phi_candidates = self.robot_phis[validate[:, 2]]
        x_candidates = self.brt_tensor[:, 0, 0, :, :, :][tuple(np.r_['-1', validate[:, :1], validate[:, 3:5]].T)][:, :4]
        error_index = np.nonzero(np.sum(np.isnan(x_candidates), axis=1))
        if error_index[0].shape[0] > 0:
            print("--error!!!")
            q_candidates = np.delete(q_candidates, error_index, axis=0)
            q_ae = np.delete(q_ae, error_index, axis=0)
            phi_candidates = np.delete(phi_candidates, error_index, axis=0)
            x_candidates = np.delete(x_candidates, error_index, axis=0)

        # 4. close to target wrt r
        sorted_indices = np.argsort(abs(x_candidates[:, 0]))
        n_total = len(sorted_indices)
        n_keep = max(1, int(n_total * thres_r))
        final_indices = sorted_indices[:n_keep]

        q_candidates = q_candidates[final_indices]
        phi_candidates = phi_candidates[final_indices]
        x_candidates = x_candidates[final_indices]
        q_ae = q_ae[final_indices]

        # calculate alpha
        # (beta, dis)
        beta = np.arctan2(AB[1], AB[0])
        dis = np.linalg.norm(q_ae[:, :2], axis=1)
        alpha = (-np.arccos(np.clip((dis - x_candidates[:, 0] * np.cos(phi_candidates)) / b,
                                    -1, 1)) *
                 np.sign(phi_candidates) + beta)
        AE_alpha = np.arctan2(q_ae[:, 1], q_ae[:, 0])

        # use joint 0 to control alpha
        q_candidates[:, 0] += alpha - AE_alpha
        q_candidates[q_candidates[:, 0] > np.pi, 0] -= 2 * np.pi
        q_candidates[q_candidates[:, 0] < -np.pi, 0] += 2 * np.pi

        return q_candidates, phi_candidates, x_candidates

    def get_full_throwing_config(self, q, phi, x):
        """
        Return full throwing configurations
        :input param robot description, q, phi, x
        :return: (q, phi, x, q_dot, blockPosInGripper, eef_velo, AE, box_position)
        calculate from the throwing aspect
        """
        r = x[0]
        z = x[1]
        r_dot = x[2]
        z_dot = x[3]

        # kinemetic forward
        AE, J = self.robot.forward(q, posture="posture1")

        # in ee_site space
        throwing_angle = np.arctan2(AE[1], AE[0]) + phi
        EB_dir = np.array([np.cos(throwing_angle), np.sin(throwing_angle)])

        # get current jacobian to calculate q_dot = J_pinv @ eef_velo
        J_xyz = J[:3, :]
        J_xyz_pinv = np.linalg.pinv(J_xyz)

        eef_velo = np.array([EB_dir[0] * r_dot, EB_dir[1] * r_dot, z_dot])
        q_dot = J_xyz_pinv @ eef_velo
        # AB = AE - EB
        box_position = AE + np.array([-r * EB_dir[0], -r * EB_dir[1], -z]) # 3 dim

        # control last one joint to make end effector towards box

        gripperPos = self.robot.data.site("ee_site").xpos.copy() - 0.5 # position based on kuka_base
        gripperRot = self.robot.data.site("ee_site").xmat.copy().reshape(3,3)

        eef_velo_dir_3d = eef_velo / np.linalg.norm(eef_velo)

        tmp = AE + eef_velo_dir_3d
        blockPosInGripper = gripperRot.T @ (tmp - gripperPos)
        velo_angle_in_eef = np.arctan2(blockPosInGripper[1], blockPosInGripper[0])

        if (velo_angle_in_eef < math.pi) and (velo_angle_in_eef > -math.pi):
            eef_angle_near = velo_angle_in_eef
        elif velo_angle_in_eef > math.pi:
            eef_angle_near = velo_angle_in_eef - math.pi
        else:
            eef_angle_near = velo_angle_in_eef + math.pi

        q[-1] = eef_angle_near

        return (q, phi, x, q_dot, blockPosInGripper, eef_velo, AE, box_position)

    def generate_throw_config(self, q_candidates, phi_candidates, x_candidates, base0):
        """
        input: q_candidates, phi_candidates, x_candidates, base0(box_position in xoy)
        output: trajs, throw_configs
        """
        n_candidates = q_candidates.shape[0]

        # get full throwing configuration and trajectories
        traj_durations =[]
        trajs = []
        throw_configs = []

        # record bad trajectories
        num_outlimit, num_hit, num_ruckiger, num_small_deviation = 0, 0, 0, 0

        for i in range(n_candidates):
            candidate_idx = i

            # 1.check joint0 limitation
            q0 = q_candidates[candidate_idx][0]
            if q0 > self.q_ul[0] or q0 < self.q_ll[0]:
                num_outlimit += 1
                continue

            throw_config_full = self.get_full_throwing_config(q_candidates[candidate_idx],
                                                              phi_candidates[candidate_idx],
                                                              x_candidates[candidate_idx])

            # 2 check hit gripper palm
            # if throw_config_full[4][2] < -0.02:
            #     num_hit += 1
            #     continue

            # 3. valid trajectory
            try:
                traj_throw = self.get_traj_from_ruckig(q0=self.q0, q0_dot=self.q0_dot,
                                                       qd=throw_config_full[0],
                                                       qd_dot=throw_config_full[3])

                if traj_throw.duration < 1e-10:
                    num_ruckiger += 1
                    continue
            except Exception as e:
                num_ruckiger += 1
                continue

            deviation = throw_config_full[-1][:2] + base0
            if np.linalg.norm(deviation) < 0.01:
                num_small_deviation += 1

            traj_durations.append(traj_throw.duration)
            trajs.append(traj_throw)
            throw_configs.append(throw_config_full)

        print("\t\t out of joint limit: {}, hit the palm: {}, ruckig error: {}, small deviation:{}".format(
                num_outlimit, num_hit, num_ruckiger, num_small_deviation))

        return trajs, throw_configs


    def get_traj_from_ruckig(self, q0, q0_dot,
                             qd, qd_dot,
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


        input_length = len(q0)
        inp = InputParameter(input_length)
        inp.current_position = q0
        inp.current_velocity = q0_dot
        inp.current_acceleration = np.zeros(input_length)

        inp.target_position = qd
        inp.target_velocity = qd_dot
        inp.target_acceleration = np.zeros(input_length)

        inp.max_velocity = np.array(self.max_velocity * margin_velocity)
        inp.max_acceleration = np.array(self.max_acceleration * margin_acceleration)
        inp.max_jerk = np.array(self.max_jerk * margin_jerk)

        otg = Ruckig(input_length)
        trajectory = Trajectory(input_length)
        _ = otg.calculate(inp, trajectory)

        return trajectory


    def solve(self, animate=False, posture=None):
        base0 = -self.box_position[:2]
        # search result for specific posture
        q_candidates, phi_candidates, x_candidates = self.brt_robot_data_matching(posture)
        if len(q_candidates) == 0:
            print("No result found")
            return 0

        trajs, throw_configs = self.generate_throw_config(
            q_candidates, phi_candidates, x_candidates, base0)

        if len(trajs) == 0:
            print("No trajectory found")
            return 0


        # select the minimum-time trajectory
        traj_durations = [traj.duration for traj in trajs]
        selected_idx = np.argmin(traj_durations)
        traj_throw = trajs[selected_idx]
        throw_config_full = throw_configs[selected_idx]

        print("box_position: ", self.box_position)
        print("AB          : ", throw_config_full[-1])
        print("deviation   : ", throw_config_full[-1] - self.box_position)
        print("throwing state: ", throw_config_full[2])

        if animate:
            self.throw_simulation_mujoco(traj_throw, throw_config_full, posture=posture)

    def multi_waypoint_solve(self, box_positions, animate=False, posture=None):
        base1 = box_positions[0][:2]
        base2 = box_positions[1][:2]

        q_candidates_1, phi_candidates_1, x_candidates_1 = (
            self.brt_robot_data_matching(posture, box_pos=box_positions[0]))
        q_candidates_2, phi_candidates_2, x_candidates_2 = (
            self.brt_robot_data_matching(posture, box_pos=box_positions[1]))

        if len(q_candidates_1) == 0 or len(q_candidates_2) == 0:
            print("No result found")
            return 0

        trajs_1, throw_configs_1 = self.generate_throw_config(q_candidates_1,
                                                              phi_candidates_1,
                                                              x_candidates_1,
                                                              base1)

        trajs_2, throw_configs_2 = self.generate_throw_config(q_candidates_2,
                                                              phi_candidates_2,
                                                              x_candidates_2,
                                                              base2)

        if len(trajs_1) == 0 or len(trajs_2) == 0:
            print("No trajectory found")
            return 0

        min_distance = float('inf')
        best_pair = None
        best_throw_config_pair = None

        for i, cfg1 in enumerate(throw_configs_1):
            vec1 = np.concatenate([cfg1[0], cfg1[3]])
            for j, cfg2 in enumerate(throw_configs_2):
                vec2 = np.concatenate([cfg2[0], cfg2[3]])
                distance = np.linalg.norm(vec1 - vec2)
                if distance < min_distance:
                    min_distance = distance
                    best_pair = np.array([[cfg1[0], cfg1[3]], [cfg2[0], cfg2[3]]])
                    best_throw_config_pair = (cfg1, cfg2)

        if not best_pair:
            print("No valid pairs found")
            return 0

        forward_trajectory_1 = self.get_traj_from_ruckig(self.q0, self.q0_dot,
                                                 best_pair[0][0], best_pair[0][1])

        forward_trajectory_2 = self.get_traj_from_ruckig(best_pair[0][0], best_pair[0][1],
                                                 best_pair[1][0], best_pair[1][1])

        backward_trajectory_1 = self.get_traj_from_ruckig(self.q0, self.q0_dot,
                                                         best_pair[1][0], best_pair[1][1])

        backward_trajectory_2 = self.get_traj_from_ruckig(best_pair[1][0], best_pair[1][1],
                                                         best_pair[0][0], best_pair[0][1])

        # TODO: integrate two trajectories
        if forward_trajectory_1.duration + forward_trajectory_2.duration < \
            backward_trajectory_1.duration + backward_trajectory_2.duration:
            final_trajectory = forward_trajectory_1.duration + forward_trajectory_2.duration
        else:
            final_trajectory = backward_trajectory_1.duration + backward_trajectory_2.duration
            best_throw_config_pair = (best_throw_config_pair[1], best_throw_config_pair[0])


        if animate:
            self.throw_simulation_mujoco(final_trajectory, best_throw_config_pair, posture=posture)


    def throw_simulation_mujoco(self, trajectory, throw_config_full, posture=None):
        ROBOT_BASE_HEIGHT = 0.5
        if throw_config_full.shape[0] != 1:
            box_position_1 = throw_config_full[0][-1]
            AE_1 = throw_config_full[0][-1]
            box_position_2 = throw_config_full[1][-1]
            AE_2 = throw_config_full[1][-1]

            # set the target box position for visualization
            target_id_1 = self.robot.model.body("box").id  # set box position
            target_id_2 = self.robot.model.body("box").id  # set box position
            target_position_1 = box_position_1
            target_position_1[2] += ROBOT_BASE_HEIGHT
            target_position_2 = box_position_2
            target_position_2[2] += ROBOT_BASE_HEIGHT
            self.robot._set_object_position(target_id_1, target_position_1)
            self.robot._set_object_position(target_id_2, target_position_2)

        else:
            box_position = throw_config_full[-1]
            AE = throw_config_full[-2]
            EB = box_position - AE

            # set the target box position for visualization
            target_id = self.robot.model.body("box").id  # set box position
            target_position = self.box_position
            target_position[2] += ROBOT_BASE_HEIGHT
            self.robot._set_object_position(target_id, target_position)



        freq = 200
        delta_t = 1.0 / freq
        # self.robot.print_simulator_info() # output similator infos


        # set the target box position for visualization
        target_id = self.robot.model.body("box").id  # set box position
        target_position = self.box_position
        target_position[2] += ROBOT_BASE_HEIGHT
        self.robot._set_object_position(target_id, target_position)


        t0, tf = 0, trajectory.duration
        plan_time = tf - t0
        sample_t = np.arange(0, tf, delta_t)
        n_steps = sample_t.shape[0]
        traj_data = np.zeros([3, n_steps, 7])
        base_traj_data = np.zeros([3, n_steps, 2])

        # initial trajectory data
        for i in range(n_steps):
            for j in range(3):
                tmp = trajectory.at_time(sample_t[i])[j]
                traj_data[j, i] = tmp[:7]
                base_traj_data[j, i] = tmp[-2:]

        # reset the joint
        q0 = traj_data[0, 0]
        self.robot._set_joints(q0.tolist(), render=True)

        tt = 0
        flag = True
        throw_flag = False

        while True:
            if flag:
                ref_full = trajectory.at_time(tt)
                ref = [ref_full[i][:7] for i in range(3)]
                self.robot._set_joints(ref[0], ref[1], render=True)
            else:
                ref_full = trajectory.at_time(plan_time)
                ref = [ref_full[i][:7] for i in range(3)]
                self.robot._set_joints(ref[0], ref[1], render=True)

            # get the state of ee_site in the frame of kuka_base
            object_id = self.robot.model.body("sphere").id

            if posture == "posture1":
                ee_pos = (self.robot.obj_x2base("thumb_site") + self.robot.obj_x2base("middle_site")) / 2
                ee_pos[2] += ROBOT_BASE_HEIGHT
                ee_vel = (self.robot.obj_v("thumb_site")+self.robot.obj_v("middle_site")) / 2
            elif posture == "posture2":
                ee_pos = (self.robot.obj_x2base("index_site") + self.robot.obj_x2base("ring_site")) / 2
                ee_pos[2] += ROBOT_BASE_HEIGHT
                ee_vel = (self.robot.obj_v("index_site") + self.robot.obj_v("ring_site")) / 2
            else:
                ee_pos = self.robot.x2base
                ee_pos[2] += ROBOT_BASE_HEIGHT
                ee_vel = self.robot.dx

            if throw_flag is False:
                self.robot._set_object_position(object_id, ee_pos, ee_vel[:3])

            if tt > plan_time - 1 * delta_t:
                # self.robot._set_hand_joints(self.robot.hand_home_pose.tolist(), render=True)
                throw_flag = True
            else:
                self.robot._set_hand_joints(self.robot.envelop_pose.tolist(), render=True)
                # stick object to the ee_site

            tt += delta_t
            if tt > trajectory.duration:
                flag = False
            time.sleep(delta_t)

            if tt > 10.0:
                break



if __name__ == "__main__":
    q_min = np.array([-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239, -2.96705972839,
                      -2.09439510239, -3.05432619099])
    q_max = np.array([2.96705972839, 2.09439510239, 2.96705972839, 2.09439510239, 2.96705972839,
                      2.09439510239, 3.05432619099])
    hedgehog_path = '../hedgehog_revised'
    brt_path = '../brt_data'
    # hedgehog_path = '../fix_hedgehog'
    # brt_path = '../fix_hedgehog'

    robot_path = '../description/iiwa7_allegro_throwing.xml'
    box_position = np.array([1.4, 0.6, 0.0])

    trajectory_generator = TrajectoryGenerator(q_max, q_min,
                                               hedgehog_path, brt_path,
                                               box_position, robot_path)
    trajectory_generator.solve(animate=True, posture="posture1")
