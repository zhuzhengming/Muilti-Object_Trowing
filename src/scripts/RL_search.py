import sys
import time

from sympy import false

sys.path.append("../")

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils.mujoco_interface import Robot
import mujoco
from mujoco import viewer


class RobotThrowEnv(gym.Env):
    def __init__(self, path_prefix):
        super(RobotThrowEnv, self).__init__()

        self.reward_params = {
            'success': 1000,
            'target_error': -20.0,
            'boundary_violation': -2.0,
            'action_scale': -5.0,
            'velocity_horizontal': -2.0,
            'velocity_vertical': -2.0,
            'motion_reward': 2.0
        }

        self.action_noise = {
            'enable': True,
            'scale': 0.05,
            'decay_rate': 0.999
        }
        self.observation_noise = {
            'enable': True,
            'scale': 0.05,
            'decay_rate': 0.999
        }

        self.q0 = np.array(
            [-0.32032486, 0.02707055, -0.22881525, -1.42611918, 1.38608943, 0.5596685, -1.34659665 + np.pi])

        self.joint_limits = {
            'q_min': np.array([-2.96705972839, -2.09439510239, -2.96705972839,
                               -2.09439510239, -2.96705972839, -2.09439510239, -3.05432619099]),
            'q_max': np.array([2.96705972839, 2.09439510239, 2.96705972839,
                               2.09439510239, 2.96705972839, 2.09439510239, -3.05432619099]),
            'q_dot_max': np.array([1.71, 1.74, 1.745, 2.269, 2.443, 3.142, 3.142]),
        }

        self.target_max = np.array([1.5, 1.5, 0.4])
        self.freq = 200

        # create mujoco simulator
        self.robot_path = '../description/iiwa7_allegro_throwing.xml'
        model = mujoco.MjModel.from_xml_path(self.robot_path)
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        self.view = viewer.launch_passive(model, data)
        self.robot = Robot(model, data, self.view, auto_sync=True)

        self.error_threshold = 0.15
        self.last_action = np.zeros(14)
        self.time_step = 0
        self.max_steps = 200
        self.release_state = 0


        # 14:current joint state 3:target, 1:timestamp 15:last action, 1: release state
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(34,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(15,),
            dtype=np.float32
        )

        self.gravity = np.array([0, 0, -9.81])
        self.episode_count = 0

    def reset(self, seed=None, options=None):
        """
        return: (obs, info)
        """
        if options is None:
            mode = 'training'
        else:
            mode = 'testing'

        self.episode_count += 1
        self.q_init = np.array(np.random.uniform(self.joint_limits['q_min'],self.joint_limits['q_max']))
        self.q_dot_init = np.zeros(7)
        self.time_step = 0
        self.last_action = np.zeros(15)
        self.release_state = 0

        self.robot.modify_joint(joints=self.q_init)

        # move the target gradually
        progress = min(self.episode_count / 5000, 1.0)
        if mode == 'testing':
            self.x_target = np.array([
                np.random.uniform(-1.5, 1.5),
                np.random.uniform(-1.5, 1.5),
                np.random.uniform(0.0, 0.4)
            ])
        elif mode == 'training':
            self.x_target = np.array([
                np.random.uniform(-1.5 * progress, 1.5 * progress),
                np.random.uniform(-1.5 * progress, 1.5 * progress),
                np.random.uniform(0.0 * progress, 0.4 * progress)
            ])

        obs = self._observation()
        info = {}

        return obs, info

    def step(self, action):
        """
        return: (obs, reward, terminated, truncated, info)
        """
        self.last_action = action.copy()
        self.time_step +=1

        release_control, q_desired, q_dot_desired = self._decode_action(action)

        # execute action
        if not self.release_state:
            self.robot.iiwa_hand_go(q=q_desired,
                                    dq=q_dot_desired,
                                    qh=np.zeros(16),
                                    render=True)

            if release_control:
                self.release_state = 1

        # calculate reward
        x_release, v_release, x_landing, v_landing, reward, reward_info =\
            self._compute_reward(q_desired)

        info = {
            'landing_error': np.linalg.norm(x_landing - self.x_target),
            'landing_velocity': np.linalg.norm(v_landing),
            'release_pos': x_release,
            'release_vel': v_release,
            'reward_components': reward_info
        }


        terminated = self.release_state == 1
        truncated = self.time_step >= self.max_steps or self._check_joint_limits()

        next_state = self._observation()

        # if terminated:
        #     self._render(np.array(x_landing))

        return next_state, reward, terminated, truncated, info

    def _render(self,x_landing):

        self.robot.view.add_marker(
            pos=self.x_target,
            size=[0.1, 0.1, 0.1],
            rgba=[0, 1, 0, 0.5],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            id=0
        )

        self.robot.view.add_marker(
            pos=x_landing,
            size=[0.08, 0.08, 0.08],
            rgba=[1, 0, 0, 1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            id=1
        )

        self.robot.view.sync()

    def _check_joint_limits(self):
        return (np.any(self.robot.q < self.joint_limits['q_min']) or
            np.any(self.robot.q > self.joint_limits['q_max']) or
            np.any(self.robot.qd > self.joint_limits['q_dot_max']) or
            np.any(self.robot.qd < -self.joint_limits['q_dot_max']))

    def _decode_action(self, action):
        release_control = (action[-1]+1) / 2 > 0.5
        joint_action = action[:-1]
        # add noise to action
        if self.action_noise['enable']:
            noise_scale = self.action_noise['scale'] * (self.action_noise['decay_rate'] ** self.episode_count)
            joint_action = joint_action + np.random.normal(0, noise_scale, size=14)
            joint_action = np.clip(joint_action, -1.0, 1.0)

        q_desired = joint_action[:7] * self.joint_limits['q_max']
        q_dot_desired = joint_action[7:14] * self.joint_limits['q_dot_max']
        return release_control, q_desired, q_dot_desired

    def _observation(self):
        self.q_init = self.robot.q
        self.q_dot_init = self.robot.dq
        q_norm = self.q_init / self.joint_limits['q_max']
        q_dot_norm = self.q_dot_init / self.joint_limits['q_dot_max']

        if self.observation_noise['enable']:
            noise_scale = self.observation_noise['scale'] * (self.observation_noise['decay_rate'] ** self.episode_count)
            q_norm += np.random.normal(0, noise_scale, size=q_norm.shape)
            q_dot_norm += np.random.normal(0, noise_scale, size=q_dot_norm.shape)
            q_norm = np.clip(q_norm, -1.0, 1.0)
            q_dot_norm = np.clip(q_dot_norm, -1.0, 1.0)

        x_target_norm = self.x_target / self.target_max
        time_norm = np.array([self.time_step/self.max_steps])
        last_action = self.last_action
        release_state = np.array([self.release_state])

        return np.concatenate([
            q_norm,
            q_dot_norm,
            x_target_norm,
            time_norm,
            last_action,
            release_state
        ]).astype(np.float32)

    def _simulate_throw(self):

        x_release = self.robot.x[:3]
        v_release = self.robot.dx[:3]
        discriminant = v_release[2] ** 2 + 2 * self.gravity[2] * x_release[2]
        if discriminant < 0 or x_release[2] <= 0:
            return np.concatenate([x_release, v_release, np.zeros(3), np.zeros(3)])

        t_flight = (v_release[2] + np.sqrt(v_release[2] ** 2 + 2 * self.gravity[2] * x_release[2])) / (-self.gravity[2])
        x_landing = x_release + v_release * t_flight + 0.5 * self.gravity * t_flight ** 2

        v_landing = v_release + self.gravity * t_flight

        return np.concatenate([x_release, v_release, x_landing, v_landing])

    def _compute_reward(self, q_desired):
        reward_info = {}
        x_release = np.zeros(3)
        v_release = np.zeros(3)
        x_landing = np.zeros(3)
        v_landing = np.zeros(3)

        if self.release_state == 1:
            landing_state = self._simulate_throw()
            x_release = np.array(landing_state[:3])
            v_release = np.array(landing_state[3:6])
            x_landing = np.array(landing_state[6:9])
            v_landing = np.array(landing_state[9:])

            # 1.position error
            error = np.linalg.norm(x_landing - self.x_target)
            reward_info['target_error'] = self.reward_params['target_error'] * error
            # 2.success reward
            if error < self.error_threshold:
                reward_info['success'] = self.reward_params['success']
            # 3.boundary penalty
            q_violation = np.sum(
                np.maximum(q_desired - self.joint_limits['q_max'], 0) +
                np.maximum(self.joint_limits['q_min'] - q_desired, 0)
            )
            reward_info['boundary'] = self.reward_params['boundary_violation'] * q_violation
            # 4.landing velocity penalty
            v_horizontal = np.linalg.norm(v_landing[:2])
            v_vertical = abs(v_landing[2])
            reward_info['velocity'] = (
                    self.reward_params['velocity_horizontal'] * v_horizontal +
                    self.reward_params['velocity_vertical'] * v_vertical
            )
            # 5.action scale penalty
            action_magnitude = np.linalg.norm(self.last_action)
            reward_info['action'] = self.reward_params['action_scale'] * action_magnitude
        else:
            # 1.boundary penalty
            q_violation = np.sum(
                np.maximum(q_desired - self.joint_limits['q_max'], 0) +
                np.maximum(self.joint_limits['q_min'] - q_desired, 0)
            )
            reward_info['boundary'] = self.reward_params['boundary_violation'] * q_violation

            # 2.action scale
            action_magnitude = np.linalg.norm(self.last_action[:-1])
            reward_info['action'] = self.reward_params['action_scale'] * action_magnitude

            # 3.speed up encourage
            ee_speed = np.linalg.norm(self.robot.dx[:3])
            reward_info['motion'] = self.reward_params['motion_reward'] * ee_speed

        total_reward = sum(reward_info.values())
        reward_info['total'] = total_reward

        return x_release, v_release, x_landing, v_landing, total_reward, reward_info



