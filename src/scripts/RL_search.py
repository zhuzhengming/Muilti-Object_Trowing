import sys
sys.path.append("../")

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ruckig import InputParameter, Ruckig, Trajectory, Result
from hedgehog import VelocityHedgehog

class RobotThrowEnv(gym.Env):
    def __init__(self, path_prefix):
        super(RobotThrowEnv, self).__init__()

        self.reward_params = {
            'target_error': -10.0,
            'boundary_violation': -2.0,
            'singularity': -0.5,
            'velocity_horizontal': -1.0,
            'velocity_vertical': -0.5,
            'infeasible_penalty': -1000
        }

        self.action_noise = {
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
            'q_ddot_max': np.array([1.35, 1.35, 2.0, 2.0, 7.5, 10.0, 10.0]),
            'jerk_max': np.array([6.75, 6.75, 10.0, 10.0, 30.0, 30.0, 30.0])
        }

        self.robot_path = '../description/iiwa7_allegro_throwing.xml'

        self.robot = VelocityHedgehog(self.joint_limits['q_min'],
                                      self.joint_limits['q_max'],
                                      self.joint_limits['q_dot_max'],
                                      self.joint_limits['q_ddot_max'],
                                      self.robot_path,
                                      train_mode=True)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        self.gravity = np.array([0, 0, -9.81])
        self.episode_count = 0

    def reset(self, seed=None, options=None):
        """
        return: (obs, info)
        """
        self.episode_count += 1
        self.q_init = self.q0[:6]
        self.q_dot_init = np.zeros(6)

        self.x_target = np.random.uniform(low=[-1.5, -1.5, 0], high=[1.5, 1.5, 0.5])

        obs = self._normalize_state()
        info = {}

        return obs, info

    def step(self, action):
        """
        return: (obs, reward, terminated, truncated, info)
        """
        truncated = False

        # add noise to action
        if self.action_noise['enable']:
            noise_scale = self.action_noise['scale'] * (self.action_noise['decay_rate'] ** self.episode_count)
            action = action + np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action, -1.0, 1.0)

        q_desired, q_dot_desired = self._denormalize_action(action)

        is_feasible = self.get_traj_from_ruckig(
            q0=self.q_init,
            q0_dot=self.q_dot_init,
            q0_dotdot=np.zeros_like(self.q_dot_init),
            qd=q_desired,
            qd_dot=q_dot_desired,
            qd_dotdot=np.zeros_like(q_dot_desired)
        )

        # calculate reward
        if not is_feasible:
            reward = self.reward_params['infeasible_penalty']
            terminated = True
            info = {
                'feasible': False,
                'reward_components': {
                    'total': reward,
                    'infeasible': reward,
                    'target_error': 0,
                    'boundary': 0,
                    'singularity': 0,
                    'velocity': 0
                }
            }
        else:
            x_release, v_release = self._compute_release_state(q_desired, q_dot_desired)
            landing_state = self._simulate_throw(x_release, v_release)
            x_landing = landing_state[:3]
            v_landing = landing_state[3:]

            reward, reward_info = self._compute_reward(x_landing, v_landing, q_desired)
            terminated = True
            info = {
                'feasible': True,
                'landing_error': np.linalg.norm(x_landing - self.x_target),
                'landing_velocity': np.linalg.norm(v_landing),
                'release_pos': x_release,
                'release_vel': v_release,
                'reward_components': reward_info
            }

        if terminated:
            next_state, next_info = self.reset()
        else:
            next_state = self._normalize_state()
            info = {}

        return next_state, reward, terminated, truncated, info

    def _denormalize_action(self, action):
        action = np.clip(action, -1.0, 1.0)
        q_desired = action[6:12] * self.joint_limits['q_max'][:6]
        q_dot_desired = action[6:12] * self.joint_limits['q_dot_max'][:6]
        return q_desired, q_dot_desired

    def _normalize_state(self):
        q_norm = self.q_init / self.joint_limits['q_max'][:6]
        q_dot_norm = self.q_dot_init / self.joint_limits['q_dot_max'][:6]
        x_target_norm = self.x_target / np.array([1.0, 1.0, 0.5])
        return np.concatenate([q_norm, q_dot_norm, x_target_norm]).astype(np.float32)

    def get_traj_from_ruckig(self, q0, q0_dot, q0_dotdot,
                             qd, qd_dot, qd_dotdot
                             ):

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


        inp = InputParameter(len(q0))
        inp.current_position = q0
        inp.current_velocity = q0_dot
        inp.current_acceleration = q0_dotdot

        inp.target_position = qd
        inp.target_velocity = qd_dot
        inp.target_acceleration = qd_dotdot

        inp.max_velocity = self.joint_limits['q_dot_max']
        inp.max_acceleration = self.joint_limits['q_ddot_max']
        inp.max_jerk = self.joint_limits['jerk_max']

        otg = Ruckig(len(q0))
        trajectory = Trajectory(len(q0))
        result = otg.calculate(inp, trajectory)

        return result==Result.Finished

    def _compute_release_state(self, q_desired, q_dot_desired):
        x_release, J = self.robot.forward(q_desired)
        v_release = J @ q_dot_desired

        return x_release[:3], v_release[:3]

    def _simulate_throw(self, x_release, v_release):
        discriminant = v_release[2] ** 2 + 2 * self.gravity[2] * x_release[2]
        if discriminant < 0 or x_release[2] <= 0:
            return np.concatenate([x_release, v_release])

        t_flight = (v_release[2] + np.sqrt(v_release[2] ** 2 + 2 * self.gravity[2] * x_release[2])) / (-self.gravity[2])
        x_landing = x_release + v_release * t_flight + 0.5 * self.gravity * t_flight ** 2

        v_landing = v_release + self.gravity * t_flight

        return np.concatenate([x_landing, v_landing])

    def _compute_reward(self, x_landing, v_landing, q_desired):
        reward_info = {}
        error = np.linalg.norm(x_landing - self.x_target)
        reward_info['target_error'] = self.reward_params['target_error'] * error

        q_violation = np.sum(
            np.maximum(q_desired - self.joint_limits['q_max'], 0) +
            np.maximum(self.joint_limits['q_min'] - q_desired, 0)
        )
        reward_info['boundary'] = self.reward_params['boundary_violation'] * q_violation

        _, J = self.robot.forward(q_desired)
        _, s, _ = np.linalg.svd(J)
        cond = s[0] / s[-1] if s[-1] > 1e-6 else 1e6
        reward_info['singularity'] = self.reward_params['singularity'] * cond if cond > 100 else 0.0

        v_horizontal = np.linalg.norm(v_landing[:2])
        v_vertical = abs(v_landing[2])
        reward_info['velocity'] = (
                self.reward_params['velocity_horizontal'] * v_horizontal +
                self.reward_params['velocity_vertical'] * v_vertical
        )

        total_reward = sum(reward_info.values())
        reward_info['total'] = total_reward

        return total_reward, reward_info

