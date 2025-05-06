import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from gym import spaces
import random
from trajectory_generator import TrajectoryGenerator


class RobotThrowingEnv(gym.Env):
    def __init__(self, solver):
        super(RobotThrowingEnv, self).__init__()

        self.q_min = np.array([-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239, -2.96705972839,
                               -2.09439510239, -3.05432619099])
        self.q_max = np.array([2.96705972839, 2.09439510239, 2.96705972839, 2.09439510239, 2.96705972839,
                               2.09439510239, 3.05432619099])

        self.box_min = np.array([-1.4, -1.4, -0.3])
        self.box_max = np.array([1.4, 1.4, 0.3])

        self.hedgehog_path = '../hedgehog_revised'
        self.brt_path = '../brt_data'
        self.xml_path = '../description/iiwa7_allegro_throwing.xml'

        self.solver = solver
        self.target_boxes = []
        self.current_target_idx = 0
        self.state = None
        self.action_space = None
        self.observation_space = spaces.Box(low=np.concatenate([self.q_min, [0]]),
                                            high=np.concatenate([self.q_max, [1]]),
                                            dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.all_time = 0.0
        self.step_num = 0
        q0 = np.random.uniform(self.q_min, self.q_max, 7)
        while True:
            self.target_boxes = [np.random.uniform(low=-1.4, high=1.4, size=(3,)) for _ in range(2)]
            if -1.0 < self.target_boxes[0][0] < 1.0 and -1.0 < self.target_boxes[0][1] < 1.0\
                    and -1.0 < self.target_boxes[1][0] < 1.0 and -1.0 < self.target_boxes[1][1] < 1.0:
                continue
        info = {}
        return self.state, info

    def step(self, action):
        terminated = False
        q0 = self.state[:7]
        step_num = int(self.state[7])

        time_duration, solution_set = self.update_action_space(q0, self.target_boxes[step_num])

        action_idx = action

        if action_idx >= len(solution_set):
            raise ValueError(f"Action index {action_idx} out of range. Available actions: {len(solution_set)}")

        chosen_q = solution_set[action_idx]
        chosen_time = time_duration[action_idx]

        reward = self.calculate_trajectory_time(chosen_time)

        # update state
        self.state = np.concatenate([chosen_q, [step_num+1]])

        self.step_num += 1
        if self.step_num >= 2:
            terminated = True

        return self.state, reward, terminated, {}, {}

    def calculate_trajectory_time(self, chosen_time):
        time_reward = chosen_time
        self.all_time += time_reward

        return 1.0 * time_reward + 2.0 * self.all_time


    def update_action_space(self, q0, box_pos):
        q_candidates = []
        time_duration = []
        self.trajectoryGenerator = TrajectoryGenerator(self.q_max, self.q_min,
                                                       self.hedgehog_path, self.brt_path,
                                                       self.xml_path, q0=q0, model_exist=True)

        trajs, throw_configs = self.trajectoryGenerator.solve(posture='posture2', box_pos=box_pos)

        for pair in zip(throw_configs, trajs):
            q_candidates.append([pair[0][0], pair[0][3]])
            time_duration.append(pair[1].duration)

        num_actions = len(q_candidates)
        self.action_space = spaces.Discrete(num_actions)
        return time_duration, q_candidates