import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from trajectory_generator import TrajectoryGenerator
from sb3_contrib.common.wrappers import ActionMasker


class RobotThrowingEnv(gym.Env):
    def __init__(self, box_num = 2):
        super(RobotThrowingEnv, self).__init__()

        self.q_min = np.array([-2.967, -2.094, -2.967, -2.094, -2.967, -2.094, -3.054])
        self.q_max = np.abs(self.q_min)
        self.q_dot_max = np.array([1.71, 1.74, 1.745, 2.269, 2.443, 3.142, 3.142])
        self.q_dot_min = -self.q_dot_max
        self.box_min = np.array([-1.3, -1.3, -0.2])
        self.box_max = np.array([1.3, 1.3, 0.2])

        self.max_steps = box_num
        self.current_step = 0
        self._episode_terminated = False

        self.observation_space = spaces.Box(
            low=np.concatenate([self.q_min, self.q_dot_min, [0], np.tile(self.box_min, 2)]),
            high=np.concatenate([self.q_max, self.q_dot_max, [1], np.tile(self.box_max, 2)]),
            dtype=np.float32
        )

        self.max_actions = 300
        self.action_space = spaces.Discrete(self.max_actions)
        self.valid_action_mask = np.zeros(self.max_actions, dtype=bool)

        self.trajectory_generator = TrajectoryGenerator(
            self.q_max, self.q_min,
            hedgehog_path='../hedgehog_revised',
            brt_path='../brt_data',
            robot_path='../description/iiwa7_allegro_throwing.xml',
            model_exist=True
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_terminated = False
        # fix q0
        # q0 = np.random.uniform(self.q_min, self.q_max)
        q0 = np.array([-1.5783, 0.1498, 0.1635, -0.7926, -0.0098, 0.6, 1.2881])
        q0_dot = np.zeros_like(self.q_dot_min)
        self.target_boxes = self._generate_valid_boxes()

        self.state = np.concatenate([
            q0,
            q0_dot,
            [0.0],
            self.target_boxes[0],
            self.target_boxes[1]
        ])

        self.current_step = 0
        self.total_time = 0.0

        self._update_action_space()

        return self.state, {}

    def step(self, action):
        if self._episode_terminated:
            raise RuntimeError("Episode already terminated")

        if not (0 <= action < self.max_actions) or not self.valid_action_mask[action]:
            self._episode_terminated = True
            return self.state, -100.0, True, False, {"reason": "invalid action"}

        chosen_q = self.current_solutions[action]
        chosen_time = self.current_durations[action]

        # update state
        self.state = np.concatenate([
            np.array(chosen_q).flatten(),
            [(self.current_step + 1) / self.max_steps],
            self.target_boxes[0],
            self.target_boxes[1]
        ])

        self.total_time += chosen_time
        reward = self._calculate_reward(chosen_time)
        self.current_step += 1

        terminated = self.current_step >= self.max_steps

        if not terminated:
            try:
                self._update_action_space()
            except Exception as e:
                print(f"Action space update failed: {e}")
                terminated = True

        return self.state, reward, terminated, False, {}

    def _generate_valid_boxes(self):
        while True:
            boxes = [
                np.array([
                    np.random.uniform(self.box_min[0], self.box_max[0]),
                    np.random.uniform(self.box_min[1], self.box_max[1]),
                    np.random.uniform(self.box_min[2], self.box_max[2])
                ]) for _ in range(2)
            ]

            if all((np.abs(box[0]) > 1.0) or (np.abs(box[1]) > 1.0) for box in boxes):
                return boxes

    def _update_action_space(self):
        q0 = self.state[:7]
        target_box = self.target_boxes[self.current_step]

        try:
            trajs, configs = self.trajectory_generator.solve(
                posture='posture2',
                box_pos=target_box,
                q0=q0
            )
        except Exception as e:
            print(f"Trajectory generation failed: {e}")
            trajs, configs = [], []

        self.current_solutions = []
        self.current_durations = []
        for config, traj in zip(configs, trajs):
            self.current_solutions.append([config[0], config[3]])
            self.current_durations.append(traj.duration)

        self.valid_action_mask.fill(False)

        num_valid = len(self.current_solutions)
        self.valid_action_mask[:num_valid] = True

        if num_valid == 0:
            self._episode_terminated = True
            return

        pad = self.max_actions - num_valid
        self.current_solutions += [np.full_like(q0, np.nan)] * pad
        self.current_durations += [np.nan] * pad

    def _calculate_reward(self, time_cost):
        step_penalty = 1 * time_cost

        total_penalty = 1 * self.total_time

        success_bonus = 10.0 / (1 + self.total_time) if self.current_step == self.max_steps-1 else 0.0

        valid_action_bonus = 1 * np.sum(self.valid_action_mask) / self.max_actions

        return -step_penalty - total_penalty + success_bonus + valid_action_bonus

    def action_mask(self):
        return self.valid_action_mask

def mask_fn(env):
    return env.action_mask()



if __name__ == "__main__":
    mode = "train"
    box_num = 2
    env = RobotThrowingEnv(box_num)
    env = ActionMasker(env, mask_fn)

    if mode == "test":
        obs, _ = env.reset()

        action = np.argmax(env.action_mask())
        next_obs, reward, done, _, _ = env.step(action)

    if mode == "train":
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path="../checkpoints/",
            name_prefix="rl_model"
        )

        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.97,
            clip_range=0.2,
            tensorboard_log="../logs"
        )

        model.learn(
            total_timesteps=2_000_000,
            callback=[checkpoint_callback],
            progress_bar=True
        )

    model.save("robotic_throwing_model")