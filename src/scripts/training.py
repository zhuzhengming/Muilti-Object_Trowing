from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from gymnasium.envs.registration import register, registry

import gymnasium as gym
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from RL_throwing import RobotThrowEnv

class EpisodeCounterCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_count += 1
                if self.episode_count % 100 == 0:
                    print(f"Global episode count: {self.episode_count}")
        return True

def training(env, eval_env):

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        max_grad_norm=0.5,
        tensorboard_log="../logs/robot_throw"
    )

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=3000, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path="../best_models/",
        log_path="../logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )

    # training
    try:
        model.learn(
            total_timesteps=50_000_000,
            callback=[EpisodeCounterCallback(), eval_callback],
            tb_log_name="ppo_robot_throw",
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")

    model.save("robot_throw_ppo_final")
    print("Model saved successfully")


def evaluate_model(model, env, n_episodes=10):
    env.eval_mode = True
    success_count = 0
    total_reward = 0
    error_list = []

    for i in range(n_episodes):
        obs, info = env.reset(options={"training_mode": False})
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            episode_reward += reward

            if done:
                if info['feasible']:
                    error = info['landing_error']
                    error_list.append(error)
                    if error < 0.2:
                        success_count += 1
                total_reward += episode_reward

                print(f"Episode {i + 1}:")
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Landing Error: {error:.4f} m")
                print(f"  Landing Velocity: {np.linalg.norm(info.get('landing_velocity', 0)):.2f} m/s")
                print("  Reward Components:")
                for k, v in info['reward_components'].items():
                    print(f"    {k}: {v:.2f}")

    print("\nEvaluation Summary:")
    print(f"Success Rate: {success_count / n_episodes * 100:.1f}%")
    print(f"Average Reward: {total_reward / n_episodes:.2f}")
    print(f"Mean Landing Error: {np.mean(error_list):.4f} m")
    print(f"Std Landing Error: {np.std(error_list):.4f} m")


if __name__ == "__main__":
    # mode = "evaluate"
    mode = "train"

    register(
        id="RobotThrowEnv-v0",
        entry_point="RL_search:RobotThrowEnv",
        max_episode_steps=1000,
        kwargs={"path_prefix": ""}
    )

    if "RobotThrowEnv-v0" in registry:
        print("registered!")
        entry_point = registry["RobotThrowEnv-v0"].entry_point
        print(f"   Entry Point: {entry_point}")
    else:
        print("not registered!")

    env = gym.make("RobotThrowEnv-v0", path_prefix="")
    eval_env = gym.make("RobotThrowEnv-v0", path_prefix="")

    if mode == "train":
        training(env, eval_env)
    elif mode == "evaluate":
        if os.path.exists("../best_models/best_model.zip"):
            model = PPO.load("../best_models/best_model", env=env)
            print("Loaded best model for evaluation")
        else:
            print("Using final model for evaluation")

        evaluate_model(model, env)