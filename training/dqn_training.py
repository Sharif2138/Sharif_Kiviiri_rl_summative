import os
import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import environment

SEED = 42


def evaluate_dqn_policy(model, num_episodes=10):
    eval_env = gym.make("DriverFatigue-v0")
    eval_env.reset(seed=SEED)
    eval_env.action_space.seed(SEED)
    
    total_rewards = []

    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)
        total_rewards.append(ep_reward)

    eval_env.close()
    return float(np.mean(total_rewards))


def run_dqn_sweeps():
    os.makedirs("models/dqn", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/dqn_monitors", exist_ok=True)
    
    np.random.seed(SEED)

    configs = [
        {"lr": 1e-3, "gamma": 0.99, "buffer_size": 10000, "batch_size": 32, "explore": 0.1},
        {"lr": 1e-4, "gamma": 0.99, "buffer_size": 10000, "batch_size": 32, "explore": 0.1},
        {"lr": 5e-4, "gamma": 0.95, "buffer_size": 50000, "batch_size": 64, "explore": 0.2},
        {"lr": 1e-3, "gamma": 0.95, "buffer_size": 50000, "batch_size": 64, "explore": 0.1},
        {"lr": 1e-4, "gamma": 0.90, "buffer_size": 20000, "batch_size": 128, "explore": 0.05},
        {"lr": 5e-4, "gamma": 0.99, "buffer_size": 10000, "batch_size": 128, "explore": 0.15},
        {"lr": 1e-3, "gamma": 0.90, "buffer_size": 50000, "batch_size": 32, "explore": 0.2},
        {"lr": 1e-4, "gamma": 0.99, "buffer_size": 100000, "batch_size": 64, "explore": 0.1},
        {"lr": 5e-4, "gamma": 0.95, "buffer_size": 20000, "batch_size": 32, "explore": 0.05},
        {"lr": 1e-3, "gamma": 0.99, "buffer_size": 100000, "batch_size": 128, "explore": 0.2},
    ]

    results = []

    print("STARTING DQN HYPERPARAMETER SWEEPS (10 RUNS)")

    for i, config in enumerate(configs, 1):
        # Create a fresh environment for each training run
        train_env = gym.make("DriverFatigue-v0")
        train_env.reset(seed=SEED)
        train_env.action_space.seed(SEED)
        
        train_env = Monitor(train_env, f"logs/dqn_monitors/dqn_run_{i}")

        model = DQN(
            "MlpPolicy",
            train_env,
            learning_rate=config["lr"],
            gamma=config["gamma"],
            buffer_size=config["buffer_size"],
            batch_size=config["batch_size"],
            exploration_fraction=config["explore"],
            seed=SEED,
            verbose=0,
            tensorboard_log="./logs/dqn_tensorboard/"
        )

        model.learn(total_timesteps=50000, tb_log_name=f"DQN_Run_{i}")
        train_env.close()

        # Evaluate over 10 test episodes
        mean_reward = evaluate_dqn_policy(model, num_episodes=10)
        print(
            f"DQN Run {i:02d}/10 | LR: {config['lr']} | Gamma: {config['gamma']} | Mean Reward: {mean_reward:.2f}")

        # Save trained weights
        model.save(f"models/dqn/dqn_run_{i}")

        row = {
            "Run": i,
            "Learning Rate": config["lr"],
            "Gamma": config["gamma"],
            "Replay Buffer Size": config["buffer_size"],
            "Batch Size": config["batch_size"],
            "Exploration Strategy": config["explore"],
            "Mean Reward": round(mean_reward, 2)
        }
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv("logs/dqn_report_table.csv", index=False)
    print(
        "\n[SUCCESS] Saved 10 DQN experiment results to 'logs/dqn_report_table.csv'!\n")


if __name__ == "__main__":
    run_dqn_sweeps()
