import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
import environment


# custom REINFORCE implementation
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

    def select_action(self, state):
        state_t = torch.FloatTensor(state)
        probs = self.forward(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


def train_reinforce_single_run(lr, gamma, hidden_dim, max_episodes=500):
    env = gym.make("DriverFatigue-v0")
    policy = PolicyNetwork(state_dim=4, action_dim=4, hidden_dim=hidden_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for episode in range(max_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []

        done = False
        while not done:
            action, log_prob = policy.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        discounted_returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            discounted_returns.insert(0, G)

        discounted_returns = torch.tensor(
            discounted_returns, dtype=torch.float32)

        if len(discounted_returns) > 1 and discounted_returns.std() > 1e-8:
            discounted_returns = (
                discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-8)

        policy_loss = []
        for log_prob, G_t in zip(log_probs, discounted_returns):
            policy_loss.append(-log_prob * G_t)

        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

    eval_rewards = []
    for _ in range(10):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            state_t = torch.FloatTensor(state)
            with torch.no_grad():
                probs = policy(state_t)
                action = torch.argmax(probs).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        eval_rewards.append(ep_reward)

    env.close()
    return float(np.mean(eval_rewards))


# Experiment runner for  REINFORCE, PPO, AND A2C
def run_policy_gradient_sweeps():
    os.makedirs("logs", exist_ok=True)
    csv_file_path = "logs/pg_report_table.csv"

    print("STARTING POLICY GRADIENT & ACTOR-CRITICAL HYPERPARAMETER SWEEPS")
    results = []

    # REINFORCE sweeps (10 runs)
    print("--> Running REINFORCE Hyperparameter Sweeps (10 Runs)...")
    reinforce_configs = [
        {"lr": 0.001,  "gamma": 0.99, "hidden": 64},
        {"lr": 0.0005, "gamma": 0.99, "hidden": 64},
        {"lr": 0.001,  "gamma": 0.95, "hidden": 128},
        {"lr": 0.0005, "gamma": 0.95, "hidden": 128},
        {"lr": 0.0001, "gamma": 0.99, "hidden": 64},
        {"lr": 0.001,  "gamma": 0.90, "hidden": 64},
        {"lr": 0.0005, "gamma": 0.90, "hidden": 128},
        {"lr": 0.0001, "gamma": 0.95, "hidden": 64},
        {"lr": 0.002,  "gamma": 0.99, "hidden": 32},
        {"lr": 0.0005, "gamma": 0.99, "hidden": 32},
    ]

    for idx, cfg in enumerate(reinforce_configs, 1):
        mean_rew = train_reinforce_single_run(
            cfg["lr"], cfg["gamma"], cfg["hidden"])
        results.append({
            "Algorithm": "REINFORCE",
            "Run": idx,
            "Learning Rate": cfg["lr"],
            "Gamma": cfg["gamma"],
            "N-Steps/Batch": f"Hidden: {cfg['hidden']}",
            "Mean Reward": round(mean_rew, 2)
        })
        print(
            f"    REINFORCE Run {idx:02d}/10 | LR: {cfg['lr']} | Gamma: {cfg['gamma']} | Mean Reward: {mean_rew:.2f}")

    # PPO sweeps (10 Runs)
    print("\n--> Running PPO Hyperparameter Sweeps (10 Runs)...")
    ppo_configs = [
        {"lr": 0.0003, "gamma": 0.99, "n_steps": 128, "batch_size": 32},
        {"lr": 0.001,  "gamma": 0.99, "n_steps": 256, "batch_size": 64},
        {"lr": 0.0001, "gamma": 0.95, "n_steps": 128, "batch_size": 32},
        {"lr": 0.0005, "gamma": 0.99, "n_steps": 512, "batch_size": 128},
        {"lr": 0.0003, "gamma": 0.90, "n_steps": 128, "batch_size": 32},
        {"lr": 0.001,  "gamma": 0.95, "n_steps": 256, "batch_size": 64},
        {"lr": 0.0005, "gamma": 0.99, "n_steps": 128, "batch_size": 64},
        {"lr": 0.0001, "gamma": 0.99, "n_steps": 256, "batch_size": 32},
        {"lr": 0.0003, "gamma": 0.95, "n_steps": 512, "batch_size": 64},
        {"lr": 0.0005, "gamma": 0.90, "n_steps": 256, "batch_size": 128},
    ]

    for idx, cfg in enumerate(ppo_configs, 1):
        env = gym.make("DriverFatigue-v0")
        model = PPO("MlpPolicy", env, learning_rate=cfg["lr"], gamma=cfg["gamma"],
                    n_steps=cfg["n_steps"], batch_size=cfg["batch_size"], verbose=0)
        model.learn(total_timesteps=25000)
        mean_rew, _ = evaluate_policy(model, env, n_eval_episodes=10)
        env.close()

        results.append({
            "Algorithm": "PPO",
            "Run": idx,
            "Learning Rate": cfg["lr"],
            "Gamma": cfg["gamma"],
            "N-Steps/Batch": f"Steps: {cfg['n_steps']}, Batch: {cfg['batch_size']}",
            "Mean Reward": round(float(mean_rew), 2)
        })
        print(
            f"    PPO Run {idx:02d}/10       | LR: {cfg['lr']} | Gamma: {cfg['gamma']} | Mean Reward: {mean_rew:.2f}")

    # A2C sweeps (10 Runs)
    print("\n--> Running A2C Hyperparameter Sweeps (10 Runs)...")
    a2c_configs = [
        {"lr": 0.0007, "gamma": 0.99, "n_steps": 5},
        {"lr": 0.001,  "gamma": 0.99, "n_steps": 10},
        {"lr": 0.0003, "gamma": 0.95, "n_steps": 5},
        {"lr": 0.0005, "gamma": 0.99, "n_steps": 20},
        {"lr": 0.0007, "gamma": 0.90, "n_steps": 5},
        {"lr": 0.001,  "gamma": 0.95, "n_steps": 10},
        {"lr": 0.0003, "gamma": 0.99, "n_steps": 20},
        {"lr": 0.0005, "gamma": 0.95, "n_steps": 5},
        {"lr": 0.0007, "gamma": 0.99, "n_steps": 10},
        {"lr": 0.0001, "gamma": 0.99, "n_steps": 5},
    ]

    for idx, cfg in enumerate(a2c_configs, 1):
        env = gym.make("DriverFatigue-v0")
        model = A2C("MlpPolicy", env, learning_rate=cfg["lr"], gamma=cfg["gamma"],
                    n_steps=cfg["n_steps"], verbose=0)
        model.learn(total_timesteps=25000)
        mean_rew, _ = evaluate_policy(model, env, n_eval_episodes=10)
        env.close()

        results.append({
            "Algorithm": "A2C",
            "Run": idx,
            "Learning Rate": cfg["lr"],
            "Gamma": cfg["gamma"],
            "N-Steps/Batch": f"Steps: {cfg['n_steps']}",
            "Mean Reward": round(float(mean_rew), 2)
        })
        print(
            f"    A2C Run {idx:02d}/10       | LR: {cfg['lr']} | Gamma: {cfg['gamma']} | Mean Reward: {mean_rew:.2f}")

    # Write all 30 results to CSV
    fieldnames = ["Algorithm", "Run", "Learning Rate",
                  "Gamma", "N-Steps/Batch", "Mean Reward"]
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(
        f"\n[SUCCESS] Saved Policy Gradient experiment results to '{csv_file_path}'!")


if __name__ == "__main__":
    run_policy_gradient_sweeps()
