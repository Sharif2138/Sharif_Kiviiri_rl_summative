import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch

import environment


# helper functions to load raw monitor data
def load_monitor_csv(filepath):
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r") as f:
        first_line = f.readline()

    if first_line.startswith("#"):
        df = pd.read_csv(filepath, skiprows=1)
    else:
        df = pd.read_csv(filepath)

    if "r" in df.columns:
        return df["r"].values
    elif "reward" in df.columns:
        return df["reward"].values
    return None


def get_best_run_indices():
    best_runs = {"DQN": 1, "PPO": 1, "A2C": 1, "REINFORCE": 1}

    # Load DQN Leaderboard
    if os.path.exists("logs/dqn_report_table.csv"):
        dqn_df = pd.read_csv("logs/dqn_report_table.csv")
        if not dqn_df.empty and "Mean Reward" in dqn_df.columns:
            best_idx = dqn_df.loc[dqn_df["Mean Reward"].idxmax()]["Run"]
            best_runs["DQN"] = int(best_idx)

    # Load PG Leaderboard
    if os.path.exists("logs/pg_report_table.csv"):
        pg_df = pd.read_csv("logs/pg_report_table.csv")
        if not pg_df.empty and "Mean Reward" in pg_df.columns:
            for algo in ["PPO", "A2C", "REINFORCE"]:
                algo_df = pg_df[pg_df["Algorithm"] == algo]
                if not algo_df.empty:
                    best_idx = algo_df.loc[algo_df["Mean Reward"].idxmax(
                    )]["Run"]
                    best_runs[algo] = int(best_idx)

    return best_runs


def rolling_window_avg(data, window=15):
    if data is None or len(data) == 0:
        return np.array([])
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values


# Main plot generation function
def generate_report_plots():
    os.makedirs("assets", exist_ok=True)
    best_runs = get_best_run_indices()

    print("      GENERATING REPORT PLOTS FROM REAL MONITOR DATA    ")
    print(f"Selected Best Model Runs: {best_runs}\n")

    # Paths to raw monitor CSVs
    monitor_paths = {
        "DQN": f"logs/dqn_monitors/dqn_run_{best_runs['DQN']}.monitor.csv",
        "PPO": f"logs/ppo_monitors/ppo_run_{best_runs['PPO']}.monitor.csv",
        "A2C": f"logs/a2c_monitors/a2c_run_{best_runs['A2C']}.monitor.csv",
        "REINFORCE": f"logs/reinforce_monitors/reinforce_run_{best_runs['REINFORCE']}.csv",
    }

    raw_data = {
        algo: load_monitor_csv(path) for algo, path in monitor_paths.items()
    }

    # 2x2 Subplot Grid of Raw Training Curves
    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False, sharey=False)
    colors = {"DQN": "#1f77b4", "PPO": "#2ca02c",
              "A2C": "#ff7f0e", "REINFORCE": "#d62728"}

    algo_mapping = [
        ("DQN", axes[0, 0]),
        ("PPO", axes[0, 1]),
        ("A2C", axes[1, 0]),
        ("REINFORCE", axes[1, 1]),
    ]

    for algo, ax in algo_mapping:
        data = raw_data[algo]
        color = colors[algo]
        run_num = best_runs[algo]

        if data is not None and len(data) > 0:
            episodes = np.arange(1, len(data) + 1)
            smoothed = rolling_window_avg(data, window=15)

            # Raw noisy episode trace
            ax.plot(episodes, data, alpha=0.25,
                    color=color, label="Raw Episode Return")
            # Smoothed trajectory
            ax.plot(episodes, smoothed, color=color,
                    linewidth=2.2, label="15-Ep Moving Average")

            ax.set_title(f"{algo} (Run {run_num:02d}) Learning Curve",
                         fontsize=12, fontweight="bold")
            ax.set_xlabel("Training Episode", fontsize=10)
            ax.set_ylabel("Total Reward", fontsize=10)
            ax.legend(loc="lower right", frameon=True)
            ax.grid(True, linestyle="--", alpha=0.6)
        else:
            ax.text(0.5, 0.5, f"No monitor data found for {algo}\n({monitor_paths[algo]})",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11, color="red")

    plt.tight_layout()
    plot1_path = "assets/cumulative_rewards_subplots.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f" Saved: {plot1_path}")

    # Training Stability & Reward Variance Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, color in colors.items():
        data = raw_data[algo]
        if data is not None and len(data) > 0:
            # Resample to 100 normalized training progress points
            smoothed = rolling_window_avg(data, window=20)
            progress = np.linspace(0, 100, len(smoothed))
            ax.plot(progress, smoothed,
                    label=f"{algo} (Run {best_runs[algo]:02d})", color=color, linewidth=2.5)

    ax.set_title("Comparative Training Stability & Convergence Speed",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Progress (%)", fontsize=11)
    ax.set_ylabel("Smoothed Reward (20-Ep Moving Avg)", fontsize=11)
    ax.axhline(0, color="black", linestyle=":",
               alpha=0.7, label="Zero-Reward Threshold")
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot2_path = "assets/training_stability.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f" Saved: {plot2_path}")

    # Final Model Leaderboard Comparison
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Read mean rewards from report tables
    means = []
    algos = ["DQN", "PPO", "A2C", "REINFORCE"]

    # Load rewards directly from CSV tables
    for algo in algos:
        val = None
        if algo == "DQN" and os.path.exists("logs/dqn_report_table.csv"):
            df = pd.read_csv("logs/dqn_report_table.csv")
            if not df.empty:
                val = df["Mean Reward"].max()
        elif os.path.exists("logs/pg_report_table.csv"):
            df = pd.read_csv("logs/pg_report_table.csv")
            df_algo = df[df["Algorithm"] == algo]
            if not df_algo.empty:
                val = df_algo["Mean Reward"].max()

        means.append(val if val is not None else 0.0)

    bar_colors = [colors[a] for a in algos]
    bars = ax.bar(algos, means, color=bar_colors, width=0.55,
                  edgecolor="black", linewidth=1.2)

    # Annotate bar tops
    for bar in bars:
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        y_pos = height + (10 if height >= 0 else -25)
        ax.annotate(f"{height:+.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                    xytext=(0, 0), textcoords="offset points",
                    ha="center", va=va, fontsize=11, fontweight="bold")

    ax.set_title("Top Evaluation Performance Across All Algorithms",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean Evaluation Reward (10 Test Episodes)", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plot3_path = "assets/generalization_test.png"
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f" Saved: {plot3_path}\n")

    print("[SUCCESS] All 3 report plots generated from real training monitor logs!\n")


if __name__ == "__main__":
    generate_report_plots()
