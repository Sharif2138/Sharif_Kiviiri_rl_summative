import argparse
import time
import sys
import gymnasium as gym
import environment


def run_simulation():
    print("  AI Fleet Road Safety Monitor - Simulation Mode")

    env = gym.make("DriverFatigue-v0", render_mode="human")
    obs, _ = env.reset()

    done = False
    step_count = 0
    total_reward = 0.0

    print("Starting Driver Shift Simulation...\n")

    while not done:
        perclos = obs[0]

        if perclos < 0.35:
            action = 0
        elif perclos < 0.65:
            action = 1
        elif perclos < 0.85:
            action = 2
        else:
            action = 3

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        step_count += 1
        total_reward += reward
        time.sleep(0.1)

        done = terminated or truncated
        print(
            f"Step {step_count:03d} | PERCLOS: {obs[0]:.2f} | Action: {action} | Step Reward: {reward:6.2f} | Total Reward: {total_reward:7.2f}")

    env.close()
    print(f"  Simulation Complete! Total Reward: {total_reward:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Driver Fatigue Reinforcement Learning System")
    parser.add_argument(
        "--mode",
        type=str,
        default="sim",
        choices=["sim", "train_dqn", "train_pg", "plot", "api"],
        help="Execution mode: 'sim' (GUI simulation), 'train_dqn', 'train_pg', 'plot' (generate graphs), or 'api' (start REST API)."
    )
    args = parser.parse_args()

    if args.mode == "sim":
        run_simulation()
    elif args.mode == "train_dqn":
        from training.dqn_training import run_dqn_sweeps
        run_dqn_sweeps()
    elif args.mode == "train_pg":
        from training.pg_training import run_policy_gradient_sweeps
        run_policy_gradient_sweeps()
    elif args.mode == "plot":
        from training.evaluate_and_plot import generate_report_plots
        generate_report_plots()
    elif args.mode == "api":
        import uvicorn
        print("\nStarting Fleet Monitoring FastAPI Server on http://localhost:8000 ...\n")
        uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
