# Fleet Driver Fatigue Monitoring — RL Summative

**Video demo:** https://vimeo.com/1214704672?fl=ip&fe=ec

A reinforcement learning system that simulates an in-cab driver fatigue monitor for
a commercial fleet vehicle. The agent watches signs of drowsiness (eye closure, yawning,
head pose, drive time) and decides when to intervene — from doing nothing, to a mild
seat vibration, to escalating the situation to a fleet manager.

Four algorithms were trained and compared on the same custom environment:
**DQN, REINFORCE, PPO, and A2C**.

## Environment

- **Name:** `DriverFatigue-v0`
- **Observation space:** `Box(4,)` — PERCLOS, yawn frequency, head pitch, continuous drive time
- **Action space:** `Discrete(4)` — monitor / mild vibration / in-cab alarm / manager escalation
- **Episode length:** up to 50 steps (a 12.5-hour simulated shift), or ends early if
  the driver crosses a critical drowsiness threshold
- **Rendering:** 3D cockpit view built with PyGame + PyOpenGL (`environment/rendering.py`)

## Project Structure

```text
project_root/
├── pyproject.toml
├── uv.lock
├── README.md
├── main.py
│
├── environment/
│   ├── __init__.py
│   ├── custom_env.py          # DriverFatigueEnv (Gymnasium env)
│   └── rendering.py           # 3D OpenGL/PyGame renderer
│
├── training/
│   ├── __init__.py
│   ├── dqn_training.py        # DQN hyperparameter sweeps (10 runs)
│   ├── pg_training.py         # REINFORCE / PPO / A2C sweeps (10 runs each)
│   └── evaluate_and_plot.py   # Generates all report plots
│
├── api/
│   └── api_server.py          # FastAPI endpoint serving the best DQN model
│
├── models/                    # Saved trained models (DQN, REINFORCE, PPO, A2C)
├── logs/                      # Monitor CSVs, TensorBoard logs, report tables
├── assets/                    # Generated plots for the report
```
## Setup

This project uses **[uv](https://docs.astral.sh/uv/)** for dependency and environment
management. No manual virtual environment or `pip install` is required.

```bash
git clone git@github.com:Sharif2138/Sharif_Kiviiri_rl_summative.git
cd Sharif_Kiviiri_rl_summative
uv sync
```

`uv sync` reads `pyproject.toml` / `uv.lock` and installs everything (Gymnasium,
Stable-Baselines3, PyTorch, PyGame, PyOpenGL, FastAPI, etc.) into a local `.venv`
automatically.

## Usage

All entry points run through `main.py` with `uv run`, so no extra activation step
is needed.

**Watch the trained agent in action (3D GUI simulation):**
```bash
uv run main.py --mode sim
```
Loads the best-performing DQN model (`models/dqn/dqn_run_1`) and runs one full shift,
rendering the cockpit view and printing step-by-step telemetry to the terminal.

**Re-run the DQN hyperparameter sweep (10 runs):**
```bash
uv run main.py --mode train_dqn
```

**Re-run the REINFORCE / PPO / A2C hyperparameter sweeps (10 runs each):**
```bash
uv run main.py --mode train_pg
```

**Regenerate all report plots from the saved logs:**
```bash
uv run main.py --mode plot
```

**Start the REST API (serves the best DQN model for real-time predictions):**
```bash
uv run main.py --mode api
```

Then send a POST request to `http://localhost:8000/predict_action` with a JSON body:

```json
{
  "perclos": 0.42,
  "yawn": 3.0,
  "head_pose": -12.5,
  "drive_time": 180.0
}
```

## Results Summary

| Algorithm | Best Mean Reward (10 eval episodes) |
|---|---|
| DQN | **352.19** |
| PPO | 335.29 |
| A2C | 290.77 |
| REINFORCE | 81.72 |

DQN was the most stable and best-performing algorithm across all hyperparameter
configurations. Full hyperparameter tables, training curves, and analysis are in the
project report (`report.pdf`).

## Report & Video

- **Report:** `report.pdf`
- **Video demo:** https://vimeo.com/1214704672?fl=ip&fe=ec

## Author

Sharif Kiviiri — ALU, ML Techniques II Summative Assignment
