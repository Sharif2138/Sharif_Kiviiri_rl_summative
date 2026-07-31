from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from stable_baselines3 import DQN
import pandas as pd

app = FastAPI(
    title="Driver Fatigue Intervention API",
    description="Real-time RL policy endpoint for fleet driver safety.",
    version="1.0.0"
)

ACTION_MAP = {
    0: "No Action (Driver Alert)",
    1: "Mild seat vibrations",
    2: "In bus red lights & mild Audio Alarm",
    3: "Fleet Manager Escalation"
}

class DriverStateRequest(BaseModel):
    perclos: float = Field(..., ge=0.0, le=1.0, description="Eye Closure Percentage (0.0 to 1.0)")
    yawn: float = Field(..., ge=0.0, le=10.0, description="Yawn Frequency")
    head_pose: float = Field(..., ge=-90.0, le=90.0, description="Head Pose Angle")
    drive_time: float = Field(..., ge=0.0, le=1440.0, description="Driving Time (minutes)")


model = None
try:
    leaderboard = pd.read_csv("logs/dqn_report_table.csv")
    best_run = int(leaderboard.loc[leaderboard["Mean Reward"].idxmax(), "Run"])
    best_model_path = f"models/dqn/dqn_run_{best_run}.zip"
    model = DQN.load(best_model_path, device="cpu")

    print(f"[SUCCESS] Loaded best DQN model (Run {best_run})")
    print(f"Model Path: {best_model_path}")

except Exception as e:
    print(f"[WARNING] Failed to load best model: {e}")

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": model is not None}


@app.post("/predict_action")
def predict_intervention(state: DriverStateRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="RL model is not loaded.")

    observation = np.array([
        state.perclos,
        state.yawn,
        state.head_pose,
        state.drive_time
    ], dtype=np.float32)

    action, _ = model.predict(observation, deterministic=True)
    action_int = int(action)

    return {
        "status": "success",
        "model": f"DQN Run {best_run}",
        "action_code": action_int,
        "recommended_intervention": ACTION_MAP[action_int],
        "observation": observation.tolist()
    }
