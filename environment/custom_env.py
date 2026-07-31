import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DriverFatigueEnv(gym.Env):
    metadata = {"render_modes": ["human", "console"], "render_fps": 5}

    def __init__(self, render_mode="console"):
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)

        low_bounds = np.array([0.0, 0.0, -90.0, 0.0], dtype=np.float32)
        high_bounds = np.array([1.0, 10.0, 90.0, 1440.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_bounds, high=high_bounds, dtype=np.float32)

        self.max_steps = 50
        self.current_step = 0
        self.state = None

        self.last_action = 0
        self.last_reward = 0.0
        self.renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.last_action = 0
        self.last_reward = 0.0

        self.state = np.array([
            self.np_random.uniform(0.0, 0.1),    
            self.np_random.uniform(0.0, 0.5),    
            self.np_random.uniform(-5.0, 5.0),   
            0.0                             
        ], dtype=np.float32)

        info = {"message": "Driver started shift fully alert."}
        return self.state, info

    def step(self, action):
        self.current_step += 1
        perclos, yawn, head_pose, drive_time = self.state

        drive_time += 15.0  

        time_factor = (drive_time / 1440.0) * 0.04
        perclos = np.clip(perclos + time_factor +
                          self.np_random.uniform(-0.02, 0.05), 0.0, 1.0)
        yawn = np.clip(yawn + (time_factor * 8.0) +
                       self.np_random.uniform(-0.3, 1.0), 0.0, 10.0)
        head_pose = np.clip(
            head_pose + self.np_random.uniform(-10.0, 10.0) - (perclos * 15.0), -90.0, 90.0)

        reward = 0.0

        if action == 0:
            if perclos < 0.3:
                reward += 2.0 
        elif action == 1:
            perclos = max(0.0, perclos - 0.15)
            reward -= 1.0 
        elif action == 2:
            perclos = max(0.0, perclos - 0.40)
            yawn = max(0.0, yawn - 2.0)
            if perclos < 0.4:
                reward -= 10.0 
            else:
                reward += 5.0
        elif action == 3:
            perclos = 0.0
            yawn = 0.0
            reward -= 15.0
       
        safety_baseline = 10.0
        fatigue_penalty = (perclos * 25.0) + (yawn * 1.5)
        reward += (safety_baseline - fatigue_penalty)

        
        terminated = False
        if perclos > 0.85:
            reward -= 100.0  
            terminated = True

        truncated = bool(self.current_step >= self.max_steps)

        
        self.state = np.array(
            [perclos, yawn, head_pose, drive_time], dtype=np.float32)
        self.last_action = action
        self.last_reward = float(reward)

        info = {"perclos": perclos, "drive_time": drive_time}

        return self.state, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "console":
            perclos, yawn, head, time = self.state
            print(f"Step {self.current_step:03d} | Drive Time: {time:4.0f}m | PERCLOS: {perclos:.2f} | Action: {self.last_action} | Reward: {self.last_reward:.1f}")
        elif self.render_mode == "human":
            if self.renderer is None:
                from environment.rendering import DriverFatigueRenderer
                self.renderer = DriverFatigueRenderer()
            self.renderer.render(self.state, self.last_action,
                                 self.current_step, self.last_reward)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
