from gymnasium.envs.registration import register

register(
    id="DriverFatigue-v0",
    entry_point="environment.custom_env:DriverFatigueEnv",
    max_episode_steps=50,
)
