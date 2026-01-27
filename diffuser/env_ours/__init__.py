from gym.envs.registration import register
register(
    id="pusht-v0",
    entry_point="diffuser.env_ours.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
register(
    id="wall-v0",
    entry_point="diffuser.env_ours.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)