from gym.envs.registration import register
from .pointmaze import U_MAZE, MEDIUM_MAZE, LARGE_MAZE, GIANT_MAZE
register(
    id="pusht-v0",
    entry_point="diffuser.env_ours.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
register(
    id='point_maze-v0',
    entry_point='diffuser.env_ours.pointmaze:PointMazeWrapper',
    max_episode_steps=300,
    kwargs={
        'maze_spec':U_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 23.85,
        'ref_max_score': 161.86,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
    }
)
register(
    id='point_maze_medium-v0',
    entry_point='diffuser.env_ours.pointmaze:PointMazeWrapper',
    max_episode_steps=300,
    kwargs={
        'maze_spec':MEDIUM_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 23.85,
        'ref_max_score': 161.86,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
    }
)
register(
    id='point_maze_large-v0',
    entry_point='diffuser.env_ours.pointmaze:PointMazeWrapper',
    max_episode_steps=300,
    kwargs={
        'maze_spec':LARGE_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 23.85,
        'ref_max_score': 161.86,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
    }
)
register(
    id='point_maze_giant-v0',
    entry_point='diffuser.env_ours.pointmaze:PointMazeWrapper',
    max_episode_steps=300,
    kwargs={
        'maze_spec':GIANT_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 23.85,
        'ref_max_score': 161.86,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
    }
)
register(
    id="wall-v0",
    entry_point="diffuser.env_ours.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
register(
    id="dmcontrol_reacher3-v0",
    entry_point="diffuser.env_ours.dmcontrol_wrapper:DMControlWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)