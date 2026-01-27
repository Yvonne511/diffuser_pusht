import os
import numpy as np
import gym
from diffuser.env_ours.pointmaze.maze_model import MazeEnv
from diffuser.env_ours.utils import aggregate_dct

STATE_RANGES = np.array([
    [0.39318362, 3.2198412],  # Range for first dimension
    [0.62660956, 3.2187355],  # Range for second dimension
    [-5.2262554, 5.2262554],  # Range for third dimension
    [-5.2262554, 5.2262554],  # Range for fourth dimension
    # [0.90001136, 3.0999563],  # Range for first dimension of target
    # [0.9000267, 3.0999668]    # Range for second dimension of target
])

class PointMazeWrapper(MazeEnv):
    def __init__(self, state_based=False, **kwargs):
        if state_based:
            kwargs['return_value'] = 'state'
        else:
            kwargs['return_value'] = 'obs'
        super().__init__(**kwargs)
        self.action_dim = self.action_space.shape[0]
    
    def sample_random_init_goal_states(self, seed, fix_goal=False):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        # TODO: fix_goal not used 
        rs = np.random.RandomState(seed)

        def generate_state():
            if self.spec.name in ('point_maze_medium', 'point_maze_large'):
                x, y = self.sample_random_xy()
                state = np.array([
                    x, 
                    y,
                    rs.uniform(low=STATE_RANGES[2][0], high=STATE_RANGES[2][1]),
                    rs.uniform(low=STATE_RANGES[3][0], high=STATE_RANGES[3][1]),
                ])
            else: # U_MAZE
                valid = False
                while not valid:
                    x, y = self.sample_random_xy()
                    valid = ((0.5 <= x <= 1.1 or 2.5 <= x <= 3.1) and (0.5 <= y <= 3.1))\
                            or ((1.1 < x < 2.5) and (2.5 <= y <= 3.1))
                state = np.array([
                    # 0.5, 6.1, 0, 0
                    x, 
                    y,
                    rs.uniform(low=STATE_RANGES[2][0], high=STATE_RANGES[2][1]),
                    rs.uniform(low=STATE_RANGES[3][0], high=STATE_RANGES[3][1]),
                ])
            return state

        init_state = generate_state()
        goal_state = generate_state()
        return init_state, goal_state
    
    def update_env(self, env_info):
        pass 
    
    def eval_state(self, goal_state, cur_state):
        success = np.linalg.norm(goal_state[:2] - cur_state[:2]) < 0.5
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            'success': success,
            'state_dist': state_dist,
        }

    def prepare(self, seed, init_state, stabilize=False):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        self.prepare_for_render()
        self.seed(seed)
        self.set_init_state(init_state)
        obs, state = self.reset(stabilize=stabilize)
        return obs, state

    def step_multiple(self, actions):
        """
        infos: dict, each key has shape (T, ...)
        """
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        only returns np arrays of observations and states
        seed: int
        init_state: (state_dim, )
        actions: (T, action_dim)
        obses: dict (T, H, W, C)
        states: (T, D)
        """
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states
