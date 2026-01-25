import os
import numpy as np
import gym
from diffuser.env_ours.pusht.pusht_env import PushTEnv
from diffuser.env_ours.utils import aggregate_dct

class PushTWrapper(PushTEnv):
    def __init__(
            self, 
            with_velocity=True,
            with_target=True,
            is_cluttered=False,
            state_based=False,
            use_sin_cos=False,
        ):
        super().__init__(
            with_velocity=with_velocity,
            with_target=with_target, 
            is_cluttered=is_cluttered,
            state_based=state_based,
            use_sin_cos=use_sin_cos,
        )
        print(f"PushT with_velocity: {self.with_velocity}  with_target: {self.with_target}  state_based: {self.state_based}")
        self.action_dim = self.action_space.shape[0]
    
    def sample_random_init_goal_states(self, seed, fix_goal=False):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        rs = np.random.RandomState(seed)
        
        def generate_state():
            base_state = [
                rs.randint(50, 450),   # agent x
                rs.randint(50, 450),   # agent y
                rs.randint(100, 400),  # block x
                rs.randint(100, 400),  # block y
            ]   
            angle = rs.randn() * 2 * np.pi - np.pi
            if self.use_sin_cos:
                angle_vals = [np.sin(angle), np.cos(angle)]  # 2 values
            else:
                angle_vals = [angle]  # keep it as 1-element list
            
            if self.with_velocity:
                state = np.array(base_state + angle_vals + [0, 0], dtype=np.float32)
            else:
                state = np.array(base_state + angle_vals, dtype=np.float32)
            return state
        
        init_state = generate_state()
        if fix_goal:
            angle = np.pi / 4
            goal_state = np.array([200, 200, 256, 256] + ([np.sin(angle), np.cos(angle)] if self.use_sin_cos else [angle]) + ([0, 0] if self.with_velocity else []), dtype=np.float32)
        else:   
            goal_state = generate_state()
        
        return init_state, goal_state
    
    def update_env(self, env_info):
        self.shape = env_info['shape']
    
    def eval_state(self, goal_state, cur_state):
        """
        Return True if the goal is reached
        [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]
        """
        # if position difference is < 20, and angle difference < np.pi/9, then success
        pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])
        if self.use_sin_cos:
            goal_angle = np.arctan2(goal_state[4], goal_state[5])
            cur_angle = np.arctan2(cur_state[4], cur_state[5])
        else:
            goal_angle = goal_state[4]
            cur_angle = cur_state[4]
        angle_diff = np.abs(goal_angle - cur_angle)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        success = pos_diff < 20 and angle_diff < np.pi / 9
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
        self.seed(seed)
        self.reset_to_state = init_state
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
