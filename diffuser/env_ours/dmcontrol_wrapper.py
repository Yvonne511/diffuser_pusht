import os
import numpy as np
import gym
from dm_control import suite
from diffuser.env_ours.utils import aggregate_dct

# Register custom dm_control tasks (e.g. four-link reacher)
import diffuser.env_ours.dmcontrol_tasks  # noqa: F401
suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)

STATE_RANGES = np.array([
    [0.39318362, 3.2198412],  # Range for first dimension
    [0.62660956, 3.2187355],  # Range for second dimension
    [-5.2262554, 5.2262554],  # Range for third dimension
    [-5.2262554, 5.2262554],  # Range for fourth dimension
])

class DMControlWrapper(gym.Env):
    '''gym wrapper for DMControl envs'''
    def __init__(self, **kwargs):
        self.domain = kwargs['domain']
        self.task = kwargs['task']
        self.state_based = kwargs.get('state_based', False)
        self.use_sin_cos = kwargs.get('use_sin_cos', False)
        self.dm_env = suite.load(domain_name=self.domain, task_name=self.task)
        scaling_factor = kwargs.get('scaling_factor', 5)
        self.dm_env.physics.model.actuator_ctrlrange *= scaling_factor
        if self.domain == 'reacher':
            model = self.dm_env.physics.model
            # Hide target geom by name (index varies by number of links)
            target_idx = self.dm_env.physics.model.name2id('target', 'geom')
            model.geom_rgba[target_idx] = [0, 0, 0, 0]
        self.action_dim = self.dm_env.action_spec().shape[0]
        self.n_joints = len(self.dm_env.physics.data.qpos)
        self.img_size = 224

    def _encode_state(self, qpos, qvel):
        """Build state vector, replacing qpos angles with (sin, cos) pairs if enabled."""
        if self.use_sin_cos:
            sin_cos = np.array([[np.sin(a), np.cos(a)] for a in qpos]).flatten()
            return np.concatenate([sin_cos, qvel]).astype(np.float32)
        return np.concatenate([qpos, qvel]).astype(np.float32)

    def _decode_qpos_qvel(self, state):
        """Recover (qpos, qvel) from a state vector, handling sin/cos encoding."""
        n = self.n_joints
        if self.use_sin_cos:
            qpos = np.array([np.arctan2(state[2 * j], state[2 * j + 1]) for j in range(n)])
            qvel = state[2 * n:]
        else:
            qpos = state[:n]
            qvel = state[n:]
        return qpos, qvel

    def _make_obs(self, state):
        if self.state_based:
            visual = state
        else:
            visual = self.dm_env.physics.render(height=self.img_size, width=self.img_size, camera_id=0)
        return {
            'visual': visual,
            'proprio': state
        }

    def render(self, mode='rgb_array'):
        return self.dm_env.physics.render(height=self.img_size, width=self.img_size, camera_id=0)

    def sample_random_init_goal_states(self, seed, fix_goal=False):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        rs = np.random.RandomState(seed)
        self.dm_env.reset()
        self.dm_env.physics.data.qpos[:] = rs.uniform(-np.pi, np.pi, size=self.n_joints)
        self.dm_env.physics.data.qvel[:] = rs.uniform(-1, 1, size=self.n_joints)
        self.dm_env.physics.forward()
        qpos = self.dm_env.physics.data.qpos.copy()
        qvel = self.dm_env.physics.data.qvel.copy()
        init_state = self._encode_state(qpos, qvel)
        self.dm_env.reset()
        self.dm_env.physics.data.qpos[:] = rs.uniform(-np.pi, np.pi, size=self.n_joints)
        self.dm_env.physics.data.qvel[:] = rs.uniform(-1, 1, size=self.n_joints)
        self.dm_env.physics.forward()
        qpos = self.dm_env.physics.data.qpos.copy()
        qvel = self.dm_env.physics.data.qvel.copy()
        goal_state = self._encode_state(qpos, qvel)
        return init_state, goal_state

    def update_env(self, env_info):
        pass

    def set_task_goal(self, goal_state):
        self.goal_state = goal_state

    def is_success(self, goal_state, cur_state):
        goal_qpos, _ = self._decode_qpos_qvel(goal_state)
        cur_qpos, _ = self._decode_qpos_qvel(cur_state)
        diff = np.abs(goal_qpos - cur_qpos)
        for idx in range(diff.shape[0]):
            diff[idx] = np.mod(diff[idx], 2 * np.pi)
            diff[idx] = min(diff[idx], 2 * np.pi - diff[idx])
        print(f"diff after: {diff}", np.linalg.norm(diff))
        return np.linalg.norm(diff) < 0.1

    def eval_state(self, goal_state, cur_state):
        goal_qpos, _ = self._decode_qpos_qvel(goal_state)
        cur_qpos, _ = self._decode_qpos_qvel(cur_state)
        diff = np.abs(goal_qpos - cur_qpos)
        for idx in range(diff.shape[0]):
            diff[idx] = np.mod(diff[idx], 2 * np.pi)
            diff[idx] = min(diff[idx], 2 * np.pi - diff[idx])
        state_dist = np.linalg.norm(diff)
        return {
            'success': state_dist < 0.1,
            'state_dist': state_dist
        }

    def prepare(self, seed, init_state, stabilize=False):
        """
        Reset with controlled init_state
        obs: dict with 'visual' (H W C) and 'proprio' (state_dim)
        state: (state_dim)
        """
        np.random.seed(seed)
        self.reset()
        qpos, qvel = self._decode_qpos_qvel(init_state)
        self.dm_env.physics.data.qpos[:] = qpos
        self.dm_env.physics.data.qvel[:] = qvel
        self.dm_env.physics.forward()
        qpos = self.dm_env.physics.data.qpos.copy()
        qvel = self.dm_env.physics.data.qvel.copy()
        state = self._encode_state(qpos, qvel)
        obs = self._make_obs(state)
        return obs, state

    def reset(self):
        time_step = self.dm_env.reset()
        qpos = self.dm_env.physics.data.qpos.copy()
        qvel = self.dm_env.physics.data.qvel.copy()
        state = self._encode_state(qpos, qvel)
        obs = self._make_obs(state)
        return obs, state

    def step(self, action):
        time_step = self.dm_env.step(action)
        qpos = self.dm_env.physics.data.qpos.copy()
        qvel = self.dm_env.physics.data.qvel.copy()
        state = self._encode_state(qpos, qvel)
        obs = self._make_obs(state)
        # reward = time_step.reward
        reward = time_step.reward if time_step.reward is not None else 0.0
        done = time_step.last()
        info = {}
        info["state"] = state
        return obs, reward, done, info

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
        return obses, states, infos
