import json
import numpy as np
from os.path import join
import pdb
import imageio
import os

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.env_ours.venv import SubprocVectorEnv
from diffuser.env_ours.utils import seed
import gym
import random
from einops import rearrange
import torch


class Parser(utils.Parser):
    dataset: str = 'pusht'
    config: str = 'config.pusht'
    goal_source: str = 'dset'  # "random_state", "dset", "fix_goal" # TODO: change here
    n_evals: int = 50

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

goal_source = args.goal_source
n_evals = args.n_evals
s = 99
frameskip= 1
goal_H = args.horizon
seed(s)

def make_env_and_datasets_ours(dataset_name):
    # load yaml config from conf/env/dataset_name.py
    import yaml
    import hydra
    from omegaconf import OmegaConf
    with open(f"diffuser/conf/env/{dataset_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    env_cfg = OmegaConf.create(cfg)
    
    if env_cfg.name == "wall" or env_cfg.name == "deformable_env" or "point_maze" in env_cfg.name:
        from diffuser.env_ours.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv(
            [
                gym.make(
                    f"{env_cfg.name}-v0", *env_cfg.args, **env_cfg.kwargs
                )
                for _ in range(n_evals)
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    f"{env_cfg.name}-v0", *env_cfg.args, **env_cfg.kwargs
                )
                for _ in range(n_evals)
            ]
        )

    dsets, orig_dset = hydra.utils.call(env_cfg.dataset)
    return env, dsets, orig_dset

env, dsets, orig_dset = make_env_and_datasets_ours(args.dataset)
dset = orig_dset['valid']
eval_seed = [s * n + 1 for n in range(n_evals)]

def prepare_targets():
    states = []
    actions = []
    observations = []
    
    if goal_source == "random_state" or goal_source == "fix_goal":
        # update env config from val trajs
        observations, states, actions, env_info = (
            sample_traj_segment_from_dset(traj_len=2)
        )
        env.update_env(env_info)

        # sample random states
        fix_goal = goal_source == "fix_goal"
        rand_init_state, rand_goal_state = env.sample_random_init_goal_states(
            eval_seed, fix_goal=fix_goal
        )
        if args.dataset == "deformable_env": # take rand init state from dset for deformable envs
            rand_init_state = np.array([x[0] for x in states])

        obs_0, state_0 = env.prepare(eval_seed, rand_init_state)
        obs_g, state_g = env.prepare(eval_seed, rand_goal_state)

        # add dim for t
        for k in obs_0.keys():
            obs_0[k] = np.expand_dims(obs_0[k], axis=1)
            obs_g[k] = np.expand_dims(obs_g[k], axis=1)

        obs_0 = obs_0
        obs_g = obs_g
        state_0 = rand_init_state  # (b, d)
        state_g = rand_goal_state
        gt_actions = None
        return obs_0, obs_g, state_0, state_g, gt_actions
    else:
        # update env config from val trajs
        observations, states, actions, env_info = (
            sample_traj_segment_from_dset(traj_len=frameskip * goal_H + 1)
        )
        env.update_env(env_info)

        # get states from val trajs
        init_state = [x[0] for x in states]
        init_state = np.array(init_state)
        actions = torch.stack(actions)
        if goal_source == "random_action":
            actions = torch.randn_like(actions)
        wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=frameskip)
        # exec_actions = self.data_preprocessor.denormalize_actions(actions)
        exec_actions = actions # actions not normalized in dataloader
        # replay actions in env to get gt obses
        rollout_obses, rollout_states, infos = env.rollout(
            eval_seed, init_state, exec_actions.numpy()
        )
        obs_0 = {
            key: np.expand_dims(arr[:, 0], axis=1)
            for key, arr in rollout_obses.items()
        }
        obs_g = {
            key: np.expand_dims(arr[:, -1], axis=1)
            for key, arr in rollout_obses.items()
        }
        state_0 = init_state  # (b, d)
        state_g = rollout_states[:, -1]  # (b, d)
        gt_actions = wm_actions
        return obs_0, obs_g, state_0, state_g, gt_actions

def sample_traj_segment_from_dset(traj_len):
    states = []
    actions = []
    observations = []
    env_info = []

    # Check if any trajectory is long enough
    valid_traj = [
        i
        for i in range(len(dset))
        if dset.get_seq_length(i) >= traj_len
    ]
    if len(valid_traj) == 0:
        raise ValueError("No trajectory in the dataset is long enough.")

    # sample init_states from dset
    for i in range(n_evals):
        max_offset = -1
        while max_offset < 0:  # filter out traj that are not long enough
            traj_id = random.randint(0, len(dset) - 1)
            obs, act, state, e_info = dset[traj_id]
            max_offset = obs["visual"].shape[0] - traj_len
        state = state.numpy()
        offset = random.randint(0, max_offset)
        print(f"traj {traj_id}  offset {offset} ")
        obs = {
            key: arr[offset : offset + traj_len]
            for key, arr in obs.items()
        }
        state = state[offset : offset + traj_len]
        act = act[offset : offset + traj_len - 1]
        actions.append(act)
        states.append(state)
        observations.append(obs)
        env_info.append(e_info)
    return observations, states, actions, env_info

obs_0, obs_g, state_0, state_g, gt_actions = prepare_targets()

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

#---------------------------------- main loop ----------------------------------#

exec_actions = []
for i in range(n_evals):
    cond = {
        0: obs_0['visual'][i, 0],
        diffusion.horizon - 1: obs_g['visual'][i, 0],
    }
    action, samples = policy(cond, batch_size=1)
    actions = samples.actions[0]
    sequence = samples.observations[0]

    exec_actions.append(actions)

    fullpath = join(args.savepath, f'{i}.png')
    renderer.composite(fullpath, samples.observations, ncol=1)

exec_actions = np.stack(exec_actions, axis=0)
env.prepare(eval_seed, state_0)
env.set_task_goal(state_g)
e_obses, e_states, infos = env.rollout(eval_seed, state_0, exec_actions)


for i in range(n_evals):
    plan_path = join(args.savepath, f'{i}.png')
    rollout_path = join(args.savepath, f'{i}_rollout.png')

    rollout = e_obses['visual'][i:i+1]
    renderer.composite(rollout_path, rollout, ncol=1)

    plan_img = imageio.imread(plan_path)
    rollout_img = imageio.imread(rollout_path)
    if plan_img.shape[0] != rollout_img.shape[0]:
        # pad shorter image to match height
        h = max(plan_img.shape[0], rollout_img.shape[0])
        def pad_h(img, h):
            pad = np.zeros((h - img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
            return np.concatenate([img, pad], axis=0)
        plan_img = pad_h(plan_img, h)
        rollout_img = pad_h(rollout_img, h)
    combined = np.concatenate([plan_img, rollout_img], axis=1)
    imageio.imsave(join(args.savepath, f'{i}_combined.png'), combined)
    os.remove(plan_path)
    os.remove(rollout_path)

    # write rollout video: put rollout and goal side-by-side (resulting frames have shape H x 2*W x C)
    roll = e_obses['rgb_array'][i]            # (T, H, W, C)
    goal = obs_g['rgb_array'][i]              # (1, H, W, C)
    goal_rep = np.repeat(goal, roll.shape[0], axis=0)  # (T, H, W, C)
    frames = np.concatenate([roll, goal_rep], axis=2).astype(np.uint8)  # concat width -> (T, H, 2*W, C)
    imageio.mimwrite(join(args.savepath, f'{i}_rollout.mp4'), frames, fps=30)

e_final_state = e_states[:, -1, :]
eval_results = env.eval_state(state_g, e_final_state)
successes = eval_results['success']

logs = {
    f"success_rate" if key == "success" else f"mean_{key}": np.mean(value) if key != "success" else np.mean(value.astype(float))
    for key, value in eval_results.items()
}

if args.dataset == 'pusht':
    logs["avg_max_coverage"] = np.mean(infos['max_coverage'][:, -1])
    logs["avg_final_coverage"] = np.mean(infos['final_coverage'][:, -1])

# save logs
with open(join(args.savepath, 'eval_logs.json'), 'w') as f:
    json.dump(logs, f, indent=4, default=float)
