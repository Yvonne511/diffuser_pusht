import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
from diffuser.env_ours.venv import SubprocVectorEnv
from diffuser.env_ours.utils import seed
import gym
import random
from einops import rearrange
import torch
import os
from tqdm import tqdm

import imageio

class Parser(utils.Parser):
    dataset: str = 'pusht'
    config: str = 'config.pusht'
    goal_source: str = 'dset'  # "random_state", "dset", "fix_goal"
    n_evals: int = 50
    replan: bool = False
    max_steps: int = 300

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

goal_source = args.goal_source
n_evals = args.n_evals
s = 99
frameskip= 1
goal_H = args.horizon
seed(s)

# args.savepath = args.savepath + "_replan_NO_aug"
args.savepath = args.savepath + "_replan"
if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)

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
        envs = SerialVectorEnv(
            [
                gym.make(
                    f"{env_cfg.name}-v0", *env_cfg.args, **env_cfg.kwargs
                )
                for _ in range(n_evals)
            ]
        )
    else:
        envs = SubprocVectorEnv(
            [
                lambda: gym.make(
                    f"{env_cfg.name}-v0", *env_cfg.args, **env_cfg.kwargs
                )
                for _ in range(n_evals)
            ]
        )

    wrapped_env = gym.make(f"{env_cfg.name}-v0", *env_cfg.args, **env_cfg.kwargs)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = dataset_name

    dsets, orig_dset = hydra.utils.call(env_cfg.dataset)
    return env, envs, dsets, orig_dset

env, envs, dsets, orig_dset = make_env_and_datasets_ours(args.dataset)
dset = orig_dset['valid']
eval_seed = [s * n + 1 for n in range(n_evals)]
# max_steps = min(args.max_steps, getattr(env, "max_episode_steps", args.max_steps))
max_steps = 1000
# max_steps = 5

def prepare_targets():
    states = []
    actions = []
    observations = []
    
    if goal_source == "random_state" or goal_source == "fix_goal":
        # update env config from val trajs
        observations, states, actions, env_info = (
            sample_traj_segment_from_dset(traj_len=2)
        )
        envs.update_env(env_info)

        # sample random states
        fix_goal = goal_source == "fix_goal"
        rand_init_state, rand_goal_state = envs.sample_random_init_goal_states(
            eval_seed, fix_goal=fix_goal
        )
        if args.dataset == "deformable_env": # take rand init state from dset for deformable envs
            rand_init_state = np.array([x[0] for x in states])

        obs_0, state_0 = envs.prepare(eval_seed, rand_init_state)
        obs_g, state_g = envs.prepare(eval_seed, rand_goal_state)

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
        envs.update_env(env_info)

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
        rollout_obses, rollout_states, infos = envs.rollout(
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

def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def plan_from_observation(observation, goal_observation):
    cond = {
        0: observation,
        diffusion.horizon - 1: goal_observation,
    }
    _, samples = policy(cond, batch_size=1)
    return samples.actions[0], samples

if args.replan:
    final_successes = []
    optimal_successes = []
    final_state_dists = []
    optimal_state_dists = []
    final_coverages = []
    optimal_coverages = []

    for i in range(n_evals):
        env.prepare(eval_seed[i], state_0[i])
        env.set_task_goal(state_g[i])

        goal_observation = obs_g['visual'][i, 0]
        observation = obs_0['visual'][i, 0]

        planned_actions, samples = plan_from_observation(observation, goal_observation)
        renderer.composite(join(args.savepath, f'{i}.png'), samples.observations, ncol=1)

        rollout_visuals = [to_numpy(observation)]
        successes = []
        state_dists = []
        coverages = []
        visuals = []
        cur_goal = obs_g['rgb_array'][i, 0]

        for t in tqdm(range(max_steps), desc="Env Steps"):
            if t > 0:
                planned_actions, _ = plan_from_observation(observation, goal_observation)

            action = planned_actions[0]
            next_obs, _, done, info = env.step(action)

            observation = next_obs['visual']
            visual = np.concatenate([next_obs['rgb_array'], cur_goal], axis=1)
            visuals.append(visual)
            rollout_visuals.append(to_numpy(observation))

            cur_state = info['state'] if 'state' in info else observation
            cur_state = to_numpy(cur_state)
            eval_result = env.eval_state(state_g[i], cur_state)
            successes.append(bool(eval_result['success']))
            state_dists.append(float(eval_result['state_dist']))

            if args.dataset == 'pusht' and 'final_coverage' in info:
                coverages.append(float(info['final_coverage']))

            if done:
                break
        
        frames = np.stack(visuals)
        print("### num frames", frames.shape)
        imageio.mimwrite(join(args.savepath, f'{i}_rollout_success_{np.any(successes)}.mp4'), frames, fps=30)

        rollout = np.stack(rollout_visuals, axis=0)[None]
        renderer.composite(join(args.savepath, f'{i}_rollout.png'), rollout, ncol=1)

        final_successes.append(successes[-1])
        optimal_successes.append(np.any(successes))
        final_state_dists.append(state_dists[-1])
        optimal_state_dists.append(np.min(state_dists))

        if args.dataset == 'pusht' and coverages:
            final_coverages.append(coverages[-1])
            optimal_coverages.append(np.max(coverages))

    logs = {
        "success_rate": np.mean(final_successes),
        "mean_state_dist": np.mean(final_state_dists),
        "optimal_success_rate": np.mean(optimal_successes),
        "optimal_state_dist": np.mean(optimal_state_dists),
    }

    if args.dataset == 'pusht':
        logs["avg_max_coverage"] = np.mean(optimal_coverages) if optimal_coverages else 0.0
        logs["avg_final_coverage"] = np.mean(final_coverages) if final_coverages else 0.0
else:
    exec_actions = []
    for i in range(n_evals):
        actions, samples = plan_from_observation(
            obs_0['visual'][i, 0], obs_g['visual'][i, 0]
        )
        exec_actions.append(actions)

        fullpath = join(args.savepath, f'{i}.png')
        renderer.composite(fullpath, samples.observations, ncol=1)

    exec_actions = np.stack(exec_actions, axis=0)
    envs.prepare(eval_seed, state_0)
    envs.set_task_goal(state_g)
    e_obses, e_states, infos = envs.rollout(eval_seed, state_0, exec_actions)

    for i in range(n_evals):
        rollout = e_obses['visual'][i:i+1]
        renderer.composite(join(args.savepath, f'{i}_rollout.png'), rollout, ncol=1)

    e_final_state = e_states[:, -1, :]
    eval_results = envs.eval_state(state_g, e_final_state)

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
