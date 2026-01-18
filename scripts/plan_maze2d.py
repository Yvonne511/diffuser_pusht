import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

env = datasets.load_environment(args.dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

#---------------------------------- main loop ----------------------------------#

n_evals = 50
s = 99

import random
import torch
random.seed(s)
np.random.seed(s)
torch.manual_seed(s)
torch.cuda.manual_seed_all(s)

for n in range(n_evals):
    seed = n * s + 1

    random_state = np.random.RandomState(seed)
    rs = np.random.RandomState(seed)
    WALL=10
    maze_unit=1
    noise = 1
    offset_x = 0.25
    offset_y = 0.25
    STATE_RANGES = np.array([
        [0.39318362, 3.2198412],  # Range for first dimension
        [0.62660956, 3.2187355],  # Range for second dimension
        [-5.2262554, 5.2262554],  # Range for third dimension
        [-5.2262554, 5.2262554],  # Range for fourth dimension
        # [0.90001136, 3.0999563],  # Range for first dimension of target
        # [0.9000267, 3.0999668]    # Range for second dimension of target
    ])
    def sample_random_xy():
        all_cells = [(i, j) 
            for i in range(env.maze_arr.shape[0]) 
            for j in range(env.maze_arr.shape[1]) 
            if env.maze_arr[i, j] != WALL]
        init_ij = all_cells[rs.randint(len(all_cells))]
        init_xy = add_noise(ij_to_xy(init_ij))
        return init_xy

    def ij_to_xy(ij):
        j, i = ij
        x = j * maze_unit - offset_x
        y = i * maze_unit - offset_y
        return x, y

    def add_noise(xy):
        rs = random_state
        random_x = rs.uniform(low=-noise, high=noise) * maze_unit / 4
        random_y = rs.uniform(low=-noise, high=noise) * maze_unit / 4
        return xy[0] + random_x, xy[1] + random_y

    def generate_state():
        x, y = sample_random_xy()
        state = np.array([
            x, 
            y,
            rs.uniform(low=STATE_RANGES[2][0], high=STATE_RANGES[2][1]),
            rs.uniform(low=STATE_RANGES[3][0], high=STATE_RANGES[3][1]),
        ])
        return state

    init_state = generate_state()
    goal_state = generate_state()

    # overwrite set state

    env.reset()

    env.set_state(init_state[:2], init_state[2:])
    observation = np.concatenate([env.sim.data.qpos, env.sim.data.qvel]).ravel()

    if args.conditional:
        print('Resetting target')
        env.set_target(goal_state[:2])

    ## set conditioning xy position to be the goal

    target = env._target
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0
    for t in range(env.max_episode_steps):

        state = env.state_vector().copy()

        ## can replan if desired, but the open-loop plans are good enough for maze2d
        ## that we really only need to plan once
        if t == 0:
            cond[0] = observation

            # action, samples, interm_samples = policy(cond, batch_size=args.batch_size)
            action, samples = policy(cond, batch_size=args.batch_size)
            actions = samples.actions[0]
            sequence = samples.observations[0]
        # pdb.set_trace()

        # ####
        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
            # pdb.set_trace()

        ## can use actions or define a simple controller based on state predictions
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        # pdb.set_trace()
        ####

        # else:
        #     actions = actions[1:]
        #     if len(actions) > 1:
        #         action = actions[0]
        #     else:
        #         # action = np.zeros(2)
        #         action = -state[2:]
        #         pdb.set_trace()



        next_observation, reward, terminal, _ = env.step(action)
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        print(
            f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
            f'{action}'
        )

        if 'maze2d' in args.dataset:
            xy = next_observation[:2]
            goal = env.unwrapped._target
            print(
                f'maze | pos: {xy} | goal: {goal}'
            )

        ## update rollout observations
        rollout.append(next_observation.copy())

        # logger.log(score=score, step=t)

        if t % args.vis_freq == 0 or terminal:
            fullpath = join(args.savepath, f'{seed}_{t}.png')

            # if t == 0: renderer.composite(fullpath, interm_samples.observations, ncol=16)
            if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)


            # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

            ## save rollout thus far
            renderer.composite(join(args.savepath, f'{seed}_rollout.png'), np.array(rollout)[None], ncol=1)

            # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

            # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

        if terminal:
            break

        observation = next_observation

    # logger.finish(t, env.max_episode_steps, score=score, value=0)

    ## save result as a json file
    json_path = join(args.savepath, f'rollout.json')
    json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
        'epoch_diffusion': diffusion_experiment.epoch}
    # json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
    with open(json_path, 'a') as f:
        f.write(json.dumps(json_data) + "\n")
