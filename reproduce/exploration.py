import argparse
import importlib

import matplotlib.pyplot as plt
import torch
import wandb

import os

from simpl.alg.spirl import ConstrainedSAC, PriorResidualNormalMLPPolicy
from simpl.collector import Buffer, LowFixedHierarchicalTimeLimitCollector
from simpl.nn import itemize
from simpl.alg.simpl import ConditionedPolicy, ConditionedQF





if __name__ == '__main__':

    import_pathes = {
        'maze_20t': 'maze.exploration_20t',
        'maze_40t': 'maze.exploration_40t',
        'kitchen': 'kitchen.exploration',
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('domain', choices=import_pathes.keys())
    parser.add_argument('-g', '--gpu', required=True, type=int)
    parser.add_argument('-w', '--worker-gpus', required=True, type=int, nargs='+')
    parser.add_argument('-s', '--spirl-pretrained-path', required=True)

    parser.add_argument('-p', '--wandb-project-name')
    parser.add_argument('-r', '--wandb-run-name')
    parser.add_argument('-a', '--save_replaybuffer_file_path', required=True, type=str)
    args = parser.parse_args()

    module = importlib.import_module(import_pathes[args.domain])
    env, train_tasks, config, visualize_env = module.env, module.train_tasks, module.config, module.visualize_env

    gpu = args.gpu
    worker_gpus = args.worker_gpus
    spirl_pretrained_path = args.spirl_pretrained_path
    policy_vis_period = args.policy_vis_period or 10
    wandb_project_name = args.wandb_project_name or 'SiMPL'
    wandb_run_name = args.wandb_run_name or args.domain + '.simpl_meta_train.' + wandb.util.generate_id()

    # args.save_replaybuffer_file_path/wandb_run_name/replaybuffer
    save_replaybuffer_filepath = str(args.save_replaybuffer_file_path) + '/' + str(wandb_run_name)
    os.makedirs(save_replaybuffer_filepath)
    os.makedirs(save_replaybuffer_filepath + '/replaybuffer')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # load pre-trained SPiRL
    load = torch.load(spirl_pretrained_path, map_location='cpu')
    horizon = load['horizon']
    high_action_dim = load['z_dim']
    spirl_low_policy = load['spirl_low_policy'].eval().requires_grad_(False)
    spirl_prior_policy = load['spirl_prior_policy'].eval().requires_grad_(False)

    # collector
    spirl_low_policy.explore = False
    collector = LowFixedHierarchicalTimeLimitCollector(
        env, spirl_low_policy, horizon=horizon, time_limit=config['time_limit']
    )

    conc_collector = ConcurrentCollector([
        LowFixedGPUWorker(collector, gpu)
        for gpu in worker_gpus
    ])

    # ready networks
    encoder = SetTransformerEncoder(state_dim, high_action_dim, config['e_dim'], **config['encoder'])

    high_policy = ContextPriorResidualNormalMLPPolicy(
        spirl_prior_policy, state_dim, high_action_dim, config['e_dim'],
        **config['policy']
    )

    qfs = [MLPQF(state_dim + config['e_dim'], high_action_dim, **config['qf']) for _ in range(config['n_qf'])]

    buffers = [
        Buffer(state_dim, high_action_dim, config['buffer_size'])
        for _ in range(len(train_tasks))
    ]

    trainer = Simpl(high_policy, spirl_prior_policy, qfs, encoder, enc_buffers, buffers, **config['simpl']).to(gpu)

    # exploration on all training tasks
    wandb.init(
        project=wandb_project_name, name=wandb_run_name,
        config={**config, 'gpu': gpu, 'spirl_pretrained_path': args.spirl_pretrained_path}
    )

    # exploration process

