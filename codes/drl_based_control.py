# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 14:50:07 2023

@author: decar
"""

import argparse
import datetime
import os
import pprint

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

from tianshou.env import DummyVectorEnv, SubprocVectorEnv

from PFALEnv import PFALEnv

# create policy
def create_policy(hidden_sizes=[64,64], gamma=0.99, actor_lr=3E-4, 
                  critic_lr=3E-4, alpha_lr=3E-4, auto_alpha=True, 
                  device="cuda" if torch.cuda.is_available() else "cpu"):
    env = PFALEnv()
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    max_action = env.action_space.high[0]
    net_a = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = ActorProb(
        net_a,
        action_shape,
        max_action=max_action,
        device=device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    net_c1 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    net_c2 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)
    
    if auto_alpha:
        alpha_lr = alpha_lr
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = 0.5


    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        # tau=0.01,
        gamma=gamma,
        alpha=alpha,
        # reward_normalization=True,
        estimation_step=1,
        action_space=env.action_space,
    )
    
    return policy

# train policy
def train_policy(policy, experiment, epoch, nstep_per_episode=3025, batch_size=1024*2, train_num=1, test_num=20, buffer_size=1000000):
    
    train_envs = DummyVectorEnv(
        [lambda: PFALEnv() for _ in range(train_num)]
    )

    test_envs = DummyVectorEnv(
        [lambda: PFALEnv() for _ in range(test_num)]
    )
    
    buffer = ReplayBuffer(buffer_size)
    
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    train_collector.collect(n_episode=20, random=True)
    
    algo_name = "sac"
    log_name = os.path.join("PFAL-v1", algo_name, str(experiment))
    log_path = os.path.join("log", log_name)
    
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)
    
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))
        
    result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            epoch,
            nstep_per_episode,
            1,
            test_num,
            batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=1,
            test_in_train=False,
        )
    pprint.pprint(result)
        
    policy.eval()
    test_envs.seed(0)
    test_collector.reset()
    result = test_collector.collect(n_episode=test_num)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')
    
    return result["rews"].mean(), result["rews"].std()

# load policy
def load_policy(policy, experiment):
    
    # path
    algo_name = "sac"
    log_name = os.path.join("PFAL-v1", algo_name, str(experiment))
    log_path = os.path.join("log", log_name)
    
    # load from existing policy
    print(f"Loading agent under {log_path}")
    policy_path = os.path.join(log_path, "policy.pth")
    if os.path.exists(policy_path):
        policy.load_state_dict(torch.load(policy_path))
        print("Successfully loaded policy.")
    else:
        print("Fail to load policy.")
    
    return policy

# load policy
def load_policy_cpu(policy, experiment):
    
    # path
    algo_name = "sac"
    log_name = os.path.join("PFAL-v1", algo_name, str(experiment))
    log_path = os.path.join("log", log_name)
    
    # load from existing policy
    print(f"Loading agent under {log_path}")
    policy_path = os.path.join(log_path, "policy.pth")
    if os.path.exists(policy_path):
        policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))
        print("Successfully loaded policy.")
    else:
        print("Fail to load policy.")
    
    return policy
