import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
from policy_engine.ddpg import LayerNorm,Actor,Critic
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import os
import logging

logger = logging.getLogger(__name__)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


"""
From: https://github.com/pytorch/pytorch/issues/1959
There's an official LayerNorm implementation in pytorch now, but it hasn't been included in 
pip version yet. This is a temporary version
This slows down training by a bit
"""
nn.LayerNorm = LayerNorm

class DivDDPGActor(object):
    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space, lr_critic, lr_actor):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_inputs = num_inputs
        self.action_space = action_space
        self.actor = Actor(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.actor_diverse = Actor(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.actor_optim = SGD(self.actor.parameters(), lr=lr_actor, momentum=0.9)
        self.critic = Critic(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space).to(self.device)
        self.critic_optim = SGD(self.critic.parameters(), lr=lr_critic, momentum=0.9)
        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    # This is where the behavioural policy is called
    def select_action(self, state, tensor_board_writer, step_number, action_noise=None, previous_action=None, param_noise=None):
        self.actor_target.eval()
        mu = self.actor((Variable(state)))
        self.actor_target.train()
        mu=mu.data()
        return mu.clamp(-1, 1)

    # This function samples from target policy for test
    def select_action_from_target_actor(self, state):
        self.actor_target.eval()
        mu = self.actor_target((Variable(state).to(self.device)))
        self.actor_target.train()
        mu = mu.data
        return mu

    def update_parameters(self, batch, tensor_board_writer, episode_number):
        state_batch = Variable(torch.cat(batch.state)).to(self.device)
        action_batch = Variable(torch.cat(batch.action)).to(self.device)
        reward_batch = Variable(torch.cat(batch.reward)).to(self.device)
        mask_batch = Variable(torch.cat(batch.mask)).to(self.device)
        next_state_batch = Variable(torch.cat(batch.next_state)).to(self.device)
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)
        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)

        # We may need to change the following line for speed up (cuda to cpu operation)
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)

        # updating critic network
        self.critic_optim.zero_grad()
        state_action_batch = self.critic((state_batch), (action_batch))
        value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        clip_grad_norm_(self.critic.parameters(), 0.5)
        tensor_board_writer.add_scalar('norm_grad_critic', self.calculate_norm_grad(self.critic), episode_number)
        self.critic_optim.step()

        # updating actor network
        self.actor_optim.zero_grad()
        policy_loss = -self.critic((state_batch), self.actor((state_batch)))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optim.step()

        # updating target policy networks with soft update
        soft_update(self.actor_target, self.actor, self.tau)
        norm_grad_actor_net = self.calculate_norm_grad(self.actor)
        tensor_board_writer.add_scalar('norm_grad_actor', norm_grad_actor_net, episode_number)
        soft_update(self.critic_target, self.critic, self.tau)
        return value_loss.item(), policy_loss.item()

    def calculate_norm_grad(self, net):
        S = 0
        for p in list(filter(lambda p: p.grad is not None, net.parameters())):
            S += p.grad.data.norm(2).item() ** 2
        return np.sqrt(S)

    def set_poly_rl_alg(self, poly_rl_alg):
        self.poly_rl_alg = poly_rl_alg

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/ddpg_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))