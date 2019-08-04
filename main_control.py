# -*- coding: utf-8 -*-
"""
Script for training Moore Machine Network(MMNet) on Gold Rush
"""

import os
import sys
import qbn
import time
import copy
import torch
import pickle
import gru_nn
import bgru_nn
import logging
import traceback
import gym, gym_x
import fsm_process
import tools as tl
import torch.nn as nn
from torch import optim
from functions import TernaryTanh
from torch.autograd import Variable
from moore_machine import MooreMachine
import torch.nn.functional as F

class ObsQBNet(nn.Module):
    """
    Quantized Bottleneck Network(QBN) for observation features
    """
    def __init__(self, input_size, x_features):
        super(ObsQBNet, self).__init__()
        self.bhx_size = x_features

        f1 = int(8 * x_features)
        self.encoder = nn.Sequential(nn.Linear(input_size, f1),
                                     nn.Tanh(),
                                     nn.Linear(f1, x_features),
                                     TernaryTanh())

        self.decoder = nn.Sequential(nn.Linear(x_features, f1),
                                     nn.Tanh(),
                                     nn.Linear(f1, input_size),
                                     nn.ReLU6())


    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded, encoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class HxQBNet(nn.Module):
    """
    Quantized Bottleneck Network(QBN) for hidden states of GRU
    """

    def __init__(self, input_size, x_features):
        super(HxQBNet, self).__init__()
        self.bhx_size = x_features
        f1 = int(8 * x_features)
        self.encoder = nn.Sequential(nn.Linear(input_size, f1),
                                     nn.Tanh(),
                                     nn.Linear(f1, x_features),
                                     TernaryTanh())

        self.decoder = nn.Sequential(nn.Linear(x_features, f1),
                                     nn.Tanh(),
                                     nn.Linear(f1, input_size),
                                     nn.Tanh())

    def forward(self, x):
        x = self.encode(x)
        return self.decode(x), x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class GRUNet(nn.Module):
    """
    Gated Recurrent Unit Network(GRUNet) definition
    """
    def __init__(self, input_size, gru_cells, total_actions):
        super(GRUNet, self).__init__()
        self.gru_units = gru_cells
        self.noise = False

        self.input_ff = nn.Sequential(nn.Linear(input_size, 16),
                                      nn.ELU(),
                                      nn.Linear(16, 8),
                                      nn.ReLU6())
        self.input_flat_size = 8
        self.input_c_features = self.input_flat_size
        self.gru = nn.GRUCell(self.input_flat_size, 32)

        self.critic_linear = nn.Linear(gru_cells, 1)
        self.actor_linear = nn.Linear(gru_cells, total_actions)

        self.apply(tl.weights_init)
        self.actor_linear.weight.data = tl.normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = tl.normalized_columns_initializer(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.gru.bias_ih.data.fill_(0)
        self.gru.bias_hh.data.fill_(0)

    def forward(self, input, input_fn=None, hx_fn=None, inspect=False):
        input, hx = input
        c_input = self.input_ff(input)
        c_input = c_input.view(-1, self.input_flat_size)
        input, input_x = input_fn(c_input) if input_fn is not None else (c_input, c_input)
        ghx = self.gru(input, hx)

        # Keep the noise during both training as well as evaluation
        # c_input = gaussian(c_input, self.training, mean=0, std=0.05, one_sided=True)
        # c_input = tl.uniform(c_input, self.noise, low=-0.01, high=0.01, enforce_pos=True)
        # ghx = tl.uniform(ghx, self.noise, low=-0.01, high=0.01)

        hx, bhx = hx_fn(ghx) if hx_fn is not None else (ghx, ghx)

        if inspect:
            return self.critic_linear(hx), self.actor_linear(hx), hx, (ghx, bhx, c_input, input_x)
        else:
            return self.critic_linear(hx), self.actor_linear(hx), hx

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.gru_units)

    def get_action_linear(self, state):
        return self.actor_linear(state)

    def transact(self, o_x, hx):
        hx = self.gru(o_x, hx)
        return hx


class MMNet(nn.Module):
    """
    Moore Machine Network(MMNet) definition
    """
    def __init__(self, net, hx_qbn=None, obs_qbn=None):
        super(MMNet, self).__init__()
        self.bhx_units = hx_qbn.bhx_size if hx_qbn is not None else None
        self.gru_units = net.gru_units
        self.obx_net = obs_qbn
        self.gru_net = net
        self.bhx_net = hx_qbn
        self.actor_linear = self.gru_net.get_action_linear

    def init_hidden(self, batch_size=1):
        return self.gru_net.init_hidden(batch_size)

    def forward(self, x, inspect=False):
        x, hx = x
        critic, actor, hx, (ghx, bhx, input_c, input_x) = self.gru_net((x, hx), input_fn=self.obx_net,
                                                                       hx_fn=self.bhx_net, inspect=True)
        if inspect:
            return critic, actor, hx, (ghx, bhx), (input_c, input_x)
        else:
            return critic, actor, hx

    def get_action_linear(self, state, decode=False):
        if decode:
            hx = self.bhx_net.decode(state)
        else:
            hx = state
        return self.actor_linear(hx)

    def transact(self, o_x, hx_x):
        hx_x = self.gru_net.transact(self.obx_net.decode(o_x), self.bhx_net.decode(hx_x))
        _, hx_x = self.bhx_net(hx_x)
        return hx_x

    def state_encode(self, state):
        return self.bhx_net.encode(state)

    def obs_encode(self, obs, hx=None):
        if hx is None:
            hx = Variable(torch.zeros(1, self.gru_units))
            if next(self.parameters()).is_cuda:
                hx = hx.cuda()
        _, _, _, (_, _, _, input_x) = self.gru_net((obs, hx), input_fn=self.obx_net, hx_fn=self.bhx_net, inspect=True)
        return input_x


if __name__ == '__main__':
    args = tl.get_args()
    env = gym.make(args.env)
    env.seed(args.env_seed)
    obs = env.reset()

    # create directories to store results
    result_dir = tl.ensure_directory_exits(os.path.join(args.result_dir, 'Classic_Control'))
    env_dir = tl.ensure_directory_exits(os.path.join(result_dir, args.env))

    gru_dir = tl.ensure_directory_exits(os.path.join(env_dir, 'gru_{}'.format(args.gru_size)))
    gru_net_path = os.path.join(gru_dir, 'model.p')
    gru_plot_dir = tl.ensure_directory_exits(os.path.join(gru_dir, 'Plots'))

    bhx_dir = tl.ensure_directory_exits(
        os.path.join(env_dir, 'gru_{}_bhx_{}{}'.format(args.gru_size, args.bhx_size, args.bhx_suffix)))
    bhx_net_path = os.path.join(bhx_dir, 'model.p')
    bhx_plot_dir = tl.ensure_directory_exits(os.path.join(bhx_dir, 'Plots'))

    ox_dir = tl.ensure_directory_exits(
        os.path.join(env_dir, 'gru_{}_ox_{}{}'.format(args.gru_size, args.ox_size, args.bhx_suffix)))
    ox_net_path = os.path.join(ox_dir, 'model.p')
    ox_plot_dir = tl.ensure_directory_exits(os.path.join(ox_dir, 'Plots'))

    data_dir = tl.ensure_directory_exits(os.path.join(env_dir, 'data'))
    bottleneck_data_path = os.path.join(data_dir, 'bottleneck_data.p')
    trajectories_data_path = os.path.join(data_dir, 'trajectories_data.p')
    gru_prob_data_path = os.path.join(data_dir, 'gru_prob_data.p')

    try:
        fsm_object = fsm_process.ProcessFSM(env)
        # ***********************************************************************************
        # Generating training data                                                          *
        # ***********************************************************************************
        no_batches = 10000
        if args.generate_train_data:
            train_data = fsm_object.generate_train_data(no_batches, args.batch_size, trajectories_data_path, args.generate_train_data, gru_dir)
        # ***********************************************************************************
        # GRU Network                                                                       *
        # ***********************************************************************************
        if args.gru_train or args.gru_test:
            tl.set_log(gru_dir, 'train' if args.gru_train else 'test')
            gru_net = GRUNet(len(obs), args.gru_size, int(env.action_space.n))

            if args.cuda:
                gru_net = gru_net.cuda()
            if args.gru_train:
                logging.info(['No Training Performed!!'])
                logging.warning('We assume that we already have a pre-trained model @ {}'.format(gru_net_path))
                tl.write_net_readme(gru_net, gru_dir, info={})
            if args.gru_test:
                test_performance = fsm_object.test_gru(gru_net, gru_net_path, args.cuda)
        # ***********************************************************************************
        # Generating BottleNeck training data                                               *
        # ***********************************************************************************
        if args.generate_bn_data:
            tl.set_log(data_dir, 'generate_bn_data')
            logging.info('Generating Data-Set for Later Bottle Neck Training')
            gru_net = GRUNet(len(obs), args.gru_size, int(env.action_space.n))
            gru_net.load_state_dict(torch.load(gru_net_path))
            gru_net.noise = False
            if args.cuda:
                gru_net = gru_net.cuda()
            gru_net.eval()
            tl.generate_bottleneck_data(gru_net, env, args.bn_episodes, bottleneck_data_path, cuda=args.cuda, eps=(0, 0.3), max_steps=args.generate_max_steps)
            tl.generate_trajectories(env, 3, 5, gru_prob_data_path, gru_net, cuda=args.cuda, render=True)

        # ***********************************************************************************
        # HX-QBN                                                                            *
        # ***********************************************************************************
        if args.bhx_train or args.bhx_test:
            tl.set_log(bhx_dir, 'train' if args.bhx_train else 'test')
            gru_net = GRUNet(len(obs), args.gru_size, int(env.action_space.n))
            gru_net.eval()
            bhx_net = HxQBNet(args.gru_size, args.bhx_size)
            if args.cuda:
                gru_net = gru_net.cuda()
                bhx_net = bhx_net.cuda()

            if not os.path.exists(gru_net_path):
                logging.info('Pre-Trained GRU model not found!')
                sys.exit(0)
            else:
                gru_net.load_state_dict(torch.load(gru_net_path))
            gru_net.noise = False
            env.spec.reward_threshold = gru_nn.test(gru_net, env, 5, log=True, cuda=args.cuda, render=False)
            logging.info('Reward Threshold:' + str(env.spec.reward_threshold))
            target_net = lambda bottle_net: MMNet(gru_net, hx_qbn=bottle_net)

            logging.info('Loading Data-Set')
            hx_train_data, hx_test_data, _, _ = tl.generate_bottleneck_data(gru_net, env, args.bn_episodes, bottleneck_data_path, cuda=args.cuda, max_steps=args.generate_max_steps)
            if args.bhx_train:
                fsm_object.bhx_train(bhx_net, hx_train_data, hx_test_data, bhx_net_path, bhx_plot_dir, args.batch_size, args.train_epochs, args.cuda, target_net, bhx_dir)
            if args.bhx_test:
                fsm_object.bhx_test(bhx_net, bhx_net_path, hx_test_data, args.cuda)

        # ***********************************************************************************
        # Obs-QBN                                                                           *
        # ***********************************************************************************
        if args.ox_test or args.ox_train:
            tl.set_log(ox_dir, 'train' if args.ox_train else 'test')
            gru_net = GRUNet(len(obs), args.gru_size, int(env.action_space.n))
            ox_net = ObsQBNet(gru_net.input_flat_size, args.ox_size)
            if args.cuda:
                gru_net = gru_net.cuda()
                ox_net = ox_net.cuda()

            if not os.path.exists(gru_net_path):
                logging.warning('Pre-Trained GRU model not found!')
                sys.exit(0)
            else:
                gru_net.load_state_dict(torch.load(gru_net_path))
            gru_net.noise = False
            env.spec.reward_threshold = gru_nn.test(gru_net, env, 5, log=True, cuda=args.cuda, render=False)
            logging.info('Reward Threshold:' + str(env.spec.reward_threshold))
            target_net = lambda bottle_net: MMNet(gru_net, obs_qbn=bottle_net)
            logging.info('Loading Data-Set ...')
            _, _, obs_train_data, obs_test_data = tl.generate_bottleneck_data(gru_net, env, args.bn_episodes, bottleneck_data_path, cuda=args.cuda)
            if args.ox_train:
                fsm_object.ox_train(ox_net, obs_train_data, obs_test_data, ox_net_path, ox_plot_dir, args.batch_size, args.train_epochs, args.cuda, target_net, ox_dir)
            if args.ox_test:
                fsm_object.ox_test(ox_net, ox_net_path, obs_test_data, args.cuda)

        # ***********************************************************************************
        # MMN                                                                               *
        # ***********************************************************************************
        if args.bgru_train or args.bgru_test or args.generate_fsm or args.evaluate_fsm:
            gru_net = GRUNet(len(obs), args.gru_size, int(env.action_space.n))
            bhx_net = HxQBNet(args.gru_size, args.bhx_size)
            ox_net = ObsQBNet(gru_net.input_c_features, args.ox_size)
            bgru_net = MMNet(gru_net, bhx_net, ox_net)
            if args.cuda:
                bgru_net = bgru_net.cuda()

            bx_prefix = 'scratch-'
            if not args.bx_scratch:
                if bgru_net.bhx_net is not None:
                    bgru_net.bhx_net.load_state_dict(torch.load(bhx_net_path))
                if bgru_net.obx_net is not None:
                    bgru_net.obx_net.load_state_dict(torch.load(ox_net_path))
                bx_prefix = ''

            gru_prefix = 'scratch-'
            if not args.gru_scratch:
                bgru_net.gru_net.load_state_dict(torch.load(gru_net_path))
                bgru_net.gru_net.noise = False
                gru_prefix = ''

            # create directories to save result
            bgru_dir_name = '{}gru_{}_{}hx_({},{})_bgru'.format(gru_prefix, args.gru_size, bx_prefix, args.bhx_size, args.ox_size)
            bgru_dir = tl.ensure_directory_exits(os.path.join(env_dir, bgru_dir_name))
            bgru_net_path = os.path.join(bgru_dir, 'model.p')
            min_moore_machine_path = os.path.join(bgru_dir, 'min_moore_machine.p')
            unmin_moore_machine_path = os.path.join(bgru_dir, 'unmin_moore_machine.p')
            bgru_plot_dir = tl.ensure_directory_exits(os.path.join(bgru_dir, 'Plots'))

            _log_tag = 'train' if args.bgru_train else ('test' if args.bgru_test else 'generate_fsm')
            _log_tag = _log_tag if not args.evaluate_fsm else 'evaluate_fsm'
            tl.set_log(bgru_dir, _log_tag)

            if args.bgru_train:
                fsm_object.bgru_train(bgru_net, gru_net, args.cuda, args.gru_scratch, trajectories_data_path, bgru_net_path, bgru_plot_dir, args.batch_size, args.train_epochs, gru_prob_data_path, bgru_dir)
            if args.bgru_test:
                fsm_object.bgru_test(bgru_net, bgru_net_path, args.cuda)
            if args.generate_fsm:
                fsm_object.generate_fsm(bgru_net, bgru_net_path, args.cuda, unmin_moore_machine_path, bgru_dir, min_moore_machine_path)
            if args.evaluate_fsm:
                fsm_object.evaluate_fsm(bgru_net, bgru_net_path, min_moore_machine_path)
        env.close()
    except Exception as ex:
        logging.error(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))