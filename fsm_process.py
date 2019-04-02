"""
Generating data, training and testing functions are implemented here.
"""

import os
import qbn
import time
import copy
import torch
import pickle
import gru_nn
import bgru_nn
import logging
import tools as tl
from torch import optim
from moore_machine import MooreMachine


class ProcessFSM():
    def __init__(self, env):
        self.env = env

    def generate_train_data(self, no_batches, batch_size, trajectories_data_path, generate_train_data, gru_dir):
        tl.set_log(gru_dir, 'generate_train_data')
        train_data = tl.generate_trajectories(self.env, no_batches, batch_size, trajectories_data_path)
        return train_data

    def train_gru(self, gru_net, gru_net_path, gru_plot_dir, train_data, batch_size, train_epochs, cuda, bn_episodes, bottleneck_data_path, generate_max_steps, gru_prob_data_path, gru_dir):
        logging.info('Training GRU!')
        start_time = time.time()
        gru_net.train()
        optimizer = optim.Adam(gru_net.parameters(), lr=1e-3)
        gru_net = gru_nn.train(gru_net, self.env, optimizer, gru_net_path, gru_plot_dir, train_data, batch_size,
                               train_epochs, cuda, trunc_k=50)
        logging.info('Generating Data-Set for Later Bottle Neck Training')
        gru_net.eval()
        tl.generate_bottleneck_data(gru_net, self.env, bn_episodes, bottleneck_data_path, cuda=cuda, max_steps=generate_max_steps)
        tl.generate_trajectories(self.env, 500, batch_size, gru_prob_data_path, gru_net.cpu())
        tl.write_net_readme(gru_net, gru_dir, info={'time_taken': time.time() - start_time})

        return gru_net

    def test_gru(self, trained_gru, gru_net_path, cuda):
        logging.info('Testing GRU!')
        trained_gru.load_state_dict(torch.load(gru_net_path))
        trained_gru.eval()
        trained_gru.noise = False
        no_episodes = 20
        perf = gru_nn.test(trained_gru, self.env, no_episodes, log=True, cuda=cuda, render=True)
        logging.info('Average Performance:{}'.format(perf))
        return perf

    def bhx_train(self, bhx_net, hx_train_data, hx_test_data, bhx_net_path, bhx_plot_dir, batch_size, train_epochs, cuda, target_net, bhx_dir):
        bhx_start_time = time.time()
        logging.info('Training HX SandGlassNet!')
        optimizer = optim.Adam(bhx_net.parameters(), lr=1e-4, weight_decay=0)
        bhx_net.train()
        bhx_net = qbn.train(bhx_net, (hx_train_data, hx_test_data), optimizer, bhx_net_path, bhx_plot_dir,
                            batch_size, train_epochs, cuda, grad_clip=5, target_net=target_net, env=self.env,
                            low=-0.02, high=0.02)
        bhx_end_time = time.time()
        tl.write_net_readme(bhx_net, bhx_dir, info={'time_taken': round(bhx_end_time - bhx_start_time, 4)})

    def bhx_test(self, bhx_net, bhx_net_path, hx_test_data, cuda):
        logging.info('Testing HX SandGlassNet')
        bhx_net.load_state_dict(torch.load(bhx_net_path))
        bhx_net.eval()
        bhx_test_mse = qbn.test(bhx_net, hx_test_data, len(hx_test_data), cuda=cuda)
        logging.info('MSE :{}'.format(bhx_test_mse))

    def ox_train(self, ox_net, obs_train_data, obs_test_data, ox_net_path, ox_plot_dir, batch_size, train_epochs, cuda, target_net, ox_dir):
        ox_start_time = time.time()
        logging.info('Training OX SandGlassNet!')
        optimizer = optim.Adam(ox_net.parameters(), lr=1e-4, weight_decay=0)
        ox_net.train()
        ox_net = qbn.train(ox_net, (obs_train_data, obs_test_data), optimizer, ox_net_path, ox_plot_dir,
                           batch_size, train_epochs, cuda, grad_clip=5, target_net=target_net, env=self.env,
                           low=-0.02, high=0.02)
        ox_end_time = time.time()
        tl.write_net_readme(ox_net, ox_dir, info={'time_taken': round(ox_end_time - ox_start_time, 4)})

    def ox_test(self, ox_net, ox_net_path, obs_test_data, cuda):
        logging.info('Testing  OX SandGlassNet')
        ox_net.load_state_dict(torch.load(ox_net_path))
        ox_net.eval()
        ox_test_mse = qbn.test(ox_net, obs_test_data, len(obs_test_data), cuda=cuda)
        logging.info('MSE : {}'.format(ox_test_mse))

    def bgru_train(self, bgru_net, gru_net, cuda, gru_scratch, trajectories_data_path, bgru_net_path, bgru_plot_dir, batch_size, train_epochs, gru_prob_data_path, bgru_dir):
        self.env.spec.reward_threshold = gru_nn.test(gru_net, self.env, 10, log=True, cuda=cuda, render=True)
        logging.info('Training Binary GRUNet!')
        bgru_net.train()
        _start_time = time.time()
        if gru_scratch:
            optimizer = optim.Adam(bgru_net.parameters(), lr=1e-3)
            train_data = tl.generate_trajectories(self.env, 3, 5, trajectories_data_path)
            bgru_net = gru_nn.train(bgru_net, self.env, optimizer, bgru_net_path, bgru_plot_dir, train_data, batch_size,
                                    train_epochs, cuda)
        else:
            optimizer = optim.Adam(bgru_net.parameters(), lr=1e-4)
            train_data = tl.generate_trajectories(self.env, 3, 5, gru_prob_data_path, copy.deepcopy(bgru_net.gru_net).cpu())
            bgru_net = bgru_nn.train(bgru_net, self.env, optimizer, bgru_net_path, bgru_plot_dir, train_data, 5,
                                     train_epochs, cuda, test_episodes=1, trunc_k=100)
        tl.write_net_readme(bgru_net, bgru_dir, info={'time_taken': round(time.time() - _start_time, 4)})

    def bgru_test(self, bgru_net, bgru_net_path, cuda):
        bgru_net.load_state_dict(torch.load(bgru_net_path))
        bgru_net.eval()
        bgru_perf = bgru_nn.test(bgru_net, self.env, 1, log=True, cuda=cuda, render=True)
        logging.info('Average Performance: {}'.format(bgru_perf))

    def generate_fsm(self, bgru_net, bgru_net_path, cuda, unmin_moore_machine_path, bgru_dir, min_moore_machine_path):
        bgru_net.load_state_dict(torch.load(bgru_net_path))
        bgru_net.eval()
        moore_machine = MooreMachine()
        moore_machine.extract_from_nn(self.env, bgru_net, 10, 0, log=True, partial=True, cuda=cuda)
        pickle.dump(moore_machine, open(unmin_moore_machine_path, 'wb'))
        moore_machine.save(open(os.path.join(bgru_dir, 'fsm.txt'), 'w'))

        moore_machine.minimize_partial_fsm(bgru_net)
        moore_machine.save(open(os.path.join(bgru_dir, 'minimized_moore_machine.txt'), 'w'))
        pickle.dump(moore_machine, open(min_moore_machine_path, 'wb'))

    def evaluate_fsm(self, bgru_net, bgru_net_path, min_moore_machine_path):
        bgru_net.load_state_dict(torch.load(bgru_net_path))
        moore_machine = pickle.load(open(min_moore_machine_path, 'rb'))
        bgru_net.cpu()
        bgru_net.eval()
        perf = moore_machine.evaluate(bgru_net, self.env, total_episodes=3, render=True, inspect=False)
        logging.info('Moore Machine Performance: {}'.format(perf))
