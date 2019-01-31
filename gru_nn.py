# -*- coding: utf-8 -*-
# Continuous Recurrent Network (CRN) training  : GRU

import logging, copy, random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tools import plot_data
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _train(net, optimizer, batch_data, batch_size, cuda=False, grad_clip=5, trunc_k=None):
    cross_entropy_loss = nn.CrossEntropyLoss().cuda() if cuda else nn.CrossEntropyLoss()
    data_obs, data_actions, _, data_len = batch_data
    _max, _min = max(data_len), min(data_len)

    data_obs = Variable(torch.Tensor(data_obs))
    data_actions = Variable(torch.LongTensor(data_actions))
    hx = Variable(net.init_hidden(batch_size))
    if cuda:
        data_obs, data_actions, hx = data_obs.cuda(), data_actions.cuda(), hx.cuda()

    loss = 0
    loss_data = []
    for i in range(_max):
        critic, actor, hx = net((data_obs[:, i, :], hx))
        if i < _min:
            loss += cross_entropy_loss(actor, data_actions[:, i])
        else:
            for act_i, act in enumerate(actor):
                if data_len[act_i] > i:
                    loss += cross_entropy_loss(act.unsqueeze(0), data_actions[act_i, i].unsqueeze(0))

        # Truncated BP
        if ((trunc_k is not None) and ((i + 1) % trunc_k == 0)) or (i == (_max - 1)):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()
            hx = Variable(hx.data)
            loss_data.append(float(loss.item()))
            loss = 0

    return net, round(sum(loss_data) / batch_size, 4)


def train(net, env, optimizer, model_path, plot_dir, train_data, batch_size, epochs, cuda=False, grad_clip=5,
          trunc_k=10,ep_check=True , rw_check=True):
    """Supervised Learning to train the policy"""
    batch_seeds = list(train_data.keys())
    test_env = copy.deepcopy(env)
    test_episodes = 300
    test_seeds = [random.randint(1000000, 10000000) for _ in range(test_episodes)]

    best_i = None
    batch_loss_data = {'actor': []}
    epoch_losses = {'actor': []}
    perf_data = []

    logger.info('Padding Sequences ...')
    for batch_i, batch_seed in enumerate(batch_seeds):
        data_obs, data_actions, _, data_len = train_data[batch_seed]
        _max, _min = max(data_len), min(data_len)
        _shape = data_obs[0][0].shape
        for i in range(len(data_obs)):
            data_obs[i] += [np.zeros(_shape)] * (_max - data_len[i])
            data_actions[i] += [-1] * (_max - data_len[i])

    for epoch in range(epochs):
        net.train()
        batch_losses = {'actor': []}
        random.shuffle(batch_seeds)
        for batch_i, batch_seed in enumerate(batch_seeds):
            net, actor_loss = _train(net, optimizer, train_data[batch_seed], batch_size, cuda, grad_clip, trunc_k)
            batch_losses['actor'].append(actor_loss)
            logger.info('epoch: {} batch: {} actor loss: {}'.format(epoch, batch_i, actor_loss))

        test_perf = test(net, test_env, test_episodes, test_seeds=test_seeds, cuda=cuda)
        batch_loss_data['actor'] += batch_losses['actor']
        epoch_losses['actor'].append(np.average(batch_losses['actor']))

        perf_data.append(test_perf)
        logger.info('epoch %d Test Performance: %f' % (epoch, test_perf))
        plot_data(verbose_data_dict(perf_data, epoch_losses, batch_loss_data), plot_dir)

        if best_i is None or perf_data[best_i] <= perf_data[-1]:
            torch.save(net.state_dict(), model_path)
            logger.info('GRU Model Saved!')
            best_i = len(perf_data) - 1 if best_i is None or perf_data[best_i] < perf_data[-1] else best_i

        if np.isnan(batch_loss_data['actor'][-1]):
            logger.info('Batch Loss : Nan')
            break
        if (len(perf_data) - 1 - best_i) > 100:
            logger.info('Early Stopping!')
            break

        _reward_threshold_check = ((env.spec.reward_threshold is not None) and len(perf_data) > 1) \
                                  and (np.average(perf_data[-10:]) == env.spec.reward_threshold)
        _epoch_loss_check = (len(epoch_losses['actor']) > 0) and (epoch_losses['actor'][-1] == 0)

        # We need to ensure complete imitation rather than just performance . Many a times, optimal
        # performance could be achieved without complete imitation of the actor
        if _epoch_loss_check and ep_check:
            logger.info('Complete Imitation of the Agent!!!')
            break
        if _reward_threshold_check and rw_check:
            logger.info('Consistent optimal performance achieved!!!')
            break

    net.load_state_dict(torch.load(model_path))
    return net


def test(net, env, total_episodes, test_seeds=None, cuda=False, log=False, render=False, max_actions=5000):
    net.eval()
    total_reward = 0
    with torch.no_grad():
        for ep in range(total_episodes):
            # _seed = test_seeds[ep] if test_seeds is not None else (ep + 10000)
            # env.seed(_seed)
            obs = env.reset()
            done = False
            ep_reward = 0
            ep_actions = []
            hx = Variable(net.init_hidden())
            all_observations = [obs]
            action_count = 0
            while not done:
                if render:
                    env.render()
                obs = Variable(torch.Tensor(obs)).unsqueeze(0)
                if cuda:
                    obs, hx = obs.cuda(), hx.cuda()
                critic, logit, hx = net((obs, hx))
                prob = F.softmax(logit, dim=1)
                #action = int(prob.multinomial(num_samples=1).data.cpu().numpy())
                action = int(prob.max(1)[1].data.cpu().numpy())
                # action = 0
                obs, reward, done, _ = env.step(action)
                action_count += 1
                done = done if action_count <= max_actions else True
                ep_actions.append(action)
                # a quick hack to prevent the agent from stucking
                max_same_action = 5000
                if action_count > max_same_action:
                    actions_to_consider = ep_actions[-max_same_action:]
                    if actions_to_consider.count(actions_to_consider[0]) == max_same_action:
                        done = True
                ep_reward += reward
                if not done:
                    all_observations.append(obs)
            total_reward += ep_reward
            if log:
                logger.info('Episode =>{} Score=> {} Actions=> {} ActionCount=> {}'.format(ep, ep_reward, ep_actions,
                                                                                           action_count))
                # logger.info(''.join([str(x) for x in all_observations]))
        return total_reward / total_episodes


def verbose_data_dict(perf_data, epoch_losses, batch_losses):
    data_dict = []
    if epoch_losses is not None and len(epoch_losses) > 0:
        data_dict.append({'title': "Actor_Loss_vs_Epoch", 'data': epoch_losses['actor'],
                          'y_label': 'Loss' + '( min: ' + str(min(epoch_losses['actor'])) + ' )', 'x_label': 'Epoch'})
    if batch_losses is not None and len(batch_losses) > 0:
        data_dict.append({'title': "Actor_Loss_vs_Batches", 'data': batch_losses['actor'],
                          'y_label': 'Loss' + '( min: ' + str(min(batch_losses['actor'])) + ' )', 'x_label': 'Batch'})
    if perf_data is not None and len(perf_data) > 0:
        data_dict.append({'title': "Test_Performance_vs_Epoch", 'data': perf_data,
                          'y_label': 'Average Episode Reward' + '( max: ' + str(max(perf_data)) + ' )',
                          'x_label': 'Epoch'})
    return data_dict
