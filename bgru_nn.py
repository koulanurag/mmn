# -*- coding: utf-8 -*-
# Moore Machine Network Training

import logging, copy, random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tools import plot_data
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _train(net, env, optimizer, batch_data, batch_size, cuda=False, grad_clip=0.5, trunc_k=10):
    mse_loss_fn = nn.MSELoss().cuda() if cuda else nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss().cuda() if cuda else nn.CrossEntropyLoss()

    data_obs, data_actions, data_actions_probs, data_len = batch_data
    _max, _min = max(data_len), min(data_len)

    data_obs = Variable(torch.Tensor(data_obs))
    data_actions_probs = Variable(torch.FloatTensor(data_actions_probs))
    data_actions = Variable(torch.LongTensor(data_actions))
    hx = Variable(net.init_hidden(batch_size))
    if cuda:
        data_obs, data_actions_probs, hx = data_obs.cuda(), data_actions_probs.cuda(), hx.cuda()
        data_actions = data_actions.cuda()

    mse_loss, ce_loss = 0, 0
    mse_loss_data, ce_loss_data = [], []
    for i in range(_max):
        critic, actor, hx = net((data_obs[:, i, :], hx))
        prob = F.softmax(actor, dim=1)
        if i < _min:
            mse_loss += mse_loss_fn(prob, data_actions_probs[:, i])
            ce_loss += ce_loss_fn(actor, data_actions[:, i])
        else:
            for act_i, act in enumerate(prob):
                if data_len[act_i] > i:
                    mse_loss += mse_loss_fn(act.unsqueeze(0), data_actions_probs[act_i, i].unsqueeze(0))
                    ce_loss += ce_loss_fn(actor[act_i].unsqueeze(0), data_actions[act_i, i].unsqueeze(0))

        # Truncated BP
        if ((trunc_k is not None) and ((i + 1) % trunc_k == 0)) or (i == (_max - 1)):
            optimizer.zero_grad()
            mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()
            hx = Variable(hx.data)
            mse_loss_data.append(float(mse_loss.item()))
            ce_loss_data.append(float(ce_loss.item()))
            mse_loss, ce_loss = 0, 0

    return net, round(sum(mse_loss_data) / batch_size, 5), round(sum(ce_loss_data) / batch_size, 5)


def train(net, env, optimizer, model_path, plot_dir, train_data, batch_size, epochs, cuda=False, test_episodes=300,
          trunc_k=10):
    """Supervised Learning to train the policy"""
    batch_seeds = list(train_data.keys())
    test_env = env
    test_episodes = test_episodes
    test_seeds = [random.randint(1000000, 10000000) for _ in range(test_episodes)]

    best_i = None
    batch_loss_data = {'actor_mse': [], 'actor_ce': []}
    epoch_losses = {'actor_mse': [], 'actor_ce': []}
    perf_data = []

    logger.info('Padding Sequences ...')
    for batch_i, batch_seed in enumerate(batch_seeds):
        data_obs, data_actions, data_action_probs, data_len = train_data[batch_seed]
        _max, _min = max(data_len), min(data_len)
        obs_shape = data_obs[0][0].shape
        act_shape = np.array(data_actions[0][0]).shape
        act_prob_shape = np.array(data_action_probs[0][0]).shape
        if _max != _min:
            for i in range(len(data_obs)):
                data_obs[i] += [np.zeros(obs_shape)] * (_max - data_len[i])
                data_actions[i] += [np.zeros(act_shape)] * (_max - data_len[i])
                data_action_probs[i] += [np.zeros(act_prob_shape)] * (_max - data_len[i])

    for epoch in range(epochs):
        # Testing before training as sometimes the combined model doesn't needs to be trained
        test_perf = test(net, env, test_episodes, test_seeds=test_seeds, cuda=cuda, log=False, render=True)
        # test_perf = 10
        perf_data.append(test_perf)
        logger.info('epoch %d Test Performance: %f' % (epoch, test_perf))
        if best_i is None or perf_data[best_i] <= perf_data[-1]:
            torch.save(net.state_dict(), model_path)
            logger.info('Binary GRU Model Saved!')
            best_i = len(perf_data) - 1 if best_i is None or perf_data[best_i] < perf_data[-1] else best_i

        # _reward_threshold_check = (env.spec.reward_threshold is not None and len(perf_data) > 1
        #                            and np.average(perf_data[-20:]) == env.spec.reward_threshold)
        _reward_threshold_check = perf_data[-1] >= env.spec.reward_threshold
        _epoch_loss_check = (len(epoch_losses['actor_mse']) > 0) and (epoch_losses['actor_mse'][-1] == 0)

        # if _reward_threshold_check or _epoch_loss_check:
        if _reward_threshold_check or _epoch_loss_check:
            logger.info('Optimal Performance achieved!!!')
            logger.info('Exiting!')
            break

        net.train()
        batch_losses = {'actor_mse': [], 'actor_ce': []}
        random.shuffle(batch_seeds)
        for batch_i, batch_seed in enumerate(batch_seeds):
            net, actor_mse_loss, actor_ce_loss = _train(net, env, optimizer, train_data[batch_seed], batch_size,
                                                        cuda=cuda, trunc_k=trunc_k)
            batch_losses['actor_mse'].append(actor_mse_loss)
            batch_losses['actor_ce'].append(actor_ce_loss)
            logger.info('epoch: {} batch: {} actor mse loss: {} actor ce loss: {}'.format(epoch, batch_i,
                                                                                          actor_mse_loss,
                                                                                          actor_ce_loss))
        batch_loss_data['actor_mse'] += batch_losses['actor_mse']
        batch_loss_data['actor_ce'] += batch_losses['actor_ce']
        epoch_losses['actor_mse'].append(np.average(batch_losses['actor_mse']))
        epoch_losses['actor_ce'].append(np.average(batch_losses['actor_ce']))
        plot_data(verbose_data_dict(perf_data, epoch_losses, batch_loss_data), plot_dir)

        if np.isnan(batch_loss_data['actor_mse'][-1]) or np.isnan(batch_loss_data['actor_ce'][-1]):
            logger.info('Actor Loss: Nan')
            break
        if (len(perf_data) - 1 - best_i) > 50:
            logger.info('Early Stopping!')
            break

    plot_data(verbose_data_dict(perf_data, epoch_losses, batch_loss_data), plot_dir)
    net.load_state_dict(torch.load(model_path))
    return net


def test(net, env, total_episodes, test_seeds=None, cuda=False, log=False, render=False, max_actions=10000):
    net.eval()
    total_reward = 0
    with torch.no_grad():
        for ep in range(total_episodes):
            # _seed = test_seeds[ep] if test_seeds is not None else (ep + 10000)
            # env.seed(_seed)
            obs = env.reset()
            done, ep_reward, ep_actions = False, 0, []
            hx = Variable(net.init_hidden())
            all_obs = [obs]
            action_count = 0
            while not done:
                if render:
                    env.render()
                obs = Variable(torch.Tensor(obs)).unsqueeze(0)
                if cuda:
                    obs, hx = obs.cuda(), hx.cuda()
                critic, logit, hx = net((obs, hx))
                prob = F.softmax(logit, dim=1)
                action = int(prob.max(1)[1].data.cpu().numpy())
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
                    all_obs.append(obs)
            total_reward += ep_reward
            # if ep_reward ==0:
            #     print('')
            if log:
                logger.info('Episode =>{} Score=> {} Actions=> {} ActionCount=> {}'.format(ep, ep_reward, ep_actions,
                                                                                           action_count))
        return total_reward / total_episodes


def verbose_data_dict(perf_data, epoch_losses, batch_losses):
    data_dict = []
    if epoch_losses is not None and len(epoch_losses['actor_mse']) > 0:
        data_dict.append({'title': "Actor_MSE_Loss_vs_Epoch", 'data': epoch_losses['actor_mse'],
                          'y_label': 'Loss' + '( min: ' + str(min(epoch_losses['actor_mse'])) + ' )',
                          'x_label': 'Epoch'})
    if epoch_losses is not None and len(epoch_losses['actor_ce']) > 0:
        data_dict.append({'title': "Actor_CE_Loss_vs_Epoch", 'data': epoch_losses['actor_ce'],
                          'y_label': 'Loss' + '( min: ' + str(min(epoch_losses['actor_ce'])) + ' )',
                          'x_label': 'Epoch'})
    if batch_losses is not None and len(batch_losses['actor_mse']) > 0:
        data_dict.append({'title': "Actor_MSE_Loss_vs_Batches", 'data': batch_losses['actor_mse'],
                          'y_label': 'Loss' + '( min: ' + str(min(batch_losses['actor_mse'])) + ' )',
                          'x_label': 'Batch'})
    if batch_losses is not None and len(batch_losses['actor_ce']) > 0:
        data_dict.append({'title': "Actor_CE_Loss_vs_Batches", 'data': batch_losses['actor_ce'],
                          'y_label': 'Loss' + '( min: ' + str(min(batch_losses['actor_ce'])) + ' )',
                          'x_label': 'Batch'})
    if perf_data is not None and len(perf_data) > 0:
        data_dict.append({'title': "Test_Performance_vs_Epoch", 'data': perf_data,
                          'y_label': 'Average Episode Reward' + '( max: ' + str(max(perf_data)) + ' )',
                          'x_label': 'Epoch'})
    return data_dict
