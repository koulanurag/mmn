# -*- coding: utf-8 -*-
# Moore Machine Network Training (Binarized encoding)

import logging, copy, random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tools import plot_data
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _train(net, env, optimizer, batch_data, batch_size, cuda=False, grad_clip=0.5):
    mse_loss = nn.MSELoss().cuda() if cuda else nn.MSELoss()
    # mse_loss = nn.SmoothL1Loss().cuda() if cuda else nn.SmoothL1Loss()
    data_obs, data_actions_probs = batch_data
    data_obs = Variable(torch.Tensor(data_obs))
    data_actions_probs = Variable(torch.FloatTensor(data_actions_probs))
    hx = net.initHidden(batch_size)
    if cuda:
        data_obs, data_actions_probs, hx = data_obs.cuda(), data_actions_probs.cuda(), hx.cuda()

    optimizer.zero_grad()
    loss = 0
    for i in range(env._max_episode_steps):
        output, hx = net((data_obs[:, i, :], hx))
        prob = F.softmax(output, dim=1)
        loss += mse_loss(prob, data_actions_probs[:, i])
    loss = loss / batch_size
    loss.backward()
    torch.nn.utils.clip_grad_value_(net.parameters(), grad_clip)
    optimizer.step()

    return net, float(loss.item())


def train(net, env, optimizer, model_path, plot_dir, train_data, batch_size, epochs, cuda=False):
    """Supervised Learning to train the policy"""
    batch_seeds = list(train_data.keys())
    test_env = copy.deepcopy(env)
    test_episodes = 100
    test_seeds = [random.randint(1000000, 10000000) for _ in range(test_episodes)]

    best_i = None
    batch_loss_data = {'actor': []}
    epoch_losses = {'actor': []}
    perf_data = []
    for epoch in range(epochs):

        # Testing before training as sometimes the combined model doesn't needs to be trained
        test_perf = test(net, test_env, test_episodes, test_seeds=test_seeds, cuda=cuda, log=False)
        perf_data.append(test_perf)
        logger.info('epoch %d Test Performance: %f' % (epoch, test_perf))
        if best_i is None or perf_data[best_i] < perf_data[-1]:
            best_i = len(perf_data) - 1
            torch.save(net.state_dict(), model_path)
            logger.info('Binary GRU Model Saved!')
        if env.spec.reward_threshold is not None and np.average(perf_data[-10:]) >= env.spec.reward_threshold:
            logger.info('Optimal Performance achieved!!!')
            break

        net.train()
        batch_losses = {'actor': []}
        random.shuffle(batch_seeds)
        for batch_i, batch_seed in enumerate(batch_seeds):
            net, actor_loss = _train(net, env, optimizer, train_data[batch_seed], batch_size, cuda=cuda)
            batch_losses['actor'].append(actor_loss)
            logger.info('epoch: %d batch: %d actor loss: %f' % (epoch, batch_i, actor_loss))
        batch_loss_data['actor'] += batch_losses['actor']
        epoch_losses['actor'].append(np.average(batch_losses['actor']))
        plot_data(verbose_data_dict(perf_data, epoch_losses, batch_loss_data), plot_dir)

        if (len(perf_data) - 1 - best_i) > 100 or np.isnan(batch_loss_data['actor'][-1]):
            logger.info('Early Stopping!')
            break

    plot_data(verbose_data_dict(perf_data, epoch_losses, batch_loss_data), plot_dir)
    net.load_state_dict(torch.load(model_path))
    return net


def test(net, env, total_episodes, test_seeds=None, cuda=False, log=False):
    net.eval()
    total_reward = 0
    with torch.no_grad():
        for ep in range(total_episodes):
            # _seed = test_seeds[ep] if test_seeds is not None else (ep + 10000)
            # env.seed(_seed)
            obs = env.reset()
            done, ep_reward, ep_actions = False, 0, []
            hx = net.initHidden()
            while not done:
                obs = Variable(torch.Tensor(obs)).unsqueeze(0)
                if cuda:
                    obs, hx = obs.cuda(), hx.cuda()
                logit, hx = net((obs, hx))
                prob = F.softmax(logit, dim=1)
                action = int(prob.max(1)[1].data.cpu().numpy())
                obs, reward, done, _ = env.step(action)
                ep_actions.append(action)
                ep_reward += reward
            total_reward += ep_reward
            if log:
                logger.info('Episode =>{} Score=> {} Actions=> {}'.format(ep, ep_reward, ep_actions))
        return total_reward / total_episodes


def verbose_data_dict(perf_data, epoch_losses, batch_losses):
    data_dict = []
    if epoch_losses is not None and len(epoch_losses['actor']) > 0:
        data_dict.append({'title': "Actor_Loss_vs_Epoch", 'data': epoch_losses['actor'],
                          'y_label': 'Loss' + '( min: ' + str(min(epoch_losses['actor'])) + ' )', 'x_label': 'Epoch'})
    if batch_losses is not None and len(batch_losses['actor']) > 0:
        data_dict.append({'title': "Actor_Loss_vs_Batches", 'data': batch_losses['actor'],
                          'y_label': 'Loss' + '( min: ' + str(min(batch_losses['actor'])) + ' )', 'x_label': 'Batch'})
    if perf_data is not None and len(perf_data) > 0:
        data_dict.append({'title': "Test_Performance_vs_Epoch", 'data': perf_data,
                          'y_label': 'Average Episode Reward' + '( max: ' + str(max(perf_data)) + ' )',
                          'x_label': 'Epoch'})
    return data_dict
