# -*- coding: utf-8 -*-
"""
Quantized BottleNeck Network(QBN) training and testing
"""

import math
import torch
import random
import logging
import numpy as np
import torch.nn as nn
from tools import plot_data
import torch.nn.functional as F
from torch.autograd import Variable

logger = logging.getLogger(__name__)


def train(net, data, optimizer, model_path, plot_dir, batch_size, epochs, cuda=False, grad_clip=None, target_net=None, env=None, low=0, high=0.05, target_test_episodes=1):
    """
    Train the QBN

    :param net: given network
    :param data: given data to train the network on
    :param optimizer: optimizer method(Adam is preferred)
    :param model_path: path to where save the model
    :param plot_dir: path to where save the plots
    :param batch_size: batch size
    :param epochs: number of training epochs
    :param cuda: check if cuda is available
    :param grad_clip: max norm of the gradients
    :param target_net:
    :param env: environment
    :param low: lower bound of noise data
    :param high: upper bound of noise data
    :param target_test_episodes: number of episodes to test on
    :return: returns the trained model
    """
    mse_loss = nn.MSELoss().cuda() if cuda else nn.MSELoss()
    train_data, test_data = data

    min_loss_i, best_perf_i = None, None
    batch_loss_data, epoch_losses, test_losses, test_perf_data = [], [], [], []
    total_batches = math.ceil(len(train_data) / batch_size)

    for epoch in range(epochs):
        net.train()
        batch_losses = []
        random.shuffle(train_data)
        for b_i in range(total_batches):
            batch_input = train_data[b_i:b_i + batch_size]
            batch_target = Variable(torch.FloatTensor(batch_input))
            batch_input = torch.FloatTensor(batch_input)
            batch_input = Variable(batch_input, requires_grad=True)

            if cuda:
                batch_input, batch_target = batch_input.cuda(), batch_target.cuda()
            batch_ouput, _ = net(batch_input)

            optimizer.zero_grad()
            loss = mse_loss(batch_ouput, batch_target)
            loss.backward()
            batch_losses.append(loss.item())
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()

            logger.info('epoch: %d batch: %d loss: %f' % (epoch, b_i, loss.item()))

        batch_loss_data += batch_losses
        epoch_losses.append(round(np.average(batch_losses), 5))
        test_losses.append(round(test(net, test_data, len(test_data), cuda=cuda), 5))
        test_perf = test_with_env(target_net(net), env, target_test_episodes, cuda=cuda)
        test_perf_data.append(test_perf)

        if (best_perf_i is None) or (test_perf_data[best_perf_i] <= test_perf_data[-1]) or test_perf_data[
            -1] == env.spec.reward_threshold:
            torch.save(net.state_dict(), model_path)
            logger.info('Bottle Net Model Saved!')
        if (best_perf_i is None) or (test_perf_data[best_perf_i] < test_perf_data[-1]):
            best_perf_i = len(test_perf_data) - 1
            logger.info('Best Perf i updated')
        if (min_loss_i is None) or (test_losses[min_loss_i] > test_losses[-1]):
            min_loss_i = len(test_losses) - 1
            logger.info('min_loss_i updated')

        plot_data(verbose_data_dict(test_losses, epoch_losses, batch_loss_data, test_perf_data), plot_dir)
        logger.info('epoch: %d test loss: %f best perf i: %d min loss i: %d' % (epoch, test_losses[-1], best_perf_i,
                                                                                min_loss_i))

        if np.isnan(batch_losses[-1]):
            logger.info('Batch Loss: Nan')
            break
        if ((len(test_losses) - 1 - min_loss_i) > 50) or (test_losses[-1] == 0):
            logger.info('Test Loss hasn\'t improved in last 50 epochs' if test_losses[-1] != 0 else 'Zero Test Loss!!')
            logger.info('Stopping!')
            break

    net.load_state_dict(torch.load(model_path))
    return net


def test(net, data, batch_size, cuda=False):
    """
    Test the trained network.

    :param net: given network
    :param data: given data to test the network on
    :param batch_size: batch size
    :param cuda: check if cuda is available
    :return: test performance
    """
    mse_loss = nn.MSELoss().cuda() if cuda else nn.MSELoss()
    net.eval()
    batch_losses = []
    total_batches = int(len(data) / batch_size)
    if len(data) % batch_size != 0:
        total_batches += 1
    with torch.no_grad():
        for b_i in range(total_batches):
            batch_input = data[b_i:b_i + batch_size]
            batch_input = Variable(torch.FloatTensor(batch_input))
            batch_target = Variable(torch.FloatTensor(batch_input))
            if cuda:
                batch_target, batch_input = batch_target.cuda(), batch_input.cuda()
            batch_ouput, _ = net(batch_input)
            loss = mse_loss(batch_ouput, batch_target)
            batch_losses.append(float(loss.item()))

    return sum(batch_losses) / len(batch_losses)


def test_with_env(net, env, total_episodes, cuda=False, log=False, render=False, max_actions=10000):
    """
    Test on an environment.

    :param net: given network
    :param env: environment
    :param total_episodes: number of episodes of testing
    :param cuda: check if cuda is available
    :param log: check to print out test log
    :param render: check to render environment
    :param max_actions: max number of actions
    :return: test performance
    """
    net.eval()
    total_reward = 0
    with torch.no_grad():
        for ep in range(total_episodes):
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
            if log:
                logger.info('Episode =>{} Score=> {} Actions=> {} ActionCount=> {}'.format(ep, ep_reward, ep_actions,
                                                                                           action_count))
        return total_reward / total_episodes


def verbose_data_dict(test_loss, epoch_losses, batch_losses, test_env_perf):
    """
    Makes data(losses and performance) into a dictionary for the sake of data plotting.

    :param test_loss: test performance
    :param epoch_losses: MSE and CE epoch loss
    :param batch_losses: MSE and CE batch loss
    :param test_env_perf: performance on test environment
    :return: data info dictionary
    """
    data_dict = [
        {'title': "Test_Loss_vs_Epoch", 'data': test_loss, 'y_label': 'Loss(' + str(min(test_loss)) + ')',
         'x_label': 'Epoch'},
        {'title': "Loss_vs_Epoch", 'data': epoch_losses, 'y_label': 'Loss(' + str(min(epoch_losses)) + ')',
         'x_label': 'Epoch'},
        {'title': "Loss_vs_Batches", 'data': batch_losses, 'y_label': 'Loss(' + str(min(batch_losses)) + ')',
         'x_label': 'Batch'}
    ]
    if len(test_env_perf) > 0:
        data_dict.append({'title': "Performance_vs_Epoch_intervals", 'data': test_env_perf,
                          'y_label': 'Score (' + str(max(test_env_perf)) + ')',
                          'x_label': 'Epoch Interval'})
    return data_dict
