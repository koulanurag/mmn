# -*- coding: utf-8 -*-
# Binary Bottle-Neck Network (BBN) Training  Module

import logging
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tools import plot_data

logger = logging.getLogger(__name__)


def train(net, data, optimizer, model_path, plot_dir, batch_size, epochs, cuda=False):
    mse_loss = nn.MSELoss().cuda() if cuda else nn.MSELoss()
    # mse_loss = nn.SmoothL1Loss().cuda() if cuda else nn.SmoothL1Loss()
    train_data, test_data = data

    best_i = None
    batch_loss_data, epoch_losses, test_losses = [], [], []
    total_batches = int(len(train_data) / batch_size)
    if len(train_data) % batch_size != 0:
        total_batches += 1

    for epoch in range(epochs):
        net.train()
        batch_losses = []
        random.shuffle(train_data)
        for b_i in range(total_batches):
            batch_input = train_data[b_i:b_i + batch_size]
            batch_input = Variable(torch.FloatTensor(batch_input), requires_grad=True)
            batch_target = Variable(torch.FloatTensor(batch_input))
            if cuda:
                batch_input, batch_target = batch_input.cuda(), batch_target.cuda()
            batch_ouput, _ = net(batch_input)

            optimizer.zero_grad()
            loss = mse_loss(batch_ouput, batch_target)
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()

            logger.info('epoch: %d batch: %d loss: %f' % (epoch, b_i, loss.item()))

        batch_loss_data += batch_losses
        epoch_losses.append(round(np.average(batch_losses), 4))
        test_losses.append(round(test(net, test_data, len(test_data), cuda=cuda), 4))
        plot_data(verbose_data_dict(test_losses, epoch_losses, batch_loss_data), plot_dir)

        if best_i is None or test_losses[best_i] > test_losses[-1]:
            best_i = len(test_losses) - 1
            torch.save(net.state_dict(), model_path)
            logger.info('Bottle Net Model Saved!')
        if (len(test_losses) - 1 - best_i) > 20 or np.isnan(batch_losses[-1]):
            logger.info('Early Stopping!')
            break

    net.load_state_dict(torch.load(model_path))
    return net


def test(net, data, batch_size, cuda=False):
    mse_loss = nn.MSELoss().cuda() if cuda else nn.MSELoss()
    # mse_loss = nn.SmoothL1Loss().cuda() if cuda else nn.SmoothL1Loss()
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


def verbose_data_dict(test_loss, epoch_losses, batch_losses):
    data_dict = [
        {'title': "Test_Loss_vs_Epoch", 'data': test_loss, 'y_label': 'Loss(' + str(min(test_loss)) + ')',
         'x_label': 'Epoch'},
        {'title': "Loss_vs_Epoch", 'data': epoch_losses, 'y_label': 'Loss(' + str(min(epoch_losses)) + ')',
         'x_label': 'Epoch'},
        {'title': "Loss_vs_Batches", 'data': batch_losses, 'y_label': 'Loss(' + str(min(batch_losses)) + ')',
         'x_label': 'Batch'}
    ]
    return data_dict
