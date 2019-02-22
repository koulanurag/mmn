"""
Useful functions that have been used through out the whole code base.
"""

import os
import torch
import pickle
import random
import logging
import argparse
import numpy as np
import matplotlib as mpl
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

# To plot graphs over a server shell since the default display is not available on server.
mpl.use('Agg')
logger = logging.getLogger(__name__)

def ensure_directory_exits(directory_path):
    """
    Creates directory if path doesn't exist.

    :param directory_path: path to directory to save stuff
    :return: existing directory path
    """

    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
    except Exception:
        pass
    return directory_path


def normalized_columns_initializer(weights, std=1.0):
    """
    Normalizing over a matrix.

    :param weights: given matrix
    :param std: standard deviation
    :return: normalized matrix
    """

    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).unsqueeze(0).sum(1).expand_as(out))
    return out


def weights_init(m):
    """
    Weight initialization.
    """

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def plot_data(data_dict, plots_dir_path):
    """
    Plotting given data.

    :param data_dict: dictionary containing data
    :param plots_dir_path: path to save plots
    """

    for x in data_dict:
        title = x['title']
        data = x['data']
        if len(data) == 1:
            plt.scatter([0], data)
        else:
            plt.plot(data)
        plt.grid(True)
        plt.title(title)
        plt.ylabel(x['y_label'])
        plt.xlabel(x['x_label'])
        plt.savefig(os.path.join(plots_dir_path, title + ".png"))
        plt.clf()

    logger.info('Plot Saved! - ' + plots_dir_path)


def write_net_readme(net, dir, info={}):
    """
    Writes the configuration of the network.

    :param net: given network
    :param dir: path to where save the file
    :param info: additional info to save
    """

    with open(os.path.join(dir, 'NET_README.txt'), 'w') as _file:
        _file.write('********Net Information********\n\n')
        _file.write(net.__str__() + '\n\n')
        if len(info.keys()) > 0:
            _file.write('INFO:' + '\n')
            for key in info.keys():
                _file.write(key + ' : ' + str(info[key]) + '\n')


def gaussian(ins, is_training, mean, std, one_sided=False):
    """
    Add Gaussian noise to data.

    :param ins: input data
    :param is_training: check if it's training
    :param mean: mean of the distribution
    :param std: standard deviation
    :return: new data with noise added
    """

    if is_training:
        noise = Variable(ins.data.new(ins.size()).normal_(mean, std))
        return ins + (abs(noise) if one_sided else noise)
    return ins


def uniform(ins, is_training, low, high, enforce_pos=False):
    output = ins
    if is_training:
        noise = Variable(ins.data.new(ins.size()).uniform_(low, high))
        output = ins + noise
        if enforce_pos:
            output[output < 0] = 0
    return output


def set_log(logPath, suffix=''):
    """
    Set logging configuration.

    :param logPath: where to write logs
    :param suffix: any suffix added to the path
    """

    logging.basicConfig(
        format="%(asctime)s [%(levelname)-5.5s] [%(name)s -> %(funcName)s]  %(message)s",
        handlers=[
            logging.FileHandler("{0}/{1}.log".format(logPath, 'logs' + ('-' if len(suffix) > 0 else '') + suffix),
                                mode='w'),
            logging.StreamHandler()
        ],
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.DEBUG)


def generate_bottleneck_data(net, env, episodes, save_path, cuda=False, eps=(0, 0), max_steps=None):
    """
    Generating bottleneck data for the given network.

    :param net: given network
    :param env: given environment
    :param episodes: number of episodes
    :param save_path: path to save data in
    :param cuda: check if cuda is available
    :param max_steps: maximum number of steps to take. used for exploration.
    :return: observation and hidden state bottleneck data
    """
    if os.path.exists(save_path):
        # unpickling after reading the file is efficient
        hx_train_data, hx_test_data, obs_train_data, obs_test_data = pickle.loads(open(save_path, "rb").read())
    else:
        logging.info('No Data Found @ path : {}'.format(save_path))
        logging.info('Generating BottleNeck Data..')
        bottleneck_data = {}
        hx_data, obs_data, action_data = [], [], []
        all_ep_rewards = []
        with torch.no_grad():
            for ep in range(episodes):
                done = False
                obs = env.reset()
                hx = Variable(net.init_hidden())
                ep_reward = 0
                act_count = 0
                exploration_start_step = random.choice(range(0, max_steps, int(0.02 * max_steps)))
                while not done:
                    env.render()
                    obs = Variable(torch.Tensor(obs)).unsqueeze(0)
                    if cuda:
                        hx = hx.cuda()
                        obs = obs.cuda()
                    critic, logit, hx, (_, _, obs_c, _) = net((obs, hx), inspect=True)
                    if exploration_start_step >= act_count and random.random() < eps[ep % len(eps)]:
                        action = env.action_space.sample()
                    # if random.random() < eps[ep % len(eps)]:
                    #     action = env.action_space.sample()
                    else:
                        prob = F.softmax(logit, dim=1)
                        action = int(prob.max(1)[1].data.cpu().numpy())
                    obs, reward, done, info = env.step(action)
                    action_data.append(action)
                    act_count += 1
                    done = done if act_count <= max_steps else True
                    if action not in bottleneck_data:
                        bottleneck_data[action] = {'hx_data': [], 'obs_data': []}
                    bottleneck_data[action]['hx_data'].append(hx.data.cpu().numpy()[0].tolist())
                    bottleneck_data[action]['obs_data'].append(obs_c.data.cpu().numpy()[0].tolist())

                    ep_reward += reward
                logging.info('episode:{} reward:{}'.format(ep, ep_reward))
                all_ep_rewards.append(ep_reward)
        logging.info('Average Performance:{}'.format(sum(all_ep_rewards) / len(all_ep_rewards)))

        hx_train_data, hx_test_data, obs_train_data, obs_test_data = [], [], [], []
        for action in bottleneck_data.keys():
            hx_train_data += bottleneck_data[action]['hx_data']
            hx_test_data += bottleneck_data[action]['hx_data']
            obs_train_data += bottleneck_data[action]['obs_data']
            obs_test_data += bottleneck_data[action]['obs_data']

            logging.info('Action: {} Hx Data: {} Obs Data: {}'.format(action, len(np.unique(bottleneck_data[action]['hx_data'], axis=0).tolist()), len(np.unique(bottleneck_data[action]['obs_data'], axis=0).tolist())))

        obs_test_data = np.unique(obs_test_data, axis=0).tolist()
        hx_test_data = np.unique(hx_test_data, axis=0).tolist()

        random.shuffle(hx_train_data)
        random.shuffle(obs_train_data)
        random.shuffle(hx_test_data)
        random.shuffle(obs_test_data)

        pickle.dump((hx_train_data, hx_test_data, obs_train_data, obs_test_data), open(save_path, "wb"))

    logging.info('Data Sizes:')
    logging.info('Hx Train:{} Hx Test:{} Obs Train:{} Obs Test:{}'.format(len(hx_train_data), len(hx_test_data), len(obs_train_data), len(obs_test_data)))

    return hx_train_data, hx_test_data, obs_train_data, obs_test_data


def get_args():
    """
    Arguments used to get input from command line.
    :return: given arguments in command line
    """
    parser = argparse.ArgumentParser(description='GRU to FSM')
    parser.add_argument('--generate_train_data', action='store_true', default=False, help='Generate Train Data')
    parser.add_argument('--generate_bn_data', action='store_true', default=False, help='Generate Bottle-Neck Data')
    parser.add_argument('--generate_max_steps', type=int, help='Maximum number of steps to be used for data generation')
    parser.add_argument('--gru_train', action='store_true', default=False, help='Train GRU Network')
    parser.add_argument('--gru_test', action='store_true', default=False, help='Test GRU Network')
    parser.add_argument('--gru_size', type=int, help="No. of GRU Cells")
    parser.add_argument('--gru_lr', type=float, default=0.001, help="No. of GRU Cells")

    parser.add_argument('--bhx_train', action='store_true', default=False, help='Train bx network')
    parser.add_argument('--ox_train', action='store_true', default=False, help='Train ox network')
    parser.add_argument('--bhx_test', action='store_true', default=False, help='Test bx network')
    parser.add_argument('--ox_test', action='store_true', default=False, help='Test ox network')
    parser.add_argument('--bgru_train', action='store_true', default=False, help='Train binary gru network')
    parser.add_argument('--bgru_test', action='store_true', default=False, help='Test binary gru network')
    parser.add_argument('--blstm_train', action='store_true', default=False, help='Train binary lstm network')
    parser.add_argument('--blstm_test', action='store_true', default=False, help='Test binary lstm network')

    parser.add_argument('--bhx_size', type=int, help="binary encoding size")
    parser.add_argument('--bhx_suffix', default='', help="suffix fo bhx folder")
    parser.add_argument('--ox_size', type=int, help="binary encoding size")

    parser.add_argument('--train_epochs', type=int, default=400, help="No. of training episodes")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size used for training")
    parser.add_argument('--bgru_lr', type=float, default=0.0001, help="Learning rate for binary GRU")
    parser.add_argument('--gru_scratch', action='store_true', default=False, help='use scratch gru for BGRU')
    parser.add_argument('--blstm_lr', type=float, default=0.0001, help="Learning rate for binary LSTM")
    parser.add_argument('--lstm_scratch', action='store_true', default=False, help='use scratch lstm for BLSTM')
    parser.add_argument('--bx_scratch', action='store_true', default=False, help='use scratch bx network for BGRU or BLSTM')
    parser.add_argument('--generate_fsm', action='store_true', default=False, help='extract fsm from fmm net')
    parser.add_argument('--evaluate_fsm', action='store_true', default=False, help='evaluate fsm')

    parser.add_argument('--bn_episodes', type=int, default=20,
                        help="No. of episodes for generating data for Bottleneck Network")
    parser.add_argument('--bn_epochs', type=int, default=100, help="No. of Training epochs")

    parser.add_argument('--no_cuda', action='store_true', default=False, help='no cuda usage')
    parser.add_argument('--env', help="Name of the environment")
    parser.add_argument('--env_seed', type=int, default=0, help="Seed for the environment")
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results")
    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    return args


def generate_trajectories(env, batches, batch_size, save_path, guide=None, cuda=False, render=False):
    """
    Generate trajectories used as training data.

    :param env: given environment
    :param batches: number of batches
    :param batch_size: batch size
    :param save_path: path to save generated data
    :param cuda: check if cuda is available
    :param render: check to render environment
    :return: generated trajectory data
    """
    if os.path.exists(save_path):
        logging.info('Loading Saved data .. ')
        # unpickling after reading the file is efficient
        _train_data = pickle.loads(open(save_path, "rb").read())
    else:
        logging.info('Generating data .. ')
        _train_data = {}
        all_ep_rewards = []
        if guide is not None:
            guide.eval()

        with torch.no_grad():
            for seed in range(batches):
                data_obs, data_actions, data_actions_prob, data_len = [], [], [], []
                for ep in range(batch_size):
                    _actions, _action_probs, _obs = [], [], []
                    done = False
                    obs = env.reset()
                    hx = None if guide is None else Variable(guide.init_hidden())
                    if hx is not None and cuda:
                        hx = hx.cuda()
                    ep_reward = 0

                    while not done:
                        if render:
                            env.render()
                        _obs.append(obs)
                        if guide is None:
                            # action = env.action_space.sample()
                            action = env.env.get_desired_action()
                            _actions.append(action)
                        else:
                            obs = Variable(torch.Tensor(obs).unsqueeze(0))
                            if cuda:
                                obs = obs.cuda()
                            critic, logit, hx, (_, _, obs_c, _) = guide((obs, hx), inspect=True)
                            prob = F.softmax(logit, dim=1)
                            action = int(prob.max(1)[1].data.cpu().numpy())
                            _action_probs.append(prob.data.cpu().numpy()[0].tolist())
                            _actions.append(action)
                        obs, reward, done, info = env.step(action)
                        ep_reward += reward

                    data_obs.append(_obs)
                    data_actions.append(_actions)
                    data_actions_prob.append(_action_probs)
                    data_len.append(len(_obs))
                    all_ep_rewards.append(ep_reward)
                    logging.info('Ep:{} Batch: {} Reward:{}'.format(seed, ep, ep_reward))

                _train_data[seed] = (data_obs, data_actions, data_actions_prob, data_len)
        logging.info('Average Performance: {}'.format(sum(all_ep_rewards) / len(all_ep_rewards)))
        pickle.dump(_train_data, open(save_path, "wb"))
    return _train_data
