import os, sys, time, random, copy, argparse, logging, traceback
import gym, gym_x
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from functions import BinaryTanh, TernaryTanh
import gru_nn, bgru_nn
import bottleneck_nn as b_nn
from moore_machine import MooreMachine
from tools import ensure_directory_exits, write_net_readme, weights_init, normalized_columns_initializer
import pickle
import numpy as np


def set_log(logPath, suffix=''):
    logging.basicConfig(
        format="%(asctime)s [%(levelname)-5.5s] [%(name)s -> %(funcName)s]  %(message)s",
        handlers=[
            logging.FileHandler("{0}/{1}.log".format(logPath, 'logs' + ('-' if len(suffix) > 0 else '') + suffix),
                                mode='w'),
            logging.StreamHandler()
        ],
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.DEBUG)


class BNNet(nn.Module):
    """Bottle Neck Network """

    def __init__(self, input_size, x_features):
        super(BNNet, self).__init__()
        self.bhx_size = x_features
        self.bin_encoder = nn.Sequential(nn.Linear(input_size, 3 * x_features),
                                         nn.ELU(),
                                         nn.Linear(3 * x_features, x_features),
                                         BinaryTanh())

        self.bin_decoder = nn.Sequential(nn.Linear(x_features, 3 * x_features),
                                         nn.ELU(),
                                         nn.Linear(3 * x_features, input_size),
                                         nn.Tanh())
        self.apply(weights_init)

    def forward(self, input):
        x = self.encode(input)
        return self.decode(x), x

    def encode(self, input):
        return self.bin_encoder(input)

    def decode(self, input):
        return self.bin_decoder(input)


class GRUNet(nn.Module):
    def __init__(self, input_size, gru_cells, total_actions):
        super(GRUNet, self).__init__()
        self.gru_units = gru_cells
        self.input_c_features = input_size

        self.gru = nn.GRUCell(self.input_c_features, gru_cells)
        self.actor_linear = nn.Linear(gru_cells, total_actions)
        self.critic_linear = nn.Linear(gru_cells, 1)  # not being used in our case

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 1.0)
        self.actor_linear.bias.data.fill_(0)

        self.gru.bias_ih.data.fill_(0)
        self.gru.bias_hh.data.fill_(0)

    def forward(self, input, input_fn=None, hx_fn=None, inspect=False):
        input, hx = input

        c_input = input
        input, input_x = input_fn(c_input) if input_fn is not None else (c_input, c_input)
        ghx = self.gru(input, hx)
        hx, bhx = hx_fn(ghx) if hx_fn is not None else (ghx, ghx)

        if inspect:
            return self.critic_linear(hx), self.actor_linear(hx), hx, (ghx, bhx, c_input, input_x)
        else:
            return self.critic_linear(hx), self.actor_linear(hx), hx

    def initHidden(self, batch_size=1):
        return torch.zeros(batch_size, self.gru_units)

    def get_action_linear(self, state):
        return self.actor_linear(state)

    def transact(self, o_x, hx):
        hx = self.gru(o_x, hx)
        return hx


class BinaryGRUNet(nn.Module):
    def __init__(self, gru_net, bhx_net=None, obx_net=None):
        super(BinaryGRUNet, self).__init__()
        self.bhx_units = bhx_net.bhx_size if bhx_net is not None else None
        self.gru_units = gru_net.gru_units
        self.obx_net = obx_net
        self.gru_net = gru_net
        self.bhx_net = bhx_net
        self.actor_linear = self.gru_net.get_action_linear

    def initHidden(self, batch_size=1):
        return self.gru_net.initHidden(batch_size)

    def forward(self, input, inspect=False):
        input, hx = input
        critic, actor, hx, (ghx, bhx, input_c, input_x) = self.gru_net((input, hx), input_fn=self.obx_net,
                                                                       hx_fn=self.bhx_net,
                                                                       inspect=True)
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
        _, _, _, (_, _, _, input_x) = self.gru_net((obs, hx), input_fn=self.obx_net, hx_fn=self.bhx_net, inspect=True)
        return input_x


def generate_bottleneck_data(net, env, episodes, save_path, cuda=False):
    if os.path.exists(save_path):
        # unpickling after reading the file is efficient
        h_train, h_test, o_train, o_test = pickle.loads(open(save_path, "rb").read())
    else:
        logging.info('No Data Found @ path : {}'.format(save_path))
        logging.info('Generating BottleNeck Data..')
        hx_data, obs_data = [], []
        all_ep_rewards = []
        with torch.no_grad():
            for ep in range(episodes):
                done = False
                obs = env.reset()
                hx = Variable(net.initHidden())
                ep_reward = 0
                while not done:
                    obs = Variable(torch.Tensor(obs)).unsqueeze(0)
                    if cuda:
                        hx = hx.cuda()
                        obs = obs.cuda()
                    critic, logit, hx, (_, _, obs_c, _) = net((obs, hx), inspect=True)
                    prob = F.softmax(logit, dim=1)
                    action = int(prob.max(1)[1].data.cpu().numpy())
                    obs, reward, done, info = env.step(action)
                    hx_data.append(list([float(x) for x in hx.data.cpu().numpy()[0]]))
                    obs_data.append(list([float(x) for x in obs_c.data.cpu().numpy()[0]]))
                    ep_reward += reward
                logging.info('episode:{} reward:{}'.format(ep, ep_reward))
                all_ep_rewards.append(ep_reward)
        logging.info('Average Performance:{}'.format(sum(all_ep_rewards) / len(all_ep_rewards)))

        random.shuffle(hx_data)
        random.shuffle(obs_data)
        h_threshold = int(0.7 * len(hx_data))
        o_threshold = int(0.7 * len(obs_data))
        h_train, h_test = hx_data[:h_threshold], hx_data[h_threshold:]
        o_train, o_test = obs_data[:o_threshold], obs_data[o_threshold:]

        # h_train = np.unique(hx_data[:h_threshold], axis=0).tolist()
        # h_test = np.unique(hx_data[h_threshold:], axis=0).tolist()
        # o_train = np.unique(obs_data[:o_threshold], axis=0).tolist()
        # o_test = np.unique(obs_data[o_threshold:], axis=0).tolist()
        pickle.dump((h_train, h_test, o_train, o_test), open(save_path, "wb"))

    logging.info('Data Sizes:')
    logging.info('Hx Train:{} Hx Test:{} Obs Train:{} Obs Test:{}'.format(len(h_train), len(h_test),
                                                                          len(o_train), len(o_test)))

    # _repeat = 50
    # h_train = np.repeat(h_train, _repeat, axis=0).tolist()
    # o_train = np.repeat(o_train, _repeat, axis=0).tolist()

    return h_train, h_test, o_train, o_test


def generate_trajectories(env, batches, batch_size, save_path, guide=None):
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
                    hx = None if guide is None else Variable(guide.initHidden())
                    ep_reward = 0

                    while not done:
                        _obs.append(obs)
                        if guide is None:
                            action = env.env.get_desired_action()
                            _actions.append(action)
                        else:
                            obs = Variable(torch.Tensor(obs).unsqueeze(0))
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
                    if ep_reward == 0:
                        print(_obs)
                    logging.info('Batch:{} Ep: {} Reward:{}'.format(seed, ep, ep_reward))

                _train_data[seed] = (data_obs, data_actions, data_actions_prob, data_len)
        logging.info('Average Performance: {}'.format(sum(all_ep_rewards) / len(all_ep_rewards)))
        pickle.dump(_train_data, open(save_path, "wb"))
    return _train_data


def get_args():
    parser = argparse.ArgumentParser(description='GRU to FSM')
    parser.add_argument('--generate_train_data', action='store_true', default=False, help='Generate Train Data')
    parser.add_argument('--gru_train', action='store_true', default=False, help='Train GRU Network')
    parser.add_argument('--gru_test', action='store_true', default=False, help='Test GRU Network')
    parser.add_argument('--gru_size', type=int, default=1, help="No. of GRU Cells")
    parser.add_argument('--gru_lr', type=float, default=0.001, help="No. of GRU Cells")
    parser.add_argument('--grad_clip', type=float, default=5, help="Value for Gradinent Norm clipping")
    parser.add_argument('--trunc_k', type=int, default=10, help="k for Truncated BackProp")

    parser.add_argument('--generate_bx_data', action='store_true', default=False, help='Generate BottleNeck Data')
    parser.add_argument('--bhx_train', action='store_true', default=False, help='Train bx network')
    parser.add_argument('--ox_train', action='store_true', default=False, help='Train ox network')
    parser.add_argument('--bhx_test', action='store_true', default=False, help='Test bx network')
    parser.add_argument('--ox_test', action='store_true', default=False, help='Test ox network')
    parser.add_argument('--bgru_train', action='store_true', default=False, help='Train binary gru network')
    parser.add_argument('--bgru_test', action='store_true', default=False, help='Test binary gru network')

    parser.add_argument('--bhx_size', type=int, default=5, help="binary encoding size")
    parser.add_argument('--ox_size', type=int, default=3, help="binary encoding size")

    parser.add_argument('--train_epochs', type=int, default=1000, help="No. of training episodes")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size used for training GRU")
    parser.add_argument('--bgru_lr', type=float, default=0.0001, help="Learning rate for binary GRU")
    parser.add_argument('--gru_scratch', action='store_true', default=False, help='use scratch gru for BGRU')
    parser.add_argument('--bx_scratch', action='store_true', default=False, help='use scratch bx network for BGRU')
    parser.add_argument('--generate_fsm', action='store_true', default=False, help='extract fsm from fmm net')
    parser.add_argument('--evaluate_fsm', action='store_true', default=False, help='evaluate fsm')

    parser.add_argument('--bn_episodes', type=int, default=5000,
                        help="No. of episodes for generating data for Bottleneck Network")
    parser.add_argument('--bn_epochs', type=int, default=100, help="No. of Training epochs")

    parser.add_argument('--no_cuda', action='store_true', default=False, help='no cuda usage')
    parser.add_argument('--env', default='TomitaA-v0', help="Name of the environment")
    parser.add_argument('--env_seed', type=int, default=0, help="Seed for the environment")
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results', 'Tomita','Binary'),
                        help="Directory Path to store results")
    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    return args


if __name__ == '__main__':
    args = get_args()
    env = gym.make(args.env)
    env.seed(args.env_seed)
    obs = env.reset()

    # create directories to store results
    result_dir = ensure_directory_exits(args.result_dir)
    env_dir = ensure_directory_exits(os.path.join(result_dir, args.env))

    gru_dir = ensure_directory_exits(os.path.join(env_dir, 'gru_{}'.format(args.gru_size)))
    gru_net_path = os.path.join(gru_dir, 'model.p')
    gru_plot_dir = ensure_directory_exits(os.path.join(gru_dir, 'Plots'))

    bhx_dir = ensure_directory_exits(os.path.join(env_dir, 'gru_{}_bhx_{}'.format(args.gru_size, args.bhx_size)))
    bhx_net_path = os.path.join(bhx_dir, 'model.p')
    bhx_plot_dir = ensure_directory_exits(os.path.join(bhx_dir, 'Plots'))

    data_dir = ensure_directory_exits(os.path.join(env_dir, 'data'))
    bottleneck_data_path = os.path.join(data_dir, 'bottleneck_data.p')
    trajectories_data_path = os.path.join(data_dir, 'trajectories_data.p')
    gru_prob_data_path = os.path.join(data_dir, 'gru_prob_data.p')

    try:
        # ***********************************************************************************
        # Generate Training Data                                                            *
        # ***********************************************************************************
        if args.generate_train_data:
            set_log(data_dir, 'generate_train_data')
            train_data = generate_trajectories(env, 2000, args.batch_size, trajectories_data_path)

        # ***********************************************************************************
        # Gru Network                                                                       *
        # ***********************************************************************************
        if args.gru_train or args.gru_test:
            set_log(gru_dir, 'train' if args.gru_train else 'test')
            gru_net = GRUNet(len(obs), args.gru_size, int(env.action_space.n))
            train_data = generate_trajectories(env, 1000, args.batch_size, trajectories_data_path)
            if args.cuda:
                gru_net = gru_net.cuda()
            if args.gru_train:
                logging.info('Training GRU!')
                start_time = time.time()
                gru_net.train()
                optimizer = optim.Adam(gru_net.parameters(), lr=1e-3)
                gru_net = gru_nn.train(gru_net, env, optimizer, gru_net_path, gru_plot_dir, train_data, args.batch_size,
                                       args.train_epochs, args.cuda, args.grad_clip, args.trunc_k)
                write_net_readme(gru_net, gru_dir, info={'time_taken': time.time() - start_time})
            if args.gru_test:
                logging.info('Testing GRU!')
                gru_net.load_state_dict(torch.load(gru_net_path))
                gru_net.eval()
                perf = gru_nn.test(gru_net, env, 2000, log=True, cuda=args.cuda)
                logging.info('Average Performance:{}'.format(perf))

        # ***********************************************************************************
        # Generate Bx Data                                                                  *
        # ***********************************************************************************
        if args.generate_bx_data:
            set_log(data_dir, 'generate_bx_data')
            logging.info('Generating Data-Set for Later Bottle Neck Training')
            gru_net = GRUNet(len(obs), args.gru_size, int(env.action_space.n))
            gru_net.load_state_dict(torch.load(gru_net_path))
            gru_net.eval()

            generate_bottleneck_data(gru_net, env, args.bn_episodes, bottleneck_data_path, cuda=args.cuda)
            generate_trajectories(env, 2000, args.batch_size, gru_prob_data_path, gru_net.cpu())

        # ***********************************************************************************
        # Binary BHX SandGlassNet                                                           *
        # ***********************************************************************************
        if args.bhx_train or args.bhx_test:
            set_log(bhx_dir, 'train' if args.bhx_train else 'test')
            gru_net = GRUNet(len(obs), args.gru_size, int(env.action_space.n))
            gru_net.eval()
            bhx_net = BNNet(args.gru_size, args.bhx_size)
            if args.cuda:
                gru_net = gru_net.cuda()
                bhx_net = bhx_net.cuda()

            if not os.path.exists(gru_net_path):
                logging.info('Pre-Trained GRU model not found!')
                sys.exit(0)
            else:
                gru_net.load_state_dict(torch.load(gru_net_path))

            logging.info('Loading Data-Set')
            hx_train_data, hx_test_data, _, _ = generate_bottleneck_data(gru_net, env, args.bn_episodes,
                                                                         bottleneck_data_path, cuda=args.cuda)
            if args.bhx_train:
                bhx_start_time = time.time()
                logging.info('Training HX SandGlassNet!')
                optimizer = optim.Adam(bhx_net.parameters(), lr=1e-3, weight_decay=1e-4)
                bhx_net.train()
                bhx_net = b_nn.train(bhx_net, (hx_train_data, hx_test_data), optimizer, bhx_net_path, bhx_plot_dir,
                                     args.batch_size, args.train_epochs, args.cuda)
                bhx_end_time = time.time()
                write_net_readme(bhx_net, bhx_dir, info={'time_taken': round(bhx_end_time - bhx_start_time, 4)})
            if args.bhx_test:
                logging.info('Testing  HX SandGlassNet')
                bhx_net.load_state_dict(torch.load(bhx_net_path))
                bhx_net.eval()
                bhx_test_mse = b_nn.test(bhx_net, hx_test_data, len(hx_test_data), cuda=args.cuda)
                logging.info('MSE :{}'.format(bhx_test_mse))

        # ***********************************************************************************
        # Binary Gru Network                                                                *
        # ***********************************************************************************
        if args.bgru_train or args.bgru_test or args.generate_fsm:
            gru_net = GRUNet(len(obs), args.gru_size, int(env.action_space.n))
            bhx_net = BNNet(args.gru_size, args.bhx_size)
            bgru_net = BinaryGRUNet(gru_net, bhx_net, None)
            if args.cuda:
                bgru_net = bgru_net.cuda()

            bx_prefix = 'scratch-'
            if not args.bx_scratch:
                if bgru_net.bhx_net is not None:
                    bgru_net.bhx_net.load_state_dict(torch.load(bhx_net_path))
                bx_prefix = ''

            gru_prefix = 'scratch-'
            if not args.gru_scratch:
                bgru_net.gru_net.load_state_dict(torch.load(gru_net_path))
                gru_prefix = ''

            # create directories to save result
            bgru_dir_name = '{}gru_{}_{}hx_({},{})_bgru'.format(gru_prefix, args.gru_size, bx_prefix, args.bhx_size,
                                                                args.ox_size)
            bgru_dir = ensure_directory_exits(os.path.join(env_dir, bgru_dir_name))
            bgru_net_path = os.path.join(bgru_dir, 'model.p')
            bgru_plot_dir = ensure_directory_exits(os.path.join(bgru_dir, 'Plots'))

            _log_tag = 'train' if args.bgru_train else ('test' if args.bgru_test else 'generate_fsm')
            _log_tag = _log_tag if not args.evaluate_fsm else 'evaluate_fsm'

            set_log(bgru_dir, _log_tag)

            if args.bgru_train:
                logging.info('Training Binary GRUNet!')
                bgru_net.train()
                _start_time = time.time()
                if args.gru_scratch:
                    optimizer = optim.Adam(bgru_net.parameters(), lr=1e-3)
                    train_data = generate_trajectories(env, 1000, args.batch_size, trajectories_data_path)
                    bgru_net = gru_nn.train(bgru_net, env, optimizer, bgru_net_path, bgru_plot_dir, train_data,
                                            args.batch_size, args.train_epochs, args.cuda)
                else:
                    optimizer = optim.Adam(bgru_net.parameters(), lr=1e-4)
                    train_data = generate_trajectories(env, 1000, args.batch_size, gru_prob_data_path,
                                                       copy.deepcopy(bgru_net.gru_net).cpu())

                    bgru_net = bgru_nn.train(bgru_net, env, optimizer, bgru_net_path, bgru_plot_dir, train_data,
                                             args.batch_size, args.train_epochs, args.cuda)
                write_net_readme(bgru_net, bgru_dir, info={'time_taken': round(time.time() - _start_time, 4)})

            if args.bgru_test:
                logging.info('Testing BGRU!')
                bgru_net.load_state_dict(torch.load(bgru_net_path))
                bgru_net.eval()
                bgru_perf = bgru_nn.test(bgru_net, env, 2000, log=True, cuda=args.cuda)
                logging.info('Average Performance: {}'.format(bgru_perf))

            if args.generate_fsm:
                bgru_net.load_state_dict(torch.load(bgru_net_path))
                bgru_net.cpu()
                bgru_net.eval()
                moore_machine = MooreMachine()
                moore_machine.extract_from_nn(copy.deepcopy(env), bgru_net, 1000, 0, log=True)
                perf = moore_machine.evaluate(bgru_net, copy.deepcopy(env), total_episodes=500)
                moore_machine.save(open(os.path.join(bgru_dir, 'fsm.txt'), 'w'))
                logging.info('Performance Before Minimization:{}'.format(perf))
                moore_machine.minimize()
                moore_machine.save(open(os.path.join(bgru_dir, 'minimized_fsm.txt'), 'w'))
                # pickle.dump(moore_machine, open(os.path.join(bgru_dir, 'moore_machine.p'), 'wb'))
                perf = moore_machine.evaluate(bgru_net, env, total_episodes=500)
                logging.info('Performance After Minimization: {}'.format(perf))
            if args.evaluate_fsm:
                bgru_net.load_state_dict(torch.load(bgru_net_path))
                moore_machine = pickle.load(open(os.path.join(bgru_dir, 'moore_machine.p'), 'rb'))
                bgru_net.cpu()
                bgru_net.eval()
                perf = moore_machine.evaluate(bgru_net, env, total_episodes=500)
                logging.info('Performance : {}'.format(perf))

    except Exception as ex:
        logging.error(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
