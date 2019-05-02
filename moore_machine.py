import os
import torch
import random
import scipy.misc
import numpy as np
import logging, sys
from collections import deque
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.autograd import Variable
from tools import ensure_directory_exits
from PIL import Image, ImageFont, ImageDraw

logger = logging.getLogger(__name__)

sys.setrecursionlimit(3000)

class MooreMachine:
    """
    Moore Machine Network definition
    """

    def __init__(self, t={}, sd={}, ss=np.array([]), os=np.array([]), start_state=0, total_actions=None):
        self.transaction = t
        self.state_desc = sd
        self.state_space = ss
        self.obs_space = os
        self.start_state = start_state
        self.minimized = False
        self.obs_minobs_map = None
        self.minobs_obs_map = None
        self.frequency = None
        self.trajectory = None
        self.total_actions = total_actions

    def __str__(self):

        msg = '***********************************************' + '\n'
        msg += 'Transaction:' + self.transaction.__str__() + '\n'
        msg += '```````````````````````````````````````````````' + '\n'
        msg += 'State Desc:' + self.state_desc.__str__() + '\n'
        msg += '```````````````````````````````````````````````' + '\n'
        msg += 'Observations:' + self.obs_space.__str__() + '\n'
        msg += '```````````````````````````````````````````````' + '\n'
        msg += 'State Space:' + self.state_space.__str__() + '\n'
        msg += '***********************************************'
        return msg

    @staticmethod
    def _get_index(source, item, force=True):
        """
        Returns index of the item in the source.

        :param source: np-array comprising of unique elements (set)
        :param item: target item(array)
        :param force: if True: In case item not found; it will add the item and return the corresponding index
        """
        _index = np.where(np.all(source == item, axis=1))[0] if len(source) != 0 else []
        if len(_index) != 0:
            _index = _index[0]
        elif force:
            source = source.tolist()
            source.append(item)
            source = np.array(source)
            _index = len(source) - 1
        else:
            _index = None
        return source, _index

    def _update_info(self, obs, curr_state, next_state, curr_action, next_action):
        """
        Records new states and transactions.

        :param obs: array of observations
        :param curr_state: current state of the environment
        :param next_state: next state of the environment
        :param curr_action: current action of the environment
        :param next_action: next action of the environment
        :return: each state's index and a set of states and observations
        """
        self.obs_space, obs_index = self._get_index(self.obs_space, obs)
        state_indices = []
        new_entries = []
        for state_info in [(curr_state, curr_action), (next_state, next_action)]:
            state, _action = state_info
            self.state_space, state_index = self._get_index(self.state_space, state)
            if state_index not in self.state_desc:
                self.state_desc[state_index] = {'action': str(_action), 'description': state}
            if self.state_desc[state_index]['action'] == str(None) and _action is not None:
                self.state_desc[state_index]['action'] = str(_action)
            state_indices.append(state_index)
        for s_i in state_indices:
            if s_i not in self.transaction:
                self.transaction[s_i] = {_: None for _ in range(len(self.obs_space))}
                new_entries += [(s_i, _) for _ in range(len(self.obs_space))]
            elif obs_index not in self.transaction[s_i]:
                for o_i in range(len(self.obs_space)):
                    if o_i not in self.transaction[s_i]:
                        self.transaction[s_i][o_i] = None
                        if s_i != state_indices[0] and o_i != obs_index:
                            new_entries.append((s_i, o_i))
        self.transaction[state_indices[0]][obs_index] = state_indices[1]

        return state_indices, new_entries

    def extract_from_nn(self, env, net, episodes, seed=0, log=True, render=False, partial=False, cuda=False):
        """
        Extract Finite State Moore Machine Network(MMNet) from a BottleNeck Gated Recurrent Unit Network(BGRUNet).

        :param env: the environment where agent is in
        :param net: BottleNeck GRUNet
        :param episodes: number of episodes
        :param log: check to print out logs
        :param render: check to render environment
        :param cuda: check if cuda is available
        """
        net.eval()
        max_actions = 10000
        random.seed(seed)
        self.total_actions = int(env.action_space.n)
        x = set([])
        with torch.no_grad():

            # collect all unique transactions
            all_ep_rewards = []
            for ep in range(episodes):
                done = False
                obs = env.reset()
                curr_state = Variable(net.init_hidden())
                if cuda:
                    curr_state = curr_state.cuda()
                curr_state_x = net.state_encode(curr_state)
                ep_reward = 0
                ep_actions = []
                while not done:
                    if render:
                        env.render()
                    curr_action = net.get_action_linear(curr_state_x, decode=True)
                    prob = F.softmax(curr_action, dim=1)
                    curr_action = int(prob.max(1)[1].cpu().data.numpy()[0])
                    obs = Variable(torch.Tensor(obs)).unsqueeze(0)
                    if cuda:
                        obs = obs.cuda()
                    critic, logit, next_state, (next_state_c, next_state_x), (_, obs_x) = net((obs, curr_state),
                                                                                              inspect=True)
                    prob = F.softmax(logit, dim=1)
                    next_action = int(prob.max(1)[1].cpu().data.numpy())

                    self._update_info(obs_x.cpu().data.numpy()[0], curr_state_x.cpu().data.numpy()[0],
                                      next_state_x.cpu().data.numpy()[0], curr_action, next_action)
                    obs, reward, done, _ = env.step(next_action)

                    done = done if len(ep_actions) <= max_actions else True
                    ep_actions.append(next_action)
                    # a quick hack to prevent the agent from stucking
                    max_same_action = 5000
                    if len(ep_actions) > max_same_action:
                        actions_to_consider = ep_actions[-max_same_action:]
                        if actions_to_consider.count(actions_to_consider[0]) == max_same_action:
                            done = True
                    curr_state = next_state
                    curr_state_x = next_state_x
                    ep_reward += reward
                    x.add(''.join([str(int(i)) for i in next_state.cpu().data.numpy()[0]]))
                if log:
                    logger.info('Episode:{} Reward: {} '.format(ep, ep_reward))
                all_ep_rewards.append(ep_reward)

            if log:
                logger.info('Average Reward:{}'.format(np.average(all_ep_rewards)))

        if not partial:
            # find missing entries in the transaction table
            unknowns = []
            for curr_state_i in self.state_desc.keys():
                if curr_state_i in self.transaction:
                    for obs_i in range(len(self.obs_space)):
                        if (obs_i not in self.transaction[curr_state_i]) or (
                                self.transaction[curr_state_i][obs_i] is None):
                            unknowns.append((curr_state_i, obs_i))
                else:
                    unknowns += [(curr_state_i, i) for i in range(len(self.obs_space))]

            # fill information for the missing transactions
            done = False
            while not done:
                done = True
                for i, (state_i, obs_i) in enumerate(unknowns):
                    state_x = self.state_desc[state_i]['description']
                    state_x = Variable(torch.FloatTensor(state_x).unsqueeze(0))

                    obs_x = self.obs_space[obs_i]
                    obs_x = torch.FloatTensor(obs_x).unsqueeze(0)
                    obs_x = Variable(obs_x)

                    if cuda:
                        state_x, obs_x = state_x.cuda(), obs_x.cuda()

                    curr_action = net.get_action_linear(state_x, decode=True)
                    prob = F.softmax(curr_action, dim=1)
                    curr_action = int(prob.max(1)[1].cpu().data.numpy()[0])

                    next_state_x = net.transact(obs_x, state_x)
                    next_action = net.get_action_linear(next_state_x, decode=True)
                    prob = F.softmax(next_action, dim=1)
                    next_action = int(prob.max(1)[1].cpu().data.numpy()[0])

                    next_state_x = next_state_x.cpu().data.numpy()[0]
                    state_x = state_x.cpu().data.numpy()[0]
                    obs_x = obs_x.cpu().data.numpy()[0]
                    state_indices, new_entries = self._update_info(obs_x, state_x, next_state_x, curr_action,
                                                                   next_action)
                    unknowns.pop(i)

                    if len(new_entries) > 0:
                        unknowns += new_entries
                        logger.info('New Unknown State-Trasactions: {}'.format(new_entries))
                    x.add(''.join([str(int(i)) for i in next_state_x]))
                    done = False
                    break

        # find index of the start_state
        start_state = Variable(net.init_hidden())
        if cuda:
            start_state = start_state.cuda()
        start_state_x = net.state_encode(start_state).data.cpu().numpy()[0]
        _, self.start_state = self._get_index(self.state_space, start_state_x, force=False)

    def map_action(self, net, s_i, obs_i):
        """
        Gets state and observation at time i in a network and gives next action.

        :param net: given network
        :param s_i: state at time i
        :param obs_i: observation at time i
        :return: next action according to the given state and observation
        """
        state_x = self.state_desc[s_i]['description']
        state_x = Variable(torch.FloatTensor(state_x).unsqueeze(0))

        obs_x = self.obs_space[obs_i]
        obs_x = torch.FloatTensor(obs_x).unsqueeze(0)
        obs_x = Variable(obs_x)
        next_state_x = net.transact(obs_x, state_x)
        next_action = net.get_action_linear(next_state_x, decode=True)
        prob = F.softmax(next_action, dim=1)
        next_action = int(prob.max(1)[1].cpu().data.numpy()[0])
        return next_action

    def minimize_partial_fsm(self, net):
        """
        Minimizing the whole Finite State Machine(FSM) to fewer states.

        :param net: given network
        """
        _states = sorted(self.transaction.keys())
        compatibility_mat = {s: {p: False if self.state_desc[s]['action'] != self.state_desc[p]['action'] else None
                                 for p in _states[:i + 1]}
                             for i, s in enumerate(_states[1:])}
        unknowns = []
        for s in compatibility_mat.keys():
            for k in compatibility_mat[s].keys():
                if compatibility_mat[s][k] is None:
                    unknowns.append((s, k))

        unknown_lengths = deque(maxlen=1000)
        while len(unknowns) != 0:
            # next 3 lines are experimental
            if len(unknown_lengths) > 0 and unknown_lengths.count(unknown_lengths[0]) == unknown_lengths.maxlen:
                s, k = unknowns[-1]
                compatibility_mat[s][k] = True

            s, k = unknowns.pop(0)
            if compatibility_mat[s][k] is None:
                compatibility_mat[s][k] = []
                for obs_i in range(len(self.obs_space)):
                    if (obs_i not in self.transaction[s]) or (self.transaction[s][obs_i] is None) or \
                            (obs_i not in self.transaction[k]) or (self.transaction[k][obs_i] is None):
                        pass
                    else:
                        next_s, next_k = self.transaction[s][obs_i], self.transaction[k][obs_i]
                        action_next_s = self.state_desc[next_s]['action']
                        action_next_k = self.state_desc[next_k]['action']
                        # if next_s != next_k and next_k != k and next_s != s:
                        if next_s != next_k and not (next_k == k and next_s == s):
                            if action_next_s != action_next_k:
                                compatibility_mat[s][k] = False
                                break
                            first, sec = sorted([next_k, next_s])[::-1]
                            if type(compatibility_mat[first][sec]).__name__ == 'bool' and not \
                                    compatibility_mat[first][sec]:
                                compatibility_mat[s][k] = False
                                break
                            elif compatibility_mat[first][sec] is None or \
                                    type(compatibility_mat[first][sec]).__name__ != 'bool':
                                compatibility_mat[s][k].append((first, sec))

            elif type(compatibility_mat[s][k]).__name__ != 'bool':
                for i, (m, n) in enumerate(compatibility_mat[s][k]):
                    if type(compatibility_mat[m][n]).__name__ == 'bool' and not compatibility_mat[m][n]:
                        compatibility_mat[s][k] = False
                        break
                    elif type(compatibility_mat[m][n]).__name__ == 'bool' and compatibility_mat[m][n]:
                        compatibility_mat[s][k].pop(i)

            if type(compatibility_mat[s][k]).__name__ != 'bool':
                if len(compatibility_mat[s][k]) == 0:
                    compatibility_mat[s][k] = True
                else:
                    unknowns.append((s, k))

            unknown_lengths.append(len(unknowns))

        new_states = []
        new_state_info = {}
        processed = {x: False for x in _states}
        belongs_to = {_: None for _ in _states}
        for s in sorted(_states):
            if not processed[s]:
                comp_pair = [sorted((s, x))[::-1] for x in _states if
                             (x != s and compatibility_mat[max(s, x)][min(s, x)])]
                if len(comp_pair) != 0:
                    _new_state = self.traverse_compatible_states(comp_pair, compatibility_mat)
                    _new_state.sort(key=len, reverse=True)
                else:
                    _new_state = [[s]]
                for d in _new_state[0]:
                    processed[d] = True
                    belongs_to[d] = len(new_states)
                new_state_info[len(new_states)] = {'action': self.state_desc[_new_state[0][0]]['action'],
                                                   'sub_states': _new_state[0]}
                new_states.append(_new_state[0])

        new_trans = {}
        for i, s in enumerate(new_states):
            new_trans[i] = {}
            for o in range(len(self.obs_space)):
                new_trans[i][o] = None
                for sub_s in s:
                    if o in self.transaction[sub_s] and self.transaction[sub_s][o] is not None:
                        new_trans[i][o] = belongs_to[self.transaction[sub_s][o]]
                        break

        # if the new_state comprising of start-state has just one sub-state ;
        # then we can merge this new_state with other new_states as the action of the start-state doesn't matter
        start_state_p = belongs_to[self.start_state]
        if len(new_states[start_state_p]) == 1:
            start_state_trans = new_trans[start_state_p]
            for state in new_trans.keys():
                if state != start_state_p and new_trans[state] == start_state_trans:
                    new_trans.pop(start_state_p)
                    new_state_info.pop(start_state_p)
                    new_state_info[state]['sub_states'] += new_states[start_state_p]

                    # This could be highly wrong (On God's Grace :D )
                    for _state in new_trans.keys():
                        for _o in new_trans[_state].keys():
                            if new_trans[_state][_o] == start_state_p:
                                new_trans[_state][_o] = state

                    start_state_p = state
                    break

        # Minimize Observation Space (Combine observations which show the same transaction behaviour for all states)
        _obs_minobs_map = {}
        _minobs_obs_map = {}
        _trans_minobs_map = {}
        min_trans = {s: {} for s in new_trans.keys()}
        obs_i = 0
        for i in range(len(self.obs_space)):
            _trans_key = [new_trans[s][i] for s in sorted(new_trans.keys())].__str__()
            if _trans_key not in _trans_minobs_map:
                obs_i += 1
                o = 'o_' + str(obs_i)
                _trans_minobs_map[_trans_key] = o
                _minobs_obs_map[o] = [i]
                for s in new_trans.keys():
                    min_trans[s][o] = new_trans[s][i]
            else:
                _minobs_obs_map[_trans_minobs_map[_trans_key]].append(i)
            _obs_minobs_map[i] = _trans_minobs_map[_trans_key]

        # Update information
        self.transaction = min_trans
        self.state_desc = new_state_info
        self.state_space = list(self.transaction.keys())
        self.start_state = start_state_p
        self.obs_minobs_map = _obs_minobs_map
        self.minobs_obs_map = _minobs_obs_map
        self.minimized = True

    @staticmethod
    def traverse_compatible_states(states, compatibility_mat):
        for i, s in enumerate(states):
            for j, s_next in enumerate(states[i + 1:]):
                compatible = True
                for m in s:
                    for n in s_next:
                        if m != n and not compatibility_mat[max(m, n)][min(m, n)]:
                            compatible = False
                            break
                    if not compatible:
                        break
                if compatible:
                    _states = states[:i] + [sorted(list(set(s + s_next)))] + states[i + j + 2:]
                    return MooreMachine.traverse_compatible_states(_states, compatibility_mat)
        return states

    def minimize(self):
        """
        Minimize observation space.
        """
        # create initial partitions (states) based on the action space
        partitions = {'s_' + str(i): [] for i in range(self.total_actions)}
        state_dict = {}

        # mapping from new partition states to original state space (un-minified) /vice-versa (for efficiency)
        for x in self.state_desc.keys():
            _key = 's_' + str(self.state_desc[x]['action'])
            partitions[_key].append(x)
            state_dict[x] = _key

        while True:
            _new_states = {}
            for i, p in enumerate(sorted(partitions.keys())):
                for s in partitions[p]:
                    _key = str(i) + '_' + "_".join([state_dict[self.transaction[s][o]]
                                                    for o in range(len(self.obs_space))])
                    if _key in _new_states:
                        _new_states[_key].append(s)
                    else:
                        _new_states[_key] = [s]
            if len(_new_states.keys()) > len(partitions.keys()):
                _partitions = {}
                for i, p in enumerate(sorted(_new_states.keys())):
                    i = 'ns_' + str(i)
                    _partitions[i] = _new_states[p]
                    for s in _new_states[p]:
                        state_dict[s] = i
                partitions = _partitions
            else:
                break

        # create new transaction table:
        new_trans = {}
        new_state_info = {}
        for p in partitions:
            if len(partitions[p]) > 0:
                new_trans[p] = {o: state_dict[self.transaction[partitions[p][0]][o]] for o in
                                range(len(self.obs_space))}
                new_state_info[p] = {'action': self.state_desc[partitions[p][0]]['action'],
                                     'sub_states': partitions[p]}

        # if the partition comprising of start-state has just one sub-state ;
        # then we can merge this partition with other partitions as the action of the start-state doesn't matter
        start_state_p = state_dict[self.start_state]
        if len(partitions[start_state_p]) == 1:
            start_state_trans = new_trans[start_state_p]
            for state in new_trans.keys():
                if state != start_state_p and new_trans[state] == start_state_trans:
                    new_trans.pop(start_state_p)
                    new_state_info.pop(start_state_p)
                    new_state_info[state]['sub_states'] += partitions[start_state_p]

                    # This could be highly wrong (On God's Grace :D )
                    for _state in new_trans.keys():
                        for _o in new_trans[_state].keys():
                            if new_trans[_state][_o] == start_state_p:
                                new_trans[_state][_o] = state

                    start_state_p = state
                    break

        # Combine observations which show the same transaction behaviour for all states
        _obs_minobs_map = {}
        _minobs_obs_map = {}
        _trans_minobs_map = {}
        min_trans = {s: {} for s in new_trans.keys()}
        obs_i = 0
        for i in range(len(self.obs_space)):
            _trans_key = [new_trans[s][i] for s in new_trans.keys()].__str__()
            if _trans_key not in _trans_minobs_map:
                obs_i += 1
                o = 'o_' + str(obs_i)
                _trans_minobs_map[_trans_key] = o
                _minobs_obs_map[o] = [i]
                for s in new_trans.keys():
                    min_trans[s][o] = new_trans[s][i]
            else:
                _minobs_obs_map[_trans_minobs_map[_trans_key]].append(i)
            _obs_minobs_map[i] = _trans_minobs_map[_trans_key]

        # Update information
        self.transaction = min_trans
        self.state_desc = new_state_info
        self.state_space = self.transaction.keys()
        self.start_state = start_state_p
        self.obs_minobs_map = _obs_minobs_map
        self.minobs_obs_map = _minobs_obs_map
        self.minimized = True

    def evaluate(self, net, env, total_episodes, log=True, render=False, inspect=False, store_obs=False, path=None, cuda=False):
        """
        Evaluate the trained network.

        :param net: trained Bottleneck GRU network
        :param env: environment
        :param total_episodes: number of episodes to test
        :param log: check to print out evaluation log
        :param render: check to render environment
        :param inspect: check for previous evaluations to not evaluate again
        :param store_obs: check to store observations again
        :param path: where to check for inspection
        :param cuda: check if cuda is available
        :return: evaluation performance on given model
        """
        net.eval()
        if inspect:
            obs_path = ensure_directory_exits(os.path.join(path, 'obs'))
            video_dir_path = ensure_directory_exits(os.path.join(path, 'eps_videos'))
            if len(os.listdir(video_dir_path)) > 0:
                sys.exit('Previous Video Files present: ' + video_dir_path)
            self.frequency = {s: {t: 0 for t in sorted((self.state_desc.keys()))} for s in
                              sorted(self.state_desc.keys())}
            self.trajectory = []

        total_reward = 0
        for ep in range(total_episodes):
            if inspect:
                ep_video_path = ensure_directory_exits(os.path.join(video_dir_path, str(ep)))
                obs, org_obs = env.reset(inspect=True)
                _shape = (org_obs.shape[1], org_obs.shape[0])
            else:
                obs = env.reset()
            done = False
            ep_reward = 0
            ep_actions = []
            ep_obs = []
            curr_state = self.start_state
            while not done:
                ep_obs.append(obs)
                obs = torch.FloatTensor(obs).unsqueeze(0)
                obs = Variable(obs)
                if cuda:
                    obs = obs.cuda()
                obs_x = list(net.obs_encode(obs).data.cpu().numpy()[0])
                _, obs_index = self._get_index(self.obs_space, obs_x, force=False)
                if store_obs:
                    obs_dir = ensure_directory_exits(os.path.join(obs_path, str(obs_index)))
                    scipy.misc.imsave(
                        os.path.join(obs_dir, str(obs_index) + '_' + str(random.randint(0, 100000)) + '.jpg'),
                        org_obs)

                if not self.minimized:
                    (obs_index, pre_index) = (obs_index, None)
                else:
                    try:
                        (obs_index, pre_index) = (self.obs_minobs_map[obs_index], obs_index)
                    except Exception as e:
                        logger.error(e)

                next_state = self.transaction[curr_state][obs_index]
                if next_state is None:
                    logger.info('None state encountered!')
                    logger.info('Exiting the script!')
                    sys.exit(0)
                if render and inspect:
                    _text = 'Current State:{} \n Obs: {} \n Next State: {} \n\n\n Total States:{} \n Total Obs: {}'
                    _text = _text.format(str(curr_state), (obs_index, pre_index).__str__(), str(next_state),
                                         len(self.state_desc.keys()), len(self.minobs_obs_map.keys()))
                    _label_img = self.text_image(_shape, _text)
                    _img = np.hstack((org_obs, _label_img))
                    env.render(inspect=inspect, img=_img)
                    if inspect:
                        frame_id = str(len(ep_obs))
                        frame_id = '0' * (10 - len(frame_id)) + frame_id
                        scipy.misc.imsave(os.path.join(ep_video_path, 'frame_' + frame_id + '.jpg'), _img)
                        self.frequency[curr_state][next_state] += 1
                        if ep == total_episodes - 1:
                            self.trajectory.append([len(ep_obs), curr_state, (obs_index, pre_index), next_state])
                elif render:
                    env.render()

                curr_state = next_state
                action = int(self.state_desc[curr_state]['action'])
                obs, reward, done, info = env.step(action)
                org_obs = info['org_obs'] if 'org_obs' in info else obs
                ep_actions.append(action)
                ep_reward += reward

                # a quick hack to prevent the agent from stucking
                max_same_action = 5000
                if len(ep_actions) > max_same_action:
                    actions_to_consider = ep_actions[-max_same_action:]
                    if actions_to_consider.count(actions_to_consider[0]) == max_same_action:
                        done = True

            total_reward += ep_reward
            if log:
                logger.info("Episode => {} Score=> {}".format(ep, ep_reward))
            if inspect:
                _parseable_path = ep_video_path.replace('(', '\(')
                _parseable_path = _parseable_path.replace(')', '\)')
                os.system("ffmpeg -f image2 -pattern_type glob -framerate 1 -i '{}*.jpg' {}{}.mp4".
                          format(os.path.join(_parseable_path, 'frame_'), os.path.join(_parseable_path, 'video_'),
                                 ep))
                os.system("rm -rf {}/*.jpg".format(_parseable_path))

        if self.minimized and store_obs:
            logger.info('Combining Sub-Observations')
            combined_obs_path = ensure_directory_exits(os.path.join(path, 'combined_obs'))
            for k in sorted(self.minobs_obs_map.keys()):
                logger.info('Observation Class:' + str(k))
                max_images_per_comb = 250  # beyond this images cannot be combined due to library/memory issues
                suffix = len(self.minobs_obs_map[k]) > max_images_per_comb
                total_parts = int(len(self.minobs_obs_map[k]) / max_images_per_comb)
                if len(self.minobs_obs_map[k]) % max_images_per_comb != 0:
                    total_parts += 1
                for p_i in range(total_parts):
                    k_image = None
                    for o_i in self.minobs_obs_map[k][p_i:p_i + max_images_per_comb]:
                        o_path = os.path.join(obs_path, str(o_i))
                        o_files = [os.path.join(o_path, f) for f in os.listdir(o_path) if
                                   os.path.isfile(os.path.join(o_path, f))]
                        o_i_image = scipy.misc.imread(random.choice(o_files))

                        o_i_image = np.hstack((self.text_image(_shape, str(o_i),
                                                               position=(_shape[0] // 2, _shape[1] // 2)),
                                               o_i_image))
                        for i in range(9):
                            o_i_image = np.hstack((o_i_image, scipy.misc.imread(random.choice(o_files))))
                        k_image = o_i_image if k_image is None else np.vstack((k_image, o_i_image))
                    k_shape = (_shape[0], len(self.minobs_obs_map[k][p_i:p_i + max_images_per_comb]) * _shape[1])
                    k_name_image = self.text_image(k_shape, str(k), position=(k_shape[0] // 2, 10), font_size=20)
                    k_image = np.hstack((k_name_image, k_image))
                    k_file_name = str(k) + (('_part_' + str(p_i + 1)) if suffix else '')
                    scipy.misc.imsave(os.path.join(combined_obs_path, k_file_name + '.jpg'), k_image)

            if inspect:
                obs_path = obs_path.replace('(', '\(').replace(')', '\)')
                os.system("rm -rf {}".format(obs_path))

        return total_reward / total_episodes

    @staticmethod
    def text_image(shape, text, position=(0, 0), font_size=25):
        font = ImageFont.truetype("arial.ttf", font_size)
        img = Image.new("RGB", shape, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text(position, text, (0, 0, 0), font=font)
        return np.array(img)

    def save(self, info_file):
        """
        Save data into given file.

        :param info_file: an opened file to write data in
        """
        info_file.write('Total Unique States:{}\n'.format(len(self.state_desc.keys())))
        if not self.minimized:
            info_file.write('Total Unique Observations:{}\n'.format(len(self.obs_space)))
        else:
            info_file.write('Total Unique Observations:{}\n'.format(len(self.minobs_obs_map.keys())))
        info_file.write('\n\nStart State: {}\n'.format(self.start_state))

        if not self.minimized:
            info_file.write('\n\nObservation Description:\n')
            t1 = PrettyTable(["Index", "Features"])
            for i, o in enumerate(self.obs_space):
                t1.add_row([i, o.__str__()])
            info_file.write(t1.__str__() + '\n')
        else:
            info_file.write('\n\nMin-Observation Space Description:\n')
            t1 = PrettyTable(["obs-tag", "Sub-Observation Space"])
            for k in sorted(self.minobs_obs_map.keys()):
                t1.add_row([k, self.minobs_obs_map[k]])
            info_file.write(t1.__str__() + '\n')

        info_file.write('\n\nStates Description:\n')
        t1 = PrettyTable(["Name", "Action", "Description" if not self.minimized else 'Sub States'])
        for k in sorted(self.state_desc.keys()):
            _state_info = self.state_desc[k]['description' if not self.minimized else 'sub_states']
            t1.add_row([k, self.state_desc[k]['action'], _state_info])
        info_file.write(t1.__str__() + '\n')
        if not self.minimized:
            column_names = [""] + [str(_) for _ in range(len(self.obs_space))]
            t = PrettyTable(column_names)
            for key in sorted(self.transaction.keys()):
                t.add_row([key] +
                          [(self.transaction[key][int(c)] if int(c) in self.transaction[key] else None) for c in
                           column_names[1:]])
        else:
            column_names = [""] + sorted(self.transaction[list(self.transaction.keys())[0]].keys())
            t = PrettyTable(column_names)
            for key in sorted(self.transaction.keys()):
                t.add_row([key] + [self.transaction[key][c] for c in column_names[1:]])

        info_file.write('\n\nTransaction Matrix:    (StateIndex_ObservationIndex x StateIndex)' + '\n')
        info_file.write(t.__str__())

        if self.frequency is not None:
            column_names = [""] + [str(_) for _ in sorted(self.frequency.keys())]
            t = PrettyTable(column_names)
            for key in sorted(self.frequency.keys()):
                t.add_row([key] + [self.frequency[key][c] for c in sorted(self.frequency.keys())])
            info_file.write('\n\nState Transaction Frequency Matrix:    (From  x To)' + '\n')
            info_file.write(t.__str__())

        if self.trajectory is not None:
            info_file.write('\n\nTrajectory info:' + '\n')
            info_file.write(self.trajectory.__str__())
        info_file.close()


if __name__ == '__main__':
    trans = {0: {0: 1, 1: 2}, 1: {0: 0, 1: 3}, 2: {0: 4, 1: 5}, 3: {0: 4, 1: 5}, 4: {0: 4, 1: 5}, 5: {0: 5, 1: 5}}
    desc = {0: {'action': 0}, 1: {'action': 0}, 2: {'action': 1}, 3: {'action': 1}, 4: {'action': 1},
            5: {'action': 0}}
    mm = MooreMachine(trans, desc, [], [_ for _ in range(2)], 0, total_actions=2)
    mm.minimize()
    correct_trans = {'ns_1': {0: 'ns_1', 1: 'ns_2'}, 'ns_0': {0: 'ns_0', 1: 'ns_0'}, 'ns_2': {0: 'ns_2', 1: 'ns_0'}}
    print(mm.transaction)
    print(mm.state_desc)
    print(mm.transaction == correct_trans)
    print(len(mm.transaction.keys()))

    trans = {0: {0: 1, 1: 2}, 1: {0: 1, 1: 3}, 2: {0: 1, 1: 2}, 3: {0: 1, 1: 4}, 4: {0: 1, 1: 2}}
    desc = {i: {'action': 0} for i in range(4)}
    desc[4] = {'action': 1}
    mm = MooreMachine(trans, desc, [], [_ for _ in range(2)], 0, total_actions=2)
    mm.minimize()
    correct_trans = {'ns_1': {0: 'ns_1', 1: 'ns_2'}, 'ns_0': {0: 'ns_1', 1: 'ns_0'}, 'ns_2': {0: 'ns_1', 1: 'ns_3'},
                     'ns_3': {0: 'ns_1', 1: 'ns_0'}}
    print(mm.transaction)
    print(mm.state_desc)
    print(mm.transaction == correct_trans)
    print(len(mm.transaction.keys()))
