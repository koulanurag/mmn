import logging
import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from prettytable import PrettyTable


logger = logging.getLogger(__name__)


class MooreMachine:
    """Moore Machine"""

    def __init__(self, t={}, sd={}, ss=np.array([]), os=np.array([]), start_state=0, total_actions=None):
        self.transaction = t
        self.state_desc = sd
        self.state_space = ss
        self.obs_space = os
        self.start_state = start_state
        self.minimized = False
        self.obs_minobs_map = None
        self.minobs_obs_map = None
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
        """ Returns index of the item in the source

        :param source: np-array comprising of unique elements (set)
        :param item: target item (array)
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
        """ Records new states and transactions"""
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

    def extract_from_nn(self, env, net, episodes, seed, log=True,render=False):
        """ Extract Finite State Moore Machine from a Binary Neural Network"""
        net.eval()
        random.seed(seed)
        self.total_actions = int(env.action_space.n)
        x = set([])
        with torch.no_grad():

            # collect all unique transactions
            all_ep_rewards = []
            for ep in range(episodes):
                done = False
                obs = env.reset()
                curr_state = Variable(net.initHidden())
                curr_state_x = net.state_encode(curr_state)
                ep_reward = 0
                while not done:
                    if render:
                        env.render()
                    curr_action = net.get_action_linear(curr_state_x, decode=True)
                    prob = F.softmax(curr_action, dim=1)
                    curr_action = int(prob.max(1)[1].cpu().data.numpy()[0])
                    obs = Variable(torch.Tensor(obs)).unsqueeze(0)

                    critic, logit, next_state, (next_state_c, next_state_x), (_, obs_x) = net((obs, curr_state),
                                                                                              inspect=True)
                    prob = F.softmax(logit, dim=1)
                    next_action = int(prob.max(1)[1].data.cpu().numpy())

                    self._update_info(obs_x.cpu().data.numpy()[0], curr_state_x.cpu().data.numpy()[0],
                                      next_state_x.cpu().data.numpy()[0], curr_action, next_action)
                    obs, reward, done, _ = env.step(next_action)
                    curr_state = next_state
                    curr_state_x = next_state_x
                    ep_reward += reward
                    x.add(''.join([str(int(i)) for i in next_state.cpu().data.numpy()[0]]))
                if log:
                    logger.info('Episode:{} Reward: {} '.format(ep, ep_reward))
                all_ep_rewards.append(ep_reward)

            if log:
                logger.info('Average Reward:{}'.format(np.average(all_ep_rewards)))

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
            start_state = net.initHidden()
            start_state_x = net.state_encode(start_state).data.cpu().numpy()[0]
            _, self.start_state = self._get_index(self.state_space, start_state_x, force=False)

        return self.obs_space, self.transaction, self.state_desc, self.start_state

    def minimize(self):
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

        # Minimize Observation Space (Combine observations which show the same transaction behaviour for all states)
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

    def evaluate(self, net, env, total_episodes, log=True,render=False):
        net.eval()
        total_reward = 0
        for ep in range(total_episodes):
            obs = env.reset()
            done = False
            ep_reward = 0
            ep_actions = []
            ep_obs = []
            curr_state = self.start_state
            while not done:
                if render:
                    env.render()
                ep_obs.append(obs)
                obs = torch.FloatTensor(obs).unsqueeze(0)
                obs = Variable(obs)
                obs_x = list(net.obs_encode(obs).data.cpu().numpy()[0])
                _, obs_index = self._get_index(self.obs_space, obs_x, force=False)
                obs_index = obs_index if not self.minimized else self.obs_minobs_map[obs_index]
                curr_state = self.transaction[curr_state][obs_index]
                action = int(self.state_desc[curr_state]['action'])
                obs, reward, done, _ = env.step(action)
                ep_actions.append(action)
                ep_reward += reward
            total_reward += ep_reward
            if log:
                logger.info("Episode => {} Score=> {}".format(ep, ep_reward))
                logger.info("Action => {} Observation=> {}".format(ep_actions, ep_obs))
        return total_reward / total_episodes

    def save(self, info_file):
        info_file.write('Total Unique States:{}\n'.format(len(self.state_desc.keys())))
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
        column_names = [""] + sorted(self.transaction[list(self.transaction.keys())[0]].keys())
        t = PrettyTable(column_names)
        for key in sorted(self.transaction.keys()):
            t.add_row([key] + [self.transaction[key][c] for c in column_names[1:]])

        info_file.write('\n\nTransaction Matrix:    (StateIndex_ObservationIndex x StateIndex)' + '\n')
        info_file.write(t.__str__())
        info_file.close()


if __name__ == '__main__':
    trans = {0: {0: 1, 1: 2}, 1: {0: 0, 1: 3}, 2: {0: 4, 1: 5}, 3: {0: 4, 1: 5}, 4: {0: 4, 1: 5}, 5: {0: 5, 1: 5}}
    desc = {0: {'action': 0}, 1: {'action': 0}, 2: {'action': 1}, 3: {'action': 1}, 4: {'action': 1}, 5: {'action': 0}}
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
