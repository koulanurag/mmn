import numpy as np
import gym
from scipy.misc import imresize


class AtariWrapper():
    def __init__(self, env):
        self.env = env

    def step(self, *args, **kwargs):
        state, reward, done, info = self.env.step(*args, **kwargs)
        state = self.__process_atari_image(state)
        return state, reward, done, info

    @property
    def action_space(self):
        return self.env.action_space

    def close(self, *args, **kwargs):
        return self.env.close(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def reset(self, *args, **kwargs):
        state = self.env.reset(*args, **kwargs)
        return self.__process_atari_image(state)

    def seed(self, *args, **kwargs):
        return self.env.seed(*args, **kwargs)

    @staticmethod
    def __process_atari_image(img):
        return imresize(img[35:195].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.0


class PongWrapper():
    def __init__(self, env):
        self.env = env

    def step(self, action):
        if action > 2:
            raise Exception('Unknown Action')
        if action == 1:
            action = 4
        elif action == 2:
            action = 5
        state, reward, done, info = self.env.step(action)
        state = self.__process_atari_image(state)
        return state, reward, done, info

    @property
    def action_space(self):
        return gym.spaces.discrete.Discrete(3)

    def close(self, *args, **kwargs):
        return self.env.close(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def reset(self, *args, **kwargs):
        state = self.env.reset(*args, **kwargs)
        return self.__process_atari_image(state)

    def seed(self, *args, **kwargs):
        return self.env.seed(*args, **kwargs)

    @staticmethod
    def __process_atari_image(img):
        return imresize(img[35:195].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.0
