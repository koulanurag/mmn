import numpy as np
import gym
from scipy.misc import imresize


class AtariWrapper():
    def __init__(self, env):
        self.env = env
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.spec = self.env.spec

    def step(self, *args, **kwargs):
        state, reward, done, info = self.env.step(*args, **kwargs)
        info['org_obs'] = state
        state = self.process_atari_image(state)
        return state, reward, done, info

    @property
    def action_space(self):
        return self.env.action_space

    def close(self, *args, **kwargs):
        return self.env.close(*args, **kwargs)

    def render(self, mode='human', inspect=False, img=None):
        if not inspect:
            return self.env.render(mode)
        else:
            if mode == 'rgb_array':
                return img
            elif mode == 'human':
                from gym.envs.classic_control import rendering
                if self.env.env.viewer is None:
                    self.env.env.viewer = rendering.SimpleImageViewer()
                self.env.env.viewer.imshow(img)
                return self.env.env.viewer.isopen

    def reset(self, inspect=False):
        state = self.env.reset()
        if inspect:
            return self.process_atari_image(state), state
        else:
            return self.process_atari_image(state)

    def seed(self, *args, **kwargs):
        return self.env.seed(*args, **kwargs)

    @staticmethod
    def process_atari_image(img):
        return imresize(img[5:195].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.0


class Crop35And195(AtariWrapper):
    def __init__(self, env):
        AtariWrapper.__init__(self, env)

    @staticmethod
    def process_atari_image(img):
        return imresize(img[35:195].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.0


class Crop15And195(AtariWrapper):
    def __init__(self, env):
        AtariWrapper.__init__(self, env)

    @staticmethod
    def process_atari_image(img):
        return imresize(img[15:195].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.0


class PongWrapper(Crop35And195):
    def __init__(self, env):
        Crop35And195.__init__(self, env)

    def step(self, action):
        if action > 2:
            raise Exception('Unknown Action')
        if action == 1:
            action = 4
        elif action == 2:
            action = 5
        state, reward, done, info = self.env.step(action)
        info['org_obs'] = state
        state = self.process_atari_image(state)
        return state, reward, done, info

    @property
    def action_space(self):
        return gym.spaces.discrete.Discrete(3)


class SpaceInvaderWrapper(Crop15And195):
    def __init__(self, env):
        Crop15And195.__init__(self, env)

    @property
    def action_space(self):
        return gym.spaces.discrete.Discrete(4)


class EnduroWrapper(AtariWrapper):
    def __init__(self, env):
        AtariWrapper.__init__(self, env)

    @staticmethod
    def process_atari_image(img):
        return imresize(img[0:155, 10:].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.0


class BeamRiderWrapper(AtariWrapper):
    def __init__(self, env):
        AtariWrapper.__init__(self, env)

    @staticmethod
    def process_atari_image(img):
        return imresize(img[30:180, 10:].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.0


class FreewayWrapper(AtariWrapper):
    def __init__(self, env):
        AtariWrapper.__init__(self, env)

    @staticmethod
    def process_atari_image(img):
        return imresize(img[25:195, 10:].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.0


class BoxingWrapper(AtariWrapper):
    def __init__(self, env):
        AtariWrapper.__init__(self, env)

    @staticmethod
    def process_atari_image(img):
        return imresize(img[15:180, 30:130].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.0


class BreakoutWrapper(AtariWrapper):
    def __init__(self, env):
        AtariWrapper.__init__(self, env)

    @staticmethod
    def process_atari_image(img):
        return imresize(img[35:195, 10:150].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.0


class QbertWrapper(AtariWrapper):
    def __init__(self, env):
        AtariWrapper.__init__(self, env)

    @staticmethod
    def process_atari_image(img):
        return imresize(img[30:190, 10:150].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.0


class BowlingWrapper(AtariWrapper):
    def __init__(self, env):
        AtariWrapper.__init__(self, env)

    @property
    def action_space(self):
        return gym.spaces.discrete.Discrete(6)

    @staticmethod
    def process_atari_image(img):
        return imresize(img[105:172, :].mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.0


class ElevatorActionWrapper(AtariWrapper):
    def __init__(self, env):
        AtariWrapper.__init__(self, env)






def atari_wrapper(env_name):
    x = env_name.lower()
    x = x.split('deterministic')[0] if 'deterministic' in x else x.split('-')[0]
    if x == 'pong':
        env = PongWrapper(gym.make(env_name))
    elif x == 'spaceinvaders':
        env = SpaceInvaderWrapper(gym.make(env_name))
    elif x == 'enduro':
        env = EnduroWrapper(gym.make(env_name))
    elif x == 'beamrider':
        env = BeamRiderWrapper(gym.make(env_name))
    elif x == 'freeway':
        env = FreewayWrapper(gym.make(env_name))
    elif x == 'boxing':
        env = BoxingWrapper(gym.make(env_name))
    elif x == 'breakout':
        env = BreakoutWrapper(gym.make(env_name))
    elif x == 'qbert':
        env = QbertWrapper(gym.make(env_name))
    elif x in ['bowling']:
        env = BowlingWrapper(gym.make(env_name))
    # elif x in ['qbert']:
    #     env = Crop15And195(gym.make(env_name))
    elif x in ['elevatoraction']:
        env = ElevatorActionWrapper(gym.make(env_name))


    else:
        env = AtariWrapper(gym.make(env_name))
    return env
