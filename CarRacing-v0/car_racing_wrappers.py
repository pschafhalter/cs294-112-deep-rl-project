import numpy as np
import gym
from gym import spaces

import utils


class DiscreteCarRacing(gym.Wrapper):
    NUM_ACTIONS = 4
    ALLOWED_ACTIONS = [
        [-1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0.8],
        # [0, 0, 0],
    ]

    def __init__(self, env):
        super(DiscreteCarRacing, self).__init__(env)
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)

    def step(self, action):
        return self.env.step(self.ALLOWED_ACTIONS[action])


class CarRacingNoIndicators(gym.Wrapper):
    """Removes the ABS and speed indicators from the observation.

    Observation shape changes from 96x96 to 84x96.
    """
    INDICATOR_MARGIN = 12  # pixels taken up by the indicator bar

    def __init__(self, env):
        super(CarRacingNoIndicators, self).__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 96, 3), dtype=np.uint8)

    def step(self, action):
        state, step_reward, done, info = self.env.step(action)
        new_state = state[0:-self.INDICATOR_MARGIN]
        return new_state, step_reward, done, info


class CarRacing6Labels(gym.Wrapper):
    """Labels each pixel using 1-hot encoding.

    Will mislabel indicators.

    Labels:
        Unlabeled: black pixels for car tires or the edge of the map.
        Car: red pixels defining the body of the car.
        Road: gray pixels.
        Grass: green pixels.
        Road markings, red: red markings for tight turns in the road.
    """
    COLORS = np.array([
        [0, 0, 0],  # Unlabeled. Tires or out of bounds
        [204, 0, 0],  # Car
        [102, 102, 102],  # Road
        [102, 204, 102],  # Grass
        [255, 0, 0],  # Road markings
        [255, 255, 255]
    ])  # Road markings
    NUM_LABELS = 6

    def __init__(self, env):
        super(CarRacing6Labels, self).__init__(env)
        observation_shape = (env.observation_space.shape[0],
                             env.observation_space.shape[1], self.NUM_LABELS)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=observation_shape, dtype=np.uint8)

    def step(self, action):
        state, step_reward, done, info = self.env.step(action)
        new_state = utils.label_img_colors(state, self.COLORS)
        return new_state, step_reward, done, info


class CarRacingRoadLabel(gym.Wrapper):
    """Only labels the road"""
    COLORS = np.array([
        [0, 0, 0],  # Unlabeled. Tires or out of bounds
        [204, 0, 0],  # Car
        [102, 102, 102],  # Road
        [102, 204, 102],  # Grass
        [255, 0, 0],  # Road markings
        [255, 255, 255]
    ])  # Road markings
    NUM_LABELS = 1

    def __init__(self, env):
        super(CarRacingRoadLabel, self).__init__(env)
        observation_shape = (env.observation_space.shape[0],
                             env.observation_space.shape[1], self.NUM_LABELS)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=observation_shape, dtype=np.uint8)

    def step(self, action):
        state, step_reward, done, info = self.env.step(action)
        new_state = utils.label_img_colors(state, self.COLORS)
        new_state = np.expand_dims(new_state[:, :, 2], 2).astype(np.uint8)
        return new_state, step_reward, done, info


class LowResGridWrapper(gym.Wrapper):
    def __init__(self, env, resolution, mode="truth", nsamples=1000):
        super(LowResGridWrapper, self).__init__(env)
        self.resolution = resolution
        self.nsamples = nsamples
        observation_shape = resolution + (env.observation_space.shape[2], )
        self.observation_space = spaces.Box(
            low=0., high=1., shape=observation_shape, dtype=np.float32)

        if mode == "truth":
            self.to_low_res = utils.to_low_res_truth
        elif mode == "sample_regions":
            self.to_low_res = utils.to_low_res_sample_regions
        elif mode == "sample_frame":
            self.to_low_res = utils.to_low_res_sample_frame
        else:
            raise ValueError(
                "Mode must be `truth`, `sample_regions` or `sample_frame`")

    def step(self, action):
        state, step_reward, done, info = super(LowResGridWrapper,
                                               self).step(action)
        new_state = self.to_low_res(state, self.resolution, self.nsamples)

        return new_state, step_reward, done, info


class LimitMaxStepsWrapper(gym.Wrapper):
    def __init__(self, env, max_steps):
        super(LimitMaxStepsWrapper, self).__init__(env)
        self.max_steps = max_steps
        self.num_steps = 0

    def step(self, action):
        self.num_steps += 1
        state, step_reward, done, info = super(LimitMaxStepsWrapper,
                                               self).step(action)
        done = done or self.num_steps >= self.max_steps
        return state, step_reward, done, info

    def reset(self):
        self.num_steps = 0
        return super(LimitMaxStepsWrapper, self).reset()


class LowResPolarWrapper(gym.Wrapper):
    """Generates grid defined by angle and distance boundaries"""

    def __init__(self, env, angles, distances, nsamples=100, origin=(74, 48)):
        super(LowResPolarWrapper, self).__init__(env)
        self.angles = angles
        self.distances = distances
        self.nsamples = nsamples
        self.origin = origin
        observation_shape = (len(distances) - 1, len(angles) - 1,
                             env.observation_space.shape[2])
        self.observation_space = spaces.Box(
            low=0., high=1., shape=observation_shape, dtype=np.float32)

    def step(self, action):
        state, step_reward, done, info = self.env.step(action)
        new_state = utils.to_low_res_polar(state, self.origin, self.angles,
                                           self.distances, self.nsamples)
        return new_state, step_reward, done, info


class GaussianNoiseWrapper(gym.Wrapper):
    """Adds Gaussian noise to the observation.
    
    Args:
        mean: mean of the Gaussian noise.
        std: standard deviation of the Gaussian noise.
        normalize: whether to normalize observations to [0, 1] along the last
                dimension.
    """

    def __init__(self, env, mean, std, normalize=True):
        super(GaussianNoiseWrapper, self).__init__(env)
        self.mean = mean
        self.std = std
        self.normalize = normalize

    def step(self, action):
        state, step_reward, done, info = self.env.step(action)
        state += np.random.normal(self.mean, self.std, state.shape)
        state = np.clip(state, 0, None)  # No negative probabilities

        if self.normalize:
            state = state / np.sum(state, axis=2, keepdims=True)

        return state, step_reward, done, info
