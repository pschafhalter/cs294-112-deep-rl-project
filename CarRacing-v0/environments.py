import ray.tune.registry

from car_racing import CarRacing
from car_racing_wrappers import *
from atari_wrappers import *

registry = {}


def register_env(env_name, env_creator):
    registry[env_name] = env_creator
    ray.tune.registry.register_env(env_name, env_creator)


def make_low_res_car_racing_env(config):
    env = CarRacing()
    env = CarRacingNoIndicators(env)
    env = CarRacing6Labels(env)

    # Cast to tuple to facilitate loading from JSON
    resolution = tuple(config["resolution"])
    env = LowResGridWrapper(env, resolution)

    noise_mean = config.get("noise_mean", 0)
    noise_std = config.get("noise_std", 0)
    if noise_mean or noise_std:
        env = GaussianNoiseWrapper(env, noise_mean, noise_std, normalize=True)

    env = DiscreteCarRacing(env)
    env = NoopResetEnv(env)
    env.override_num_noops = 50
    if config.get("framestack", True):
        env = FrameStack(env, 4)
    env = StopOnMinReward(env, -5)
    env = LimitMaxStepsWrapper(env, 1500)

    rollout_directory = config.get("rollout_directory", "/tmp/rollouts")
    env = gym.wrappers.Monitor(env, rollout_directory, resume=True)
    return env


register_env("low-res-car-racing", make_low_res_car_racing_env)


def make_low_res_car_racing_road_label_env(config):
    env = CarRacing()
    env = CarRacingNoIndicators(env)
    env = CarRacingRoadLabel(env)

    # Cast to tuple to facilitate loading from JSON
    resolution = tuple(config["resolution"])
    env = LowResGridWrapper(env, resolution)

    noise_mean = config.get("noise_mean", 0)
    noise_std = config.get("noise_std", 0)
    if noise_mean or noise_std:
        env = GaussianNoiseWrapper(env, noise_mean, noise_std, normalize=True)

    env = DiscreteCarRacing(env)
    env = NoopResetEnv(env)
    env.override_num_noops = 50
    if config.get("framestack", True):
        env = FrameStack(env, 4)
    env = StopOnMinReward(env, -5)
    env = LimitMaxStepsWrapper(env, 1500)

    rollout_directory = config.get("rollout_directory", "/tmp/rollouts")
    env = gym.wrappers.Monitor(env, rollout_directory, resume=True)
    return env


register_env("low-res-car-racing-road-label",
             make_low_res_car_racing_road_label_env)


def make_polar_car_racing_road_label_env(config):
    env = CarRacing()
    env = CarRacingNoIndicators(env)
    env = CarRacingRoadLabel(env)

    angles = config["angles"]
    distances = config["distances"]
    nsamples = config.get("nsamples_per_region", 100)

    env = LowResPolarWrapper(env, angles, distances, nsamples)

    noise_mean = config.get("noise_mean", 0)
    noise_std = config.get("noise_std", 0)
    if noise_mean or noise_std:
        env = GaussianNoiseWrapper(env, noise_mean, noise_std, normalize=True)

    env = DiscreteCarRacing(env)
    env = NoopResetEnv(env)
    env.override_num_noops = 50
    if config.get("framestack", True):
        env = FrameStack(env, 4)
    env = StopOnMinReward(env, -5)
    env = LimitMaxStepsWrapper(env, 1500)

    rollout_directory = config.get("rollout_directory", "/tmp/rollouts")
    env = gym.wrappers.Monitor(env, rollout_directory, resume=True)
    return env


register_env("polar-car-racing-road-label",
             make_polar_car_racing_road_label_env)
