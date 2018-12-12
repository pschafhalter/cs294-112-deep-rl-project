#!/usr/bin/env python

import argparse
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

import ray
from ray.rllib.agents.agent import get_agent_class

import environments
import models


def visualize_grid(observation):
    if observation.shape[-1] > 1:
        img = observation[:, :, 2]  # Gets road label
    else:
        img = observation[:, :, 0]
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("car_racing", img)
    cv2.waitKey(1)


def visualize_polar(distances, angles, observation):
    if observation.shape[-1] == 4:  # hack for framestack
        img = observation[:, :, 0]
    elif observation.shape[-1] > 1:
        img = observation[:, :, 2]  # Gets road label
    else:
        img = observation[:, :, 0]

    plt.ion()

    if not hasattr(visualize_polar, "fig"):
        visualize_polar.fig = plt.figure()
    else:
        visualize_polar.ax.remove()
    visualize_polar.ax = visualize_polar.fig.add_subplot(
        111, projection="polar", label=str(np.random.random()))
    visualize_polar.ax.set_theta_direction(-1)
    visualize_polar.ax.set_theta_zero_location("N")

    for i, (dist_min, dist_max) in enumerate(
            zip(distances[-2::-1], distances[:0:-1])):
        for j, (angle_min, angle_max) in enumerate(
                zip(angles[:-1], angles[1:])):
            width = angle_max - angle_min
            angle = (angle_min + angle_max) / 2
            height = dist_max - dist_min
            visualize_polar.ax.bar(
                angle,
                height,
                width=width,
                bottom=dist_min,
                alpha=img[i, j],
                color="blue")

    visualize_polar.fig.canvas.draw()
    visualize_polar.fig.canvas.flush_events()


def rollout(env,
            env_config,
            agent,
            render_env=False,
            render_observation=False,
            max_steps=1500):
    observation = env.reset()
    done = False
    reward_total = 0.0
    steps = 0
    while not done and steps < max_steps:
        action = agent.compute_action(observation)
        observation, reward, done, _ = env.step(action)
        reward_total += reward
        steps += 1
        if render_env:
            env.render()
        if render_observation:
            if "angles" in env_config:
                visualize_polar(env_config["distances"], env_config["angles"],
                                observation)
            else:
                visualize_grid(observation)

    return reward_total


def run_rollouts(agent,
                 env_creator,
                 env_config,
                 num_rollouts=1,
                 render_env=False,
                 render_observation=False,
                 max_steps=1500):

    env = env_creator(env_config)

    reward_totals = [
        rollout(env, env_config, agent, render_env, render_observation,
                max_steps) for _ in range(num_rollouts)
    ]

    return reward_totals


def run(agent_name,
        config,
        checkpoint,
        num_rollouts=1,
        render_env=False,
        render_observation=False,
        max_steps=1500):
    cls = get_agent_class(agent_name)
    agent = cls(config=config)
    agent.restore(checkpoint)

    env_creator = environments.registry[config["env"]]
    env_config = config.get("env_config", {})

    reward_totals = run_rollouts(agent, env_creator, env_config, num_rollouts,
                                 render_env, render_observation, max_steps)

    print(reward_totals)
    print("mean = {}, std = {}".format(
        np.mean(reward_totals), np.std(reward_totals)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("agent", type=str)
    parser.add_argument("config_filename", type=str)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--rollout_dir", type=str, default="/tmp/rollouts")
    parser.add_argument(
        "--render_env", default=False, action="store_const", const=True)
    parser.add_argument(
        "--render_obs", default=False, action="store_const", const=True)

    args = parser.parse_args()

    with open(args.config_filename, "r") as f:
        config = json.load(f)

    config["num_workers"] = 1

    env_config = config.get("env_config", {})
    env_config["rollout_directory"] = args.rollout_dir
    config["env_config"] = env_config

    ray.init()

    run(args.agent, config, args.checkpoint, args.num_rollouts,
        args.render_env, args.render_obs, args.max_steps)
