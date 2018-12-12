#!/usr/bin/env python

import argparse
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

import environments
import models


def visualize_grid(observation):
    if observation.shape[-1] > 1:
        img = observation[:, :, 2]  # Gets road label
    else:
        img = observation[:, :, 0]
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("car_racing", img)
    cv2.waitKey(0)


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

if __name__ == "__main__":
    import time

    env = environments.make_low_res_car_racing_road_label_env({"resolution": (6,6)})
    obs = env.reset()
    for _ in range():
        obs, _, __, ___ = env.step(2)
        env.render()

    visualize_grid(obs)
