import ray
from ray.tune import run_experiments, grid_search

from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.misc import flatten
import tensorflow as tf
import tensorflow.contrib.slim as slim


class LowResModel(Model):
    def _build_layers(self, inputs, num_outputs, options):
        dim = options["dim"]
        if dim == 84:
            filters = [
                [16, [8, 8], 4],
                [32, [4, 4], 2],
                [256, [11, 11], 1],
            ]
        elif dim == 42:
            filters = [
                [16, [4, 4], 2],
                [32, [4, 4], 2],
                [256, [11, 11], 1],
            ]
        elif dim == 21:
            filters = [
                [16, [2, 2], 1],
                [32, [4, 4], 2],
                [256, [11, 11], 1],
            ]

        activation = tf.nn.relu

        with tf.name_scope("vision_net"):
            for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
                inputs = slim.conv2d(
                    inputs,
                    out_size,
                    kernel,
                    stride,
                    activation_fn=activation,
                    scope="conv{}".format(i))
            out_size, kernel, stride = filters[-1]
            fc1 = slim.conv2d(
                inputs,
                out_size,
                kernel,
                stride,
                activation_fn=activation,
                padding="VALID",
                scope="fc1")
            fc2 = slim.conv2d(
                fc1,
                num_outputs, [1, 1],
                activation_fn=None,
                normalizer_fn=None,
                scope="fc2")
            return flatten(fc2), flatten(fc1)

ModelCatalog.register_custom_model("LowResModel", LowResModel)


if __name__ == "__main__":
    ray.init()
    run_experiments({
        "low-res-pong-v3": {
            "run": "A3C",
            "env": "PongNoFrameskip-v4",
            # "checkpoint_freq": 100,
            "config": {
                "num_workers": 8,
                "model": {
                    "dim": grid_search([84, 42, 21]),
                    "custom_model": "LowResModel",
                },
            },
            "stop" : {
                "episode_reward_mean": 20
            },
            }
        })
