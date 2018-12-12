import ray
from ray.tune import run_experiments, grid_search

from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.misc import flatten
import tensorflow as tf
from tensorflow import layers


class LowResModel(Model):
    def _build_layers(self, inputs, num_outputs, options):
        with tf.name_scope("fc_net"):
            last_layer = layers.conv2d(
                    inputs,
                    16,
                    (4, 4),
                    activation=tf.nn.relu)
            last_layer = layers.conv2d(
                    inputs,
                    32,
                    (4, 4),
                    activation=tf.nn.relu)
            last_layer = flatten(last_layer)
            output = tf.layers.dense(last_layer, num_outputs, activation=None)
            return output, last_layer

ModelCatalog.register_custom_model("LowResModel", LowResModel)


if __name__ == "__main__":
    ray.init()
    run_experiments({
        "low-res-car-pong": {
            "run": "A3C",
            "env": "PongNoFrameskip-v4",
            "checkpoint_freq": 100,
            "config": {
                "num_gpus": 0.33,
                "num_workers": 3,
                "model": {
                    "custom_model": "LowResModel",
                },
                # "model": {"dim": grid_search([84, 42, 21, 10])},
            },
            "stop" : {
                "episode_reward_mean": 1000
            },
            }
        })
