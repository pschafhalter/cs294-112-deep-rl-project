import tensorflow as tf
from tensorflow import layers

from ray.rllib.models import ModelCatalog, Model
from ray.rllib.models.misc import (
        flatten, normc_initializer, get_activation_fn)


class FullyConnectedNetwork(Model):
    """Fully connected model with variable number of hidden layers.
    
    Set `fcnet_hiddens` (list of integers) to define the hidden layers.
    Set `fcnet_activation` (string, defaults to "relu") to define the
        activation function.

    Based on Ray RLlib's FullyConnectedNetwork implementation.
    """
    def _build_layers(self, inputs, num_outputs, options):
        hiddens = options.get("fcnet_hiddens", [256, 256])
        activation = get_activation_fn(options.get("fcnet_activation", "relu"))

        with tf.name_scope("fc_net"):
            last_layer = flatten(inputs)
            for size in hiddens:
                last_layer = layers.dense(
                        last_layer,
                        size,
                        kernel_initializer=normc_initializer(1.0),
                        activation=activation)
            output = layers.dense(
                    last_layer,
                    num_outputs,
                    kernel_initializer=normc_initializer(1.0),
                    activation=None)
            return output, last_layer


ModelCatalog.register_custom_model("FullyConnectedNetwork", FullyConnectedNetwork)


class ReducedKhanElibolModel(Model):
    """1 conv layer followed by a 256 FC layer.

    Original model uses 2 conv layers."""
    def _build_layers(self, inputs, num_outputs, options):
        with tf.name_scope("KhanElibolModel"):
            last_layer = layers.conv2d(
                    inputs,
                    32,
                    (4, 4),
                    activation=tf.nn.relu)

            last_layer = flatten(last_layer)
            last_layer = layers.dense(
                    last_layer,
                    256,
                    kernel_initializer=normc_initializer(0.01),
                    activation = tf.nn.relu)
            output = layers.dense(
                    last_layer,
                    num_outputs,
                    kernel_initializer=normc_initializer(0.01),
                    activation = None)
            return output, last_layer


ModelCatalog.register_custom_model("ReducedKhanElibolModel", ReducedKhanElibolModel)


class LowResKhanElibolModel(Model):
    """2 conv layers followed by a 256 FC layer.

    inputs -> 4x4 conv w/ 16 outputs -> 2x2 conv w/ 32 outputs.
    Relu activation for all layers
    """
    def _build_layers(self, inputs, num_outputs, options):
        with tf.name_scope("KhanElibolModel"):
            last_layer = layers.conv2d(
                    inputs,
                    16,
                    (4, 4),
                    activation=tf.nn.relu)

            last_layer = layers.conv2d(
                    last_layer,
                    32,
                    (2, 2),
                    activation=tf.nn.relu)

            last_layer = flatten(last_layer)
            last_layer = layers.dense(
                    last_layer,
                    256,
                    kernel_initializer=normc_initializer(0.01),
                    activation = tf.nn.relu)
            output = layers.dense(
                    last_layer,
                    num_outputs,
                    kernel_initializer=normc_initializer(0.01),
                    activation = None)
            return output, last_layer


ModelCatalog.register_custom_model("LowResKhanElibolModel", LowResKhanElibolModel)
