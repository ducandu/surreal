# Copyright 2019 ducandu GmbH, All Rights Reserved
# (this is a modified version of the Apache 2.0 licensed RLgraph file of the same name).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import copy
import tensorflow as tf

from surreal.components.models.model import Model
from surreal.spaces import ContainerSpace
from surreal.utils.keras import keras_from_spec


class DistributionAdapter(Model):
    """
    A Component that cleans up a neural network's flat output and gets it ready for parameterizing a
    Distribution Component.

    Processing steps include:
    - Sending the raw, flattened NN output through a Dense layer whose number of units matches the flattened
    output space.
    - Reshaping (according to the output Space).
    - Translating the reshaped outputs into distribution parameters.
    """
    def __init__(self, output_space, kernel_initializer="glorot_uniform", bias_initializer="zeros", activation=None,
                 pre_network=None):
        """
        Args:
            output_space (Optional[Space]): The output Space within which this Component will create parameters.

            kernel_initializer (Optional[any]): An optional Initializer spec that will be used to initialize the
                weights of `self.output_layer`. Default: None (use default initializer, which is Xavier).

            bias_initializer (Optional[any]): An optional Initializer spec that will be used to initialize the
                biases of `self.output_layer`. Default: None (use default initializer, which is 0.0).

            activation (Optional[str]): The activation function to use for `self.output_layer`.
                Default: None (=linear).

            pre_network (Optional[tf.keras.models.Model]): A keras Model coming before the
                last action layer. If None, only the action layer itself is applied.
        """
        super().__init__()

        # Build the action layer for this adapter based on the given action-space.
        self.output_space = output_space.with_batch()
        assert not isinstance(self.output_space, ContainerSpace),\
            "ERROR: DistributionAdapter cannot handle ContainerSpaces!"

        units, self.final_shape = self.get_units_and_shape()
        assert isinstance(units, int) and units > 0, "ERROR: `units` must be int and larger 0!"

        self.output_layer = tf.keras.layers.Dense(
            units=units,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        )

        # Do we have a pre-NN?
        self.pre_network = keras_from_spec(pre_network)

    def get_units_and_shape(self):
        """
        Returns the number of units in the layer that will be added and the shape of the output according to the
        action space.

        Returns:
            Tuple:
                int: The number of units for the action layer.
                shape: The final shape for the output space.
        """
        raise NotImplementedError

    def get_parameters_from_adapter_outputs(self, inputs):
        """
        Args:
            inputs (tf.Tensor): The adapter layer output(s) already reshaped.

        Returns:
            tuple:
                parameters (tf.Tensor): The parameters, ready to be passed to a Distribution object's
                    get_distribution method (usually some probabilities or loc/scale pairs).
                probs (tf.Tensor): probs in categorical case. In all other cases: None.
                log_probs (tf.Tensor): log(probs) in categorical case. In all other cases: None.
        """
        raise NotImplementedError

    def call(self, inputs):
        if self.pre_network is not None:
            raw_out = self.output_layer(self.pre_network(inputs))
        else:
            raw_out = self.output_layer(inputs)

        # Reshape the output according to our output_space.
        reshaped = tf.reshape(raw_out, self.final_shape)

        # Return parameters.
        return self.get_parameters_from_adapter_outputs(reshaped)

    def copy(self, trainable=None):
        # Hide non-copyable members.
        output_layer = self.output_layer
        pre_network = self.pre_network
        self.output_layer = None
        self.pre_network = None

        # Do the copy.
        copy_ = copy.deepcopy(self)
        copy_.output_layer = self.clone_component(output_layer, trainable=trainable)
        if pre_network is not None:
            copy_.pre_network = self.clone_component(pre_network, trainable=trainable)

        # Put everything back in place and clone keras models.
        self.output_layer = output_layer
        self.pre_network = pre_network

        # Do a sample call to build the copy, then sync weights.
        if self.init_args is not None:
            copy_(*self.init_args, **self.init_kwargs)
            copy_.sync_from(self, tau=1.0)

        return copy_

    def _get_weights_list(self):
        return (self.pre_network.variables if self.pre_network is not None else []) + \
                self.output_layer.variables
