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

import tensorflow as tf

from surreal.components.distribution_adapters.distribution_adapter import DistributionAdapter


class PlainOutputAdapter(DistributionAdapter):
    """
    Output adapter for simple Float output Spaces.
    """
    def __init__(self, output_space, weights_spec=None, biases_spec=None, activation=None,
                 pre_network=None, squashing_function="sigmoid"):
        super().__init__(output_space, weights_spec, biases_spec, activation, pre_network)
        self.squashing_function = squashing_function
        assert self.squashing_function in ["sigmoid", "tanh"]

    def get_units_and_shape(self):
        units = self.output_space.flat_dim
        new_shape = self.output_space.get_shape(include_main_axes=True)
        new_shape = tuple([i if i is not None else -1 for i in new_shape])
        return units, new_shape

    def get_parameters_from_adapter_outputs(self, adapter_outputs):
        if self.output_space.low != float("-inf"):
            # Squash values using our squashing function.
            if self.output_space.high != float("inf"):
                if self.squashing_function == "sigmoid":
                    return (tf.math.sigmoid(adapter_outputs) * (self.output_space.high - self.output_space.low)) + \
                           self.output_space.low
                else:
                    return ((tf.math.tanh(adapter_outputs) + 1.0) *
                            (self.output_space.high - self.output_space.low) / 2.0) + self.output_space.low
            # TODO: one-sided bounds.
            else:
                raise NotImplementedError
        # TODO: one-sided bounds.
        elif self.output_space.high != float("inf"):
            raise NotImplementedError
        else:
            return adapter_outputs
