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

from math import log
import tensorflow as tf

from surreal.components.distribution_adapters.distribution_adapter import DistributionAdapter
from surreal.utils.util import SMALL_NUMBER


class BetaDistributionAdapter(DistributionAdapter):
    """
    Action adapter for the Beta distribution
    """
    def get_units_and_shape(self):
        # Add moments (2x for each action item).
        units = 2 * self.output_space.flat_dim
        if self.output_space.shape == ():
            new_shape = self.output_space.get_shape(include_main_axes=True) + (2,)
        else:
            shape = self.output_space.get_shape(include_main_axes=True)
            new_shape = tuple(list(shape[:-1]) + [shape[-1] * 2])

        new_shape = tuple([i if i is not None else -1 for i in new_shape])
        return units, new_shape

    def get_parameters_from_adapter_outputs(self, adapter_outputs):
        # Stabilize both alpha and beta (currently together in last_nn_layer_output).
        parameters = tf.clip_by_value(
            adapter_outputs, clip_value_min=log(SMALL_NUMBER), clip_value_max=-log(SMALL_NUMBER)
        )
        parameters = tf.math.log((tf.math.exp(parameters) + 1.0)) + 1.0
        alpha, beta = tf.split(parameters, num_or_size_splits=2, axis=-1)

        if self.output_space.shape == ():
            alpha = tf.squeeze(alpha, axis=-1)
            beta = tf.squeeze(beta, axis=-1)

        return tuple([alpha, beta])
