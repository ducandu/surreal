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
from surreal.utils.util import MIN_LOG_STDDEV, MAX_LOG_STDDEV


class SquashedNormalDistributionAdapter(DistributionAdapter):
    """
    Action adapter for the Squashed-normal distribution.
    """
    def get_units_and_shape(self):
        # Add moments (2x for each action item).
        units = 2 * self.output_space.flat_dim  # Those two dimensions are the mean and log sd
        if self.output_space.shape == ():
            new_shape = self.output_space.get_shape(include_main_axes=True) + (2,)
        else:
            shape = self.output_space.get_shape(include_main_axes=True)
            new_shape = tuple(list(shape[:-1]) + [shape[-1] * 2])
        new_shape = tuple([i if i is not None else -1 for i in new_shape])
        return units, new_shape

    def get_parameters_from_adapter_outputs(self, adapter_outputs):
        mean, log_sd = tf.split(adapter_outputs, num_or_size_splits=2, axis=-1)
        log_sd = tf.clip_by_value(log_sd, MIN_LOG_STDDEV, MAX_LOG_STDDEV)
        # Turn log sd into sd to ascertain always positive stddev values.
        sd = tf.math.exp(log_sd)

        return tuple([mean, sd])
