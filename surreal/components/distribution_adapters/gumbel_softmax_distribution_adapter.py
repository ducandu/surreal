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

from surreal.components.distribution_adapters.distribution_adapter import DistributionAdapter


class GumbelSoftmaxDistributionAdapter(DistributionAdapter):
    """
    Action adapter for the GumbelSoftmax distribution.
    """
    def get_units_and_shape(self):
        units = self.output_space.flat_dim_with_categories  #+ 1  # +1 temperature node
        new_shape = self.output_space.get_shape(include_main_axes=True, with_category_rank=True)
        new_shape = [i if i is not None else -1 for i in new_shape]
        #new_shape[-1] += 1  # Add the temperature node.
        return units, tuple(new_shape)

    def get_parameters_from_adapter_outputs(self, adapter_outputs):
        return adapter_outputs
        # For Gumbel, we also learn the temperature parameter.
        #log_temp, logits = tf.split(
        #    adapter_outputs, num_or_size_splits=(1, adapter_outputs.shape.as_list()[-1] - 1), axis=-1
        #)
        #log_temp = tf.squeeze(log_temp, axis=-1)
        #log_temp = tf.clip_by_value(log_temp, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)
        ## Turn log sd into sd to ascertain always positive stddev values.
        #temp = tf.math.exp(log_temp)

        # Return temp and logits for GumbelSoftmaxDistribution.
        #return tuple([temp, logits])
