# Copyright 2019 ducandu GmbH, All Rights Reserved.
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

from surreal.components.distribution_adapters.normal_distribution_adapter import NormalDistributionAdapter
from surreal.spaces import Float


class NormalMixtureDistributionAdapter(NormalDistributionAdapter):
    def __init__(self, action_space, num_mixtures=1, **kwargs):
        """
        Args:
            num_mixtures (int): The mixture's size (number of sub-distributions to categorically sample from).
                Default: 1 (no mixture).
        """
        super(NormalMixtureDistributionAdapter, self).__init__(action_space, **kwargs)
        self.num_mixtures = num_mixtures

    def get_units_and_shape(self):
        if self.num_mixtures == 1:
            return super(NormalMixtureDistributionAdapter, self).get_units_and_shape()

        new_shape = list(self.output_space.get_shape(with_category_rank=True))
        last_dim = new_shape[-1]
        # num_mixtures=categorical nodes + num_mixtures * "normal" outputs (mean + std = 2 nodes)
        new_shape = tuple(new_shape[:-1] + [self.num_mixtures + self.num_mixtures * 2 * new_shape[-1]])
        units = self.num_mixtures + self.num_mixtures * last_dim * 2

        return units, new_shape

    def get_parameters_from_adapter_outputs(self, adapter_outputs):
        # Shortcut: If no mixture distribution, let DistributionAdapter parent deal with everything.
        if self.num_mixtures == 1:
            return super(NormalMixtureDistributionAdapter, self).get_parameters_from_adapter_outputs(
                adapter_outputs
            )

        # Continuous actions.
        # For now, assume unbounded outputs.
        assert isinstance(self.output_space, Float) and self.output_space.unbounded

        parameters = {}

        # Nodes encode the following:
        # - [num_mixtures] (for categorical)
        # - [num_mixtures * 2 * last-action-dim] (for each item in the mix: 1 mean node, 1 log-std node)

        # Unbounded -> Mixture Multivariate Normal distribution.
        last_dim = self.output_space.get_shape()[-1]
        categorical, means, log_sds = tf.split(adapter_outputs, num_or_size_splits=[
            self.num_mixtures, self.num_mixtures * last_dim, self.num_mixtures * last_dim
        ], axis=-1)

        # Parameterize the categorical distribution, which will pick one of the mixture ones.
        parameters["categorical"] = categorical

        # Split into one for each item in the Mixture.
        means = tf.split(means, num_or_size_splits=self.num_mixtures, axis=-1)
        # Turn log sd into sd to ascertain always positive stddev values.
        sds = tf.split(tf.math.exp(log_sds), num_or_size_splits=self.num_mixtures, axis=-1)

        # Store each mixture item's parameters in DataOpDict.
        for i in range(self.num_mixtures):
            mean = means[i]
            sd = sds[i]
            parameters["parameters{}".format(i)] = tuple([mean, sd])

        return parameters
