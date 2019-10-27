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

import numpy as np
import tensorflow as tf

from surreal.components.distribution_adapters.distribution_adapter import DistributionAdapter
from surreal.spaces import Float


class MixtureDistributionAdapter(DistributionAdapter):

    def __init__(self, output_space, *sub_adapters, **kwargs):
        """
        Args:
            sub_adapters (Union[DistributionAdapter,dict]): The sub-Adapters' specs.
        """
        super(MixtureDistributionAdapter, self).__init__(output_space, **kwargs)
        self.sub_adapters = [DistributionAdapter.make(s) for s in sub_adapters]
        self.num_mixtures = len(self.sub_adapters)

    def get_units_and_shape(self):
        sub_adapters_units_and_shapes = [s.get_units_and_shape() for s in self.sub_adapters]

        new_shape = list(self.output_space.get_shape(with_category_rank=True))
        # num_mixtures=categorical nodes + sub-adapter's nodes.
        new_shape = tuple(new_shape[:-1] + [self.num_mixtures + np.sum(sub_adapters_units_and_shapes[:, 0])])
        units = int(self.num_mixtures + np.sum(sub_adapters_units_and_shapes[:, 0]))

        return units, new_shape

    def get_parameters_from_adapter_outputs(self, adapter_outputs):
        # Continuous actions.
        # For now, assume unbounded outputs.
        assert isinstance(self.output_space, Float) and self.output_space.unbounded

        # Nodes encode the following:
        # - [num_mixtures] (for categorical)
        # - [rest] (for each item in the mix)

        # Assume that all sub-distribution shave the same type (and thus use the same number of outputs).
        split = tf.split(adapter_outputs, num_or_size_splits=[self.num_mixtures, adapter_outputs.shape[-1] / self.num_mixtures], axis=-1)

        # Parameterize the categorical distribution, which will pick one of the mixture ones.
        parameters = {"categorical": split[0]}
        # Get parameters of sub-adapters.
        for i, s in enumerate(self.sub_adapters):
            parameters["parameters{}".format(i)] = s.get_parameters_from_adapter_outputs(split[i+1])

        return parameters
