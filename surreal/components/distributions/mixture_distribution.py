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

import tensorflow_probability as tfp

from surreal.components.distributions.categorical import Categorical
from surreal.components.distributions.distribution import Distribution
from surreal.utils.errors import SurrealError


class MixtureDistribution(Distribution):
    """
    A mixed distribution of n sub-distribution components and a categorical which determines,
    from which sub-distribution we sample.
    """
    def __init__(self, *sub_distributions, num_experts=None):
        """
        Args:
            sub_distributions (List[Union[string,Distribution]]): The type-strings or actual Distribution objects
                that define the n sub-distributions of this MixtureDistribution.

            num_experts (Optional[int]): If provided and len(`sub_distributions`) == 1, clone the given single
                sub_distribution `num_experts` times to get all sub_distributions.
        """
        super(MixtureDistribution, self).__init__()

        self.sub_distributions = []
        # Default is some Normal.
        if len(sub_distributions) == 0:
            sub_distributions = ["normal"]
        # If only one given AND num_experts is provided, clone the sub_distribution config.
        if len(sub_distributions) == 1 and num_experts is not None:
            self.sub_distributions = [Distribution.make(
                {"type": sub_distributions[0]} if isinstance(sub_distributions[0], str) else sub_distributions[0]
            ) for _ in range(num_experts)]
        # Sub-distributions are given as n single configs.
        else:
            for i, s in enumerate(sub_distributions):
                self.sub_distributions.append(Distribution.make({"type": s} if isinstance(s, str) else s))

        # The categorical distribution to pick from our n experts when sampling.
        self.categorical = Categorical()

    def parameterize_distribution(self, parameters):
        """
        Args:
            parameters (DataOpDict): The parameters to use for parameterizations of the different sub-distributions
                including the main-categorical one. Keys must be "categorical", "parameters0", "parameters1", etc..

        Returns:
            tfp.Distribution: The ready-to-be-sampled mixed distribution.
        """
        # Must be a Dict with keys: 'categorical', 'parameters0', 'parameters1', etc...
        assert "categorical" in parameters, "`parameters` for MixtureDistribution needs key: 'categorical'!"
        assert parameters["categorical"].shape[-1] == len(self.sub_distributions), \
            "`categorical` parameters does not have same size as len(`self.sub_distributions`)!"
        for i, s in enumerate(self.sub_distributions):
            assert "parameters{}".format(i) in parameters, \
                "`parameters` for MixtureDistribution needs key: 'parameters{}'!".format(i)

        components = []
        for i, s in enumerate(self.sub_distributions):
            components.append(s.parameterize_distribution(parameters["parameters{}".format(i)]))

        return tfp.distributions.Mixture(
            cat=self.categorical.parameterize_distribution(parameters["categorical"]),
            components=components
        )

    def _sample_deterministic(self, distribution):
        return distribution.mean()
