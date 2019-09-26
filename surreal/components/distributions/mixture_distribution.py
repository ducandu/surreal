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
    def __init__(self, *sub_distributions):
        """
        Args:
            sub_distributions (List[Union[string,Distribution]]): The type-strings or actual Distribution objects
                that define the n sub-distributions of this MixtureDistribution.
        """
        super(MixtureDistribution, self).__init__()

        self.sub_distributions = []
        for i, s in enumerate(sub_distributions):
            if isinstance(s, str):
                self.sub_distributions.append(Distribution.make(
                    {"type": s, "scope": "sub-distribution-{}".format(i)}
                ))
            else:
                self.sub_distributions.append(Distribution.make(s))

        self.categorical = Categorical()

        #self.add_components(self.categorical, *self.sub_distributions)

    def parameterize_distribution(self, parameters):
        """
        Args:
            parameters (DataOpDict): The parameters to use for parameterizations of the different sub-distributions
                including the main-categorical one. Keys must be "categorical", "parameters0", "parameters1", etc..

        Returns:
            tfp.Distribution: The ready-to-be-sampled mixed distribution.
        """
        # Must be a Dict with keys: 'categorical', 'parameters0', 'parameters1', etc...
        if "categorical" not in parameters:
            raise SurrealError("`parameters` for MixtureDistribution needs key: 'categorical'!")
        for i, s in enumerate(self.sub_distributions):
            sub_space = parameters.get("parameters{}".format(i))
            if sub_space is None:
                raise SurrealError("`parameters` for Mixed needs key: 'parameters{}'!".format(i))

        components = []
        for i, s in enumerate(self.sub_distributions):
            components.append(s.get_distribution(parameters["parameters{}".format(i)]))

        return tfp.distributions.Mixture(
            cat=self.categorical.get_distribution(parameters["categorical"]),
            components=components
        )

    def _sample_deterministic(self, distribution):
        return distribution.mean()
