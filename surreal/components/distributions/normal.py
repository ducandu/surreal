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

from surreal.components.distributions.distribution import Distribution


class Normal(Distribution):
    """
    A Gaussian Normal distribution object defined by a tuple: mean, variance,
    which is the same as "loc_and_scale".
    """
    def parameterize_distribution(self, parameters):
        """
        Args:
            parameters (DataOpTuple): Tuple holding the mean and stddev parameters.
        """
        # Must be a Tuple of len 2 (loc and scale).
        return tfp.distributions.Normal(loc=parameters[0], scale=parameters[1])

    def _sample_deterministic(self, distribution):
        return distribution.mean()
