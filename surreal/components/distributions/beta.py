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


class Beta(Distribution):
    """
    A Beta distribution is defined on the interval [0, 1] and parameterized by shape parameters
    alpha and beta (also called concentration parameters).

    PDF(x; alpha, beta) = x**(alpha - 1) (1 - x)**(beta - 1) / Z
        with Z = Gamma(alpha) Gamma(beta) / Gamma(alpha + beta)
        and Gamma(n) = (n - 1)!

    """
    def __init__(self, low=0.0, high=1.0):
        super(Beta, self).__init__()

        self.low = low
        self.high = high

    #def check_input_spaces(self, input_spaces, action_space=None):
    #    # Must be a Tuple of len 2 (alpha and beta).
    #    in_space = input_spaces["parameters"]
    #    sanity_check_space(in_space, allowed_types=[Tuple])
    #    assert len(in_space) == 2, "ERROR: Expected Tuple of len=2 as input Space to Beta!"
    #    sanity_check_space(in_space[0], allowed_types=[Float])
    #    sanity_check_space(in_space[1], allowed_types=[Float])

    def parameterize_distribution(self, parameters):
        """
        Args:
            parameters (DataOpTuple): Tuple holding the alpha and beta parameters.
        """
        # Note: concentration0==beta, concentration1=alpha (!)
        return tfp.distributions.Beta(concentration1=parameters[0], concentration0=parameters[1])

    def _sample_deterministic(self, distribution):
        mean = distribution.mean()
        return self._squash(mean)

    def _sample_stochastic(self, distribution):
        raw_values = super(Beta, self)._sample_stochastic(distribution)
        return self._squash(raw_values)

    def _log_prob(self, distribution, values):
        raw_values = self._unsquash(values)
        return super(Beta, self)._log_prob(distribution, raw_values)

    def _squash(self, raw_values):
        return raw_values * (self.high - self.low) + self.low

    def _unsquash(self, values):
        return (values - self.low) / (self.high - self.low)
