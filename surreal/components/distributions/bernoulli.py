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
from surreal.utils.util import convert_dtype


class Bernoulli(Distribution):
    """
    A Bernoulli distribution object defined by a single value p, the probability for True (rather than False).
    """
    def parameterize_distribution(self, parameters):
        """
        Args:
            parameters (tf.Tensor): The logit value that distribution returns True (must be passed through sigmoid to
                yield actual True-prob).
        """
        return tfp.distributions.Bernoulli(logits=parameters, dtype=convert_dtype("bool"))

    def _sample_deterministic(self, distribution):
        return distribution.prob(True) >= 0.5
