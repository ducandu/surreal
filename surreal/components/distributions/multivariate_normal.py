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


class MultivariateNormal(Distribution):
    """
    A multivariate Gaussian distribution.
    """
    def __init__(self, parameterize_via_diagonal=True):
        """
        Args:
            parameterize_via_diagonal (bool): Whether we are parameterizing via the diagonal stddev values.
        """
        super().__init__()
        self.parameterize_via_diagonal = parameterize_via_diagonal

    def parameterize_distribution(self, parameters):
        if self.parameterize_via_diagonal is True:
            return tfp.distributions.MultivariateNormalDiag(
                loc=parameters[0], scale_diag=parameters[1]
            )
        # TODO: support parameterization through full covariance matrix.
        else:
            raise NotImplementedError
        #else:
        #    mean, covariance_matrix = tf.split(parameters, num_or_size_splits=[1, self.num_events], axis=-1)
        #    mean = tf.squeeze(mean, axis=-1)
        #    return tfp.distributions.MultivariateNormalFullCovariance(
        #        loc=mean, covariance_matrix=covariance_matrix
        #    )

    def _sample_deterministic(self, distribution):
        return distribution.mean()
