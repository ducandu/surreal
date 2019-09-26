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
    def __init__(self, parameterize_via_diagonal=True, parameterize_via_covariance=False):
        """
        Args:
            parameterize_via_diagonal (bool): Whether we are parameterizing via the diagonal stddev values.
                Note that
            parameterize_via_covariance (bool): Whether we are parameterizing via the full covariance values.
        """
        super(MultivariateNormal, self).__init__()
        self.parameterize_via_diagonal = parameterize_via_diagonal
        self.parameterize_via_covariance = parameterize_via_covariance
        assert self.parameterize_via_diagonal != self.parameterize_via_covariance, \
            "ERROR: Exactly one of `parameterize_via_diagonal` and `parameterize_via_covariance` must be True!"

    #def check_input_spaces(self, input_spaces, action_space=None):
    #    # Must be a Tuple of len 2 (mean and stddev OR mean and full co-variance matrix).
    #    in_space = input_spaces["parameters"]
    #    sanity_check_space(in_space, allowed_types=[Tuple])
    #    assert len(in_space) == 2, "ERROR: Expected Tuple of len=2 as input Space to MultivariateNormal!"
    #    sanity_check_space(in_space[0], allowed_types=[Float])
    #    sanity_check_space(in_space[1], allowed_types=[Float])

    #    if self.parameterize_via_diagonal:
    #        # Make sure mean and stddev have the same last rank.
    #        assert in_space[0].shape[-1] == in_space[1].shape[-1],\
    #            "ERROR: `parameters` in_space must have the same last rank for mean as for (diagonal) stddev values!"

    def parameterize_distribution(self, parameters):
        if self.parameterize_via_diagonal:
            return tfp.distributions.MultivariateNormalDiag(
                loc=parameters[0], scale_diag=parameters[1]
            )
        # TODO: support parameterization through full covariance matrix.
        #else:
        #    mean, covariance_matrix = tf.split(parameters, num_or_size_splits=[1, self.num_events], axis=-1)
        #    mean = tf.squeeze(mean, axis=-1)
        #    return tfp.distributions.MultivariateNormalFullCovariance(
        #        loc=mean, covariance_matrix=covariance_matrix
        #    )

    def _sample_deterministic(self, distribution):
            return distribution.mean()
