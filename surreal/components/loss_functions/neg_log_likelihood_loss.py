# Copyright 2019 ducandu GmbH. All Rights Reserved.
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

from surreal.components.distributions.distribution import Distribution
from surreal.components.loss_functions.loss_function import LossFunction


class NegLogLikelihoodLoss(LossFunction):
    """
    Calculates the negative log-likelihood loss by passing the labels through a given distribution
    (parameterized by `predictions`) and inverting the sign.

    L(params,labels) = -log(Dparams.pdf(labels))
    Where:
        Dparams: Parameterized distribution object.
        pdf: Prob. density function of the distribution.
    """
    def __init__(self, distribution):
        """
        Args:
            distribution (Union[Distribution,dict]): The distribution component to use for calculating the neg log
                likelihood of a sample.
        """
        super().__init__()
        self.distribution = Distribution.make(distribution)

    def call(self, parameters, labels):
        """
        Args:
            parameters (any): Parameters to parameterize `self.distribution`.

            labels (any): Labels that will be passed (as `values`) through the pdf function of the distribution
                to get the loss.

        Returns:
            any: The loss value(s) (one single value for each batch item).
        """
        # Get the distribution's log-likelihood for the labels, given the parameterized distribution.
        neg_log_likelihood = -self.distribution.log_prob(parameters, labels)
        # If necessary, reduce over all non-batch/non-time ranks.
        neg_log_likelihood = tf.reduce_sum(
            neg_log_likelihood, axis=list(range(len(neg_log_likelihood.shape) - 1, 0, -1))
        )
        return neg_log_likelihood
