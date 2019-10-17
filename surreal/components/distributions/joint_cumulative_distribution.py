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

import tensorflow as tf

from surreal.components.distributions.distribution import Distribution
from surreal.utils.nest import flatten_alongside


class JointCumulativeDistribution(Distribution):
    """
    A joint cumulative distribution consisting of an arbitrarily nested container of n sub-distributions
    assumed to be all independent(!) of each other, such that:
    For e.g. n=2 and random variables X and Y: P(X and Y) = P(X)*P(Y) for all x and y.
    - Sampling returns a ContainerDataOp.
    - log_prob returns the sum of all single log prob terms (joint log prob).
    - entropy returns the sum of all single entropy terms (joint entropy).
    """
    def __init__(self, distributions, num_main_axes=1):
        """
        Args:
            distributions (any): Arbitrarily nested structure containing the specifications of the single
                sub-distributions.

            num_main_axes (Optional[int]): An optional hint as to how many main_axes to keep, when reducing
                probs/log-probs over the individual distributions.
        """
        super().__init__()

        self.distributions = distributions
        # Create the flattened sub-distributions and add them.
        self.flattened_sub_distributions = tf.nest.flatten(tf.nest.map_structure(
            lambda spec: Distribution.make(spec, scope="sub-distribution"), distributions
        ))
        self.num_main_axes = num_main_axes

    def log_prob(self, parameters, values):
        """
        Override `log_prob` as we have to add all the resulting log-probs together
        (joint log-prob of individual ones).
        """
        distributions = self.parameterize_distribution(parameters)
        all_log_probs = self._log_prob(distributions, values)
        return self._reduce_over_sub_distributions(all_log_probs)

    def parameterize_distribution(self, parameters):
        return tf.nest.pack_sequence_as(
            self.distributions,
            [self.flattened_sub_distributions[i].parameterize_distribution(params)
             for i, params in enumerate(flatten_alongside(parameters, self.distributions))]
        )

    def _sample_deterministic(self, distribution):
        return tf.nest.pack_sequence_as(
            distribution,
            [self.flattened_sub_distributions[i]._sample_deterministic(distr)
             for i, distr in enumerate(tf.nest.flatten(distribution))]
        )

    def _sample_stochastic(self, distribution, seed=None):
        return tf.nest.pack_sequence_as(
            distribution,
            [self.flattened_sub_distributions[i]._sample_stochastic(distr, seed=seed)
             for i, distr in enumerate(tf.nest.flatten(distribution))]
        )

    def _log_prob(self, distribution, values):
        return tf.nest.pack_sequence_as(
            distribution,
            [self.flattened_sub_distributions[i]._log_prob(distr, val)
             for i, (distr, val) in enumerate(zip(tf.nest.flatten(distribution), tf.nest.flatten(values)))]
        )

    def _prob(self, distribution, values):
        return tf.nest.pack_sequence_as(
            distribution,
            [self.flattened_sub_distributions[i]._prob(distr, val)
             for i, (distr, val) in enumerate(zip(tf.nest.flatten(distribution), tf.nest.flatten(values)))]
        )

    def _reduce_over_sub_distributions(self, log_probs):
        num_ranks_to_keep = self.num_main_axes
        log_probs_list = []
        for log_prob in tf.nest.flatten(log_probs):
            # Reduce sum over all ranks to get the joint log llh.
            log_prob = tf.reduce_sum(log_prob, axis=list(range(len(log_prob.shape) - 1, num_ranks_to_keep - 1, -1)))
            log_probs_list.append(log_prob)
        return tf.reduce_sum(tf.stack(log_probs_list, axis=0), axis=0)

    def _entropy(self, distribution):
        num_ranks_to_keep = self.num_main_axes
        all_entropies = []
        for distr in tf.next.flatten(distribution):
            entropy = distr.entropy()
            # Reduce sum over all ranks to get the joint log llh.
            entropy = tf.reduce_sum(entropy, axis=list(range(len(entropy.shape) - 1, num_ranks_to_keep - 1, -1)))
            all_entropies.append(entropy)
        return tf.reduce_sum(tf.stack(all_entropies, axis=0), axis=0)
