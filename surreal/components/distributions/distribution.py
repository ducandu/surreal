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

from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf

from surreal.makeable import Makeable


class Distribution(Makeable, metaclass=ABCMeta):
    """
    A distribution class that can incorporate a backend-specific distribution object that gets its parameters
    from an external source (e.g. a NN).
    """
    def __init__(self, seed=None):
        """
        Keyword Args:
            seed (Optional[int]): An optional random seed to use when sampling stochastically.
        """
        super().__init__()

        self.seed = seed

    @abstractmethod
    def parameterize_distribution(self, parameters):
        """
        Parametrizes this distribution (normally from an NN-output vector). Returns the tfp Distribution object.

        Args:
            parameters (any): The input(s) used to parameterize this distribution. This is normally a cleaned up
                single NN-output (e.g.: the two values for mean and variance for a univariate Gaussian
                distribution).

        Returns:
            tfp.distributions.Distribution: The parameterized distribution object.
        """
        raise NotImplementedError

    def sample_stochastic(self, parameters):
        distribution = self.parameterize_distribution(parameters)
        return self._sample_stochastic(distribution, seed=self.seed)

    def sample_deterministic(self, parameters):
        distribution = self.parameterize_distribution(parameters)
        return self._sample_deterministic(distribution)

    def sample(self, parameters, deterministic=False):
        distribution = self.parameterize_distribution(parameters)
        return self._sample(distribution, deterministic)

    def sample_and_log_prob(self, parameters, deterministic=False):
        distribution = self.parameterize_distribution(parameters)
        actions = self._sample(distribution, deterministic)
        log_probs = self._log_prob(distribution, actions)
        return actions, log_probs

    def entropy(self, parameters):
        """
        Returns the entropy value of the distribution.

        Args:
            parameters (any): The parameters to parameterize the tfp distribution whose entropy to
                calculate.

        Returns:
            any: The distribution's entropy.
        """
        distribution = self.parameterize_distribution(parameters)
        return distribution.entropy()

    def log_prob(self, parameters, values):
        distribution = self.parameterize_distribution(parameters)
        return self._log_prob(distribution, values)

    def prob(self, parameters, values):
        """
        Probability density/mass function.

        Args:
            parameters (any): The parameters to parameterize the tfp distribution whose entropy to
                calculate.

            values (any): Values for which to compute the probabilities/likelihoods given `distribution`.

        Returns:
            any: The probability/likelihood of the given values.
        """
        distribution = self.parameterize_distribution(parameters)
        likelihood = self._prob(distribution, values)
        return likelihood

    def kl_divergence(self, parameters, other_parameters):
        distribution = self.parameterize_distribution(parameters)
        other_distribution = self.parameterize_distribution(other_parameters)
        return self._kl_divergence(distribution, other_distribution)

    def _sample(self, distribution, deterministic=False):
        """
        Takes a sample from the (already parameterized) distribution. The parametrization also includes a possible
        batch size.

        Args:
            distribution (tfp.distributions.Distribution): The (already parameterized) distribution to use for
                sampling.

            deterministic (Union[bool,tf.Tensor]): Whether to return the maximum-likelihood result, instead of a random
                sample. Can be used to pick deterministic actions from discrete ("greedy") or continuous (mean-value)
                distributions.

        Returns:
            any: The taken sample(s).
        """
        # Fixed boolean input (not a tf.Tensor).
        if isinstance(deterministic, (bool, np.ndarray)):
            # Don't do `is True` here in case `deterministic` is np.ndarray!
            if deterministic:
                return self._sample_deterministic(distribution)
            else:
                return self._sample_stochastic(distribution)

        return tf.cond(
            pred=deterministic,
            true_fn=lambda: self._sample_deterministic(distribution),
            false_fn=lambda: self._sample_stochastic(distribution)
        )

    @abstractmethod
    def _sample_deterministic(self, distribution):
        """
        Returns a deterministic sample for a given distribution.

        Args:
            distribution (tfp.distributions.Distribution): The (already parameterized) distribution
                whose sample to take.

        Returns:
            any: The sampled value.
        """
        raise NotImplementedError

    def _sample_stochastic(self, distribution, seed=None):
        """
        Returns an actual sample for a given distribution.

        Args:
            distribution (tfp.distributions.Distribution): The (already parameterized) distribution
                whose sample to take.

        Returns:
            any: The drawn sample.
        """
        return distribution.sample(seed=seed or self.seed)

    def _log_prob(self, distribution, values):
        """
        Log of the probability density/mass function.

        Args:
            distribution (tfp.distributions.Distribution): The (already parameterized) distribution whose log-likelihood
                value to calculate.

            values (any): Values for which to compute the log probabilities/likelihoods.

        Returns:
            any: The log probability/likelihood of the given values.
        """
        return distribution.log_prob(value=values)

    def _prob(self, distribution, values):
        """
        Probability density/mass function.

        Args:
            distribution (tfp.distributions.Distribution): The (already parameterized) distribution whose likelihood
                value to calculate.

            values (any): Values for which to compute the probabilities/likelihoods.

        Returns:
            any: The probability/likelihood of the given values.
        """
        return distribution.prob(value=values)

    def _kl_divergence(self, distribution, distribution_b):
        """
        Kullback-Leibler (KL) divergence between two distribution objects.

        Args:
            distribution (tf.distributions.Distribution): The (already parameterized) distribution 1.
            distribution_b (tf.distributions.Distribution): The other distribution object.

        Returns:
            any: The KL-divergence between the two distributions.
        """
        pass
        # TODO: never tested. tf throws error: NotImplementedError: No KL(distribution_a || distribution_b) registered for distribution_a type Bernoulli and distribution_b type ndarray
        #return tf.distributions.kl_divergence(
        #    distribution_a=distribution_a,
        #    distribution_b=distribution_b,
        #    allow_nan_stats=True,
        #    name=None
        #)
