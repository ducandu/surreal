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

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from surreal.components.distributions.distribution import Distribution
from surreal.utils.util import SMALL_NUMBER


class SquashedNormal(Distribution):
    """
    A Squashed with tanh Normal distribution object defined by a tuple: mean, standard deviation.
    The distribution will never return low or high exactly, but `low`+SMALL_NUMBER or `high`-SMALL_NUMBER respectively.
    """
    def __init__(self, low=-1.0, high=1.0):
        """
        Args:
            low (float): The lowest possible sampling value (excluding this value).
            high (float): The highest possible sampling value (excluding this value).
        """
        super().__init__()

        assert np.all(np.less(low, high))
        self.low = low
        self.high = high

    def sample_and_log_prob(self, parameters, deterministic=True):
        distribution = self.parameterize_distribution(parameters)
        scaled_outputs, log_prob = self._sample_and_log_prob(distribution, deterministic)
        return scaled_outputs, log_prob

    def parameterize_distribution(self, parameters):
        return tfp.distributions.Normal(loc=parameters[0], scale=parameters[1])

    def _sample_deterministic(self, distribution):
        mean = distribution.mean()
        return self._squash(mean)

    def _sample_stochastic(self, distribution, seed=None):
        sample = self._squash(distribution.sample(seed=seed or self.seed))
        return sample

    def _log_prob(self, distribution, values):
        unsquashed_values = self._unsquash(values)
        log_prob = distribution.log_prob(value=unsquashed_values)
        unsquashed_values_tanhd = tf.math.tanh(unsquashed_values)
        log_prob -= tf.math.reduce_sum(tf.math.log(1 - unsquashed_values_tanhd ** 2 + SMALL_NUMBER), axis=-1, keepdims=True)
        return log_prob

    def _sample_and_log_prob(self, distribution, deterministic):
        raw_action = tf.cond(
            pred=deterministic,
            true_fn=lambda: distribution.mean(),
            false_fn=lambda: distribution.sample()
        )
        action = tf.math.tanh(raw_action)
        log_prob = distribution.log_prob(raw_action)
        log_prob -= tf.math.reduce_sum(tf.math.log(1 - action ** 2 + SMALL_NUMBER), axis=-1, keepdims=True)

        scaled_action = (action + 1) / 2 * (self.high - self.low) + self.low
        return scaled_action, log_prob

    def _squash(self, raw_values):
        # Make sure raw_values are not too high/low (such that tanh would return exactly 1.0/-1.0,
        # which would lead to +/-inf log-probs).
        return (tf.clip_by_value(tf.math.tanh(raw_values), -1.0+SMALL_NUMBER, 1.0-SMALL_NUMBER) + 1.0) / 2.0 * (self.high - self.low) + self.low

    def _unsquash(self, values):
        return tf.math.atanh((values - self.low) / (self.high - self.low) * 2.0 - 1.0)
