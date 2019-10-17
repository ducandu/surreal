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
import tensorflow_probability as tfp

from surreal.components.distributions.distribution import Distribution


class GumbelSoftmax(Distribution):
    """
    The Gumbel Softmax distribution [1] (also known as the Concrete [2] distribution) is a close cousin of the relaxed
    one-hot categorical distribution, whose tfp implementation we will use here plus
    adjusted `sample_...` and `prob/log_prob` methods. See discussion at [0].

    [0] https://stackoverflow.com/questions/56226133/soft-actor-critic-with-discrete-action-space

    [1] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017): https://arxiv.org/abs/1611.01144
    [2] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
        https://arxiv.org/abs/1611.00712
    """
    def __init__(self, temperature=1.0):
        """
        Args:
            temperature (float): Temperature parameter. For low temperatures, the expected value approaches
                a categorical random variable. For high temperatures, the expected value approaches a uniform
                distribution.
        """
        super(GumbelSoftmax, self).__init__()
        self.temperature = temperature

    def parameterize_distribution(self, parameters):
        return tfp.distributions.RelaxedOneHotCategorical(temperature=self.temperature, logits=parameters)

    def _sample_deterministic(self, distribution):
        # Simply return our probs as the deterministic output vector.
        return distribution._distribution.probs

    def _log_prob(self, distribution, values):
        """
        Override since the implementation of tfp.RelaxedOneHotCategorical yields positive values.
        """
        if values.shape != distribution.logits.shape:
            values = tf.cast(tf.one_hot(values, distribution.logits.shape.as_list()[-1]), dtype=tf.float32)
            assert values.shape == distribution.logits.shape

        # [0]'s implementation (see line below) seems to be an approximation to the actual Gumbel Softmax density.
        return -tf.reduce_sum(-values * tf.nn.log_softmax(distribution.logits, axis=-1), axis=-1)

        # We will use instead here the exact formula from [1] (see `self._prob()`).
        #return tf.math.log(self._prob(distribution, values))

    def _prob(self, distribution, values):
        """
        Override since the implementation of tfp.RelaxedOneHotCategorical yields seemingly wrong values (log probs
        always seem to be returned positive).
        """
        raise NotImplementedError
        # Density formula from [1]:
        #num_categories = float(values.shape[-1])
        #density = tf.math.exp(tf.math.lgamma(num_categories)) * tf.math.pow(self.temperature, num_categories - 1) * \
        #        (tf.reduce_sum(distribution.probs / tf.math.pow(values, self.temperature), axis=-1) ** -num_categories) * \
        #        tf.reduce_prod(distribution.probs / tf.math.pow(values, self.temperature + 1.0), axis=-1)

        #return density
