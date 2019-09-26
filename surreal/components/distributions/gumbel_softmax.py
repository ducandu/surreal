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
    The Gumbel Softmax distribution is also known as a relaxed one-hot categorical or concrete distribution.

    Gumbel Softmax: https://arxiv.org/abs/1611.01144

    Concrete: https://arxiv.org/abs/1611.00712
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
        """
        Returns the argmax (int) of a relaxed one-hot vector. See `_graph_fn_sample_stochastic` for details.
        """
        # Cast to float again because this is called from a tf.cond where the other option calls a stochastic
        # sample returning a float.
        argmax = tf.argmax(input=distribution._distribution.probs, axis=-1, output_type=tf.int32)
        sample = tf.cast(argmax, dtype=tf.float32)
        # Argmax turns (?, n) into (?,), not (?, 1)
        # TODO: What if we have a time rank as well?
        if len(sample.shape) == 1:
            sample = tf.expand_dims(sample, -1)
        return sample
