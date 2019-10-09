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
    adjusted `sample_...` and `prob/log_prob` methods.
    See discussion at: https://stackoverflow.com/questions/56226133/soft-actor-critic-with-discrete-action-space

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
        raise NotImplementedError
        ## Cast to float again because this is called from a tf.cond where the other option calls a stochastic
        ## sample returning a float.
        #argmax = tf.argmax(input=distribution._distribution.probs, axis=-1, output_type=tf.int32)
        #sample = tf.cast(argmax, dtype=tf.float32)
        ## Argmax turns (?, n) into (?,), not (?, 1)
        ## TODO: What if we have a time rank as well?
        #if len(sample.shape) == 1:
        #    sample = tf.expand_dims(sample, -1)
        #return sample

    #def _sample_stochastic(self, distribution, seed=None):
    #    """
    #    Returns the argmax (int) of a RelaxedOneHotCategorical-sampled relaxed one-hot vector.
    #    """
    #    noise = tf.random.uniform(shape=distribution.probs.shape)
    #    noisy_logits = distribution.logits - tf.math.log(-tf.math.log(noise))
    #    return tf.argmax(noisy_logits, axis=-1)

    def _log_prob(self, distribution, values):
        """
        Override since the implementation of tfp.RelaxedOneHotCategorical yields positive values.
        """
        if values.shape != distribution.logits.shape:
            values = tf.cast(tf.one_hot(values, distribution.logits.shape.as_list()[-1]), dtype=tf.float32)
            assert values.shape == distribution.logits.shape
        return -tf.reduce_sum(-values * tf.nn.log_softmax(distribution.logits, axis=-1), axis=-1)

    def _prob(self, distribution, values):
        """
        TODO
        """
        raise NotImplementedError

