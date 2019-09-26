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
from surreal.utils import util


class Categorical(Distribution):
    """
    A categorical distribution object defined by a n values {p0, p1, ...} that add up to 1, the probabilities
    for picking one of the n categories.
    """
    def parameterize_distribution(self, parameters):
        return tfp.distributions.Categorical(logits=parameters, dtype=util.convert_dtype("int"))

    def _sample_deterministic(self, distribution):
        return tf.argmax(input=distribution.probs, axis=-1, output_type=util.convert_dtype("int"))
