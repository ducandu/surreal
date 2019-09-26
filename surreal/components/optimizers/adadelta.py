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

from surreal.components.optimizers.optimizer import Optimizer


class Adadelta(Optimizer):
    """
    Adadelta optimizer which adapts learning rate over time:

    https://arxiv.org/abs/1212.5701
    """
    def __init__(self, learning_rate=0.001, *, rho=0.95, epsilon=1e-8, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)

        self.rho = rho
        self.epsilon = epsilon

        self.optimizer = tf.keras.optimizers.Adadelta(
            learning_rate=self.learning_rate(time_percentage=0.0),
            rho=self.rho,
            epsilon=self.epsilon,
            decay=self.decay
            #clip_norm=self.clip_norm,  # TODO: add when natively supported in tf 2.0
            #clip_value=self.clip_value
        )

