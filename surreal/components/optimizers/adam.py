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


class Adam(Optimizer):
    """
    Adaptive momentum optimizer:
    https://arxiv.org/abs/1412.6980
    """
    def __init__(self, learning_rate=0.001, *, beta_1=0.9, beta_2=0.999, epsilon=1e-8, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)

        self.beta1 = beta_1
        self.beta2 = beta_2
        self.epsilon = epsilon

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate(time_percentage=0.0),
            beta_1=self.beta1,
            beta_2=self.beta2,
            epsilon=self.epsilon,
            decay=self.decay,
            #clip_norm=self.clip_norm,  # TODO: add when natively supported in tf 2.0
            #clip_value=self.clip_value
        )


