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


class RMSProp(Optimizer):
    """
    RMSProp Optimizer as discussed by Hinton:

    https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """
    def __init__(self, learning_rate=0.001, *, rho=0.9, epsilon=1e-8, momentum=0.0, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)

        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon

        self.optimizer = tf.keras.optimizers.RMSProp(
            learning_rate=self.learning_rate(time_percentage=0.0),
            rho=self.rho,
            momentum=self.momentum,
            epsilon=self.epsilon,
            #clip_norm=self.clip_norm,  # TODO: add when natively supported in tf 2.0
            #clip_value=self.clip_value
        )
