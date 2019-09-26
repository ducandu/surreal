# Copyright 2019 ducandu GmbH. All Rights Reserved.
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

from surreal.components.preprocessors.preprocessor import Preprocessor
from surreal.utils.util import SMALL_NUMBER, force_list


class Normalize(Preprocessor):
    """
    Normalizes an input over all axes individually (denoted as `Xi` below) according to the following formula:

    Xi = (Xi - min(Xi)) / (max(Xi) - min(Xi) + epsilon),
        where:
        Xi is one entire axis of values.
        max(Xi) is the max value along this axis.
        min(Xi) is the min value along this axis.
        epsilon is a very small constant number (to avoid dividing by 0).
    """
    def __init__(self, axes=-1):
        super(Normalize, self).__init__()
        self.axes = force_list(axes)

    def call(self, inputs):
        min_value = inputs
        max_value = inputs

        for axis in self.axes:
            min_value = np.min(min_value, axis)
            max_value = np.max(max_value, axis)

        # Add some small constant to never let the range be zero.
        return (inputs - min_value) / (max_value - min_value + SMALL_NUMBER)
