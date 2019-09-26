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
from surreal.spaces import Space
from surreal.utils.util import SMALL_NUMBER


class MovingStandardize(Preprocessor):
    """
    Standardizes inputs using a moving estimate of mean and std.
    """
    def __init__(self, input_space):
        """
        Args:
            input_space (Space): The input space
        """
        super().__init__()

        self.input_space = Space.make(input_space)

        # How many samples have we seen (after last reset)?
        self.sample_count = None
        # Current estimate of the mean.
        self.mean_est = None
        # Current estimate of the sum of stds.
        self.std_sum_est = None

        self.reset()

    def reset(self):
        self.sample_count = 0
        self.mean_est = np.zeros(self.input_space.get_shape(include_main_axes=True), dtype=np.float32)
        self.std_sum_est = np.zeros(self.input_space.get_shape(include_main_axes=True), dtype=np.float32)

    def call(self, inputs):
        # https://www.johndcook.com/blog/standard_deviation/
        #inputs = np.asarray(inputs, dtype=np.float32)
        self.sample_count += 1
        if self.sample_count == 1:
            self.mean_est[...] = inputs
        else:
            update = inputs - self.mean_est
            self.mean_est[...] += update / self.sample_count
            self.std_sum_est[...] += update * update * (self.sample_count - 1) / self.sample_count

        # Subtract mean.
        result = inputs - self.mean_est

        # Estimate variance via sum of variance.
        if self.sample_count > 1:
            var_estimate = self.std_sum_est / (self.sample_count - 1)
        else:
            var_estimate = np.square(self.mean_est)
        std = np.sqrt(var_estimate) + SMALL_NUMBER

        standardized = result / std
        return standardized
