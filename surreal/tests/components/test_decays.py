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
import unittest

from surreal.components.misc.decay_components import Decay, Constant, LinearDecay, ExponentialDecay
from surreal.spaces import *
from surreal.tests.test_util import check


class TestDecays(unittest.TestCase):
    """
    Tests time-step dependent TimeDependentParameter Component classes.
    """
    time_percentage_space = Float(main_axes="B")

    def test_constant(self):
        constant = Constant.make(2.0)
        input_ = np.array([0.5, 0.1, 1.0, 0.9, 0.02, 0.01, 0.99, 0.23])
        out = constant(input_)
        check(out, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    def test_linear_decay(self):
        linear_decay = LinearDecay.make({"from": 2.0, "to": 0.5})
        input_ = np.array([0.5, 0.1, 1.0, 0.9, 0.02, 0.01, 0.99, 0.23])
        out = linear_decay(input_)
        check(out, 2.0 - input_ * (2.0 - 0.5))

    def test_linear_decay_with_step_function(self):
        linear_decay = LinearDecay.make({"from": 2.0, "to": 0.5, "begin_time_percentage": 0.5, "end_time_percentage": 0.6})
        input_ = np.array([0.5, 0.1, 1.0, 0.9, 0.02, 0.01, 0.99, 0.23, 0.51, 0.52, 0.55, 0.59])
        out = linear_decay(input_)
        check(out, np.array([2.0, 2.0, 0.5, 0.5, 2.0, 2.0, 0.5, 2.0, 1.85, 1.7, 1.25, 0.65]))

    def test_linear_parameter_using_global_time_step(self):
        max_time_steps = 100
        linear_decay = Decay.make("linear-decay", from_=2.0, to_=0.5, max_time_steps=max_time_steps)
        # Call without any parameters -> force component to use GLOBAL_STEP, which should be 0 right now -> no decay.
        for time_step in range(30):
            out = linear_decay()
            check(out, 2.0 - (time_step / max_time_steps) * (2.0 - 0.5))

    def test_polynomial_parameter(self):
        polynomial_decay = Decay.make(type="polynomial-decay", from_=2.0, to_=0.5, power=2.0)
        input_ = np.array([0.5, 0.1, 1.0, 0.9, 0.02, 0.01, 0.99, 0.23])
        out = polynomial_decay(input_)
        check(out, (2.0 - 0.5) * (1.0 - input_) ** 2 + 0.5)

    def test_polynomial_parameter_using_global_time_step(self):
        max_time_steps = 10
        polynomial_decay = Decay.make("polynomial-decay", from_=3.0, to_=0.5, max_time_steps=max_time_steps)
        # Call without any parameters -> force component to use internal `current_time_step`.
        # Go over the max time steps and expect time_percentage to be capped at 1.0.
        for time_step in range(50):
            out = polynomial_decay()
            check(out, (3.0 - 0.5) * (1.0 - min(time_step / max_time_steps, 1.0)) ** 2 + 0.5)

    def test_exponential_parameter(self):
        exponential_decay = Decay.make(type="exponential-decay", from_=2.0, to_=0.5, decay_rate=0.5)
        input_ = np.array([0.5, 0.1, 1.0, 0.9, 0.02, 0.01, 0.99, 0.23])
        out = exponential_decay(input_)
        check(out, 0.5 + (2.0 - 0.5) * 0.5 ** input_)

    def test_exponential_parameter_using_global_time_step(self):
        max_time_steps = 10
        decay_rate = 0.1
        exponential_decay = ExponentialDecay.make(
            from_=3.0, to_=0.5, max_time_steps=max_time_steps, decay_rate=decay_rate
        )
        # Call without any parameters -> force component to use internal `current_time_step`.
        # Go over the max time steps and expect time_percentage to be capped at 1.0.
        for time_step in range(100):
            out = exponential_decay()
            check(out, 0.5 + (3.0 - 0.5) * decay_rate ** min(time_step / max_time_steps, 1.0))
