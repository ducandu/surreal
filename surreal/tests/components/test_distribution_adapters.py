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
import unittest

from surreal.components.distribution_adapters.plain_output_adapter import PlainOutputAdapter
from surreal.spaces import *
from surreal.tests import check
from surreal.utils.numpy import dense, relu


class TestDistributionAdapters(unittest.TestCase):

    def test_plain_output_adapter(self):
        input_space = Float(-1.0, 1.0, shape=(5,), main_axes="B")
        output_space = Float(shape=(3,), main_axes="B")

        adapter = PlainOutputAdapter(output_space)

        # Simple function call -> Expect output for all int-inputs.
        input_ = input_space.sample(6)
        result = adapter(input_)
        weights = adapter.get_weights()
        expected = dense(input_, weights[0], weights[1])

        check(result, expected)

    def test_plain_output_adapter_with_pre_network(self):
        input_space = Float(-1.0, 1.0, shape=(5,), main_axes="B")
        output_space = Float(shape=(3,), main_axes="B")

        adapter = PlainOutputAdapter(output_space, pre_network=tf.keras.models.Sequential(
            tf.keras.layers.Dense(units=10, activation="relu")
        ))

        # Simple function call -> Expect output for all int-inputs.
        input_ = input_space.sample(6)
        result = adapter(input_)
        weights = adapter.get_weights()
        expected = dense(relu(dense(input_, weights[0], weights[1])), weights[2], weights[3])

        check(result, expected)

    def test_copying_an_adapter(self):
        input_space = Float(-1.0, 1.0, shape=(5,), main_axes="B")
        output_space = Float(shape=(3,), main_axes="B")

        adapter = PlainOutputAdapter(output_space, pre_network=None)

        # Simple function call -> Expect output for all int-inputs.
        input_ = input_space.sample(3)
        result = adapter(input_)
        weights = adapter.get_weights()
        expected = dense(input_, weights[0], weights[1])

        check(result, expected)

        new_adapter = adapter.copy()
        new_weights = new_adapter.get_weights()
        # Check all weights.
        check(weights, new_weights)
        # Do a pass and double-check.
        result = new_adapter(input_)
        check(result, expected)
