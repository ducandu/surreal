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

import numpy as np
import tensorflow as tf
import unittest

from surreal.components.distribution_adapters import BernoulliDistributionAdapter, CategoricalDistributionAdapter, \
    MixtureDistributionAdapter, PlainOutputAdapter, NormalDistributionAdapter, BetaDistributionAdapter
from surreal.spaces import *
from surreal.tests import check
from surreal.utils.numpy import dense, relu
from surreal.utils.util import MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT, SMALL_NUMBER


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

    def test_bernoulli_adapter(self):
        input_space = Float(shape=(16,), main_axes="B")
        output_space = Bool(shape=(2,), main_axes="B")

        adapter = BernoulliDistributionAdapter(output_space=output_space, activation="relu")
        batch_size = 32
        inputs = input_space.sample(batch_size)
        out = adapter(inputs)
        weights = adapter.get_weights()

        # Parameters are the plain logits (no sigmoid).
        expected = relu(dense(inputs, weights[0], weights[1]))
        check(out, expected, decimals=5)

    def test_categorical_adapter(self):
        input_space = Float(shape=(16,), main_axes="B")
        output_space = Int(2, shape=(3, 2), main_axes="B")

        adapter = CategoricalDistributionAdapter(
            output_space=output_space, kernel_initializer="ones", activation="relu"
        )
        batch_size = 2
        inputs = input_space.sample(batch_size)
        out = adapter(inputs)
        weights = adapter.get_weights()
        expected = np.reshape(relu(dense(inputs, weights[0], weights[1])), newshape=(batch_size, 3, 2, 2))
        check(out, expected, decimals=5)

    def test_normal_adapter(self):
        input_space = Float(shape=(8,), main_axes="B")
        output_space = Float(shape=(3, 2), main_axes="B")

        adapter = NormalDistributionAdapter(output_space=output_space, activation="linear")
        batch_size = 3
        inputs = input_space.sample(batch_size)
        out = adapter(inputs)
        weights = adapter.get_weights()
        expected = np.split(np.reshape(dense(inputs, weights[0], weights[1]), newshape=(batch_size, 3, 4)), 2, axis=-1)
        expected[1] = np.clip(expected[1], MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)
        expected[1] = np.exp(expected[1])
        check(out, expected, decimals=5)

    def test_beta_adapter(self):
        input_space = Float(shape=(8,), main_axes="B")
        output_space = Float(shape=(3, 2), main_axes="B")

        adapter = BetaDistributionAdapter(output_space=output_space)
        batch_size = 5
        inputs = input_space.sample(batch_size)
        out = adapter(inputs)
        weights = adapter.get_weights()
        expected = np.reshape(dense(inputs, weights[0], weights[1]), newshape=(batch_size, 3, 4))
        expected = np.clip(expected, np.log(SMALL_NUMBER), -np.log(SMALL_NUMBER))
        expected = np.log(np.exp(expected) + 1.0) + 1.0
        expected = np.split(expected, 2, axis=-1)
        check(out, expected, decimals=5)

    def test_mixture_adapter(self):
        input_space = Float(shape=(16,), main_axes="B")
        output_space = Float(shape=(3,), main_axes="B")

        adapter = MixtureDistributionAdapter(
            output_space, "normal-distribution-adapter", "beta-distribution-adapter",
            activation="relu"  # Don't do this in real life! This is just to test.
        )
        batch_size = 2
        inputs = input_space.sample(batch_size)
        out = adapter(inputs)
        weights = adapter.get_weights()
        params0 = np.split(dense(inputs, weights[2], weights[3]), 2, axis=-1)
        params0[1] = np.exp(np.clip(params0[1], MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT))

        params1 = dense(inputs, weights[4], weights[5])
        params1 = np.clip(params1, np.log(SMALL_NUMBER), -np.log(SMALL_NUMBER))
        params1 = np.log(np.exp(params1) + 1.0) + 1.0
        params1 = np.split(params1, 2, axis=-1)

        expected = {
            "categorical": relu(dense(inputs, weights[0], weights[1])), "parameters0": params0, "parameters1": params1
        }
        check(out, expected, decimals=5)

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
