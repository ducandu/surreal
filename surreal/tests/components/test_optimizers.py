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
import random
import unittest

from surreal.components.optimizers.adam import Adam
from surreal.components.optimizers.sgd import SGD
from surreal.tests.test_util import check


class TestOptimizers(unittest.TestCase):

    # Some Loss function: Var (v) should be close to 1.0.
    # The derivative is: dL/dv = 2v - 2.0
    @staticmethod
    def L(v):
        return (v - 1.0) ** 2

    def test_gradient_tape(self):
        # Var to optimize.
        var = tf.Variable(random.random())
        # Derivative of loss is dL/dv = 2*(v-1.0) = 2v - 2
        expected_grad = 2 * var.numpy() - 2.0

        # Must use gradient tape as we are in eager mode. In graph mode, we would do `get_gradients`, which does
        # not work here.
        with tf.GradientTape() as t:
            loss = self.L(var)

        check(t.gradient(loss, var), expected_grad)

    def test_apply_gradients(self):
        lr = random.random()
        optimizer = SGD(learning_rate=lr)

        # Var to optimize.
        var = tf.Variable(random.random())
        var_value_orig = var.numpy()
        # Derivative of loss is dL/dv = 2*(v-1.0) = 2v - 2
        expected_grad = 2 * var_value_orig - 2.0

        # Must use gradient tape as we are in eager mode. In graph mode, we would do `get_gradients`, which does
        # not work here.
        with tf.GradientTape() as t:
            loss = self.L(var)

        optimizer.apply_gradients(grads_and_vars=[(t.gradient(loss, var), var)])

        # Check against variable now. Should change by -learning_rate * grad.
        var_value_after = var.numpy()
        expected_new_value = var_value_orig - (lr * expected_grad)
        check(var_value_after, expected_new_value)

    def test_minimize(self):
        # Test case not working w/o graph mode.
        return
        lr = random.random()
        optimizer = Adam(learning_rate=lr)

        # Var to optimize.
        var = tf.Variable(random.random())
        var_value_orig = var.numpy()
        # Derivative of loss is dL/dv = 2*(v-1.0) = 2v - 2
        expected_grad = 2 * var_value_orig - 2.0

        # Must use gradient tape as we are in eager mode. In graph mode, we would do `get_gradients`, which does
        # not work here.
        with tf.GradientTape() as t:
            loss = self.L(var)

        optimizer.minimize(loss, [var])

        # Check against variable now. Should change by -learning_rate * grad.
        var_value_after = var.numpy()
        expected_new_value = var_value_orig - (lr * expected_grad)
        check(var_value_after, expected_new_value)
