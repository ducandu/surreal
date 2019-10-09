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
import unittest
import tensorflow as tf

from surreal.components.networks.network import Network
from surreal.spaces import *
from surreal.tests import check
from surreal.utils.errors import SurrealError
from surreal.utils.numpy import dense, one_hot, softmax, relu


class TestNetworks(unittest.TestCase):

    class MyModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.dense = tf.keras.layers.Dense(10)
            self.dense2 = tf.keras.layers.Dense(5)

        def call(self, inputs):
            return tf.concat([self.dense(inputs), self.dense2(inputs)], axis=-1)

    def test_subclassing_network_with_primitive_int_output_space(self):
        input_space = Float(-1.0, 1.0, shape=(5,), main_axes="B")
        output_space = Int(3, main_axes="B")

        # Using keras subclassing.
        network = self.MyModel()
        nn = Network(network=network, output_space=output_space)

        # Simple function call -> Expect output for all int-inputs.
        input_ = input_space.sample(6)
        result = nn(input_)
        weights = nn.get_weights()
        expected = dense(np.concatenate(
            [dense(input_, weights[0], weights[1]), dense(input_, weights[2], weights[3])],
            axis=-1
        ), weights[4], weights[5])

        check(result, expected)

        # Function call with value -> Expect output for only that int-value
        input_ = input_space.sample(6)
        values = output_space.sample(6)
        result = nn(input_, values)
        weights = nn.get_weights()
        expected = dense(np.concatenate(
            [dense(input_, weights[0], weights[1]), dense(input_, weights[2], weights[3])],
            axis=-1
        ), weights[4], weights[5])
        expected = np.sum(expected * one_hot(values, depth=output_space.num_categories), axis=-1)

        check(result, expected)

    def test_func_api_network_with_primitive_int_output_space_and_distribution(self):
        input_space = Float(-1.0, 1.0, shape=(3,), main_axes="B")
        output_space = Int(5, main_axes="B")

        # Using keras functional API to create network.
        i = tf.keras.layers.Input(shape=(3,))
        d = tf.keras.layers.Dense(10)(i)
        e = tf.keras.layers.Dense(5)(i)
        o = tf.concat([d, e], axis=-1)
        network = tf.keras.Model(inputs=i, outputs=o)

        # Use default distributions (i.e. categorical for Int).
        nn = Network(
            network=network,
            output_space=output_space,
            distributions="default"
        )
        input_ = input_space.sample(1000)
        result = nn(input_)
        # Check the sample for a proper mean value.
        check(np.mean(result), 2, decimals=0)

        # Function call with value -> Expect probabilities for given int-values.
        input_ = input_space.sample(6)
        values = output_space.sample(6)
        result = nn(input_, values)
        weights = nn.get_weights()
        expected = dense(np.concatenate(
            [dense(input_, weights[0], weights[1]), dense(input_, weights[2], weights[3])],
            axis=-1
        ), weights[4], weights[5])
        expected = softmax(expected)
        expected = np.sum(expected * one_hot(values, depth=output_space.num_categories), axis=-1)

        check(result, expected)

        # Function call with "likelihood" option set -> Expect sample plus probabilities for sampled int-values.
        input_ = input_space.sample(1000)
        sample, probs = nn(input_, likelihood=True)
        check(np.mean(sample), 2, decimals=0)
        check(np.mean(probs), 1.0 / output_space.num_categories, decimals=1)

    def test_layer_network_with_container_output_space(self):
        # Using keras layer as network spec.
        layer = tf.keras.layers.Dense(10)

        nn = Network(
            network=layer,
            output_space=Dict({"a": Float(shape=(2, 3)), "b": Int(3)})
        )
        # Simple call -> Should return dict with "a"->float(2,3) and "b"->float(3,)
        input_ = Float(-1.0, 1.0, shape=(5,), main_axes="B").sample(5)
        result = nn(input_)
        weights = nn.get_weights()
        expected_a = np.reshape(dense(dense(input_, weights[0], weights[1]), weights[2], weights[3]), newshape=(-1, 2, 3))
        expected_b = dense(dense(input_, weights[0], weights[1]), weights[4], weights[5])

        check(result, dict(a=expected_a, b=expected_b))

    def test_layer_network_with_container_output_space_and_distributions(self):
        input_space = Float(-1.0, 1.0, shape=(10,), main_axes="B")
        output_space = Dict({"a": Float(shape=(2, 3)), "b": Int(3)}, main_axes="B")

        # Using keras layer as network spec.
        layer = tf.keras.layers.Dense(10)

        nn = Network(
            network=layer,
            output_space=output_space,
            distributions=True
        )
        # Simple call -> Should return sample dict with "a"->float(2,3) and "b"->int(3).
        input_ = input_space.sample(1000)
        result = nn(input_)
        check(np.mean(result["a"]), 0.0, decimals=0)
        check(np.mean(result["b"]), 1, decimals=0)

        # Call with value -> Should return likelihood of value.
        input_ = input_space.sample(3)
        value = output_space.sample(3)
        likelihood = nn(input_, value)
        self.assertTrue(likelihood.shape == (3,))
        self.assertTrue(likelihood.dtype == np.float32)

    def test_layer_network_with_container_output_space_and_one_distribution(self):
        input_space = Float(-1.0, 1.0, shape=(5,), main_axes="B")
        output_space = Dict({"a": Float(shape=(2, 3)), "b": Int(3)}, main_axes="B")
        # Using keras layer as network spec.
        layer = tf.keras.layers.Dense(10)

        nn = Network(
            network=layer,
            output_space=output_space,
            # Only one output component is a distribution, the other not (Int).
            distributions=dict(a=True)
        )
        # Simple call -> Should return sample dict with "a"->float(2,3) and "b"->int(3,).
        input_ = input_space.sample(1000)
        result = nn(input_)
        check(np.mean(result["a"]), 0.0, decimals=0)
        check(np.mean(np.sum(softmax(result["b"]), axis=-1)), 1.0, decimals=5)

        # Call with value -> Should return likelihood of "a"-value and output for "b"-value.
        input_ = input_space.sample(3)
        value = output_space.sample(3)
        result, likelihood = nn(input_, value)
        self.assertTrue(result["a"] is None)  # a is None b/c value was already given for likelihood calculation
        self.assertTrue(result["b"].shape == (3,))  # b is the (batched) output values for the given int-numbers
        self.assertTrue(result["b"].dtype == np.float32)
        self.assertTrue(likelihood.shape == (3,))  # (total) likelihood is some float
        self.assertTrue(likelihood.dtype == np.float32)

        # Extract only the "b" value-output (one output for each int category).
        # Also: No likelihood output b/c "a" was invalidated.
        del value["a"]
        value["b"] = None
        result = nn(input_, value)
        self.assertTrue(result["a"] is None)
        self.assertTrue(result["b"].shape == (3, 3))
        self.assertTrue(result["b"].dtype == np.float32)

        value = output_space.sample(3)
        value["a"] = None
        del value["b"]
        result = nn(input_, value)
        self.assertTrue(result is None)

    def test_layer_network_with_container_output_space_and_mix_of_distributions_and_no_distributions(self):
        input_space = Float(-1.0, 1.0, shape=(5,), main_axes="B")
        output_space = Dict({
            "a": Float(shape=(2, 3)), "b": Int(3), "c": Float(-0.1, 1.0, shape=(2,)), "d": Int(3, shape=(2,))
        }, main_axes="B")
        # Using keras layer as network spec.
        layer = tf.keras.layers.Dense(10)

        nn = Network(
            network=layer,
            output_space=output_space,
            # Only two output components are distributions (a and b), the others not (c=Float, d=Int).
            distributions=dict(a="default", b=True)
        )
        # Simple call -> Should return sample dict.
        input_ = input_space.sample(10000)
        result = nn(input_)
        check(np.mean(result["a"]), 0.0, decimals=0)
        check(np.mean(result["b"]), 1.0, decimals=0)
        check(np.mean(result["d"]), 0.0, decimals=0)
        self.assertTrue(result["d"].shape == (10000, 2, 3))
        self.assertTrue(result["d"].dtype == np.float32)

        # Change limits of input a little to get more chances of reaching extreme float outputs (for "c").
        input_space = Float(-10.0, 10.0, shape=(5,), main_axes="B")
        input_ = input_space.sample(10000)
        result = nn(input_)
        self.assertFalse(np.any(result["c"].numpy() > 1.0))
        self.assertTrue(np.any(result["c"].numpy() > 0.9))
        self.assertFalse(np.any(result["c"].numpy() < -0.1))
        self.assertTrue(np.any(result["c"].numpy() < 0.0))

        # Call with (complete) value -> Should return likelihood of "a"+"b"-values and outputs for "c"/"d"-values.
        input_ = input_space.sample(100)
        value = output_space.sample(100)
        # Delete float value ("c") from value otherwise this would create an error as we can't get a likelihood value
        # for a non-distribution float output.
        del value["c"]
        result, likelihood = nn(input_, value)
        # a is None b/c value was already given for likelihood calculation
        self.assertTrue(result["a"] is None)
        # b is None b/c value was already given for likelihood calculation
        self.assertTrue(result["b"] is None)
        # c is None b/c value was not given for "c" as it would result in ERROR (float component w/o distribution).
        self.assertTrue(result["c"] is None)
        # d are the (batched) output values for the given int-numbers
        self.assertTrue(result["d"].shape == (100, 2))
        self.assertTrue(result["d"].dtype == np.float32)

        self.assertTrue(likelihood.shape == (100,))  # (total) likelihood is some float
        self.assertTrue(likelihood.dtype == np.float32)

        # Calculate likelihood only for "a" component.
        del value["b"]
        # Extract only the "c" sample-output.
        value["c"] = None
        # We don't want outputs for "d".
        del value["d"]
        result, likelihood = nn(input_, value)
        # a and b are None as we are using these for likelihood calculations only.
        self.assertTrue(result["a"] is None)
        self.assertTrue(result["b"] is None)
        # c is a float sample.
        self.assertTrue(result["c"].shape == (100, 2))
        self.assertTrue(result["c"].dtype == np.float32)
        self.assertFalse(np.any(result["c"].numpy() > 1.0))
        self.assertFalse(np.any(result["c"].numpy() < -0.1))
        # d is None (no output desired).
        self.assertTrue(result["d"] is None)

        value = output_space.sample(100)
        # Calculate likelihood only for "b" component.
        value["a"] = None
        # Do nothing for "c".
        # Leave "d" and expect output-values for each given int.
        del value["c"]

        result, likelihood = nn(input_, value)

        self.assertTrue(result["a"] is None)
        self.assertTrue(result["b"] is None)
        self.assertTrue(result["c"] is None)

        # d are the (batched) output values for the given int-numbers
        self.assertTrue(result["d"].shape == (100, 2))
        self.assertTrue(result["d"].dtype == np.float32)

        self.assertTrue(likelihood.shape == (100,))  # (total) likelihood is some float
        self.assertTrue(likelihood.dtype == np.float32)

        # Add "c" again, but as None (will not cause ERROR then and return the output).
        value["c"] = None
        result, likelihood = nn(input_, value)
        self.assertTrue(result["c"].shape == (100, 2))
        self.assertTrue(result["c"].dtype == np.float32)
        self.assertFalse(np.any(result["c"].numpy() > 1.0))
        self.assertFalse(np.any(result["c"].numpy() < -0.1))

    def test_dueling_network(self):
        input_space = Float(-1.0, 1.0, shape=(2,), main_axes="B")
        output_space = Dict({"A": Float(shape=(4,)), "V": Float()}, main_axes="B")  # V=single node
        # Using keras layer as main network spec.
        layer = tf.keras.layers.Dense(5)

        nn = Network(
            network=layer,
            output_space=output_space,
            # Only two output components are distributions (a and b), the others not (c=Float, d=Int).
            adapters=dict(A=dict(pre_network=tf.keras.layers.Dense(2)), V=dict(pre_network=tf.keras.layers.Dense(3)))
        )
        # Simple call -> Should return sample dict.
        input_ = input_space.sample(10)
        result = nn(input_)

        weights = nn.get_weights()
        expected_a = dense(dense(dense(input_, weights[0], weights[1]), weights[2], weights[3]), weights[4], weights[5])
        expected_v = np.reshape(
            dense(dense(dense(input_, weights[0], weights[1]), weights[6], weights[7]), weights[8], weights[9]),
            newshape=(10,)
        )
        check(result["A"], expected_a, decimals=5)
        check(result["V"], expected_v, decimals=5)

    def test_func_api_network_with_manually_handling_container_input_space(self):
        # Simple vector plus image as inputs (see e.g. SAC).
        input_space = Dict(A=Float(-1.0, 1.0, shape=(2,)), B=Float(-1.0, 1.0, shape=(2, 2, 3)), main_axes="B")
        output_space = Float(shape=(3,), main_axes="B")  # simple output

        # Using keras functional API to create network.
        keras_input = input_space.create_keras_input()
        # Simply flatten an concat everything, then output.
        o = tf.keras.layers.Flatten()(keras_input["B"])
        o = tf.concat([keras_input["A"], o], axis=-1)
        network = tf.keras.Model(inputs=keras_input, outputs=o)

        # Use no distributions.
        nn = Network(
            network=network,
            output_space=output_space,
            distributions=False
        )

        # Simple function call.
        input_ = input_space.sample(6)
        result = nn(input_)
        weights = nn.get_weights()
        expected = dense(np.concatenate([input_["A"], np.reshape(input_["B"], newshape=(6, -1))], axis=-1), weights[0], weights[1])

        check(result, expected)

        # Function call with value -> Expect error as we only have float outputs (w/o distributions).
        input_ = input_space.sample(6)
        values = output_space.sample(6)
        error = True
        try:
            nn(input_, values)
            error = False
        except SurrealError:
            pass
        self.assertTrue(error)

    def test_func_api_network_with_automatically_handling_container_input_space(self):
        # Simple vectors plus image as inputs (see e.g. SAC).
        input_space = Dict(A=Float(-1.0, 1.0, shape=(2,)), B=Int(5), C=Float(-1.0, 1.0, shape=(2, 2, 3)), main_axes="B")
        output_space = Float(shape=(3,), main_axes="B")  # simple output

        # Only define a base-core network and let the automation handle the complex input structure via
        # `pre-concat` nets.
        core_nn = tf.keras.models.Sequential()
        core_nn.add(tf.keras.layers.Dense(3, activation="relu"))
        core_nn.add(tf.keras.layers.Dense(3))

        # Use no distributions.
        nn = Network(
            network=core_nn,
            input_space=input_space,
            pre_concat_networks=dict(
                # leave "A" out -> "A" input will go unaltered into concat step.
                B=lambda i: tf.one_hot(i, depth=input_space["B"].num_categories, axis=-1),
                C=tf.keras.layers.Flatten()
            ),
            output_space=output_space,
            distributions=False
        )

        # Simple function call.
        input_ = input_space.sample(6)
        result = nn(input_)
        weights = nn.get_weights()
        expected = dense(dense(relu(dense(np.concatenate([
            input_["A"],
            one_hot(input_["B"], depth=input_space["B"].num_categories),
            np.reshape(input_["C"], newshape=(6, -1))
        ], axis=-1), weights[0], weights[1])), weights[2], weights[3]), weights[4], weights[5])

        check(result, expected)

    def test_copying_a_network(self):
        # Using keras layer as network spec.
        layer = tf.keras.layers.Dense(4)

        nn = Network(network=layer, output_space=Dict({"a": Float(shape=(2,)), "b": Int(2)}))
        # Simple call -> Should return dict with "a"->float(2,) and "b"->float(2,)
        input_ = Float(-1.0, 1.0, shape=(5,), main_axes="B").sample(5)
        _ = nn(input_)
        weights = nn.get_weights()
        expected_a = dense(dense(input_, weights[0], weights[1]), weights[2], weights[3])
        expected_b = dense(dense(input_, weights[0], weights[1]), weights[4], weights[5])

        # Do the copy.
        nn_copy = nn.copy()
        result = nn_copy(input_)
        check(result, dict(a=expected_a, b=expected_b))
