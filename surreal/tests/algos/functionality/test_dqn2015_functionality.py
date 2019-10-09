# Copyright 2019 ducandu GmbH, All Rights Reserved
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

from collections import namedtuple
import copy
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import unittest

import surreal.debug as debug
# Override debug setting. Needed for some of the tests.
debug.KeepLastMemoryBatch = True

from surreal.algos.dqn2015 import DQN2015, DQN2015Config, DQN2015Loss
from surreal.components.preprocessors.preprocessor import Preprocessor
from surreal.envs import GridWorld
from surreal.tests.test_util import check
from surreal.utils.numpy import dense, one_hot


class TestDQN2015Functionality(unittest.TestCase):
    """
    Tests the DQN2015 algo functionality (loss functions, execution logic, etc.).
    """
    def test_dqn2015_loss_function(self):
        # Batch of size=2.
        input_ = {
            "x": np.random.random(size=(2, 2)),  # states don't matter for this test as Q-funcs are faked.
            "a": np.array([0, 1]),
            "r": np.array([9.4, -1.23]),
            "t": np.array([False, False]),
            "x_": np.random.random(size=(2, 2))  # states don't matter for this test as Q-funcs are faked.
        }
        # Fake q-nets. Just have to be callables, returning some q-values.
        q_net = lambda s, a: np.array([10.0, -90.6])
        target_q_net = lambda s_: np.array([[12.0, -8.0], [22.3, 10.5]])

        """
        Calculation:
        batch of 2, gamma=1.0
        Qt(s'a') = [12 -8] [22.3 10.5] -> max(a') = [12] [22.3]
        Q(s,a)  = [10.0] [-90.6]
        L = E(batch)| 0.5((r + gamma max(a')Qt(s'a') ) - Q(s,a))^2 |
        L = (0.5(9.4 + 1.0*12 - 10.0)^2 + 0.5(-1.23 + 1.0*22.3 - -90.6)^2) / 2
        L = (0.5(129.96) + 0.5(12470.1889)) / 2
        L = (64.98 + 6235.09445) / 2
        L = 3150.037225
        """
        # Batch size=2 -> Expect 2 values returned by `loss_per_item`.
        expected_loss_per_item = np.array([64.979996, 6235.09445], dtype=np.float32)
        # Expect the mean over the batch.
        expected_loss = expected_loss_per_item.mean()
        out = DQN2015Loss()(input_, q_net, target_q_net, namedtuple("FakeDQN2015Config", ["gamma"])(gamma=1.0))
        check(out.numpy(), expected_loss, decimals=2)

    def test_dqn2015_functionality(self):
        # Fake q-net/qt-net used for this test.
        def q(s, a):
            return np.sum(dense(dense(s, weights_q[0], weights_q[1]), weights_q[2], weights_q[3]) * one_hot(a, depth=4), axis=-1)

        def qt(s):
            return dense(dense(s, weights_qt[0], weights_qt[1]), weights_qt[2], weights_qt[3])

        env = GridWorld("2x2", actors=1)
        state_space = env.actors[0].state_space.with_batch()
        action_space = env.actors[0].action_space.with_batch()

        # Add the preprocessor.
        preprocessor = Preprocessor(
            lambda inputs_: tf.one_hot(inputs_, depth=state_space.num_categories)
        )
        preprocessed_space = preprocessor(state_space)

        # Add the Q-network.
        i = K.layers.Input(shape=preprocessed_space.shape, dtype=preprocessed_space.dtype)
        o = K.layers.Dense(2, activation="linear")(i)  # keep it very simple
        # o = K.layers.Dense(256)(o)
        q_network = K.Model(inputs=i, outputs=o)

        # Create a very simple DQN2015.
        dqn = DQN2015(config=DQN2015Config.make(
            "../configs/dqn2015_grid_world_2x2_functionality.json",
            preprocessor=preprocessor,
            q_network=q_network,
            state_space=state_space,
            action_space=action_space
        ), name="my-dqn")

        check(dqn.Q.get_weights(), dqn.Qt.get_weights())

        # Point actor(s) to the algo.
        for actor in env.actors:
            actor.set_algo(dqn)

        # Set our weights fixed.
        weights = [
            np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]]),  # hidden layer kernel
            np.array([0.0, 0.0]),  # hidden layer bias
            np.array([[-0.1, -0.2, -0.3, -0.4], [0.1, 0.2, 0.3, 0.4]]),  # output layer kernel
            np.array([0.0, 0.0, 0.0, 0.0])  # output layer bias
        ]
        dqn.Q.set_weights(weights)

        # Perform one step in the env.
        expected_action = np.argmax(dqn.Q(dqn.Phi(env.state)), axis=-1)
        env.run(ticks=1)  # ts=0 -> do nothing
        # Check action taken.
        check(dqn.a.value, expected_action)
        # Check state of the env after action taken.
        check(env.state[0], 1)
        check(env.reward[0], -0.1)
        check(env.terminal[0], False)
        # Check memory of dqn (after one time step, should still be empty).
        check(dqn.memory.size, 0)

        # Perform one step in the env.
        expected_action = np.argmax(dqn.Q(dqn.Phi(env.state)), axis=-1)
        env.run(ticks=1)  # ts=1 -> no sync, no update
        # Check action taken.
        check(dqn.a.value, expected_action)
        # Check state of the env after action taken.
        check(env.state[0], 1)
        check(env.reward[0], -0.1)
        check(env.terminal[0], False)
        # Check memory of dqn.
        check(dqn.memory.size, 1)
        check(dqn.memory.memory, [
            np.array([2, 0, 0, 0]),
            np.array([-0.1, 0., 0., 0.]),
            np.array([False, False, False, False]),
            np.array([[1., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]),
            np.array([[0., 1., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]])
        ])

        # Perform one step in the env.
        # What are the weights after the update?
        weights_q_before_update = dqn.Q.get_weights()
        weights_q = copy.deepcopy(weights_q_before_update)
        weights_qt = dqn.Qt.get_weights()

        # Check action taken (action is picked before! update).
        expected_action = np.argmax(dqn.Q(dqn.Phi(np.array([1]))), axis=-1)

        env.run(ticks=1)  # ts=2 -> no sync, do update
        weights_q_after_update = dqn.Q.get_weights()
        check(dqn.a.value, expected_action)

        # Check new weight values after the update.
        loss = DQN2015Loss()(dqn.memory.last_records_pulled, q, qt, dqn.config)
        for i, matrix in enumerate(weights_q_before_update):
            for idx in np.ndindex(matrix.shape):
                weights_q = copy.deepcopy(weights_q_before_update)
                weights_q[i][idx] += 0.0001
                lossd = DQN2015Loss()(dqn.memory.last_records_pulled, q, qt, dqn.config)
                dL_over_dw = (lossd - loss) / 0.0001
                check(weights_q_after_update[i][idx], weights_q_before_update[i][idx] - dL_over_dw * dqn.optimizer.learning_rate(0.0), decimals=4)

        # Check state of the env after action taken.
        check(env.state[0], 1)
        check(env.reward[0], -0.1)
        check(env.terminal[0], False)
        # Check memory of dqn.
        check(dqn.memory.size, 2)
        check(dqn.memory.memory, [
            np.array([2, 2, 0, 0]),
            np.array([-0.1, -0.1, 0., 0.]),
            np.array([False, False, False, False]),
            np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]),
            np.array([[0., 1., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]])
        ])

        env.terminate()
