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

import logging
import numpy as np
import os
import tensorflow as tf
import unittest

from surreal.algos.dddqn import DDDQN, DDDQNConfig, dueling
from surreal.components import Preprocessor
import surreal.debug as debug
from surreal.envs import GridWorld, OpenAIGymEnv
from surreal.tests.test_util import check
from surreal.utils.numpy import one_hot


class TestDDDQNShortLearningTasks(unittest.TestCase):
    """
    Tests the DDDQN algo on shorter-than-1min learning problems.
    """
    logging.getLogger().setLevel(logging.INFO)

    def test_dddqn_learning_on_grid_world_2x2(self):
        # Create an Env object.
        env = GridWorld("2x2", actors=1)

        # Add the preprocessor.
        preprocessor = Preprocessor(
            lambda inputs_: tf.one_hot(inputs_, depth=env.actors[0].state_space.num_categories)
        )
        # Create a Config.
        dqn_config = DDDQNConfig.make(
            "{}/../configs/dddqn_grid_world_2x2_learning.json".format(os.path.dirname(__file__)),
            preprocessor=preprocessor,
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )

        # Create an Algo object.
        algo = DDDQN(config=dqn_config, name="my-dddqn")

        # Point actor(s) to the algo.
        env.point_all_actors_to_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=1500, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        n = 10
        mean_last_n = np.mean(env.historic_episodes_returns[-n:])
        print("Avg return over last {} episodes: {}".format(n, mean_last_n))
        self.assertTrue(mean_last_n >= 0.6)

        # Check learnt Q-function (using our dueling layer).
        a_and_v = algo.Q(one_hot(np.array([0, 0, 0, 0, 1, 1, 1, 1]), depth=4))
        q = dueling(a_and_v, np.array([0, 1, 2, 3, 0, 1, 2, 3]))
        print(q)
        self.assertTrue(q[1] < min(q[2:]) and q[1] < q[0])  # q(s=0,a=right) is the worst
        check(q[5], 1.0, atol=0.2)  # Q(1,->) is close to 1.0.
        #self.assertTrue(q[5] > max(q[:4]) and q[5] > max(q[6:]))  # q(s=1,a=right) is the best
        #check(q, [0.8, -5.0, 0.9, 0.8, 0.8, 1.0, 0.9, 0.9], decimals=1)  # a=up,down,left,right

        env.terminate()

    def test_dddqn_learning_on_cart_pole_with_4_actors(self):
        # Create an Env object.
        env = OpenAIGymEnv("CartPole-v0", actors=4)

        # Create a Config.
        dqn_config = DDDQNConfig.make(
            "{}/../configs/dddqn_cart_pole_learning_n_actors.json".format(os.path.dirname(__file__)),  # TODO: filename wrong (num actors)
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )

        # Create an Algo object.
        algo = DDDQN(config=dqn_config, name="my-dqn")

        # Point actor(s) to the algo.
        env.point_all_actors_to_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=2000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        last_n = 10
        mean_last_episodes = np.mean(env.historic_episodes_returns[-last_n:])
        print("Avg return over last {} episodes: {}".format(last_n, mean_last_episodes))
        self.assertTrue(mean_last_episodes > 160.0)

        env.terminate()
