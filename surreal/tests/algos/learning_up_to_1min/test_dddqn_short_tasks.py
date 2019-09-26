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
import numpy as np
import unittest

from surreal.algos.dddqn import DDDQN, DDDQNConfig, dueling
from surreal.components import Preprocessor
import surreal.debug as debug
from surreal.envs import GridWorld, OpenAIGymEnv
from surreal.utils.numpy import one_hot
from surreal.tests.test_util import check


class TestDDDQNShortLearningTasks(unittest.TestCase):
    """
    Tests the DDDQN algo on shorter-than-1min learning problems.
    """
    def test_learning_on_grid_world(self):
        # Create an Env object.
        env = GridWorld("2x2", actors=1)

        # Add the preprocessor.
        preprocessor = Preprocessor(
            lambda inputs_: tf.one_hot(inputs_, depth=env.actors[0].state_space.num_categories)
        )
        # Create a DQN2015Config.
        dqn_config = DDDQNConfig.make(
            "../configs/dddqn_gridworld_2x2_learning.json",
            preprocessor=preprocessor,
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )

        # Create an Algo object.
        algo = DDDQN(config=dqn_config, name="my-dddqn")

        # Point actor(s) to the algo.
        for actor in env.actors:
            actor.set_algo(algo)

        # Run and wait for env to complete.
        # TODO: Maybe we need 500 steps more to learn correct Q-values consistently.
        env.run(ticks=1000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        mean_last_10 = np.mean(env.historic_episodes_returns[-10:])
        print("Avg return over last 10 episodes: {}".format(mean_last_10))
        self.assertTrue(mean_last_10 >= 0.6)

        # Check learnt Q-function (using our dueling layer).
        a_and_v = algo.Q(one_hot(np.array([0, 0, 0, 0, 1, 1,  1, 1]), depth=4))
        q = dueling(a_and_v, np.array([0, 1, 2, 3, 0, 1, 2, 3]), 4)
        check(q, [0.8, -5.0, 0.9, 0.8, 0.8, 1.0, 0.9, 0.9], decimals=1)  # a=up,down,left,right

    def test_learning_on_cartpole_with_4_actors(self):
        # Create an Env object.
        env = OpenAIGymEnv("CartPole-v0", actors=4)

        # Create a DQN2015Config.
        dqn_config = DDDQNConfig.make(
            "../configs/dddqn_cartpole_learning_4_actors.json",
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )

        # Create an Algo object.
        algo = DDDQN(config=dqn_config, name="my-dqn")

        # Point actor(s) to the algo.
        for actor in env.actors:
            actor.set_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=2000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        mean_last_10 = np.mean(env.historic_episodes_returns[-10:])
        print("Avg return over last 10 episodes: {}".format(mean_last_10))
        self.assertTrue(mean_last_10 > 130.0)
