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

from surreal.algos.dqn2015 import DQN2015, DQN2015Config
from surreal.components import Preprocessor
import surreal.debug as debug
from surreal.envs import OpenAIGymEnv, GridWorld
from surreal.tests.test_util import check


class TestDQN2015ShortLearningTasks(unittest.TestCase):
    """
    Tests the DQN2015 algo on shorter-than-1min learning problems.
    """
    def test_learning_on_grid_world(self):
        # Create an Env object.
        env = GridWorld("2x2", actors=1)

        # Add the preprocessor.
        preprocessor = Preprocessor(
            lambda inputs_: tf.one_hot(inputs_, depth=env.actors[0].state_space.num_categories)
        )
        # Create a DQN2015Config.
        dqn_config = DQN2015Config.make(  # type: DQN2015Config
            "../configs/dqn2015_gridworld_2x2_learning.json",
            preprocessor=preprocessor,
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )

        # Create an Algo object.
        algo = DQN2015(config=dqn_config, name="my-dqn")

        # Point actor(s) to the algo.
        for actor in env.actors:
            actor.set_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=3000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        mean_last_10 = np.mean(env.historic_episodes_returns[-10:])
        print("Avg return over last 10 episodes: {}".format(mean_last_10))
        self.assertTrue(mean_last_10 >= 0.3)

        # Check learnt Q-function.
        check(algo.Q(
            np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        ), [[0.8, -5.0, 0.9, 0.8], [0.8, 1.0, 0.9, 0.9]], decimals=1)  # a=up,down,left,right

    def test_learning_on_4x4_grid_world_with_8_actors(self):
        # Create an Env object.
        env = GridWorld("4x4", actors=8)

        # Add the preprocessor.
        preprocessor = Preprocessor(
            lambda inputs_: tf.one_hot(inputs_, depth=env.actors[0].state_space.num_categories)
        )

        # Create a DQN2015Config.
        dqn_config = DQN2015Config.make(  # type: DQN2015Config
            "../configs/dqn2015_gridworld_4x4_learning_8_actors.json",
            preprocessor=preprocessor,
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )

        # Create an Algo object.
        algo = DQN2015(config=dqn_config, name="my-dqn")

        # Point actor(s) to the algo.
        for actor in env.actors:
            actor.set_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=3000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        mean_last_10 = np.mean(env.historic_episodes_returns[-10:])
        print("Avg return over last 10 episodes: {}".format(mean_last_10))
        self.assertTrue(mean_last_10 >= 0.1)

        # Check learnt Q-function for states 0 and 1, action=down (should be larger 0.0, ideally 0.5).
        action_values = algo.Q(preprocessor(np.array([0, 1])))
        self.assertTrue(action_values[0][2] >= 0.0)
        self.assertTrue(action_values[1][2] >= 0.0)

    def test_learning_on_cartpole_with_4_actors(self):
        # Create an Env object.
        env = OpenAIGymEnv("CartPole-v0", actors=4)

        # Create a DQN2015Config.
        dqn_config = DQN2015Config.make(  # type: DQN2015Config
            "../configs/dqn2015_cartpole_learning_4_actors.json",
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )

        # Create an Algo object.
        algo = DQN2015(config=dqn_config, name="my-dqn")

        # Point actor(s) to the algo.
        for actor in env.actors:
            actor.set_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=3000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        mean_last_10 = np.mean(env.historic_episodes_returns[-10:])
        print("Avg return over last 10 episodes: {}".format(mean_last_10))
        self.assertTrue(mean_last_10 > 130.0)
