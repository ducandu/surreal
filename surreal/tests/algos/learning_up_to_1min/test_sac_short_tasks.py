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

from surreal.algos.sac import SAC, SACConfig
from surreal.components.preprocessors.preprocessor import Preprocessor
import surreal.debug as debug
from surreal.envs import OpenAIGymEnv, GridWorld
from surreal.tests.test_util import check


class TestSACShortLearningTasks(unittest.TestCase):
    """
    Tests the SAC algo on shorter-than-1min learning problems.
    """
    def test_sac_learning_on_grid_world_2x2(self):
        # Create an Env object.
        env = GridWorld("2x2", actors=1)

        # Add the preprocessor (not really necessary, as NN will automatically one-hot, but faster as states
        # are then stored in memory already preprocessed and won't have to be preprocessed again for batch-updates).
        preprocessor = Preprocessor(
            lambda inputs_: tf.one_hot(inputs_, depth=env.actors[0].state_space.num_categories)
        )

        # Create a Config.
        config = SACConfig.make(
            "../configs/sac_grid_world_2x2_learning.json",
            preprocessor=preprocessor,
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )

        # Create an Algo object.
        algo = SAC(config=config, name="my-sac")

        # Point actor(s) to the algo.
        for actor in env.actors:
            actor.set_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=1500, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        mean_last_10 = np.mean(env.historic_episodes_returns[-10:])
        print("Avg return over last 10 episodes: {}".format(mean_last_10))
        self.assertTrue(mean_last_10 >= 0.8)

        env.terminate()

    def test_sac_learning_on_cart_pole_with_n_actors(self):
        # Create an Env object.
        env = OpenAIGymEnv("CartPole-v0", actors=1)

        # Create a Config.
        config = SACConfig.make(
            "../configs/sac_cart_pole_learning_n_actors.json",
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )

        # Create an Algo object.
        algo = SAC(config=config, name="my-sac")

        # Point actor(s) to the algo.
        for actor in env.actors:
            actor.set_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=3000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        last_n = 10
        mean_last_episodes = np.mean(env.historic_episodes_returns[-last_n:])
        print("Avg return over last {} episodes: {}".format(last_n, mean_last_episodes))
        self.assertTrue(mean_last_episodes > 160.0)

        env.terminate()

    def test_sac_learning_on_pendulum(self):
        # Create an Env object.
        env = OpenAIGymEnv("Pendulum-v0", actors=1)

        # Create a Config.
        config = SACConfig.make(
            "../configs/sac_pendulum_learning.json",
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )

        # Create an Algo object.
        algo = SAC(config=config, name="my-sac")

        # Point actor(s) to the algo.
        for actor in env.actors:
            actor.set_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=10000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        mean_last_10 = np.mean(env.historic_episodes_returns[-10:])
        print("Avg return over last 10 episodes: {}".format(mean_last_10))
        self.assertTrue(mean_last_10 >= -200.0)

        env.terminate()
