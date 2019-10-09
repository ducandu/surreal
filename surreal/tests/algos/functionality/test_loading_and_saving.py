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

import tensorflow as tf
import unittest

from surreal.algos.dqn2015 import DQN2015, DQN2015Config
from surreal.envs import GridWorld


class TestLoadingAndSavingOfAlgos(unittest.TestCase):
    """
    Tests loading and saving (with and without weights) of the Algo class.
    """
    def test_saving_then_loading_to_get_exact_same_algo(self):
        env = GridWorld("2x2", actors=1)
        state_space = env.actors[0].state_space.with_batch()
        action_space = env.actors[0].action_space.with_batch()

        # Create a very simple DQN2015.
        dqn = DQN2015(config=DQN2015Config.make(
            "../configs/dqn2015_grid_world_2x2_learning.json",
            preprocessor=lambda inputs_: tf.one_hot(inputs_, depth=state_space.num_categories),
            state_space=state_space,
            action_space=action_space
        ), name="my-dqn")

        # Point actor(s) to the algo.
        for actor in env.actors:
            actor.set_algo(dqn)

        dqn.save("test.json")

        env.terminate()