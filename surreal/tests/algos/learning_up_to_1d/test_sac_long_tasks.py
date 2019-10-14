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

import numpy as np
import unittest

from surreal.algos.sac import SAC, SACConfig
from surreal.components.preprocessors import Preprocessor, GrayScale, ImageResize, ImageCrop, Sequence
import surreal.debug as debug
from surreal.envs import OpenAIGymEnv


class TestSACLongLearningTasks(unittest.TestCase):
    """
    Tests the SAC algo on up-to-1-day learning problems.
    """
    def test_sac_learning_on_space_invaders(self):
        # Create an Env object.
        env = OpenAIGymEnv(
            "SpaceInvaders-v4", actors=64, fire_after_reset=False, episodic_life=True, max_num_noops_after_reset=6,
            frame_skip=(2, 5)
        )

        preprocessor = Preprocessor(
            ImageCrop(x=5, y=29, width=150, height=167),
            GrayScale(keepdims=True),
            ImageResize(width=84, height=84, interpolation="bilinear"),
            lambda inputs_: ((inputs_ / 128) - 1.0).astype(np.float32),  # simple preprocessor: [0,255] to [-1.0,1.0]
            Sequence(sequence_length=4, adddim=False)
        )
        # Create a DQN2015Config.
        config = SACConfig.make(
            "../configs/sac_space_invaders_learning.json",
            preprocessor=preprocessor,
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space,
            summaries=["Ls_critic[0]", "Ls_critic[1]", "L_actor", "L_alpha", "alpha", ("actions", "a_soft.value[0]")]
        )
        # Create an Algo object.
        algo = SAC(config=config, name="my-sac")

        # Point actor(s) to the algo.
        for actor in env.actors:
            actor.set_algo(algo)

        # Run and wait for env to complete.
        env.run(ticks=100000, sync=True, render=debug.RenderEnvInLearningTests)

        # Check last n episode returns.
        n = 10
        mean_last_10 = np.mean(env.historic_episodes_returns[-n:])
        print("Avg return over last 10 episodes: {}".format(mean_last_10))
        self.assertTrue(mean_last_10 > 150.0)

        env.terminate()
