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

import unittest

from surreal.envs.grid_world import GridWorld
from surreal.tests.test_util import check


class TestGridWorld(unittest.TestCase):
    """
    Tests creation, resetting and stepping through a deterministic GridWorld.
    """
    def test_2x2_grid_world_with_2_actors(self):
        """
        Tests a minimalistic 2x2 GridWorld with two Actors.
        """
        env = GridWorld(world="2x2", actors=2)

        # Simple test runs with fixed actions.
        # X=player's position
        s = env._reset(0)  # ["XH", " G"]  X=player's position
        check(s, 0)
        s = env._reset(1)  # ["XH", " G"]  X=player's position
        check(s, 0)

        env.act([2, 1])  # down: [" H", "XG"], # right: [" X", " G"]
        check(env.state, [1, 0])
        check(env.reward, [-0.1, -5.0])
        check(env.terminal, [False, True])
        env.act([1, 2])  # right: [" H", " X"], # down: [" H", "XG"]
        check(env.state, [0, 1])  # 0=state got already reset (flow envs).
        check(env.reward, [1.0, -0.1])
        check(env.terminal, [True, False])

        s = env._reset(0)
        check(s, 0)
        s = env._reset(1)
        check(s, 0)
        env.act([1, 1])  # both Actors move right: [" X", " G"] -> in the hole
        check(s, [0, 0])
        check(env.reward, [-5.0, -5.0])
        check(env.terminal, [True, True])

        # Run against a wall.
        s = env._reset(0)  # ["XH", " G"]  X=player's position
        check(s, 0)

        env.act([3, 0])  # left: ["XH", " G"], up: ["XH", " G"]
        check(env.state, [0, 0])
        check(env.reward, [-0.1, -0.1])
        check(env.terminal, [False, False])
        env.act([2, 0])  # down: [" H", "XG"], up: ["XH", " G"]
        check(env.state, [1, 0])
        check(env.reward, [-0.1, -0.1])
        check(env.terminal, [False, False])
        env.act([0, 2])  # up: ["XH", " G"], down: [" H", "XG"]
        check(env.state, [0, 1])
        check(env.reward, [-0.1, -0.1])
        check(env.terminal, [False, False])
        env.act([1, 1])  # right: [" X", " G"], right: [" H", " X"]
        check(env.state, [0, 0])
        check(env.reward, [-5.0, 1.0])
        check(env.terminal, [True, True])
