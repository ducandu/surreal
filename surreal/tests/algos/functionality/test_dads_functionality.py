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

import logging
import os
import unittest

from surreal.algos.dads import DADS, DADSConfig
from surreal.envs import GridWorld


class TestDADSFunctionality(unittest.TestCase):
    """
    Tests the SAC algo functionality (loss functions, execution logic, etc.).
    """
    logging.getLogger().setLevel(logging.INFO)

    def test_dads_compilation(self):
        """
        Tests the c'tor of SAC.
        """
        env = GridWorld("4-room", actors=2)
        # Create a Config (for any Atari game).
        config = DADSConfig.make(
            "{}/../configs/dads_grid_world_4room_learning.json".format(os.path.dirname(__file__)),
            state_space=env.actors[0].state_space,
            action_space=env.actors[0].action_space
        )
        dads = DADS(config, name="my-dads")
        print("DADS built ({}).".format(dads))

