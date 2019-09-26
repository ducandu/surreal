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
import time

from surreal.envs import Env
import surreal.spaces as spaces


class RandomEnv(Env):
    """
    An Env producing random states no matter what actions come in.
    """
    def __init__(self, state_space, action_space, reward_space=None, terminal_prob=0.1, deterministic=False):
        """
        Args:
            reward_space (Union[dict,Space]): The reward Space from which to randomly sample for each step.
            terminal_prob (Union[dict,Space]): The probability with which an episode ends for each step.
            deterministic (bool): Convenience flag to seed the environment automatically upon construction.
        """
        super(RandomEnv, self).__init__()

        self.state_space = spaces.Space.make(state_space)
        self.action_space = spaces.Space.make(action_space)

        self.reward_space = spaces.Space.make(reward_space)
        self.terminal_prob = terminal_prob

        if deterministic is True:
            np.random.seed(10)
        self.last_state = np.random.get_state()

    def seed(self, seed=None):
        if seed is None:
            seed = time.time()
        np.random.seed(seed)
        self.last_state = np.random.get_state()
        return seed

    def reset(self):
        return self.step()[0]  # 0=state

    def reset_flow(self):
        return self.reset()

    def step(self, actions=None):
        if actions is not None:
            assert self.action_space.contains(actions), \
                "ERROR: Given action ({}) in step is not part of action Space ({})!".format(actions, self.action_space)

        # Set the seed to the last observed state for this instance.
        np.random.set_state(self.last_state)
        # Do the random sampling (using numpy).
        state = self.state_space.sample()
        reward = self.reward_space.sample()
        terminal = np.random.choice([True, False], p=[self.terminal_prob, 1.0 - self.terminal_prob])
        # Store the current state of the RNG.
        self.last_state = np.random.get_state()
        return state, reward, terminal, None

    def step_flow(self, actions=None):
        ret = self.step(actions)
        return ret[0], ret[1], ret[2]

    def __str__(self):
        return "RandomEnv()"
