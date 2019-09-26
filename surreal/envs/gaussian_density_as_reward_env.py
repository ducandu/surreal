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
from scipy import stats

from surreal.envs.env import Env
from surreal.spaces import Float


class GaussianDensityAsRewardEnv(Env):
    """
    Env where the reward is always the Gaussian probability density at the given
    single-float action.
    The state is a uniformly random value between -1 and 1 and determines the loc of the pdf.
    The scale of the pdf is fixed in the ctor.

    The optimal policy would be to pick actions that are closest to the current state.
    """
    def __init__(self, episode_length=5, scale=0.1):
        super(GaussianDensityAsRewardEnv, self).__init__(
            state_space=Float(shape=(1,)), action_space=Float(shape=(1,), low=-2.0, high=2.0)
        )
        self.episode_length = episode_length
        self.episode_step = 0
        self.loc = None
        self.scale = scale

    def seed(self, seed=None):
        pass

    def reset(self):
        self.episode_step = 0
        self.loc = np.random.uniform(size=(1, )) * 2 - 1
        return self.loc

    def step(self, actions, **kwargs):
        reward = stats.norm.pdf(actions, loc=self.loc, scale=self.scale)[0]
        self.episode_step += 1
        self.loc = np.random.uniform(size=(1,)) * 2 - 1
        return self.loc, reward, self.episode_step >= self.episode_length, None

    def get_max_reward(self):
        max_reward_per_step = stats.norm(loc=0.0, scale=self.scale).pdf(0.0)
        return self.episode_length * max_reward_per_step

    def __str__(self):
        return self.__class__.__name__ + "(episode-len={}, scale={})".format(self.episode_length, self.scale)
