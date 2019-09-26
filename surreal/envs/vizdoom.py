# Copyright 2019 ducandu GmbH, All Rights Reserved (this is a modified version of the Apache 2.0 licensed RLgraph file of the same name).
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

import vizdoom
import numpy as np
import time

from surreal.utils.rlgraph_errors import RLGraphError
from surreal.utils.util import convert_dtype as dtype_
from surreal.envs import Env
from surreal.spaces import *


class VizDoomEnv(Env):
    """
    VizDoom Integration: https://github.com/mwydmuch/ViZDoom.
    """
    def __init__(self, config_file, visible=False, mode=vizdoom.Mode.PLAYER, screen_format=vizdoom.ScreenFormat.GRAY8, screen_resolution=vizdoom.ScreenResolution.RES_640X480):
        """
        Args:
            config_file (str): The config file to configure the DoomGame object.
            visible (bool): 
            mode (vizdoom.Mode): The playing mode of the game.
            screen_format (vizdoom.ScreenFormat): The screen (pixel) format of the game.
            screen_resolution (vizdoom.ScreenResolution): The screen resolution (width x height) of the game.
        """
        # Some restrictions on the settings.
        assert screen_format in [vizdoom.ScreenFormet.RGB24, vizdoom.ScreedFormat.GRAY8], "ERROR: `screen_format` must be either GRAY8 or RGB24!"
        assert screen_resolution in [vizdoom.ScreenResolution.RES_640X480], "ERROR: `screen_resolution` must be 640x480!"

        self.game = vizdoom.DoomGame()
        self.game.load_config(config_file)
        self.game.set_window_visible(False)
        self.game.set_mode(mode)
        self.game.set_screen_format(screen_format)
        self.game.set_screen_resolution(screen_resolution)
        self.game.init()
 
        # Calculate action and state Spaces for Env c'tor.
        state_space = Int(255, shape=(480, 480, 1 if screen_format == vizdoom.ScreenFormat.GRAY8 else 3))  # image of size [resolution] with [screen-format] channels
        action_space = Int(1, shape=(self.game.get_available_buttons_size(),))

        super(VizDoomEnv, self).__init__(state_space=state_space, action_space=action_space)

    def seed(self, seed=None):
        if seed is None:
            seed = time.time()
        self.game.set_seed(seed)
        return seed

    def reset(self):
        self.game.newEpisode()
        return self.game.getState, 0.0, self.game.isEpisodeFinished(), dict(is_dead=self.game.isPlayerDead, )

    def terminate(self):
        #self.gym_env.close()
        self.game = None

    def step(self, actions=None):
        if self.visualize:
            self.gym_env.render()
        state, reward, terminal, info = self.gym_env.act(actions)
        return state, reward, terminal, info

    @staticmethod
    def translate_space(space):
        """
        Translates an openAI space into an RLGraph Space object.

        Args:
            space (gym.spaces.Space): The openAI Space to be translated.

        Returns:
            Space: The translated Rlgraph Space.
        """
        if isinstance(space, gym.spaces.Discrete):
            if space.n == 2:
                return Bool()
            else:
                return Int(space.n)
        elif isinstance(space, gym.spaces.MultiBinary):
            return Bool(shape=(space.n,))
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return Int(low=np.zeros((space.nvec.ndim,), dtype_("uint8", "np")), high=space.nvec)
        elif isinstance(space, gym.spaces.Box):
            return Float(low=space.low, high=space.high)
        elif isinstance(space, gym.spaces.Tuple):
            return Tuple(*[OpenAIGymEnv.translate_space(s) for s in space.spaces])
        elif isinstance(space, gym.spaces.Dict):
            return Dict({k: OpenAIGymEnv.translate_space(v) for k, v in space.spaces.items()})
        else:
            raise RLGraphError("Unknown openAI gym Space class for state_space!")

    def __str__(self):
        return "OpenAIGym({})".format(self.gym_env)

