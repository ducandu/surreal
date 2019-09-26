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


from surreal.algos.rl_algo import RLAlgo
from surreal.spaces import Space
from surreal.makeable import Makeable


class Actor(Makeable):
    """
    An Actors lives inside an Env, has a BrainID (linking it to some RLAlgo) and can pass actions from the
    algorithm back to the Env.
    """
    def __init__(self, name, state_space, action_space, reward_space=None, rl_algo=None):
        """
        Args:
            name (str): Some name for this Actor.
            state_space (Space): The state Space that this Actor will receive from the Env.
            action_space (Space): The action Space that this Actor will be able to execute on.
            reward_space (Optional[Space]: The reward space that this actor will use.
                Default: float.

            rl_algo (Optional[RLAlgo]): The RLAlgo that this Actor will query for actions given some observation
                state from the Env.
        """
        super().__init__()

        # Some unique name for this Actor.
        self.name = name
        # The Algo controlling this Actor.
        self.rl_algo = rl_algo  # type: RLAlgo

        # The state Space (observations of this Actor).
        self.state_space = Space.make(state_space)
        # The action Space.
        self.action_space = Space.make(action_space)
        # The reward Space (will default to float if None).
        self.reward_space = Space.make(reward_space)
        #if reward_space is None:
        #    self.reward_space = Float(main_axes="B")

    def set_algo(self, rl_algo):
        """
        Sets the RLAlgo of this Actor.

        Args:
            rl_algo (RLAlgo): The (new) RLAlgo to use.
        """
        self.rl_algo = rl_algo
