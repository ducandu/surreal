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


from abc import ABCMeta, abstractmethod

from surreal.actors.actor import Actor
from surreal.envs.env import Env


class SingleActorEnv(Env, metaclass=ABCMeta):
    """
    An Env that can naturally only support one actor (usually the "player"). To integrate it into the Env API
    - if more than one Actors is required - these envs will multiply themselves and thus emulate a regular
    multi-Actor Env.
    """
    def __init__(self, actors, state_space, action_space, reward_space=None, **kwargs):
        """
        Args:
            state_space (Space): The (single) state Space used for all copies of this env and their single Actors.
            action_space (Space): The (single) action Space used for all copies of this env and their single Actors.
        """
        # Actors given as number. Create new Actors.
        if isinstance(actors, int):
            num_actors = actors
            actors = []
            for i in range(num_actors):
                actors.append(Actor(
                    name="actor{}".format(i),
                    action_space=action_space, state_space=state_space,
                    reward_space=reward_space
                ))

        # Call the super's constructor.
        super().__init__(actors=actors, **kwargs)

    @abstractmethod
    def _reset(self, actor_slot, **kwargs):
        """
        Resets an internal (single actor) env at the given slot.

        Args:
            actor_slot (int): The Actor, whose internal env to reset.

        Returns:
            any: The (single Actor/non-batched) state after the reset.
        """
        raise NotImplementedError
