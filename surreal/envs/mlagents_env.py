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

import numpy as np
from mlagents.envs.environment import UnityEnvironment

from surreal.envs.vector_env import VectorEnv
from surreal.spaces import Dict, Tuple, Int
from surreal.spaces.space_utils import get_space_from_data


class MLAgentsEnv(VectorEnv):
    """
    An Env sitting behind a tcp connection and communicating through this adapter.
    Note: Communication between Unity and Python takes place over an open socket without authentication.
    Ensure that the network where training takes place is secure.
    """
    def __init__(self, file_name=None, worker_id=0, base_port=5005, seed=0, docker_training=False, no_graphics=False,
                 timeout_wait=30, train_mode=True, **kwargs):
        """
        Args:
            file_name (Optional[str]): Name of Unity environment binary.
            base_port (int): Port number to connect to Unity environment. `worker_id` increments on top of this.
            worker_id (int): Number to add to `base_port`. Used for asynchronous agent scenarios.
            docker_training (bool): Informs this class, whether the process is being run within a container.
                Default: False.
            no_graphics (bool): Whether to run the Unity simulator in no-graphics mode. Default: False.
            timeout_wait (int): Time (in seconds) to wait for connection from environment.
            train_mode (bool): Whether to run in training mode, speeding up the simulation. Default: True.
        """
        # First create the UnityMLAgentsEnvironment to get state and action core, then create RLgraph Env
        # instance.
        self.mlagents_env = UnityEnvironment(
            file_name, worker_id, base_port, seed, docker_training, no_graphics
        )
        all_brain_info = self.mlagents_env.reset()
        # Get all possible information from AllBrainInfo.
        # TODO: Which scene do we pick?
        self.scene_key = next(iter(all_brain_info))
        first_brain_info = all_brain_info[self.scene_key]
        num_environments = len(first_brain_info.agents)

        state_space = {}
        if len(first_brain_info.vector_observations[0]) > 0:
            state_space["vector"] = get_space_from_data(first_brain_info.vector_observations[0])
            # TODO: This is a hack.
            if state_space["vector"].dtype == np.float64:
                state_space["vector"].dtype = np.float32
        if len(first_brain_info.visual_observations) > 0:
            state_space["visual"] = get_space_from_data(first_brain_info.visual_observations[0])
        if first_brain_info.text_observations[0]:
            state_space["text"] = get_space_from_data(first_brain_info.text_observations[0])

        if len(state_space) == 1:
            self.state_key = next(iter(state_space))
            state_space = state_space[self.state_key]
        else:
            self.state_key = None
            state_space = Dict(state_space)
        brain_params = next(iter(self.mlagents_env.brains.values()))
        if brain_params.vector_action_space_type == "discrete":
            highs = brain_params.vector_action_space_size
            # MultiDiscrete (Tuple(Int)).
            if any(h != highs[0] for h in highs):
                action_space = Tuple([Int(h) for h in highs])
            # Normal Int:
            else:
                action_space = Int(
                    low=np.zeros_like(highs, dtype=np.int32),
                    high=np.array(highs, dtype=np.int32),
                    shape=(len(highs),)
                )
        else:
            action_space = get_space_from_data(first_brain_info.action_masks[0])
        if action_space.dtype == np.float64:
            action_space.dtype = np.float32

        super(MLAgentsEnv, self).__init__(
            num_environments=num_environments, state_space=state_space, action_space=action_space, **kwargs
        )

        # Caches the last observation we made (after stepping or resetting).
        self.last_state = None

    def get_env(self):
        return self

    def reset(self, index=0):
        # Reset entire MLAgentsEnv iff global_done is True.
        if self.mlagents_env.global_done is True or self.last_state is None:
            self.reset_all()
        return self.last_state[index]

    def reset_all(self):
        all_brain_info = self.mlagents_env.reset()
        self.last_state = self._get_state_from_brain_info(all_brain_info)
        return self.last_state

    def step(self, actions, text_actions=None, **kwargs):
        # MLAgents Envs don't like tuple-actions.
        if isinstance(actions[0], tuple):
            actions = [list(a) for a in actions]
        all_brain_info = self.mlagents_env.act(
            # TODO: Only support vector actions for now.
            vector_action=actions, memory=None, text_action=text_actions, value=None
        )
        self.last_state = self._get_state_from_brain_info(all_brain_info)
        r = self._get_reward_from_brain_info(all_brain_info)
        t = self._get_terminal_from_brain_info(all_brain_info)
        return self.last_state, r, t, None

    def render(self):
        # TODO: If no_graphics is True, maybe user can render through this method manually?
        pass

    def terminate(self):
        self.mlagents_env.close()

    def terminate_all(self):
        return self.terminate()

    def __str__(self):
        return "MLAgentsEnv(port={}{})".format(
            self.mlagents_env.port, " [loaded]" if self.mlagents_env._loaded else ""
        )

    def _get_state_from_brain_info(self, all_brain_info):
        brain_info = all_brain_info[self.scene_key]
        if self.state_key is None:
            return {"vector": list(brain_info.vector_observations), "visual": list(brain_info.visual_observations),
                    "text": list(brain_info.text_observations)}
        elif self.state_key == "vector":
            return list(brain_info.vector_observations)
        elif self.state_key == "visual":
            return list(brain_info.visual_observations)
        elif self.state_key == "text":
            return list(brain_info.text_observations)

    def _get_reward_from_brain_info(self, all_brain_info):
        brain_info = all_brain_info[self.scene_key]
        return [np.array(r_, dtype=np.float32) for r_ in brain_info.rewards]

    def _get_terminal_from_brain_info(self, all_brain_info):
        brain_info = all_brain_info[self.scene_key]
        return brain_info.local_done
