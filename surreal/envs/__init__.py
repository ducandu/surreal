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

from surreal.envs.env import Env
from surreal.envs.deterministic_env import DeterministicEnv
from surreal.envs.gaussian_density_as_reward_env import GaussianDensityAsRewardEnv
from surreal.envs.grid_world import GridWorld
from surreal.envs.openai_gym_env import OpenAIGymEnv
from surreal.envs.random_env import RandomEnv
from surreal.envs.vector_env import VectorEnv
from surreal.envs.sequential_vector_env import SequentialVectorEnv

Env.__lookup_classes__ = dict(
    deterministic=DeterministicEnv,
    deterministicenv=DeterministicEnv,
    gaussiandensity=GaussianDensityAsRewardEnv,
    gaussiandensityasreward=GaussianDensityAsRewardEnv,
    gaussiandensityasrewardenv=GaussianDensityAsRewardEnv,
    gridworld=GridWorld,
    gridworldenv=GridWorld,
    openai=OpenAIGymEnv,
    openaigym=OpenAIGymEnv,
    openaigymenv=OpenAIGymEnv,
    random=RandomEnv,
    randomenv=RandomEnv,
    sequentialvector=SequentialVectorEnv,
    sequentialvectorenv=SequentialVectorEnv
)

try:
    import deepmind_lab

    # If import works: Can import our Adapter.
    from rlgraph.environments.deepmind_lab import DeepmindLabEnv

    Env.__lookup_classes__.update(dict(
        deepmindlab=DeepmindLabEnv,
        deepmindlabenv=DeepmindLabEnv,
    ))
    # TODO travis error on this, investigate.
except Exception:
    pass


try:
    import mlagents

    # If import works: Can import our Adapter.
    from rlgraph.environments.mlagents_env import MLAgentsEnv

    Env.__lookup_classes__.update(dict(
        mlagents=MLAgentsEnv,
        mlagentsenv=MLAgentsEnv,
    ))
    # TODO travis error on this, investigate.
except Exception:
    pass


__all__ = ["Env"] + \
          list(set(map(lambda x: x.__name__, Env.__lookup_classes__.values())))
