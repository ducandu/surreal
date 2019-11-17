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

from surreal.config import AlgoConfig
from surreal.algos.rl_algo import RLAlgo, RLAlgoEvent
from surreal.algos.dads import DADS, DADSConfig
from surreal.algos.dqn2015 import DQN2015, DQN2015Loss, DQN2015Config
from surreal.algos.dddqn import DDDQN, DDDQNLoss, DDDQNConfig
from surreal.algos.sac import SAC, SACLoss, SACConfig

__all__ = [
    "RLAlgo", "RLAlgoEvent", ""
    "DADS", "DADSConfig",
    "DQN2015", "DQN2015Loss", "DQN2015Config",
    "DDDQN", "DDDQNLoss", "DDDQNConfig",
    "SAC", "SACLoss", "SACConfig"
]

_config_classes = [DADSConfig, DQN2015Config, DDDQNConfig, SACConfig]

AlgoConfig.__lookup_classes__ = {
    **{cls.__name__: cls for cls in _config_classes},
    **{cls.get_algo_class().__name__: cls for cls in _config_classes}
}
