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

from surreal.envs import Env


class VectorEnv(Env):
    """
    Abstract multi-environment class to support stepping through multiple envs at once.
    """
    def __init__(self, num_environments, **kwargs):
        super(VectorEnv, self).__init__(**kwargs)
        self.num_environments = num_environments

    def get_env(self):
        """
        Returns an underlying sub-environment instance.

        Returns:
            Env: Environment instance.
        """
        raise NotImplementedError

    def reset(self, index=0):
        """
        Resets the given sub-environment.

        Returns:
            any: New state for sub-environment.
        """
        raise NotImplementedError

    def reset_all(self):
        """
        Resets all envs.

        Returns:
            any: New states for envs.
        """
        raise NotImplementedError

    def terminate_all(self):
        raise NotImplementedError
