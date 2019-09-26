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

import os

from surreal.version import __version__
from surreal.makeable import Makeable
from surreal.config import Config


if "SURREAL_HOME" in os.environ:
    SURREAL_HOME = os.environ.get("SURREAL_HOME")
else:
    SURREAL_HOME = os.path.expanduser('~')
    SURREAL_HOME = os.path.join(SURREAL_HOME, ".surreal/")

PATH_EPISODE_LOGS = SURREAL_HOME + "episodes/"
PATH_LOSS_LOGS = SURREAL_HOME + "losses/"
PATH_PREPROCESSING_LOGS = SURREAL_HOME + "preprocessing/"

# Create dirs if necessary:
for dir in [SURREAL_HOME, PATH_EPISODE_LOGS, PATH_LOSS_LOGS, PATH_PREPROCESSING_LOGS]:
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError:
            pass

__all__ = ["__version__", "Config", "Makeable",
           "SURREAL_HOME", "PATH_EPISODE_LOGS", "PATH_LOSS_LOGS", "PATH_PREPROCESSING_LOGS"
           ]
