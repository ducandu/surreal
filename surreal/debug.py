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

import logging
import json
import os
from time import sleep

from surreal import SURREAL_HOME

"""
Global debug settings module. Reads settings from optional ~/.surreal/debug.json file, otherwise assigns default values
to debug globals. If file exists, will also warn of possible slowdowns due to debugging settings being "on".
"""

# Now check the .surreal/debug.json file on whether some of these settings should be overridden.
DEBUG_SETTINGS = dict()
try:
    with open(os.path.expanduser(os.path.join(SURREAL_HOME, "debug.json"))) as f:
        DEBUG_SETTINGS = json.load(f)
    logging.warning("debug.json file found in home directory! This could possibly mean slowdowns in execution.")
    for i in range(2, 0, -1):
        logging.warning(i)
        sleep(1)
except json.decoder.JSONDecodeError as e:
    raise e
except (ValueError, FileNotFoundError):
    pass

# Assert proper syncing functionality of Models by comparing all values before
# (should be dissimilar) and after (should be similar) syncing.
AssertModelSync = DEBUG_SETTINGS.pop("AssertModelSync", False)

# Whether to reference the last batch pulled from a memory in the memories own `last_batch_pulled` property.
KeepLastMemoryBatch = DEBUG_SETTINGS.pop("KeepLastMemoryBatch", False)

# Store every nth trajectory (an entire episode's states, actions, rewards).
StoreEveryNthEpisode = DEBUG_SETTINGS.pop("StoreEveryNthEpisode", False)

# Whether to log loss values during training.
StoreLosses = DEBUG_SETTINGS.pop("StoreLosses", False)

# Store preprocessing every n calls. None for off.
# Note: Will only log the first item in each batch.
StorePreprocessingEveryNCalls = DEBUG_SETTINGS.pop("StorePreprocessingEveryNCalls", False)
# Store the preprocessing steps only for the first n components of a Preprocessor.
# False = log all steps; 0 = log only the inputs, not any preprocessed data.
StorePreprocessingOnlyForFirstNPreprocessorComponents = \
    DEBUG_SETTINGS.pop("StorePreprocessingOnlyForFirstNPreprocessorComponents", False)

# Whether to render all learning tests.
RenderEnvInLearningTests = DEBUG_SETTINGS.pop("RenderEnvInLearningTests", False)

# If rendering an env, maximally render n actors.
IfRenderingRenderMaximallyNActors = DEBUG_SETTINGS.pop("IfRenderingRenderMaximallyNActors", 1)

assert not DEBUG_SETTINGS, "ERROR: Unknown debug settings found in debug.json file: {}".format(DEBUG_SETTINGS)
