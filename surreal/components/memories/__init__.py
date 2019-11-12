# Copyright 2019 ducandu GmbH, All Rights Reserved
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

from surreal.components.memories.memory import Memory
from surreal.components.memories.fifo_buffer import FIFOBuffer
from surreal.components.memories.prioritized_replay_buffer import PrioritizedReplayBuffer
from surreal.components.memories.replay_buffer import ReplayBuffer
from surreal.components.memories.trajectory_buffer import TrajectoryBuffer

Memory.__lookup_classes__ = dict(
    fifobuffer=FIFOBuffer,
    fifo=FIFOBuffer,
    prioritizedreplay=PrioritizedReplayBuffer,
    prioritizedreplaybuffer=PrioritizedReplayBuffer,
    replaybuffer=ReplayBuffer,
    replaymemory=ReplayBuffer,
    trajectorybuffer=TrajectoryBuffer
)
Memory.__default_constructor__ = ReplayBuffer

__all__ = ["Memory"] + \
          list(set(map(lambda x: x.__name__, Memory.__lookup_classes__.values())))
