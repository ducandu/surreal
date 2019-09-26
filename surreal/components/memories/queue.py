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

from collections import deque

from surreal.components.memories.memory import Memory


class Queue(Memory):
    def __init__(self, record_space, capacity=1000):
        super().__init__(record_space=record_space, capacity=capacity)

        self.deque = deque([], maxlen=self.capacity)

    def add_records(self, records, single=False):
        num_records, flat_records = self.get_number_and_flatten_records(records, single)
        # Make sure records roughly matches our memory.
        assert len(flat_records) == len(self.deque)

        update_indices = np.arange(self.index, self.index + num_records) % self.capacity
        for i in range(len(self.memory)):
            for j, k in enumerate(update_indices):
                self.memory[i][k] = flat_records[i][j]
        self.index = (self.index + num_records) % self.capacity
        self.size = min(self.size + num_records, self.capacity)


