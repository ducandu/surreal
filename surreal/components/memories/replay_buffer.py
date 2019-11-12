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

import numpy as np

from surreal.components.memories.memory import Memory
from surreal.debug import KeepLastMemoryBatch
from surreal.utils.errors import SurrealError


class ReplayBuffer(Memory):
    """
    Implements a standard replay memory to sample randomized batches of arbitrary data.
    """
    def add_records(self, records, single=False):
        num_records, flat_records = self.get_number_and_flatten_records(records, single)

        # Determine our insertion indices.
        indices = np.arange(self.index, self.index + num_records) % self.capacity

        # Add values to the indices.
        for i, memory_bin in enumerate(self.memory):
            for j, k in enumerate(indices):
                memory_bin[k] = flat_records[i][j]

        self.index = (self.index + num_records) % self.capacity
        self.size = min(self.size + num_records, self.capacity)

    def get_records_with_indices(self, num_records=1):
        if self.size <= 0:
            raise SurrealError("ReplayBuffer is empty.")

        # Calculate the indices to pull from the memory.
        # If num_records is <= our size, return w/o replacement (duplicates), otherwise, allow duplicates.
        indices = np.random.choice(
            np.arange(0, self.size), size=int(num_records), replace=True if num_records > self.size else False
        )
        indices = (self.index - 1 - indices) % self.capacity
        records = self.get_records_at_indices(indices)

        if KeepLastMemoryBatch is True:
            self.last_records_pulled = records

        return records, indices
