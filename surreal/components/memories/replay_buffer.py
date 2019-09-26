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
import tensorflow as tf

from surreal.components.memories.memory import Memory
from surreal.debug import KeepLastMemoryBatch
from surreal.utils.errors import SurrealError


class ReplayBuffer(Memory):
    """
    Implements a standard replay memory to sample randomized batches of arbitrary data.
    """
    def __init__(self, record_space, capacity=1000):
        super().__init__(record_space=record_space, capacity=capacity)

        # Create the main memory as a flattened OrderedDict from any arbitrarily nested Space.
        self.memory = tf.nest.map_structure(lambda space: space.create_variable(
            name="memory", trainable=False, initializer=0,
            is_python=True, local=False, use_resource=False
        ), self.flat_record_space)

        # Current index into the buffer.
        self.index = 0

    def add_records(self, records, single=False):
        num_records, flat_records = self.get_number_and_flatten_records(records, single)
        # Make sure records roughly matches our memory.
        assert len(flat_records) == len(self.memory)

        update_indices = np.arange(self.index, self.index + num_records) % self.capacity
        for i in range(len(self.memory)):
            for j, k in enumerate(update_indices):
                self.memory[i][k] = flat_records[i][j]
        self.index = (self.index + num_records) % self.capacity
        self.size = min(self.size + num_records, self.capacity)

    def get_records(self, num_records=1):
        if self.size <= 0:
            raise SurrealError("ReplayBuffer is empty.")

        # Calculate the indices to pull from the memory.
        # If num_records is <= our size, return w/o replacement (duplicates), otherwise, allow duplicates.
        indices = np.random.choice(np.arange(0, self.size), size=int(num_records),
                                   replace=True if num_records > self.size else False)
        indices = (self.index - 1 - indices) % self.capacity
        records = [np.array([var[i] for i in indices]) for var in self.memory]
        records = tf.nest.pack_sequence_as(self.record_space.structure, records)

        if KeepLastMemoryBatch is True:
            self.last_records_pulled = records

        return records
