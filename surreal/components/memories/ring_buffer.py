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

import numpy as np
import tensorflow as tf

from surreal.components.memories.memory import Memory
from surreal.spaces import Dict


class RingBuffer(Memory):
    """
    Simple ring-buffer to be used for on-policy sampling based on sample count or episodes.
    Fetches most recently added memories.
    """
    def __init__(self, record_space, capacity=1000, terminal_key="terminal"):
        """
        Args:
            terminal_key (str): The name of the terminal indicator in the record Space. The record Space (a Dict) must
                have this key in the top-level.
        """
        super().__init__(record_space, capacity)

        assert isinstance(self.record_space, Dict), "ERROR: RingBuffer requires Dict as `record_space`!"
        assert terminal_key in self.record_space, \
            "ERROR: `record_space` of RingBuffer must contain '{}' as key!".format(terminal_key)
        self.terminal_key = terminal_key
        # The index in the flattened (sorted by key) Space.
        self.flat_terminal_index = list(sorted(self.record_space.keys())).index(self.terminal_key)

        # The current index into the memory.
        self.index = 0
        # Number of episodes.
        self.num_episodes = 0
        # Terminal indices contiguously arranged.
        self.episode_indices = np.zeros(shape=(self.capacity,), dtype=np.int32)

    def add_records(self, records, single=False):
        num_records, flat_records = self.get_number_and_flatten_records(records, single)

        update_indices = np.arange(self.index, self.index + num_records) % self.capacity

        # Newly inserted episodes.
        inserted_episodes = np.sum(records[self.terminal_key].astype(np.int32), 0)

        # Episodes previously existing in the range we inserted to as indicated
        # by count of terminals in the that slice.
        episodes_in_insert_range = 0
        # Count terminals in inserted range.
        for index in update_indices:
            episodes_in_insert_range += int(self.memory[self.flat_terminal_index][index])
        num_episode_update = self.num_episodes - episodes_in_insert_range + inserted_episodes
        self.episode_indices[:self.num_episodes - episodes_in_insert_range] = \
            self.episode_indices[episodes_in_insert_range:self.num_episodes]

        # Insert new episodes starting at previous count minus the ones we removed,
        # ending at previous count minus removed + inserted.
        slice_start = self.num_episodes - episodes_in_insert_range
        slice_end = num_episode_update

        mask = update_indices[records[self.terminal_key]]
        self.episode_indices[slice_start:slice_end] = mask

        # Update indices.
        self.num_episodes = int(num_episode_update)
        self.index = (self.index + num_records) % self.capacity
        self.size = min(self.size + num_records, self.capacity)

        # Updates all the necessary sub-variables in the record.
        for i in enumerate(self.memory):
            for j, val in zip(update_indices, flat_records[i]):
                self.memory[i][j] = val

    def get_records(self, num_records=1):
        available_records = min(num_records, self.size)
        indices = np.arange(self.index - available_records, self.index) % self.capacity

        #records = []
        #for i, variable in enumerate(self.memory):
            #records.append(self.read_variable(
            #    variable, indices, dtype=self.flat_record_space[i].dtype,
            #    shape=self.flat_record_space[i].shape
            #))

        records = [np.array([var[i] for i in indices]) for var in self.memory]
        records = tf.nest.pack_sequence_as(self.record_space.structure, records)

        return records

    def get_episodes(self, num_episodes=1):
        stored_episodes = self.num_episodes
        available_episodes = min(num_episodes, self.num_episodes)

        if stored_episodes == available_episodes:
            start = 0
        else:
            start = self.episode_indices[stored_episodes - available_episodes - 1] + 1

        # End index is just the pointer to the most recent episode.
        limit = self.episode_indices[stored_episodes - 1]
        if start >= limit:
            limit += self.capacity - 1

        indices = np.arange(start, limit + 1) % self.capacity

        #records = []
        #for i, variable in enumerate(self.memory):
        #    records.append(
        #        self.read_variable(variable, indices, dtype=self.flat_record_space[i].dtype,
        #                           shape=self.flat_record_space[i].shape)
        #    )
        records = [np.array([var[i] for i in indices]) for var in self.memory]
        records = tf.nest.pack_sequence_as(self.record_space.structure, records)
        return records
