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
import operator

from surreal.components.misc.segment_tree import SegmentTree, MinSumSegmentTree
from surreal.components.memories.memory import Memory
from surreal.debug import KeepLastMemoryBatch
from surreal.spaces import Dict
from surreal.utils.errors import SurrealError
from surreal.utils.util import SMALL_NUMBER


class PrioritizedReplayBuffer(Memory):
    """
    Implements an in-memory prioritized replay.
    """
    def __init__(self, record_space, capacity=1000, *, alpha=1.0, beta=1.0, next_record_setup=None):
        """
        Args:
            alpha (float): Degree to which prioritization is applied, 0.0 implies no
                prioritization (uniform, same behavior as ReplayBuffer), 1.0 full prioritization.

            beta (float): Importance weight factor, 0.0 for no importance correction, 1.0 for full correction.
        """
        super().__init__(record_space=record_space, capacity=capacity, next_record_setup=next_record_setup)

        # Records are allowed to carry their own weight when being added to
        self.index_record_weight = None
        if isinstance(self.record_space, Dict) and "_weight" in self.record_space:
            self.index_record_weight = list(self.record_space.keys()).index("_weight")

        self.alpha = alpha
        self.beta = beta

        self.max_priority = 1.0

        self.default_new_weight = np.power(self.max_priority, self.alpha)

        self.priority_capacity = 1
        while self.priority_capacity < self.capacity:
            self.priority_capacity *= 2

        # Create segment trees, initialize with neutral elements.
        sum_values = [0.0 for _ in range(2 * self.priority_capacity)]
        sum_segment_tree = SegmentTree(sum_values, self.priority_capacity, operator.add)
        min_values = [float("inf") for _ in range(2 * self.priority_capacity)]
        min_segment_tree = SegmentTree(min_values, self.priority_capacity, min)

        self.merged_segment_tree = MinSumSegmentTree(
            sum_tree=sum_segment_tree,
            min_tree=min_segment_tree,
            capacity=self.priority_capacity
        )

    def add_records(self, records, single=False):
        num_records, flat_records = self.get_number_and_flatten_records(records, single)

        # Determine our insertion indices.
        indices = np.arange(start=self.index, stop=self.index + num_records) % self.capacity

        for i, insert_index in enumerate(indices):
            if self.index_record_weight is None:
                self.merged_segment_tree.insert(insert_index, self.default_new_weight)
            else:
                self.merged_segment_tree.insert(insert_index, flat_records[i][self.index_record_weight])

        # Add values to the indices.
        for i, memory_bin in enumerate(self.memory):
            for j, k in enumerate(indices):
                memory_bin[k] = flat_records[i][j]

        # Update indices
        self.index = (self.index + num_records) % self.capacity
        self.size = min(self.size + num_records, self.capacity)

    def get_records_with_indices(self, num_records=1):
        if self.size <= 0:
            raise SurrealError("PrioritizedReplayBuffer is empty.")

        # Calculate the indices to pull from the memory.
        indices = []
        prob_sum = self.merged_segment_tree.sum_segment_tree.get_sum(0, self.size)  # -1?
        samples = np.random.random(size=(num_records,)) * prob_sum  # TODO: check: available_records instead or num_records?
        for sample in samples:
            indices.append(self.merged_segment_tree.sum_segment_tree.index_of_prefixsum(prefix_sum=sample))

        indices = np.asarray(indices)
        records = self.get_records_at_indices(indices)

        if KeepLastMemoryBatch is True:
            self.last_records_pulled = records

        return records, indices

    def get_records_with_indices_and_weights(self, num_records=1):
        records, indices = self.get_records_with_indices(num_records=num_records)

        sum_prob = self.merged_segment_tree.sum_segment_tree.get_sum()
        min_prob = self.merged_segment_tree.min_segment_tree.get_min_value() / sum_prob + SMALL_NUMBER
        max_weight = (min_prob * self.size) ** (-self.beta)
        weights = []
        for index in indices:
            sample_prob = self.merged_segment_tree.sum_segment_tree.get(index) / sum_prob
            weight = (sample_prob * self.size) ** (-self.beta)
            weights.append(weight / max_weight)

        weights = np.asarray(weights)

        return records, indices, weights

    def update_records(self, indices, weights):
        """
        Overwrites the current weights at the given index positions with the new values.

        Args:
            indices (np.ndarray 1D): The list of memory indices to overwrite.
            weights (np.ndarray 1D): The new weight values (shape must match that of `indices`).
        """
        # Safety net for inf weights.
        weights = np.where(np.isfinite(weights), weights, self.max_priority)
        # Update weights at given indices one by one.
        for index, weight in zip(indices, weights):
            priority = np.power(weight, self.alpha)
            self.merged_segment_tree.insert(index, priority)
            self.max_priority = max(self.max_priority, priority)
