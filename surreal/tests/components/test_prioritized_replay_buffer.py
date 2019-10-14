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

from collections import Counter
import numpy as np
import unittest

from surreal.components.memories.prioritized_replay_buffer import PrioritizedReplayBuffer
from surreal.spaces import Dict, Float
from surreal.tests.test_util import check


class TestPrioritizedReplayBuffer(unittest.TestCase):
    """
    Tests insertion and (weighted) sampling of the PrioritizedReplayBuffer Component.
    """
    record_space = Dict(
        states=dict(state1=float, state2=Float(shape=(2,))),
        actions=dict(action1=int),
        reward=float,
        terminals=bool,
        main_axes="B"
    )

    capacity = 10
    alpha = 1.0
    beta = 1.0
    max_priority = 1.0

    def test_insert(self):
        memory = PrioritizedReplayBuffer(
            record_space=self.record_space,
            capacity=self.capacity,
            alpha=self.alpha,
            beta=self.beta
        )

        # Assert indices 0 before insert.
        self.assertEqual(memory.size, 0)
        self.assertEqual(memory.index, 0)

        # Insert single record (no batch rank).
        data = self.record_space.sample()
        memory.add_records(data)
        self.assertTrue(memory.size == 1)
        self.assertTrue(memory.index == 1)

        # Insert single record (w/ batch rank).
        data = self.record_space.sample(1)
        memory.add_records(data)
        self.assertTrue(memory.size == 2)
        self.assertTrue(memory.index == 2)

        # Insert batched records.
        data = self.record_space.sample(5)
        memory.add_records(data)
        self.assertTrue(memory.size == 7)
        self.assertTrue(memory.index == 7)

        # Insert over capacity.
        data = self.record_space.sample(100)
        memory.add_records(data)
        self.assertTrue(memory.size == 10)
        self.assertTrue(memory.index == 7)

    def test_update_records(self):
        memory = PrioritizedReplayBuffer(record_space=self.record_space, capacity=self.capacity)

        # Insert record samples.
        num_records = 2
        data = self.record_space.sample(num_records)
        memory.add_records(data)
        self.assertTrue(memory.size == num_records)
        self.assertTrue(memory.index == num_records)

        # Fetch records, their indices and weights.
        batch, indices, weights = memory.get_records_with_indices_and_weights(num_records)
        check(weights, np.ones(shape=(num_records,)))
        self.assertEqual(num_records, len(indices))
        self.assertTrue(memory.size == num_records)
        self.assertTrue(memory.index == num_records)

        # Update weight of index 0 to very small.
        memory.update_records(np.array([0]), np.array([0.01]))
        # Expect to sample almost only index 1 (which still has a weight of 1.0).
        for _ in range(100):
            _, indices, weights = memory.get_records_with_indices_and_weights(num_records=1000)
            self.assertGreaterEqual(np.sum(indices), 980)

        # Update weight of index 1 to very small as well.
        # Expect to sample equally.
        for _ in range(100):
            rand = np.random.random()
            memory.update_records(np.array([0, 1]), np.array([rand, rand]))
            _, indices, _ = memory.get_records_with_indices_and_weights(num_records=1000)
            self.assertGreaterEqual(np.sum(indices), 400)
            self.assertLessEqual(np.sum(indices), 600)

        # Update weights to be 1:2.
        # Expect to sample double as often index 1 over index 0 (1.0 = 2* 0.5).
        for _ in range(100):
            rand = np.random.random() * 10
            memory.update_records(np.array([0, 1]), np.array([rand, rand * 2]))
            _, indices, _ = memory.get_records_with_indices_and_weights(num_records=1000)
            self.assertGreaterEqual(np.sum(indices), 600)
            self.assertLessEqual(np.sum(indices), 750)

        # Update weights to be 1:4.
        # Expect to sample quadruple as often index 1 over index 0.
        for _ in range(100):
            rand = np.random.random() * 10
            memory.update_records(np.array([0, 1]), np.array([rand, rand * 4]))
            _, indices, _ = memory.get_records_with_indices_and_weights(num_records=1000)
            self.assertGreaterEqual(np.sum(indices), 750)
            self.assertLessEqual(np.sum(indices), 850)

        # Update weights to be 1:9.
        # Expect to sample 9 times as often index 1 over index 0.
        for _ in range(100):
            rand = np.random.random() * 10
            memory.update_records(np.array([0, 1]), np.array([rand, rand * 9]))
            _, indices, _ = memory.get_records_with_indices_and_weights(num_records=1000)
            self.assertGreaterEqual(np.sum(indices), 850)
            self.assertLessEqual(np.sum(indices), 950)

        # Insert more record samples.
        num_records = 10
        data = self.record_space.sample(num_records)
        memory.add_records(data)
        self.assertTrue(memory.size == self.capacity)
        self.assertTrue(memory.index == 2)

        # Update weights to be 1.0 to 10.0 and sample a < 10 batch.
        memory.update_records(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                              np.array([0.1, 1., 3., 8., 16., 32., 64., 128., 256., 512.]))
        counts = Counter()
        for _ in range(1000):
            _, indices, _ = memory.get_records_with_indices_and_weights(num_records=np.random.randint(1, 6))
            for i in indices:
                counts[i] += 1
        print(counts)
        self.assertTrue(
            counts[9] >= counts[8] >= counts[7] >= counts[6] >= counts[5] >=
            counts[4] >= counts[3] >= counts[2] >= counts[1] >= counts[0]
        )

    def test_segment_tree_insert_values(self):
        """
        Tests if segment tree inserts into correct positions.
        """
        memory = PrioritizedReplayBuffer(
            record_space=self.record_space,
            capacity=self.capacity,
            alpha=self.alpha,
            beta=self.beta
        )

        priority_capacity = 1
        while priority_capacity < self.capacity:
            priority_capacity *= 2

        sum_segment_values = memory.merged_segment_tree.sum_segment_tree.values
        min_segment_values = memory.merged_segment_tree.min_segment_tree.values

        self.assertEqual(sum(sum_segment_values), 0)
        self.assertEqual(sum(min_segment_values), float("inf"))
        self.assertEqual(len(sum_segment_values), 2 * priority_capacity)
        self.assertEqual(len(min_segment_values), 2 * priority_capacity)

        # Insert 1 Element.
        observation = self.record_space.sample(size=1)
        memory.add_records(observation)

        # Check insert positions
        # Initial insert is at priority capacity
        print(sum_segment_values)
        print(min_segment_values)
        start = priority_capacity

        while start >= 1:
            self.assertEqual(sum_segment_values[start], 1.0)
            self.assertEqual(min_segment_values[start], 1.0)
            start = int(start / 2)

        # Insert another Element.
        observation = self.record_space.sample(size=1)
        memory.add_records(observation)

        # Index shifted 1
        start = priority_capacity + 1
        self.assertEqual(sum_segment_values[start], 1.0)
        self.assertEqual(min_segment_values[start], 1.0)
        start = int(start / 2)
        while start >= 1:
            # 1 + 1 is 2 on the segment.
            self.assertEqual(sum_segment_values[start], 2.0)
            # min is still 1.
            self.assertEqual(min_segment_values[start], 1.0)
            start = int(start / 2)

    def test_tree_insert(self):
        """
        Tests inserting into the segment tree and querying segments.
        """
        memory = PrioritizedReplayBuffer(record_space=self.record_space, capacity=4        )
        tree = memory.merged_segment_tree.sum_segment_tree
        tree.insert(2, 1.0)
        tree.insert(3, 3.0)
        self.assertTrue(np.isclose(tree.get_sum(), 4.0))
        self.assertTrue(np.isclose(tree.get_sum(0, 2), 0.0))
        self.assertTrue(np.isclose(tree.get_sum(0, 3), 1.0))
        self.assertTrue(np.isclose(tree.get_sum(2, 3), 1.0))
        self.assertTrue(np.isclose(tree.get_sum(2, -1), 1.0))
        self.assertTrue(np.isclose(tree.get_sum(2, 4), 4.0))

    def test_prefixsum_idx(self):
        """
        Tests fetching the index corresponding to a prefix sum.
        """
        memory = PrioritizedReplayBuffer(record_space=self.record_space, capacity=4)
        tree = memory.merged_segment_tree.sum_segment_tree
        tree.insert(2, 1.0)
        tree.insert(3, 3.0)

        self.assertEqual(tree.index_of_prefixsum(0.0), 2)
        self.assertEqual(tree.index_of_prefixsum(0.5), 2)
        self.assertEqual(tree.index_of_prefixsum(0.99), 2)
        self.assertEqual(tree.index_of_prefixsum(1.01), 3)
        self.assertEqual(tree.index_of_prefixsum(3.0), 3)
        self.assertEqual(tree.index_of_prefixsum(4.0), 3)

        memory = PrioritizedReplayBuffer(record_space=self.record_space, capacity=4)
        tree = memory.merged_segment_tree.sum_segment_tree
        tree.insert(0, 0.5)
        tree.insert(1, 1.0)
        tree.insert(2, 1.0)
        tree.insert(3, 3.0)
        self.assertEqual(tree.index_of_prefixsum(0.0), 0)
        self.assertEqual(tree.index_of_prefixsum(0.55), 1)
        self.assertEqual(tree.index_of_prefixsum(0.99), 1)
        self.assertEqual(tree.index_of_prefixsum(1.51), 2)
        self.assertEqual(tree.index_of_prefixsum(3.0), 3)
        self.assertEqual(tree.index_of_prefixsum(5.50), 3)
