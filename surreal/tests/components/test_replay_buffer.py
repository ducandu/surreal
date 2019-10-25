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

import unittest

from surreal.components.memories.replay_buffer import ReplayBuffer
from surreal.spaces import Dict, Bool, Int
from surreal.tests.test_util import check


class TestReplayBuffer(unittest.TestCase):
    """
    Tests sampling and insertion behaviour of the replay_memory module.
    """
    record_space = Dict(
        states=dict(state1=float, state2=float),
        actions=dict(action1=Int(3, shape=(3,))),
        reward=float,
        terminals=Bool(),
        next_states=dict(state1=float, state2=float),
        main_axes="B"
    )

    def test_insert(self):
        """
        Simply tests insert op without checking internal logic.
        """
        memory = ReplayBuffer(record_space=self.record_space, capacity=4)
        # Assert indices 0 before insert.
        self.assertEqual(memory.size, 0)
        self.assertEqual(memory.index, 0)

        # Insert one single record (no batch rank) and check again.
        data = self.record_space.sample()
        memory.add_records(data)
        self.assertTrue(memory.size == 1)
        self.assertTrue(memory.index == 1)

        # Insert one single record (with batch rank) and check again.
        data = self.record_space.sample(1)
        memory.add_records(data)
        self.assertTrue(memory.size == 2)
        self.assertTrue(memory.index == 2)

        # Insert two records (batched).
        data = self.record_space.sample(2)
        memory.add_records(data)
        self.assertTrue(memory.size == 4)
        self.assertTrue(memory.index == 0)

        # Insert one single record (no batch rank, BUT with `single` indicator set for performance reasons)
        # and check again.
        data = self.record_space.sample()
        memory.add_records(data, single=True)
        self.assertTrue(memory.size == 4)
        self.assertTrue(memory.index == 1)

    def test_insert_over_capacity(self):
        """
        Tests if insert correctly manages capacity.
        """
        capacity = 10
        memory = ReplayBuffer(record_space=self.record_space, capacity=capacity)
        # Assert indices 0 before insert.
        self.assertEqual(memory.size, 0)
        self.assertEqual(memory.index, 0)

        # Insert one more element than capacity.
        data = self.record_space.sample(size=capacity + 1)
        memory.add_records(data)

        # Size should be equivalent to capacity when full.
        self.assertEqual(memory.size, capacity)
        # Index should be one over capacity due to modulo.
        self.assertEqual(memory.index, 1)

    def test_get_records(self):
        """
        Tests if retrieval correctly manages capacity.
        """
        capacity = 10
        memory = ReplayBuffer(record_space=self.record_space, capacity=capacity)

        # Insert 1 record.
        data = self.record_space.sample(1)
        memory.add_records(data)

        # Assert we can now fetch 2 elements.
        retrieved_data = memory.get_records(num_records=1)
        self.assertEqual(1, len(retrieved_data["terminals"]))
        check(data, retrieved_data)

        # Test duplicate sampling.
        retrieved_data = memory.get_records(num_records=5)
        self.assertEqual(5, len(retrieved_data["terminals"]))
        # Only one record in the memory -> returned samples should all be the exact same.
        check(retrieved_data["reward"][0], retrieved_data["reward"][1])
        check(retrieved_data["reward"][0], retrieved_data["reward"][2])
        check(retrieved_data["reward"][0], retrieved_data["reward"][3])
        check(retrieved_data["reward"][0], retrieved_data["reward"][4])

        # Now insert another one.
        data = self.record_space.sample()  # w/o batch rank
        memory.add_records(data)
        # Pull exactly two records and make sure they are NOT(!) the same.
        retrieved_data = memory.get_records(num_records=2)
        self.assertEqual(2, len(retrieved_data["terminals"]))
        self.assertNotEqual(retrieved_data["reward"][0], retrieved_data["reward"][1])

        # Now insert over capacity.
        data = self.record_space.sample(capacity)
        memory.add_records(data)

        # Assert we can fetch exactly capacity elements.
        retrieved_data = memory.get_records(num_records=capacity)
        self.assertEqual(capacity, len(retrieved_data["terminals"]))
