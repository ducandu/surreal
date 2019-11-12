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
import unittest

from surreal.components.memories.trajectory_buffer import TrajectoryBuffer
from surreal.spaces import Dict
from surreal.tests.test_util import check


class TestTrajectoryBuffer(unittest.TestCase):
    """
    Tests the ring buffer. The ring buffer has very similar tests to
    the replay memory as it supports similar insertion and retrieval semantics,
    but needs additional tests on episode indexing and its latest semantics.
    """

    record_space = Dict(
        states=dict(state1=float, state2=float),
        actions=dict(action1=float),
        rewards=float,
        terminals=bool,
        main_axes="B"
    )
    capacity = 10

    def test_get_trajectories(self):
        """
        Test if we can accurately retrieve the most recent trajectories.
        """
        memory = TrajectoryBuffer(self.record_space, capacity=self.capacity, terminal_key="terminals")

        batch_size = 2

        # Insert seq1=F F T, seq2=F F F T seq3=T (using 4 batches of size 2).
        records = self.record_space.sample(batch_size)
        records["terminals"] = np.array([False, False])
        memory.add_records(records)
        self.assertTrue(memory.size == 2)
        # MEMORY: [F F x] x=idx

        records = self.record_space.sample(batch_size)
        records["terminals"] = np.array([False, False])
        memory.add_records(records)
        self.assertTrue(memory.size == 4)
        # MEMORY: [F F | F F x] x=idx

        records = self.record_space.sample(batch_size)
        records["terminals"] = np.array([True, False])
        memory.add_records(records)
        self.assertTrue(memory.size == 6)
        # MEMORY: [F F | F F | T F x] x=idx

        records = self.record_space.sample(batch_size)
        records["terminals"] = np.array([True, True])
        memory.add_records(records)
        self.assertTrue(memory.size == 8)
        # MEMORY: [F F | F F | T F | T T x] x = idx

        check(memory.index, 8)

        # We should now be able to retrieve 1 episode of length 4 (seq2).
        # [F (F) | F (F) | T (F) | T (T)] "()"=retrieved episode.
        trajectories = memory.get_trajectories(num_trajectories=1)
        check(trajectories["terminals"], [False, False, False, True])
        check(trajectories["rewards"], memory.memory[1][np.array([1, 3, 5, 7])])

        # We should now be able to retrieve 2 episodes of length 4 and 1 (seq2 and seq3).
        trajectories = memory.get_trajectories(num_trajectories=2)
        check(trajectories["terminals"], [False, False, False, True, True])
        check(trajectories["rewards"], memory.memory[1][np.array([1, 3, 5, 7, 6])])
        # [F (F) | F (F) | T (F) | {T} (T)] "(){}"=retrieved episodes.

        # We should not be able to retrieve all 3 episodes.
        trajectories = memory.get_trajectories(num_trajectories=3)
        check(trajectories["terminals"], [False, False, False, True, True, False, False, True])
        check(trajectories["rewards"], memory.memory[1][np.array([1, 3, 5, 7, 6, 0, 2, 4])])

        # Add more records that do not end in terminals.
        for i in range(2):
            records = self.record_space.sample(batch_size)
            records["terminals"] = np.array([False, False])
            memory.add_records(records)
            check(memory.index, i * 2)
            self.assertTrue(memory.size == 10)

        # MEMORY: [F F | Fx F | T F | T T | F F] x=idx

        # Check if we can get 4 episodes (the most recent one should be duplicated as we don't have 4 complete ones in
        # memory yet):
        trajectories = memory.get_trajectories(num_trajectories=4)
        check(trajectories["terminals"], [False, False, True, True, False, True, False, False, True])
        check(trajectories["rewards"], memory.memory[1][np.array([3, 5, 7, 6, 2, 4, 3, 5, 7])])

        # Complete a 4th episode.
        records = self.record_space.sample(batch_size)
        records["terminals"] = np.array([True, False])
        memory.add_records(records)
        self.assertTrue(memory.size == 10)
        # MEMORY: [F F | T F | Tx F | T T | F F] x=idx
        check(memory.index, 4)

        # Check if we can get 5 episodes (the most recent one should be duplicated as we don't have 5 complete ones in
        # memory):
        trajectories = memory.get_trajectories(num_trajectories=5)
        check(trajectories["terminals"], [False, False, True, False, True, True, True, False, False, True])
        check(trajectories["rewards"], memory.memory[1][np.array([8, 0, 2, 5, 7, 6, 4, 8, 0, 2])])
