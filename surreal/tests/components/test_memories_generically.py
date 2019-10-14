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
import unittest

from surreal.components.memories import PrioritizedReplayBuffer, ReplayBuffer
from surreal.spaces import Bool, Dict, Float, Int
from surreal.tests.test_util import check
from surreal.utils.errors import SurrealError


class TestMemoriesGenerically(unittest.TestCase):
    """
    Tests different generic functionalities of Memories.
    """
    record_space = Dict(
        states=dict(state1=float, state2=Float(shape=(2,))),
        actions=dict(action1=int),
        reward=float,
        terminals=bool,
        main_axes="B"
    )
    record_space_no_next_state = Dict(s=dict(s1=float, s2=float), a=dict(a1=Int(10)), r=float, t=Bool(), main_axes="B")

    capacity = 10
    alpha = 1.0
    beta = 1.0
    max_priority = 1.0

    def test_next_state_handling(self):
        """
        Tests if next-states can be stored efficiently (not using any space!) in the memory.

        NOTE: The memory does not care about terminal signals, it will always return the n-next-in-memory state
        regardless of whether this is a useful state (terminal=False) or not (terminal=True). In case of a
        terminal=True, the next state (whether it be the true terminal state, the reset state, or any other random
        state) does not matter anyway.
        """
        capacity = 10
        batch_size = 2

        # Test all classes of memories.
        for class_ in [ReplayBuffer, PrioritizedReplayBuffer]:
            memory = class_(record_space=self.record_space_no_next_state, capacity=capacity,
                            next_record_setup=dict(s="s_"))

            # Insert n records (inserts must always be batch-size).
            data = dict(
                s=dict(s1=np.array([0.0, 1.0]), s2=np.array([2.0, 3.0])),
                a=np.array([0, 1]), r=np.array([-0.0, -1.0]), t=np.array([False, True]),
                s_=dict(s1=np.array([0.1, 1.1]), s2=np.array([2.1, 3.1]))
            )
            memory.add_records(data)

            # Check, whether inserting the wrong batch size raises Exception.
            try:
                data = self.record_space_no_next_state.sample(batch_size + 1)
                data["s_"] = self.record_space_no_next_state["s"].sample(batch_size)
                memory.add_records(data)
                assert False, "ERROR: Should not get here. Error is expected."
            except SurrealError:
                pass

            # Assert we can now fetch n elements.
            retrieved_data = memory.get_records(num_records=1)
            self.assertEqual(1, len(retrieved_data["t"]))

            # Check the next state.
            if retrieved_data["s"]["s1"][0] == 0.0:
                self.assertTrue(retrieved_data["s_"]["s1"] == 0.1 and retrieved_data["s_"]["s2"] == 2.1)
            else:
                self.assertTrue(retrieved_data["s"]["s1"] == 1.0)
                self.assertTrue(retrieved_data["s_"]["s1"] == 1.1 and retrieved_data["s_"]["s2"] == 3.1)

            # Insert another 2xn records and then check for correct next-state returns when getting records.
            data = dict(
                s=dict(s1=np.array([0.1, 1.1]), s2=np.array([2.1, 3.1])),
                a=np.array([2, 3]), r=np.array([-2.0, -3.0]), t=np.array([False, False]),
                s_=dict(s1=np.array([0.2, 1.2]), s2=np.array([2.2, 3.2]))
            )
            memory.add_records(data)
            data = dict(
                s=dict(s1=np.array([0.2, 1.2]), s2=np.array([2.2, 3.2])),
                a=np.array([4, 5]), r=np.array([-4.0, -5.0]), t=np.array([True, True]),
                s_=dict(s1=np.array([0.3, 1.3]), s2=np.array([2.3, 3.3]))
            )
            memory.add_records(data)

            for _ in range(20):
                retrieved_data = memory.get_records(num_records=2)
                self.assertEqual(2, len(retrieved_data["t"]))

                # Check the next states (always 0.1 larger than state).
                for i in range(2):
                    check(retrieved_data["s"]["s1"][i], retrieved_data["s_"]["s1"][i] - 0.1)
                    check(retrieved_data["s"]["s2"][i], retrieved_data["s_"]["s2"][i] - 0.1)

            self.assertTrue(memory.size == 6)

            # Insert up to capacity and check again.
            data = dict(
                s=dict(s1=np.array([0.3, 1.3]), s2=np.array([2.3, 3.3])),
                a=np.array([6, 7]), r=np.array([-6.0, -7.0]), t=np.array([True, False]),
                s_=dict(s1=np.array([0.4, 1.4]), s2=np.array([2.4, 3.4]))
            )
            memory.add_records(data)
            data = dict(
                s=dict(s1=np.array([0.4, 1.4]), s2=np.array([2.4, 3.4])),
                a=np.array([8, 9]), r=np.array([-8.0, -9.0]), t=np.array([False, False]),
                s_=dict(s1=np.array([0.5, 1.5]), s2=np.array([2.5, 3.5]))
            )
            memory.add_records(data)

            for _ in range(20):
                retrieved_data = memory.get_records(num_records=3)
                self.assertEqual(3, len(retrieved_data["t"]))

                # Check the next states (always 0.1 larger than state).
                for i in range(3):
                    check(retrieved_data["s"]["s1"][i], retrieved_data["s_"]["s1"][i] - 0.1)
                    check(retrieved_data["s"]["s2"][i], retrieved_data["s_"]["s2"][i] - 0.1)

            self.assertTrue(memory.size == 10)

            # Go a little bit (one batch) over capacity and check again.
            data = dict(
                s=dict(s1=np.array([0.5, 1.5]), s2=np.array([2.5, 3.5])),
                a=np.array([10, 11]), r=np.array([-10.0, -11.0]), t=np.array([True, True]),
                s_=dict(s1=np.array([0.6, 1.6]), s2=np.array([2.6, 3.6]))
            )
            memory.add_records(data)

            for _ in range(20):
                retrieved_data = memory.get_records(num_records=4)
                self.assertEqual(4, len(retrieved_data["t"]))

                # Check the next states (always 0.1 larger than state).
                for i in range(4):
                    check(retrieved_data["s"]["s1"][i], retrieved_data["s_"]["s1"][i] - 0.1)
                    check(retrieved_data["s"]["s2"][i], retrieved_data["s_"]["s2"][i] - 0.1)

            self.assertTrue(memory.size == 10)

    def test_next_state_handling_with_n_step(self):
        """
        Tests if next-states can be stored efficiently (not using any space!) in the memory using an n-step memory.

        NOTE: The memory does not care about terminal signals, it will always return the n-next-in-memory state
        regardless of whether this is a useful state (terminal=False) or not (terminal=True). In case of a
        terminal=True, the next state (whether it be the true terminal state, the reset state, or any other random
        state) does not matter anyway.
        """
        capacity = 10
        batch_size = 2
        # Test all classes of memories.
        for class_ in [ReplayBuffer, PrioritizedReplayBuffer]:
            memory = class_(record_space=self.record_space_no_next_state, capacity=capacity,
                            next_record_setup=dict(s="s_", n_step=3))

            # Insert n records (inserts must always be batch-size).
            data = dict(
                s=dict(s1=np.array([0.0, 1.0]), s2=np.array([2.0, 3.0])),
                a=np.array([0, 1]), r=np.array([-0.0, -1.0]), t=np.array([False, True]),
                s_=dict(s1=np.array([0.3, 1.3]), s2=np.array([2.3, 3.3]))  # s' is now the n-step s'
            )
            memory.add_records(data)

            # Check, whether inserting the wrong batch size raises Exception.
            try:
                data = self.record_space_no_next_state.sample(batch_size + 1)
                data["s_"] = self.record_space_no_next_state["s"].sample(batch_size)
                memory.add_records(data)
                assert False, "ERROR: Should not get here. Error is expected."
            except SurrealError:
                pass

            # Assert we cannot pull samples yet. n-step is 3, so we need at least 3 elements in memory.
            try:
                memory.get_records(num_records=1)
                assert False, "ERROR: Should not get here. Error is expected."
            except SurrealError:
                pass

            # Insert another 2xn records and then check for correct next-state returns when getting records.
            data = dict(
                s=dict(s1=np.array([0.1, 1.1]), s2=np.array([2.1, 3.1])),
                a=np.array([2, 3]), r=np.array([-2.0, -3.0]), t=np.array([False, False]),
                s_=dict(s1=np.array([0.4, 1.4]), s2=np.array([2.4, 3.4]))  # s' is now the n-step s'
            )
            memory.add_records(data)
            data = dict(
                s=dict(s1=np.array([0.2, 1.2]), s2=np.array([2.2, 3.2])),
                a=np.array([4, 5]), r=np.array([-4.0, -5.0]), t=np.array([True, True]),
                s_=dict(s1=np.array([0.5, 1.5]), s2=np.array([2.5, 3.5]))  # s' is now the n-step s'
            )
            memory.add_records(data)

            for _ in range(20):
                retrieved_data = memory.get_records(num_records=2)
                self.assertEqual(2, len(retrieved_data["t"]))

                # Check the next states (always 0.1 larger than state).
                for i in range(2):
                    check(retrieved_data["s"]["s1"][i], retrieved_data["s_"]["s1"][i] - 0.3)
                    check(retrieved_data["s"]["s2"][i], retrieved_data["s_"]["s2"][i] - 0.3)

            self.assertTrue(memory.size == 6)

            # Insert up to capacity and check again.
            data = dict(
                s=dict(s1=np.array([0.3, 1.3]), s2=np.array([2.3, 3.3])),
                a=np.array([6, 7]), r=np.array([-6.0, -7.0]), t=np.array([True, False]),
                s_=dict(s1=np.array([0.6, 1.6]), s2=np.array([2.6, 3.6]))
            )
            memory.add_records(data)
            data = dict(
                s=dict(s1=np.array([0.4, 1.4]), s2=np.array([2.4, 3.4])),
                a=np.array([8, 9]), r=np.array([-8.0, -9.0]), t=np.array([False, False]),
                s_=dict(s1=np.array([0.7, 1.7]), s2=np.array([2.7, 3.7]))
            )
            memory.add_records(data)

            for _ in range(20):
                retrieved_data = memory.get_records(num_records=3)
                self.assertEqual(3, len(retrieved_data["t"]))

                # Check the next states (always 0.1 larger than state).
                for i in range(3):
                    check(retrieved_data["s"]["s1"][i], retrieved_data["s_"]["s1"][i] - 0.3)
                    check(retrieved_data["s"]["s2"][i], retrieved_data["s_"]["s2"][i] - 0.3)

            self.assertTrue(memory.size == 10)

            # Go a little bit (two batches) over capacity and check again.
            data = dict(
                s=dict(s1=np.array([0.5, 1.5]), s2=np.array([2.5, 3.5])),
                a=np.array([10, 11]), r=np.array([-10.0, -11.0]), t=np.array([True, True]),
                s_=dict(s1=np.array([0.8, 1.8]), s2=np.array([2.8, 3.8]))
            )
            memory.add_records(data)
            data = dict(
                s=dict(s1=np.array([0.6, 1.6]), s2=np.array([2.6, 3.6])),
                a=np.array([10, 11]), r=np.array([-10.0, -11.0]), t=np.array([False, False]),
                s_=dict(s1=np.array([0.9, 1.9]), s2=np.array([2.9, 3.9]))
            )
            memory.add_records(data)

            for _ in range(20):
                retrieved_data = memory.get_records(num_records=4)
                self.assertEqual(4, len(retrieved_data["t"]))

                # Check the next states (always 0.1 larger than state).
                for i in range(4):
                    check(retrieved_data["s"]["s1"][i], retrieved_data["s_"]["s1"][i] - 0.3)
                    check(retrieved_data["s"]["s2"][i], retrieved_data["s_"]["s2"][i] - 0.3)

            self.assertTrue(memory.size == 10)
