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

from surreal.components.memories.fifo_buffer import FIFOBuffer
from surreal.spaces import Dict
from surreal.tests.test_util import check


class TestFIFOBufferMemory(unittest.TestCase):
    """
    Tests the FIFOBuffer Component.
    """
    record_space = Dict(
        states=dict(state1=float, state2=float),
        actions=dict(action1=float),
        rewards=float,
        terminals=bool,
        main_axes="B"
    )
    capacity = 10

    def when_full(self, buffer):
        print("Executing `when_full` on buffer={}".format(buffer))
        raise Exception  # to catch

    def test_fifo_buffer(self):
        fifo_buffer = FIFOBuffer(record_space=self.record_space, capacity=self.capacity, when_full=self.when_full)

        # Not full.
        data = self.record_space.sample(self.capacity - 1)
        fifo_buffer.add_records(data)
        self.assertTrue(fifo_buffer.size == self.capacity - 1)

        # Full.
        data = self.record_space.sample(2)
        try:
            fifo_buffer.add_records(data)
            # Expect when_full to be called.
            raise AssertionError
        except Exception:
            pass

        self.assertTrue(fifo_buffer.size == self.capacity)
        all_data = fifo_buffer.flush()
        self.assertTrue(fifo_buffer.size == 0)
        self.assertTrue(fifo_buffer.index == 0)

        self.assertTrue(len(all_data["states"]["state1"]) == self.capacity)
        self.assertTrue(len(all_data["states"]["state2"]) == self.capacity)
        self.assertTrue(len(all_data["rewards"]) == self.capacity)
        self.assertTrue(all_data["rewards"].dtype == np.float32)
