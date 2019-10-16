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

from surreal.components.memories.replay_buffer import ReplayBuffer


class FIFOBuffer(ReplayBuffer):
    """
    Implements a simple FIFO buffer with a `when_full` event handler and an extra `flush` method.
    """
    def __init__(self, record_space, capacity=1000, *, when_full=None, **kwargs):
        """
        Args:
            when_full (Optional[callable]): A handler - taking this buffer as argument - which will be called when the
                buffer is full.
        """
        super().__init__(record_space=record_space, capacity=capacity, **kwargs)
        self.when_full = when_full

    def add_records(self, records, single=False):
        super().add_records(records, single=single)

        # When full, call the even handler, if given passing in ourselves as only argument.
        if self.when_full is not None and self.size  == self.capacity:
            self.when_full(self)

    def flush(self):
        """
        Empties the buffer and returns all records in FIFO-order.

        Returns:
            any: All records of this buffer in the correct FIFO-order.
        """
        pass  # TODO: implement