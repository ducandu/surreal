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

from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf

from surreal.makeable import Makeable
from surreal.utils.util import get_batch_size


class Memory(Makeable, metaclass=ABCMeta):
    """
    Abstract memory component.
    """
    def __init__(self, record_space, capacity=1000):
        """
        Args:
            record_space (Space): The Space of the records that will go into this memory. May be ContainerSpace.
            capacity (int): Maximum capacity of the memory.
        """
        super().__init__()

        self.capacity = capacity
        assert "B" not in record_space, \
            "ERROR: `record_space` ({}) for Memory must be non-batched.".format(record_space)
        self.record_space = record_space.with_batch(dimension=self.capacity)
        self.flat_record_space = tf.nest.flatten(self.record_space)

        # The current size of the memory.
        self.size = 0

        # If debug.KeepLastMemoryBatch is set: Store last pulled batch in this property.
        self.last_records_pulled = None

    @abstractmethod
    def add_records(self, records):
        """
        Inserts records into this memory (`records` must be batched).

        Args:
            records (FlattenedDataOp): FlattenedDataOp containing record data. Keys must match keys in record
                space.
        """
        raise NotImplementedError

    @abstractmethod
    def get_records(self, num_records=1):
        """
        Returns a number of records according to the retrieval strategy implemented by
        the memory.

        Args:
            num_records (int): Number of records to return.

        Returns:
            any: The retrieved records.
        """
        raise NotImplementedError

    def get_size(self):
        """
        Returns the current size of the memory.

        Returns:
            int: The current size of the memory.
        """
        return self.size

    def get_number_and_flatten_records(self, records, single):
        """
        Returns the number of records (even if a single, non-batched record is provided) and the flattened records.

        Args:
            records (any): The records to insert.

            single (bool): Optional flag to indicate that we are being passed a single record. This will avoid a
                `Space.contains()` check on our record_space, but is otherwise ok to leave as False, even if the
                incoming record is single/non-batched.

        Returns:
            Tuple:
                - int: The number of records.
                - list: The flattened records.
        """
        flat_records = tf.nest.flatten(records)
        # Single (non-batched) record.
        if single is True or self.flat_record_space[0].get_shape(include_main_Axes=True) == \
                (self.capacity,) + flat_records[0].shape:
            num_records = 0
        else:
            num_records = get_batch_size(flat_records[0])
        # Non batched, single entry -> Add batch rank.
        if num_records == 0:
            flat_records = [np.array([r]) for r in flat_records]
            num_records = 1

        # Make sure `records` roughly matches our record_space.
        assert len(flat_records) == len(self.flat_record_space), \
            "ERROR: Structure of `records` does not seem to match `self.record_space`!"

        return num_records, flat_records
