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
from collections import deque
import numpy as np
import tensorflow as tf

from surreal.makeable import Makeable
from surreal.spaces import Dict
from surreal.utils.errors import SurrealError
from surreal.utils.util import get_batch_size


class Memory(Makeable, metaclass=ABCMeta):
    """
    Abstract memory component.
    """
    def __init__(self, record_space, capacity=1000, *, next_record_setup=None):
        """
        Args:
            record_space (Space): The Space of the records that will go into this memory. May be ContainerSpace.
            capacity (int): Maximum capacity of the memory.

            next_record_setup (Optional[Dict[str,str]]): If given, will specify, which field(s) (keys) in the
                record_space should return the corresponding field(s) (values) of the "next" records, thereby
                respecting a possible batch size with which entries come in.
                If `next_record_setup` is provided, the batch size sent to `add_records` must always be
                the same, otherwise an error is raised.
                The setup reduces the amount of memory needed by 50% in some cases (e.g. Atari RL environments).
                All records passed to `add_records` must have the "next" fields given as specified in the value(s) of
                the Dict.
                E.g.
                `next_record_setup`=dict(s="s_", a="a_"): This will always return the s_ and a_ fields in all
                    records produced by `get_records`, whose values are taken from the respective s/a fields of
                    the next record. In return, the memory expects s_ and a_ to be set in each call to `add_records`.

                For an n-step setup, the extra key: `n_step` is required in `next_record_setup`. This key holds the
                number of steps to span over. Corresponds to the `n_step` config parameter in RLAlgos that support
                n-step learning.
        """
        super().__init__()

        assert "B" not in record_space, \
            "ERROR: `record_space` ({}) for Memory must be non-batched.".format(record_space)
        self.capacity = capacity

        self.record_space = record_space.with_batch(dimension=self.capacity)
        self.flat_record_space = tf.nest.flatten(self.record_space)

        # Iff `next_record_setup` is given, incoming batches must always be of the same size.
        self.batch_size = None
        # The dict carrying all next-record setup information.
        self.next_record_setup = None
        self.n_step = 1
        if next_record_setup:
            assert isinstance(record_space, Dict), "ERROR: Memory's `record_space` must be a Dict Space!"
            # We have an n-step setup.
            self.n_step = next_record_setup.pop("n_step", 1)
            # Figure out, which fields correspond to which flattened memory bins.
            self.next_record_setup = {}
            global i
            i = -1

            def numerate(s):
                global i
                i += 1
                return i

            keys_to_bins = tf.nest.map_structure(numerate, self.record_space.structure)
            for field, next_field in next_record_setup.items():
                self.next_record_setup[field] = (next_field, tf.nest.flatten(keys_to_bins[field]))

        # Extra space for storing next-states of the last n (before `self.index`, where n=batch size) records.
        self.next_records = deque(maxlen=self.n_step)

        # Create the main memory as a flattened OrderedDict from any arbitrarily nested Space.
        self.memory = tf.nest.map_structure(lambda space: space.create_variable(), self.flat_record_space)

        # The current size of the memory.
        self.size = 0

        # Current index into the buffer.
        self.index = 0

        # If debug.KeepLastMemoryBatch is set: Store last pulled batch in this property.
        self.last_records_pulled = None

    @abstractmethod
    def add_records(self, records):
        """
        Inserts records into this memory (`records` must be batched). In case `self.next_record_setup` is
        given, records must already include the respective next-fields. The values of these will already be inserted
        here at the respective positions to avoid invalid sampling (of these next-values) at the edge of the memory.

        Args:
            records (FlattenedDataOp): FlattenedDataOp containing record data. Keys must match keys in record
                space.
        """
        raise NotImplementedError

    def get_records(self, num_records=1):
        """
        Returns a number of records according to the retrieval strategy implemented by
        the memory.

        Args:
            num_records (int): Number of records to return.

        Returns:
            any: The retrieved records.
        """
        records, _ = self.get_records_with_indices(num_records)
        # Only return the records, not indices.
        return records

    @abstractmethod
    def get_records_with_indices(self, num_records=1):
        """
        Same as `get_records`, but also returns the corresponding list of indices for the records pulled.

        Args:
            num_records (int): Number of records to return.

        Returns:
            Tuple:
                any: The retrieved records.
                List[int]: The list of indices of the retrieved records.
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
        # Extract next-values from records before flattening.
        flat_next_records = None
        if self.next_record_setup:
            next_records = {}
            for field, (next_field, bins) in self.next_record_setup.items():
                next_value = records[next_field]
                del records[next_field]
                next_records[field] = next_value
            flat_next_records = tf.nest.flatten(next_records)

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

        # Check for correct batch size.
        if self.next_record_setup:
            if self.batch_size is None:
                self.batch_size = num_records
                assert self.capacity % self.batch_size == 0, \
                    "ERROR: `batch_size` set to {}. But `capacity` must be a multiple of memory's `batch_size`!".\
                    format(self.batch_size)
            elif num_records != self.batch_size:
                raise SurrealError(
                    "Incoming batch has wrong size ({}). Must always be {}!".format(num_records, self.batch_size)
                )

        # Make sure `records` roughly matches our record_space.
        assert len(flat_records) == len(self.flat_record_space), \
            "ERROR: Structure of `records` does not seem to match `self.record_space`!"

        # We have an `next_record_setup`.
        if self.next_record_setup:
            # Add the next-values to our "reserve" area.
            self.next_records.append(flat_next_records)

        return num_records, flat_records

    def inject_next_values_if_necessary(self, indices, records):
        """
        If required (`self.next_record_setup` is defined), injects into `records` the necessary next-values.
        Either pulls next-values from some records (n-steps) ahead or from `self.next_records` depending on
        `self.index` and the `indices` of the records.

        Args:
            indices (List[int]): The indices of the records to pull.
            records (List[any]): The actual records (already pulled) that now need to be extended by the next-values.
        """
        if self.next_record_setup:
            # The critical range is the index range for which we cannot simply go ahead n-steps to get the
            # next-values as the records n-steps ahead are unrelated (from a much earlier insertion) to the records at
            # `indices`. Therefore, we must use the `self.next_records` area to get the correct next-values.
            critical_range = [i % self.capacity for i in range(self.index - self.batch_size * self.n_step, self.index)]
            # Loop through all next-record setups.
            for field, (next_field, memory_bins) in self.next_record_setup.items():
                next_values = []
                for next_var, var in enumerate(memory_bins):
                    a = []
                    for i in indices:
                        # i is within last batch -> Take next-values from reserve area.
                        if i in critical_range:
                            pos_in_critical_range = critical_range.index(i)
                            # Not enough records in memory yet to produce an n-step sample.
                            if len(self.next_records) <= pos_in_critical_range // self.batch_size:
                                raise SurrealError(
                                    "Memory with n-step={} not ready yet to pull records from. Insert enough samples "
                                    "first to reach n-step capability. Current size={}.".format(self.n_step, self.size)
                                )
                            a.append(self.next_records[pos_in_critical_range // self.batch_size][next_var][pos_in_critical_range % self.batch_size])
                        # i is not within last batch -> Take next-values from next records (n-steps ahead) in memory.
                        else:
                            a.append(self.memory[var][(i + self.batch_size * self.n_step) % self.capacity])
                    next_values.append(np.array(a))
                records[next_field] = tf.nest.pack_sequence_as(self.record_space[field].structure, next_values)

        return records
