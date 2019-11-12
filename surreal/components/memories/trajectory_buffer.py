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

from surreal.components.memories.replay_buffer import ReplayBuffer
from surreal.spaces import Dict
from surreal.utils.nest import keys_to_flattened_struct_indices


class TrajectoryBuffer(ReplayBuffer):
    """
    Episode trajectory buffer to be used for on-policy sampling.
    The method `get_trajectories` returns most recently added (complete) trajectories.
    """
    def __init__(self, record_space, *, terminal_key="t", **kwargs):
        """
        Args:
            terminal_key (str): The name of the terminal indicator in the record Space. The record Space (a Dict) must
                have this key in the top-level.
        """
        super().__init__(record_space=record_space, enforce_batch_size=True, **kwargs)

        # TODO: setting: ok to return partial trajectories (that don't end in terminal=True, but traject_interrupt=True)

        assert isinstance(self.record_space, Dict), "ERROR: RingBuffer requires Dict as `record_space`!"
        assert terminal_key in self.record_space, \
            "ERROR: `record_space` of RingBuffer must contain '{}' as key!".format(terminal_key)
        self.terminal_key = terminal_key
        # The index in the flattened (sorted by key) Space.
        self.flat_terminal_index = keys_to_flattened_struct_indices(self.record_space.structure)[self.terminal_key]

    def get_trajectories(self, num_trajectories=1):
        """
        Returns an array (1D sequence) of the last n complete episodes that we can find in the buffer.

        Args:
            num_trajectories (int): The number of (completed) trajectories to pull starting from the end
                (one before index).

        Returns:
            any:
        """
        # The indices to pull (in the correct trajectory-forward order, from the latest episode back to earlier ones
        # until `num_episodes` collected).
        num_collected = 0  # How many have trajectories we collected so far?
        indices = []  # Collect lookup-indices in the right order (trajectories) in this list.
        last_trajectory = None  # The most recent trajectory found.
        # Start one before the current index.
        end_indices_orig = [self.index - i - 1 for i in range(self.batch_size)]
        end_indices = end_indices_orig[:]
        while num_collected < num_trajectories:
            # Collect next trajectory ending in some end_idx.
            end_idx = end_indices.pop(0)
            end_idx = end_idx % self.capacity
            trajectory = [end_idx]  # The currently collected trajectory.
            # Move backwards through the memory and collect one trajectory ending in end_idx and reaching back till
            # one after the previous terminal.
            idx = end_idx - self.batch_size
            idx = idx % self.capacity
            while not self.memory[self.flat_terminal_index][idx]:
                trajectory.append(idx)
                idx -= self.batch_size
                idx = idx % self.capacity
                # If we either reached unwritten sections of the memory OR
                # we crossed the index (going into unconnected territory) -> Stop here.
                if (self.size < self.capacity and idx >= self.index) or \
                        (self.size == self.capacity and ((self.index - idx) % self.capacity) <= self.batch_size):
                    break
            # Trajectory done -> Reverse it.
            trajectory = list(reversed(trajectory))
            # Check, if trajectory is terminated.
            if self.memory[self.flat_terminal_index][trajectory[-1]]:
                # This trajectory is good: Add it to list of indices.
                indices.extend(trajectory)
                if last_trajectory is None:
                    last_trajectory = trajectory
                num_collected += 1
            # Add another (valid) end-point (to begin a backward trajectory search) to our list.
            new_end_idx = trajectory[0] - self.batch_size
            if self.size == self.capacity or new_end_idx > 0:
                # We went once around the buffer. If we got no other end-index anymore, stop here and fill up return
                # with duplicate episodes (same behavior as ReplayBuffer, but for trajectories).
                if (new_end_idx % self.capacity) in end_indices_orig:
                    if len(end_indices) == 0:
                        if last_trajectory is not None:
                            for _ in range(num_trajectories - num_collected):
                                indices.extend(last_trajectory)
                        break
                else:
                    end_indices.append(new_end_idx)

        return self.get_records_at_indices(np.array(indices))
