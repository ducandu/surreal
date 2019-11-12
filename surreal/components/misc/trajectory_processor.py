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

import numpy as np

from surreal.makeable import Makeable


class TrajectoryProcessor(Makeable):
    """
    A component that owns a collection of episode trajectory calculation methods (e.g. discounting rewards).
    """
    @staticmethod
    def get_trajectory_lengths(traject_interruptions):
        """
        Returns the lengths of episode trajectories by the `traject_interruptions` indicators (bools), where True
        indicates the end of a trajectory segment (could be an actual terminal or not).

        Args:
            traject_interruptions (np.ndarray[bool]): The flags for where in the trajectory are interruptions.
                Note: These may be actual terminals.

        Returns:
            np.ndarray[int]: The (partial) trajectory lengths.
        """
        trajectory_lengths = []
        length = 0
        for interruption_flag in traject_interruptions:
            length += 1
            # Interruption: Store the trajectory len so far and reset length counter.
            if interruption_flag:
                trajectory_lengths.append(length)
                length = 0

        # Append final sequence, if any.
        if length > 0:
            trajectory_lengths.append(length)

        return trajectory_lengths

    @staticmethod
    def get_trajectory_decays(traject_interruptions, decay=0.9):
        """
        Returns the decay factors for a trajectory, characterized by `traject_interruptions`.
        That is, a sequence with interruptions is used to compute for each partial trajectory the decay
        values and the length of the sequence.

        Example:
        decay = 0.5, traject_interruptions = [F F T F T] will return lengths [3, 2] and
        decays [1 0.5 0.25 1 0.5] (decay^0, decay^1, ..decay^k) where k = sequence length for each sub-sequence.

        Args:
            traject_interruptions (np.ndarray[bool]): The flags for where in the trajectory are interruptions.
                Note: These may be actual terminals.

            decay (float): The decay factor (discount factor).

        Returns:
            Tuple[np.ndarray[int],np.ndarray[float]]: [Sequence lengths, decays].
        """
        trajectory_lengths = []
        decays = []

        length = 0
        for interruption_flag in traject_interruptions:
            # Compute decay based on sequence length.
            decays.append(pow(decay, length))
            length += 1
            # Interruption: Store the trajectory len so far and reset length counter.
            if interruption_flag:
                trajectory_lengths.append(length)
                length = 0

        # Append final sequence, if any.
        if length > 0:
            trajectory_lengths.append(length)

        return trajectory_lengths, decays

    @staticmethod
    def reverse_apply_decays_to_sequence(values, traject_interruptions, decay=0.9):
        """
        Computes decays for sequence indices and applies them (in reverse manner to a sequence of values).
        Useful to compute discounted reward estimates across a sequence of estimates.

        Args:
            values (np.ndarray[float]): The state values V(s).

            traject_interruptions (np.ndarray[bool]): The flags for where in the trajectory are interruptions.
                Note: These may be actual terminals.

            decay (float): The decay factor (discount factor).

        Returns:
            np.ndarray[float]: Decayed sequence values.
        """
        # Scan sequences in reverse:
        decayed_values = []
        i = len(values.data) - 1
        prev_v = 0
        for v in reversed(values.data):
            # Arrived at new trajectory, start over.
            if traject_interruptions[i]:
                prev_v = 0

            # Accumulate prior value.
            accum_v = v + decay * prev_v
            decayed_values.append(accum_v)
            prev_v = accum_v
            i -= 1

        # Reverse, convert, and return final.
        return list(reversed(decayed_values))

    @staticmethod
    def bootstrap_values(values, r, t, traject_interruptions, discount=0.99):
        """
        Inserts value estimates at the end of each trajectory for a given sequence and computes deltas
        for generalized advantage estimation. That is, 0 is inserted after teach terminal and the final value of the
        sub-sequence if the sequence does not end with a terminal. We then compute for each subsequence

        delta = r + discount * bootstrapped_values[1:] - bootstrapped_values[:-1]

        Args:
            values (np.ndarray[float]): The state values V(s).
            r (np.ndarray): Rewards in sample trajectory.
            t (np.ndarray): Terminals in sample trajectory.

            traject_interruptions (np.ndarray[bool]): The flags for where in the trajectory are interruptions.
                Note: These may be actual terminals.

            discount (float): Discount to apply to delta computation.

        Returns:
            Sequence of deltas.
        """
        deltas = []
        start_index = 0
        if len(values) > 1:
            last_trajectory = np.expand_dims(traject_interruptions[-1], axis=-1)
            traject_interruptions = np.concatenate((traject_interruptions[:-1], np.ones_like(last_trajectory)), axis=0)

        for i in range(len(values)):
            if traject_interruptions[i]:
                # Compute deltas for this sub-sequence.
                # Cannot do this all at once because we would need the correct offsets for each sub-sequence.
                baseline_slice = list(values[start_index:i + 1])
                if t[i]:
                    baseline_slice.append(0)
                else:
                    baseline_slice.append(values[-1])
                adjusted_v = np.array(baseline_slice)

                # +1 because we want to include i-th value.
                delta = r[start_index:i + 1] + discount * adjusted_v[1:] - adjusted_v[:-1]
                deltas.extend(delta)
                start_index = i + 1

        return np.array(deltas)
