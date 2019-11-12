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
import unittest

from surreal.components.misc.trajectory_processor import TrajectoryProcessor
from surreal.spaces import Float, Bool
from surreal.tests.test_util import check


class TestTrajectoryProcessor(unittest.TestCase):

    input_spaces = dict(
        sequence_indices=Bool(main_axes="B"),
        terminals=Bool(main_axes="B"),
        values=Float(main_axes="B"),
        rewards=Float(main_axes="B"),
        decay=float
    )

    @staticmethod
    def decay_td_sequence(td_errors, decay=0.99, value_next=0.0):
        discounted_td_errors = np.zeros_like(td_errors)
        running_add = value_next
        for t in reversed(range(0, td_errors.size)):
            running_add = running_add * decay + td_errors[t]
            discounted_td_errors[t] = running_add
        return discounted_td_errors

    @staticmethod
    def deltas(baseline, reward, discount, terminals, sequence_values):
        """
        Computes expected 1-step TD errors over a sequence of rewards, terminals, sequence-indices:

        delta = reward + discount * bootstrapped_values[1:] - bootstrapped_values[:-1]
        """
        deltas = []
        start_index = 0
        i = 0
        for _ in range(len(baseline)):
            if np.all(sequence_values[i]):
                # Compute deltas for this sub-sequence.
                # Cannot do this all at once because we would need the correct offsets for each sub-sequence.
                baseline_slice = list(baseline[start_index:i + 1])

                # Boot-strap: If also terminal, with 0, else with last value.
                if np.all(terminals[i]):
                    print("Appending boot-strap val 0 at index.", i)
                    baseline_slice.append(0)
                else:
                    print("Appending boot-strap val {} at index {}.".format(baseline[i], i))
                    baseline_slice.append(baseline[i])

                adjusted_v = np.asarray(baseline_slice)

                print("adjusted_v", adjusted_v)
                print("adjusted_v[1:]", adjusted_v[1:])
                print("adjusted_v[:-1]",  adjusted_v[:-1])

                # +1 because we want to include i-th value.
                delta = reward[start_index:i + 1] + discount * adjusted_v[1:] - adjusted_v[:-1]
                deltas.extend(delta)
                start_index = i + 1
            i += 1

        return np.array(deltas)

    def test_calc_sequence_lengths(self):
        """
        Tests counting sequence lengths based on terminal configurations.
        """
        processor = TrajectoryProcessor()
        input_ = np.asarray([False, False, False, False])
        out = processor.get_trajectory_lengths(input_)
        check(out, [4])

        input_ = np.asarray([False, False, True, False])
        out = processor.get_trajectory_lengths(input_)
        check(out, [3, 1])

        input_ = np.asarray([True, True, True, True])
        out = processor.get_trajectory_lengths(input_)
        check(out, [1, 1, 1, 1])

        input_ = np.asarray([True, False, False, True])
        out = processor.get_trajectory_lengths(input_)
        check(out, [1, 3])

    def test_bootstrapping(self):
        """
        Tests boot-strapping for GAE purposes.
        """
        processor = TrajectoryProcessor()
        discount = 0.99

        # No terminals - just boot-strap with final sequence index.
        values = np.array([1.0, 2.0, 3.0, 4.0])
        rewards = np.array([0, 0, 0, 0])
        sequence_indices = np.array([False, False, False, True])
        terminals = np.array([False, False, False, False])

        expected_deltas = self.deltas(values, rewards, discount, terminals, sequence_indices)
        deltas = processor.bootstrap_values(values, rewards, terminals, sequence_indices)
        check(deltas, expected_deltas, decimals=5)

        # Final index is also terminal.
        values = np.asarray([1.0, 2.0, 3.0, 4.0])
        rewards = np.asarray([0, 0, 0, 0])
        sequence_indices = np.asarray([False, False, False, True])
        terminals = np.asarray([False, False, False, True])

        expected_deltas = self.deltas(values, rewards, discount, terminals, sequence_indices)
        deltas = processor.bootstrap_values(values, rewards, terminals, sequence_indices)
        check(deltas, expected_deltas, decimals=5)

        # Mixed: i = 1 is also terminal, i = 3 is only sequence.
        values = np.asarray([1.0, 2.0, 3.0, 4.0])
        rewards = np.asarray([0, 0, 0, 0])
        sequence_indices = np.asarray([False, True, False, True])
        terminals = np.asarray([False, True, False, False])

        expected_deltas = self.deltas(values, rewards, discount, terminals, sequence_indices)
        deltas = processor.bootstrap_values(values, rewards, terminals, sequence_indices)
        check(deltas, expected_deltas, decimals=5)

    def test_calc_decays(self):
        """
        Tests counting sequence lengths based on terminal configurations.
        """
        processor = TrajectoryProcessor()
        decay_value = 0.5

        input_ = np.asarray([False, False, False, False])
        expected_decays = [1.0, 0.5, 0.25, 0.125]
        lengths, decays = processor.get_trajectory_decays(input_, decay_value)

        # Check lengths and decays.
        check(x=lengths, y=[4])
        check(x=decays, y=expected_decays)

        input_ = np.asarray([False, False, True, False])
        expected_decays = [1.0, 0.5, 0.25, 1.0]
        lengths, decays = processor.get_trajectory_decays(input_, decay_value)

        check(x=lengths, y=[3, 1])
        check(x=decays, y=expected_decays)

        input_ = np.asarray([True, True, True, True])
        expected_decays = [1.0, 1.0, 1.0, 1.0]
        lengths, decays = processor.get_trajectory_decays(input_, decay_value)

        check(x=lengths, y=[1, 1, 1, 1])
        check(x=decays, y=expected_decays)

    def test_reverse_apply_decays_to_trajectory(self):
        """
        Tests reverse decaying a sequence of 1-step TD errors for GAE.
        """
        processor = TrajectoryProcessor()
        decay_value = 0.5

        td_errors = np.asarray([0.1, 0.2, 0.3, 0.4])
        indices = np.array([0, 0, 0, 1])
        expected_output_sequence_manual = np.asarray([
            0.1 + 0.5 * 0.2 + 0.25 * 0.3 + 0.125 * 0.4,
            0.2 + 0.5 * 0.3 + 0.25 * 0.4,
            0.3 + 0.5 * 0.4,
            0.4
        ])
        expected_output_sequence_numpy = self.decay_td_sequence(td_errors, decay=decay_value)
        check(expected_output_sequence_manual, expected_output_sequence_numpy)
        out = processor.reverse_apply_decays_to_trajectory(td_errors, indices, decay_value)
        check(out, expected_output_sequence_manual)
