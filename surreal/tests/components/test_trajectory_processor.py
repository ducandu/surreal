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
                    baseline_slice.append(0)
                else:
                    baseline_slice.append(baseline[i])

                adjusted_v = np.asarray(baseline_slice)

                # +1 because we want to include i-th value.
                delta = reward[start_index:i + 1] + discount * adjusted_v[1:] - adjusted_v[:-1]
                deltas.extend(delta)
                start_index = i + 1
            i += 1

        return np.array(deltas)

    gamma = 0.99
    gae_lambda = 1.0
    rewards_space = Float(main_axes="B")
    values_space = Float(main_axes="B")
    terminals_space = Bool(main_axes="B")

    @staticmethod
    def discount(x, gamma):
        # Discounts a single sequence.
        discounted = []
        prev = 0
        index = 0
        # Apply discount to value.
        for val in reversed(x):
            decayed = prev + val * pow(gamma, index)
            discounted.append(decayed)
            index += 1
            prev = decayed
        return list(reversed(discounted))

    @staticmethod
    def discount_all(values, decay, terminal):
        # Discounts multiple sub-sequences by keeping track of terminals.
        discounted = []
        i = len(values) - 1
        prev_v = 0.0
        for v in reversed(values):
            # Arrived at new sequence, start over.
            if np.all(terminal[i]):
                prev_v = 0.0

            # Accumulate prior value.
            accum_v = v + decay * prev_v
            discounted.append(accum_v)
            prev_v = accum_v

            i -= 1
        return list(reversed(discounted))

    @staticmethod
    def gae_helper(baseline, reward, terminals, sequence_indices, gamma, gae_lambda):
        # Bootstrap adjust.
        deltas = []
        start_index = 0
        i = 0
        sequence_indices[-1] = True
        for _ in range(len(baseline)):
            if np.all(sequence_indices[i]):
                # Compute deltas for this subsequence.
                # Cannot do this all at once because we would need the correct offsets for each sub-sequence.
                baseline_slice = list(baseline[start_index:i + 1])

                if np.all(terminals[i]):
                    baseline_slice.append(0)
                else:
                    baseline_slice.append(baseline[i])
                adjusted_v = np.asarray(baseline_slice)

                # +1 because we want to include i-th value.
                delta = reward[start_index:i + 1] + gamma * adjusted_v[1:] - adjusted_v[:-1]
                deltas.extend(delta)
                start_index = i + 1
            i += 1

        deltas = np.asarray(deltas)
        return np.asarray(TestTrajectoryProcessor.discount_all(deltas, gamma * gae_lambda, terminals))

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

    def test_gae_with_manual_numbers_and_lambda_0_5(self):
        lambda_ = 0.5
        lg = lambda_ * self.gamma
        processor = TrajectoryProcessor()

        r = np.array([0.1, 0.2, 0.3])
        V = np.array([1.0, 2.0, 3.0])
        t = np.array([False, False, False])
        # Final interrupt signal must always be True.
        i = np.array([False, False, True])

        # Test TD-error outputs.
        td = np.array([1.08, 1.17, 0.27])
        out = processor.get_td_errors(V, r, t, i, None, self.gamma)
        check(out, td, decimals=5)

        expected_gaes_manual = np.array([
            td[0] + lg * td[1] + lg * lg * td[2],
            td[1] + lg * td[2],
            td[2]
        ])
        expected_gaes_helper = self.gae_helper(V, r, t, i, self.gamma, lambda_)
        check(expected_gaes_manual, expected_gaes_helper, decimals=5)
        advantages = processor.get_gae_values(V, r, t, i, None, self.gamma, lambda_)
        check(advantages, expected_gaes_manual)

    def test_gae_single_non_terminal_sequence(self):
        processor = TrajectoryProcessor()

        r = self.rewards_space.sample(10)
        V = self.values_space.sample(10)
        t = self.terminals_space.sample(size=10, fill_value=False)
        # Final interrupt signal must always be True.
        i = np.array([False] * 10)

        # Assume sequence indices = terminals here.
        advantage_expected = self.gae_helper(V, r, t, i, self.gamma, self.gae_lambda)
        advantage = processor.get_gae_values(V, r, t, i, None, self.gamma, self.gae_lambda)
        check(advantage, advantage_expected, decimals=5)

    def test_gae_multiple_sequences(self):
        processor = TrajectoryProcessor()

        r = self.rewards_space.sample(10)
        V = self.values_space.sample(10)
        t = [False] * 10
        t[5] = True
        t = np.asarray(t)
        i = [False] * 10
        i[5] = True
        i = np.asarray(i)

        advantage_expected = self.gae_helper(V, r, t, i, self.gamma, self.gae_lambda)
        advantage = processor.get_gae_values(V, r, t, i, None, self.gamma, self.gae_lambda)
        check(advantage, advantage_expected, decimals=5)
