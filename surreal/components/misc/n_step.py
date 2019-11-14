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

from collections import deque
import copy
import numpy as np
import tensorflow as tf

from surreal.makeable import Makeable


class NStep(Makeable):
    """
    An n-step helper component using a Deque that's n elements in capacity.
    It takes a record of s, a, r, s', t values and returns a matching n-step s, a, r^, s'^, t^, where
    r^ is the n-step discounted reward, s'^ is the n-step next state (after n steps of earlier if a terminal
    is observed) and t^ is the n-step terminal signal.
    This new tuple can then be used just like a regular 1-step one within e.g. Q-learning.
    """
    def __init__(self, gamma, n_step=1, n_step_only=True):
        """
        Args:
            gamma (float): The discount factor (for reward discounting over the n-steps).
            n_step (int): The number of steps (n) to "look ahead/back" when converting 1-step tuples into n-step ones.

            n_step_only (bool): Whether to exclude samples that are shorter than `n_step` AND don't have a terminal
                at the end.
        """
        super().__init__()

        self.gamma = gamma
        self.n_step = n_step
        self.n_step_only = n_step_only
        # Our deque to store 1-step tuples (may be batched) over n steps.
        self.queue = deque([], maxlen=self.n_step)

    def __call__(self, s, a, r, t, s_):
        """
        Converts the given tuple (maybe batched data) into an n-step tuple using our deque n-step-memory.

        Args:
            s (any): The batch of states.
            a (any): The batch of actions.
            r (any): The batch of (single-step) rewards.

            t (any): The batch of terminal signals (whether s_ is terminal, in which case s_ may already be the next
                reset state).

            s_ (any): The batch of next states (s').

        Returns:
            Dict[s,a,r^,t^,s'^]: The dict of the n-step conversion of the incoming tuple.
        """
        self.queue.append(dict(s=s, a=a, r=r, t=t))
        records = {"s": [], "a": [], "r": [], "t": [], "s_": [], "n": []}
        # TODO: What if complex container state space?
        if isinstance(s_, tf.Tensor):
            s_ = s_.numpy()
        else:
            s_ = copy.deepcopy(s_)
        if isinstance(t, tf.Tensor):
            t = t.numpy()
        else:
            t = copy.deepcopy(t)
        batch_size = t.shape[0]
        r_sum = 0.0
        num_steps = np.array([1] * batch_size)
        batch_indices_with_at_least_one_record_already = {}
        # N-step loop (moving back in deque).
        for i in reversed(range(len(self.queue))):
            record = self.queue[i]
            # Add up rewards as we move back.
            r_sum += record["r"]

            # Batch loop.
            for batch_index in range(batch_size):
                # If we are only collecting exactly n-step samples (of ones that terminate), do only one
                # sample per batch item.
                if self.n_step_only is True and batch_index in batch_indices_with_at_least_one_record_already:
                    continue
                # Reached n-steps OR a terminal (s' at i is already a reset-state (first one in episode)).
                if i == 0 or self.queue[i - 1]["t"][batch_index]:
                    # Do not include samples smaller than n-steps w/o a terminal?
                    if self.n_step_only is False or num_steps[batch_index] == self.n_step or t[batch_index]:
                        # Add done n-step record to our records.
                        records["s"].append(record["s"][batch_index])
                        records["a"].append(record["a"][batch_index])
                        records["r"].append(r_sum[batch_index])
                        records["t"].append(t[batch_index])
                        records["s_"].append(s_[batch_index])
                        records["n"].append(num_steps[batch_index])
                        batch_indices_with_at_least_one_record_already[batch_index] = True
                    if i > 0 and self.queue[i - 1]["t"][batch_index]:
                        r_sum[batch_index] = 0.0
                        num_steps[batch_index] = 0
                        s_[batch_index] = record["s"][batch_index]  # the reset-state
                        t[batch_index] = True

            # Keep multiplying by discount factor.
            r_sum *= self.gamma
            num_steps += 1

        # Return all records (non-horizontally).
        if len(records["s"]) > 0:
            return dict(
                s=np.array(records["s"]), a=np.array(records["a"]), r=np.array(records["r"]),
                t=np.array(records["t"]), s_=np.array(records["s_"]), n=np.array(records["n"])
            )
