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

import tensorflow as tf

from surreal.makeable import Makeable
from surreal.components.misc.trajectory_processor import TrajectoryProcessor


class GeneralizedAdvantages(Makeable):
    """
    A component to calculate generalized advantage estimations (GAE) [1].

    [1] High-Dimensional Continuous Control Using Generalized Advantage Estimation - Schulman et al. - U. Berkeley 2015
        https://arxiv.org/abs/1506.02438
    """
    def __init__(self, discount=1.0, lambda_=None, clip_value=0.0, **kwargs):
        """
        Args:
            discount (float): The discount factor gamma.

            lambda_ (float): The lambda value. Determines the weighting of the different n-step chunks, 0.0 meaning
                purely 1-step, 1.0 meaning a exponentially decaying mix of n-step rewards till infinity (Monte-Carlo).
                See paper for details.

            clip_value (float): The clipping value to use on the rewards.
        """
        super().__init__()
        self.lambda_ = lambda_ if lambda_ is not None else kwargs.get("lambda", 1.0)
        self.discount = discount
        self.trajectory_processor = TrajectoryProcessor()
        self.clip_value = clip_value

    def get_td_errors(self, values, r, t, traject_interruptions):
        """
        Returns the 1-step TD Errors (r + gamma V(s') - V(s)) after clipping rewards if applicable (see c'tor).

        Args:
            values (np.ndarray[float]): The state values V(s).
            r (np.ndarray): Rewards in sample trajectory.
            t (np.ndarray): Terminals in sample trajectory.

            traject_interruptions (np.ndarray[bool]): The flags for where in the trajectory are interruptions.
                Note: These may be actual terminals.

        Returns:
            np.ndarray: 1-step TD errors.
        """
        r = tf.clip_by_value(r, -self.clip_value, self.clip_value)

        # Next, we need to set the next value after the end of each sub-sequence to 0/its prior value
        # depending on terminal, then compute 1-step TD-errors: delta = r[:] + gamma * v[1:] - v[:-1]
        # -> where len(v) = len(r) + 1 b/c v contains the extra (bootstrapped) last value.
        # Terminals indicate boot-strapping. Sequence indices indicate episode fragments in case of a multi-environment.
        td_errors = self.trajectory_processor.bootstrap_values(values, r, t, traject_interruptions, self.discount)
        return td_errors

    def get_gae_values(self, values, r, t, traject_interruptions):
        """
        Returns advantage values based on GAE ([1]). Clips rewards if applicable (see c'tor).

        Args:
            values (np.ndarray[float]): The state values V(s).
            r (np.ndarray): Rewards in sample trajectory.
            t (np.ndarray): Terminals in sample trajectory.

            traject_interruptions (np.ndarray[bool]): The flags for where in the trajectory are interruptions.
                Note: These may be actual terminals.

        Returns:
            np.ndarray: Advantage estimation values.
        """
        deltas = self.get_td_errors(values, r, t, traject_interruptions)

        gae_discount = self.lambda_ * self.discount
        # Apply gae discount to each sub-sequence.
        # Note: sequences are indicated by sequence indices, which may not be terminal.
        gae_values = self.trajectory_processor.reverse_apply_decays_to_sequence(
            deltas, traject_interruptions, decay=gae_discount
        )
        return gae_values
