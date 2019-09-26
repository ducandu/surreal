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

from abc import ABCMeta
import tensorflow as tf

from surreal.components.misc.decay_components import Decay
from surreal.makeable import Makeable
from surreal.utils.util import force_list


class Optimizer(Makeable, metaclass=ABCMeta):
    """
    A local optimizer performs optimization irrespective of any distributed semantics, i.e.
    it has no knowledge of other machines and does not implement any communications with them.
    """
    def __init__(self, learning_rate=0.001, decay=0.0, clip_norm=None, clip_value=None):
        """
        Args:
            learning_rate (float): The fixed learning rate (decays must be implemented eagerly
                via optimizer.lr=.. for now).

            decay (float): Learning rate decay.

            clip_norm (Optional[float]): The value to clip-by-norm to.
                Note: Both `clip_norm` and `clip_value` may be defined.

            clip_value (Optional[float]: The value to clip-by-value to.
                Note: Both `clip_norm` and `clip_value` may be defined.
        """
        super().__init__()

        # The wrapped, tf optimizer object.
        self.optimizer = None  # type: tf.keras.optimizers.Optimizer

        self.learning_rate = Decay.make(learning_rate)
        self.decay = decay
        self.clip_norm = clip_norm
        self.clip_value = clip_value

    def minimize(self, loss, variables, time_percentage=None):
        """
        Takes one step toward minimizing some error (`loss`) term that depends on `variables`.

        Args:
            loss (tf.Tensor: The loss to minimize.
            variables (List[tf.Variable]: The list of variables to minimize.
        """
        grads_and_vars = self.get_gradients(loss, variables)
        self.apply_gradients(grads_and_vars, time_percentage=time_percentage)

    def get_gradients(self, loss, variables):
        """
        Calculates the gradients of the given loss tensor over each of the given variables.

        Args:
            loss (tf.Tensor: The loss to minimize.
            variables (List[tf.Variable]: The list of variables to minimize.

        Returns:
            List[Tuple[grad,var]]: List of tuples of grad (tensor)/var (tf.Variable) pairs. The length of the returned
                list is the same as the number of variables given.
        """
        var_list = list(variables.values()) if isinstance(variables, dict) else force_list(variables)
        grads_and_vars = self.optimizer.get_gradients(loss=loss, params=var_list)

        return grads_and_vars

    def apply_gradients(self, grads_and_vars, time_percentage=None):
        """
        Applies a given `grads_and_vars` list using the Optimizer's logic (e.g. learning rate, decay, etc..).

        Args:
            grads_and_vars (List[Tuple[grad,var]]): List of tuples of grad (tensor)/var (tf.Variable) pairs. The grads
                hereby are applied to the respective vars.

            time_percentage (float): The time-percentage value (starting from 0.0 e.g. at beginning of learning to
                1.0 at the end of learning). If None, keep current `learning_rate`.
        """
        if time_percentage is not None:
            self.optimizer.learning_rate = self.learning_rate(time_percentage)

        # Clip by norm.
        if self.clip_norm is not None:
            for i, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:
                    grads_and_vars[i] = (tf.clip_by_norm(grad, self.clip_norm), var)
        # Clip by value.
        if self.clip_value is not None:
            for i, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:
                    grads_and_vars[i] = (tf.clip_by_value(grad, -self.clip_value, self.clip_value), var)

        self.optimizer.apply_gradients(grads_and_vars=grads_and_vars)
