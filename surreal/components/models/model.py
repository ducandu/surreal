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

from abc import ABCMeta, abstractmethod
import copy
import tensorflow as tf

from surreal.makeable import Makeable
from surreal.debug import AssertModelSync
from surreal.tests.test_util import check


class Model(Makeable, metaclass=ABCMeta):
    """
    An abstract/generic model class that can hold (and register) one or many tf.keras.models.Models and
    offers copying, access to trainable vars, etc..
    """
    def __init__(self):
        super().__init__()

        # Whether a complete pass through (call) has been performed (after which all keras models herein should be
        # built.
        self.built = False

        # The args and kwargs of the first call.
        self.init_args = None
        self.init_kwargs = None

    def __call__(self, *args, **kwargs):
        if self.init_args is None:
            self.init_args = args
            self.init_kwargs = kwargs

        results = self.call(*args, **kwargs)
        # Set built to True.
        self.built = True
        return results

    @abstractmethod
    def call(self, *args, **kwargs):
        """
        Implement Model's logic (forward pass) when called via direct `self()`.

        Args:
            *args: Args to process.
            **kwargs: Kwargs to process.

        Returns:
            any: The result of a Model forward pass.
        """
        raise NotImplementedError

    def sync_from(self, other_model, tau=1.0):
        """
        Syncs all of this Model's weights from the `other_model` using the formula:
        new_weights = old_weights sync_tau

        [new weights] = tau * [other_model's weights] + (1.0 - tau) * [old weights]

        Args:
            other_model (Model): The other Model to sync from.
            tau (float): Teh tau parameter used for soft-syncing (see formula above).
        """
        other_values = other_model.get_weights(as_ref=False)

        if AssertModelSync is True:
            try:
                check(self.get_weights(), other_values)
                print("WARNING: model weights were equal.")
            except AssertionError:
                pass

        if tau == 1.0:
            self.set_weights(other_values)
        else:
            our_vars = self.get_weights(as_ref=True)
            for our_var, other_var in zip(our_vars, other_values):
                tf.keras.backend.set_value(our_var, tau * other_var + (1.0 - tau) * our_var)

        if AssertModelSync is True:
            check(self.get_weights(), other_values)

    @abstractmethod
    def copy(self, trainable=None):
        """
        Creates an exact clone of this Model (including sub-keras Models AND their exact network weights).

        Args:
            trainable (Optional[bool]): Whether the copy's variables will be trainable or not.
                Use None for not changing anything (in case original is mixed trainable/non-trainable).

        Returns:
            Model: A copy of this Model.
        """
        raise NotImplementedError

    def get_weights(self, as_ref=False):
        """
        Returns a flattened list of all weights (as values or as tf.Variable refs) of this Model.

        Args:
            as_ref (bool): Whether to return the actual tf.Variables.
                Default: False (only return current values as numpy ndarrays).

        Returns:
            List[Union[np.ndarray,tf.Variable]]: The weights of this model as numpy values or tf Variables.
        """
        # Return tf.Variables.
        if as_ref is True:
            return self._get_weights_list()
        # Return numpy ndarrays (as values).
        else:
            return tf.nest.map_structure(lambda s: s.numpy(), self._get_weights_list())

    def variables(self):
        """
        Returns:
            List[tf.Variable]: Same as when calling `self.get_weights(as_ref=True)`.
        """
        return self.get_weights(as_ref=True)

    def set_weights(self, values):
        our_vars = self.get_weights(as_ref=True)
        assert len(our_vars) == len(values)
        for our_var, other_value in zip(our_vars, values):
            tf.keras.backend.set_value(our_var, other_value)

    @staticmethod
    def clone_component(component, trainable=None):
        """
        Clones a keras/lambda/callable component of this Model and returns a copy.

        Args:
            component (any): The component to clone.
            trainable (Optional[bool]): Whether the copy should have trainable variables or not.
                None for leaving `trainable` as is in the original Model.

        Returns:
            Any: The cloned component.
        """
        # Model -> Clone.
        if isinstance(component, tf.keras.models.Model):
            clone = tf.keras.models.clone_model(component)
        # A layer -> Construct from config.
        elif isinstance(component, tf.keras.layers.Layer):
            clone = component.__class__.from_config(component.get_config())
        # Try our luck with deepcopy.
        else:
            clone = copy.deepcopy(component)

        if trainable is not None:
            clone.trainable = trainable

        return clone

    @abstractmethod
    def _get_weights_list(self):
        """
        Returns a flattened list of all weights (as tf.Variable refs) of this Model.

        Returns:
            List[Union[np.ndarray,tf.Variable]]: The weights of this model.
        """
        raise NotImplementedError
