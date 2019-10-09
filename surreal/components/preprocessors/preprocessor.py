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

import cv2
import tensorflow as tf

from surreal import PATH_PREPROCESSING_LOGS
from surreal.debug import StorePreprocessingEveryNCalls, StorePreprocessingOnlyForFirstNPreprocessorComponents
from surreal.makeable import Makeable
from surreal.spaces import Space
from surreal.spaces.space_utils import get_space_from_data


class Preprocessor(Makeable):
    """
    A generic preprocessor holding other Preprocessors that are called (or reset) one after the other.
    """
    def __init__(self, *components, **dict_components):
        """
        Args:
            components: The single preprocessors to stack up.
        """
        # Make sure preprocessor is either only defined via (sequential) components list or (horizontal) dict.
        assert not dict_components or len(components) == 0

        # In case a list is given.
        if len(components) == 1 and isinstance(components[0], list):
            components = components[0]

        if dict_components:
            self.components = [tf.nest.flatten(tf.nest.map_structure(lambda c: Preprocessor.make(c) if not callable(c) else c, dict_components))]
            #self.components = [tf.nest.flatten(tf.nest.map_structure(lambda c: Preprocessor.make(c), dict_components))]
        else:
            self.components = [Preprocessor.make(c) if not callable(c) else c for c in components]
            #self.components = [Preprocessor.make(c) for c in components]

        self.has_state = any(hasattr(c, "has_state") and c.has_state is True for c in self.components)

        # Simple counter of how many times we have been called.
        self.num_calls = 0

        # Make sure everything is ok.
        assert all(callable(c) or isinstance(c, list) and all(callable(sc) for sc in c) for c in self.components), \
            "ERROR: All components (and dict sub-components) of a Preprocessor must be callable!"

    def reset(self, batch_position=None):
        """
        Calls `reset()` on each component preprocessor (only if it has a method `reset`).

        Args:
            batch_position (Optional[int]): If applicable, an optional batch position, which is to be reset only,
                leaving the remaining state of the preprocessor intact. This only really applies to preprocessors
                that have `self.has_state` = True.
        """
        for c in self.components:
            # In case a component is a lambda or other callable object (which doesn't have a `reset` method).
            if hasattr(c, "reset") and callable(c.reset):
                c.reset(batch_position)

    def __call__(self, inputs):
        self.num_calls += 1
        if isinstance(inputs, Space):
            sample = inputs.sample()
            preprocessed_sample = self.call(sample)
            main_axes_after_preprocessing = inputs.main_axes
            out_space = get_space_from_data(preprocessed_sample, main_axes=main_axes_after_preprocessing)
            return out_space
        else:
            return self.call(inputs)

    def call(self, inputs):
        """
        Calls each single preprocessor once in the correct order passing the output of the previous preprocessing
        step into the next one.

        Args:
            inputs (any): The inputs to preprocess.

        Returns:
            any: The preprocessed inputs.
        """
        # Store the preprocessing?
        if StorePreprocessingEveryNCalls and self.num_calls % StorePreprocessingEveryNCalls == 0:
            Preprocessor._debug_store(PATH_PREPROCESSING_LOGS + "pp_{:03d}_{:02d}".format(self.num_calls, 0), inputs)

        in_ = inputs
        for i, c in enumerate(self.components):
            # A flattened dict-preprocessor.
            if isinstance(c, list):
                assert isinstance(in_, dict)
                in_flattened = tf.nest.flatten(in_)
                for j, sub_component in enumerate(c):
                    if sub_component is not None:
                        in_flattened[j] = sub_component(in_flattened[j])
                in_ = tf.nest.pack_sequence_as(in_, in_flattened)
            # A single preprocessor.
            else:
                in_ = c(in_)

            if StorePreprocessingEveryNCalls and self.num_calls % StorePreprocessingEveryNCalls == 0 and (
                    StorePreprocessingOnlyForFirstNPreprocessorComponents is False or
                    i < StorePreprocessingOnlyForFirstNPreprocessorComponents
            ):
                Preprocessor._debug_store(PATH_PREPROCESSING_LOGS + "pp_{:03d}_{:02d}".format(self.num_calls, i+1), in_)
        return in_

    @staticmethod
    def _debug_store(path, data):
        # Probably a batch of images.
        if len(data.shape) == 4 and (data.shape[3] == 1 or data.shape[3] == 3):
            cv2.imwrite(path+".png", data[0])
        else:
            raise NotImplementedError

    @classmethod
    def make(cls, spec=None, **kwargs):
        if callable(spec):
            return super().make(_args=[spec])
        return super().make(spec, **kwargs)
