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

from surreal import PATH_PREPROCESSING_LOGS
from surreal.debug import StorePreprocessingEveryNCalls, StorePreprocessingOnlyForFirstNPreprocessorComponents
from surreal.makeable import Makeable
from surreal.spaces import Space
from surreal.spaces.space_utils import get_space_from_data


class Preprocessor(Makeable):
    """
    A generic preprocessor holding other Preprocessors that are called (or reset) one after the other.
    """
    def __init__(self, *components):
        """
        Args:
            components: The single preprocessors to stack up.
        """
        # In case a list is given.
        if len(components) == 1 and isinstance(components[0], list):
            components = components[0]
        self.components = components

        self.has_state = any(hasattr(c, "has_state") and c.has_state is True for c in self.components)

        # Simple counter of how many times we have been called.
        self.num_calls = 0

        assert all(callable(c) for c in self.components), "ERROR: All components of a Preprocessor must be callable!"

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

        i = inputs
        for j, c in enumerate(self.components):
            i = c(i)
            if StorePreprocessingEveryNCalls and self.num_calls % StorePreprocessingEveryNCalls == 0 and (
                    StorePreprocessingOnlyForFirstNPreprocessorComponents is False or
                    j < StorePreprocessingOnlyForFirstNPreprocessorComponents
            ):
                Preprocessor._debug_store(PATH_PREPROCESSING_LOGS + "pp_{:03d}_{:02d}".format(self.num_calls, j+1), i)
        return i

    @staticmethod
    def _debug_store(path, data):
        # Probably a batch of images.
        if len(data.shape) == 4 and (data.shape[3] == 1 or data.shape[3] == 3):
            cv2.imwrite(path+".png", data[0])
        else:
            raise NotImplementedError

