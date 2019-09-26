# Copyright 2019 ducandu GmbH. All Rights Reserved.
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

import cv2
import numpy as np

from surreal.components.preprocessors.preprocessor import Preprocessor

cv2.ocl.setUseOpenCL(False)


class ImageResize(Preprocessor):
    """
    Resizes one or more images to a new size without touching the color channel.
    """
    def __init__(self, width, height, interpolation="bilinear"):
        """
        Args:
            width (int): The new width.
            height (int): The new height.
            interpolation (str): One of "bilinear", "area". Default: "bilinear" (which is also the default for both
                cv2 and tf).
        """
        super().__init__()

        self.width = width
        self.height = height

        assert interpolation in ["bilinear", "area"]

        self.cv2_interpolation = cv2.INTER_LINEAR if interpolation == "bilinear" else cv2.INTER_AREA

    def call(self, inputs):
        #if isinstance(inputs, list):
        #    inputs = np.asarray(inputs)
        had_single_color_dim = (inputs.shape[-1] == 1)

        # Batch of samples.
        if inputs.ndim == 4:
            resized = []
            for i in range(len(inputs)):
                resized.append(cv2.resize(
                    inputs[i], dsize=(self.width, self.height), interpolation=self.cv2_interpolation)
                )
            resized = np.asarray(resized)

        # Single sample.
        else:
            resized = cv2.resize(
                inputs, dsize=(self.width, self.height), interpolation=self.cv2_interpolation
            )

        # cv2.resize removes the color rank, if its dimension is 1 (e.g. gray-scale), add it back here.
        if had_single_color_dim is True:
            resized = np.expand_dims(resized, axis=-1)

        return resized
