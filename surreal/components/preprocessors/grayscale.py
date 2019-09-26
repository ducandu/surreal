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


class GrayScale(Preprocessor):
    """
    A simple gray scale converter for RGB images of arbitrary dimensions (normally, an image is 2D).
    Using weights: (0.299, 0.587, 0.114), which are the magic RGB-weights for "natural" gray-scaling results.
    """
    def __init__(self, keepdims=False):
        """
        Args:
            keepdims (bool): Whether to keep the color-depth rank in the pre-processed tensor (default: False).
        """
        super(GrayScale, self).__init__()

        # Whether to keep the last rank with dim=1.
        self.keepdims = keepdims

    def call(self, inputs):
        """
        Gray-scales images of arbitrary rank.
        Normally, the images' rank is 3 (width/height/colors), but can also be: batch/width/height/colors, or any other.
        However, the last rank must be of size: len(self.weights).

        Args:
            inputs (tensor): Single image or a batch of images to be gray-scaled (last rank=n colors, where
                n=len(self.weights)).

        Returns:
            any: The gray-scaled image.
        """
        # The reshaped weights used for the gray-scaling operation.
        #if isinstance(inputs, list):
        #    inputs = np.asarray(inputs)
        if inputs.ndim == 4:
            gray_scaled = []
            for i in range(len(inputs)):
                scaled = cv2.cvtColor(inputs[i], cv2.COLOR_RGB2GRAY)
                gray_scaled.append(scaled)
            scaled_images = np.asarray(gray_scaled)

            #if self.keepdims:
            #    scaled_images = scaled_images[:, :, :, np.newaxis]
        else:
            # Sample by sample.
            scaled_images = cv2.cvtColor(inputs, cv2.COLOR_RGB2GRAY)

        # Keep last dim.
        if self.keepdims:
            scaled_images = np.reshape(scaled_images, newshape=scaled_images.shape + (1,))

        return scaled_images
