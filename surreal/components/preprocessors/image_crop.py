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

import numpy as np

from surreal.components.preprocessors.preprocessor import Preprocessor


class ImageCrop(Preprocessor):
    """
    Crops one or more images to a new size without touching the color channel.
    """
    def __init__(self, x=0, y=0, width=0, height=0):
        """
        Args:
            x (int): Start x coordinate.
            y (int): Start y coordinate.
            width (int): Width of resulting image.
            height (int): Height of resulting image.
        """
        super().__init__()
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        assert self.x >= 0
        assert self.y >= 0
        assert self.width > 0
        assert self.height > 0

    def call(self, inputs):
        """
        Images come in with either a batch dimension or not.
        """
        #if isinstance(inputs, list):
        #    inputs = np.asarray(inputs)
        # Preserve batch dimension.
        if len(inputs.shape) >= 4:
            return inputs[:, self.y:self.y + self.height, self.x:self.x + self.width]
        else:
            return inputs[self.y:self.y + self.height, self.x:self.x + self.width]
