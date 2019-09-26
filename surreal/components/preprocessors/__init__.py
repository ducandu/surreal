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

from surreal.components.preprocessors.grayscale import GrayScale
from surreal.components.preprocessors.image_crop import ImageCrop
from surreal.components.preprocessors.image_resize import ImageResize
from surreal.components.preprocessors.moving_standardize import MovingStandardize
from surreal.components.preprocessors.normalize import Normalize
from surreal.components.preprocessors.preprocessor import Preprocessor
from surreal.components.preprocessors.sequence import Sequence

Preprocessor.__lookup_classes__ = dict(
    grayscale=GrayScale,
    imagecrop=ImageCrop,
    imageresize=ImageResize,
    movingstandardize=MovingStandardize,
    normalize=Normalize,
    preprocessor=Preprocessor,
    sequence=Sequence
)

__all__ = list(set(map(lambda x: x.__name__, Preprocessor.__lookup_classes__.values())))
