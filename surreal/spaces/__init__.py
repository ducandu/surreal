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

from functools import partial
import numpy as np

from surreal.spaces.primitive_spaces import PrimitiveSpace, Bool, Int, Float, Text
from surreal.spaces.container_spaces import ContainerSpace, Dict, Tuple
from surreal.spaces.space import Space
import surreal.spaces.space_utils

Space.__lookup_classes__ = dict({
    "bool": Bool,
    bool: Bool,
    np.bool_: Bool,
    "int": Int,
    int: Int,
    np.uint8: partial(Int, dtype=np.uint8),
    "uint8": partial(Int, dtype=np.uint8),
    np.int8: partial(Int, dtype=np.int8),
    "int8": partial(Int, dtype=np.int8),
    np.int32: Int,
    "int32": Int,
    np.int64: partial(Int, dtype=np.int64),
    "int64": partial(Int, dtype=np.int64),
    "continuous": Float,
    "float": Float,
    "float32": Float,
    float: Float,
    np.float32: Float,
    np.float64: partial(Float, dtype=np.float64),
    "float64": partial(Float, dtype=np.float64),
    "list": Tuple,
    "tuple": Tuple,
    "sequence": Tuple,
    dict: Dict,
    "dict": Dict,
    str: Text,
    "str": Text,
    "text": Text
})

# Default Space: A float from 0.0 to 1.0.
Space.__default_constructor__ = partial(Float, 1.0)

__all__ = ["Space", "PrimitiveSpace", "Float", "Int", "Bool", "Text",
           "ContainerSpace", "Dict", "Tuple"]
