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

from surreal.utils.numpy import sigmoid, softmax, relu, one_hot
from surreal.utils.errors import SurrealError, SurrealSpaceError, SurrealObsoletedError
from surreal.utils.nest import flatten_alongside, keys_to_flattened_struct_indices
from surreal.utils.util import convert_dtype, get_shape, get_rank, force_tuple, force_list, \
    LARGE_INTEGER, SMALL_NUMBER, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT, default_dict, get_batch_size, complement_struct

__all__ = [
    "SurrealError", "SurrealSpaceError", "SurrealObsoletedError",
    "flatten_alongside", "keys_to_flattened_struct_indices",
    "complement_struct",
    "convert_dtype", "get_shape", "get_rank", "force_tuple", "force_list",
    "sigmoid", "softmax", "relu", "one_hot",
    "default_dict", "get_batch_size",
    "LARGE_INTEGER", "SMALL_NUMBER", "MIN_LOG_NN_OUTPUT", "MAX_LOG_NN_OUTPUT"
]
