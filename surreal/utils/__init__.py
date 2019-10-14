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

#from surreal.utils.define_by_run_ops import print_call_chain
#from surreal.utils.initializer import Initializer
from surreal.utils.numpy import sigmoid, softmax, relu, one_hot
#from surreal.utils.ops import DataOp, SingleDataOp, DataOpDict, DataOpTuple, ContainerDataOp, FlattenedDataOp
#from surreal.utils.pytorch_util import pytorch_one_hot, PyTorchVariable
from surreal.utils.errors import SurrealError, SurrealSpaceError, SurrealObsoletedError
from surreal.utils.util import convert_dtype, get_shape, get_rank, force_tuple, force_list, \
    LARGE_INTEGER, SMALL_NUMBER, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT, default_dict, get_batch_size, complement_struct
#    tf_logger, print_logging_handler, root_logger, logging_formatter, , get_num_return_values, \
#    , get_method_type

__all__ = [
    "SurrealError", "SurrealSpaceError", "SurrealObsoletedError",
    "complement_struct",
    #"Initializer", ,
    "convert_dtype", "get_shape", "get_rank", "force_tuple", "force_list",
    #"logging_formatter", "root_logger", "tf_logger", "print_logging_handler",
    "sigmoid", "softmax", "relu", "one_hot",
    "default_dict", "get_batch_size",
    #"DataOp", "SingleDataOp", "DataOpDict", "DataOpTuple", "ContainerDataOp", "FlattenedDataOp",
    #"pytorch_one_hot", "PyTorchVariable",
    "LARGE_INTEGER", "SMALL_NUMBER", "MIN_LOG_NN_OUTPUT", "MAX_LOG_NN_OUTPUT"
]
