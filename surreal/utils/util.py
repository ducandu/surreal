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
import tensorflow as tf

from surreal.utils.errors import SurrealError

# Some small floating point number. Can be used as a small epsilon for numerical stability purposes.
SMALL_NUMBER = 1e-6
# Some large int number. May be increased here, if needed.
LARGE_INTEGER = 100000000
# Min and Max outputs (clipped) from an NN-output layer interpreted as the log(stddev) of some loc + scale distribution.
MIN_LOG_STDDEV = -20
MAX_LOG_STDDEV = 2

# Logging config for testing.
#logging_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%y-%m-%d %H:%memory:%S')
#root_logger = logging.getLogger('')
#root_logger.setLevel(level=logging.INFO)
#tf_logger = logging.getLogger('tensorflow')
#tf_logger.setLevel(level=logging.INFO)

#print_logging_handler = logging.StreamHandler(stream=sys.stdout)
#print_logging_handler.setFormatter(logging_formatter)
#print_logging_handler.setLevel(level=logging.INFO)
#root_logger.addHandler(print_logging_handler)


# TODO: Consider making "to" non-optional: https://github.com/rlgraph/rlgraph/issues/34
def convert_dtype(dtype, to="tf"):
    """
    Translates any type (tf, numpy, python, etc..) into the respective tensorflow/numpy data type.

    Args:
        dtype (any): String describing a numerical type (e.g. 'float'), numpy data type, tf dtype,
            or python numerical type.
        to (str): Either one of 'tf' (tensorflow), 'np' (numpy), 'str' (string).
            Default="tf".

    Returns:
        TensorFlow, Numpy, string, representing a data type (depending on `to` parameter).
    """
    dtype = str(dtype)
    if "bool" in dtype:
        return np.bool_ if to == "np" else tf.bool
    elif "float64" in dtype:
        return np.float64 if to == "np" else tf.float64
    elif "float" in dtype:
        return np.float32 if to == "np" else tf.float32
    elif "int64" in dtype:
        return np.int64 if to == "np" else tf.int64
    elif "uint8" in dtype:
        return np.uint8 if to == "np" else tf.uint8
    elif "int16" in dtype:
        return np.int16 if to == "np" else tf.int16
    elif "int" in dtype:
        return np.int32 if to == "np" else tf.int32
    elif "str" in dtype:
        return np.unicode_ if to == "np" else tf.string

    raise SurrealError("Error: Type conversion to '{}' for type '{}' not supported.".format(to, str(dtype)))


def get_rank(tensor):
    """
    Returns the rank (as a single int) of an input tensor.

    Args:
        tensor (Union[tf.Tensor,torch.Tensor,np.ndarray]): The input tensor.

    Returns:
        int: The rank of the given tensor.
    """
    if isinstance(tensor, np.ndarray):
        return tensor.ndim
    return tensor.get_shape().ndims


def get_shape(op, flat=False, no_batch=False):
    """
    Returns the (static) shape of a given DataOp as a tuple.

    Args:
        op (DataOp): The input op.
        flat (bool): Whether to return the flattened shape (the product of all ints in the shape tuple).
            Default: False.
        no_batch (bool): Whether to exclude a possible 0th batch rank from the returned shape.
            Default: False.

    Returns:
        tuple: The shape of the given op.
        int: The flattened dim of the given op (if flat=True).
    """
    # Dict.
    if isinstance(op, dict):
        shape = tuple([get_shape(op[key]) for key in sorted(op.keys())])
    # Tuple-op.
    elif isinstance(op, tuple):
        shape = tuple([get_shape(i) for i in op])
    # Numpy ndarrays.
    elif isinstance(op, np.ndarray):
        shape = op.shape
    # Primitive op (e.g. tensorflow)
    else:
        op_shape = op.get_shape()
        # Unknown shape (e.g. a cond op).
        if op_shape.ndims is None:
            return None
        shape = tuple(op_shape.as_list())

    # Remove batch rank?
    if no_batch is True and shape[0] is None:
        shape = shape[1:]

    # Return as-is or as flat shape?
    if flat is False:
        return shape
    else:
        return int(np.prod(shape))


def get_batch_size(tensor):
    """
    Returns the (dynamic) batch size (dim of 0th rank) of an input tensor.

    Args:
        tensor (SingleDataOp): The input tensor.

    Returns:
        SingleDataOp: The op holding the batch size information of the given tensor.
    """
    # Simple numpy array?
    if isinstance(tensor, np.ndarray):
        if tensor.shape == ():
            return 0
        return tensor.shape[0]
    return tf.shape(tensor)[0]


def force_list(elements=None, to_tuple=False):
    """
    Makes sure `elements` is returned as a list, whether `elements` is a single item, already a list, or a tuple.

    Args:
        elements (Optional[any]): The inputs as single item, list, or tuple to be converted into a list/tuple.
            If None, returns empty list/tuple.
        to_tuple (bool): Whether to use tuple (instead of list).

    Returns:
        Union[list,tuple]: All given elements in a list/tuple depending on `to_tuple`'s value. If elements is None,
            returns an empty list/tuple.
    """
    ctor = list
    if to_tuple is True:
        ctor = tuple
    return ctor() if elements is None else ctor(elements) \
        if type(elements) in [list, tuple] else ctor([elements])


force_tuple = partial(force_list, to_tuple=True)


def strip_list(elements):
    """
    Loops through elements if it's a tuple, otherwise processes elements as follows:
    If a list (or np.ndarray) of length 1, extracts that single item, otherwise leaves
    the list/np.ndarray untouched.

    Args:
        elements (any): The input single item, list, or np.ndarray to be converted into
            single item(s) (if length is 1).

    Returns:
        any: Single element(s) (the only one in input) or the original input list.
    """
    # `elements` is a tuple (e.g. from a function return). Process each element separately.
    if isinstance(elements, tuple):
        ret = []
        for el in elements:
            ret.append(el[0] if isinstance(el, (np.ndarray, list)) and len(el) == 1 else el)
        return tuple(ret)
    # `elements` is not a tuple: Process only `elements`.
    else:
        return elements[0] if isinstance(elements, (np.ndarray, list)) and len(elements) == 1 else \
            elements


def default_dict(original, defaults):
    """
    Updates the `original` dict with values from `defaults`, but only for those keys that
    do not exist yet in `original`.
    Changes `original` in place, but leaves `defaults` as is.

    Args:
        original (Optional[dict]): The dict to (soft)-update. If None, return `defaults`.
        defaults (dict): The dict to update from.
    """
    if original is None:
        return defaults

    for key in defaults:
        if key not in original:
            original[key] = defaults[key]
    return original


def clip(x, min_val, max_val):
    """
    Clips x between min_ and max_val.

    Args:
        x (float): The input to be clipped.
        min_val (float): The min value for x.
        max_val (float): The max value for x.

    Returns:
        float: The clipped value.
    """
    return max(min_val, min(x, max_val))


def complement_struct(struct, reference_struct, value=None):
    """
    Takes struct and fills it with Nones where ever `reference_struct` has some value that does not
    exist in struct.

    Args:
        struct (any): The structure to complement (may be a primitive as well).
        reference_struct (any): The reference structure considered "complete".
    """
    if isinstance(reference_struct, dict):
        for key in reference_struct:
            # Missing key -> complement with None value.
            if key not in struct:
                struct[key] = tf.nest.map_structure(lambda s: value, reference_struct[key])
            # Exists -> Recursive call this function on value.
            else:
                struct[key] = complement_struct(struct[key], reference_struct[key], value)
        return struct
    elif isinstance(reference_struct, (tuple, list)):
        for i, value in enumerate(reference_struct):
            # Missing value -> complement with None value.
            if i >= len(struct):
                struct.append(tf.nest.map_structure(lambda s: value, reference_struct[i]))
            # Exists -> Recursive call this function on value.
            else:
                struct[i] = complement_struct(struct[i], reference_struct[i], value)
        return struct
    else:
        return value if isinstance(struct, (dict, tuple, list)) and len(struct) == 0 else struct
