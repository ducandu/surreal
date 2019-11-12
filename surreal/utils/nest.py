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

import tensorflow as tf


def flatten_alongside(input_, alongside=None, op_tuple_list=None):
    """
    Flattens some data alongside some other structure (not any further down the nested input_).

    Args:
        input_ (any): The structure to flatten.
        alongside (Optional[dict]): If given, flatten only according to this dictionary, not any further down.

        op_tuple_list (Optional[list]): Private list of flattened gathered ops. Only used for recursive calls to
            this function.

    Returns:
        list: The flattened (list) representation of `input_`.
    """
    ret = False

    # Are we in the non-recursive (first) call?
    if op_tuple_list is None:
        # Flatten a SingleDataOp -> return FlattenedDataOp with only-key="".
        # OR: flatten_alongside something, and that something is not flattened (its only key is "").
        if not isinstance(input_, (dict, list, tuple)) or \
                (alongside is not None and len(alongside) == 1 and "" in alongside):
            return [input_]

        op_tuple_list = []
        ret = True

    if isinstance(input_, dict):
        if not isinstance(alongside, dict):
            op_tuple_list.append(input_)
        else:
            #assert alongside is None or isinstance(alongside, dict)
            for key in sorted(input_.keys()):
                # If key is in `alongside` structure, keep iterating here.
                if alongside is None or (isinstance(alongside, dict) and key in alongside):
                    flatten_alongside(input_[key], op_tuple_list=op_tuple_list, alongside=alongside[key])
                # Otherwise, stop flattening process.
                else:
                    op_tuple_list.append(input_)
    elif isinstance(input_, (list, tuple)):
        if not isinstance(alongside, (list, tuple)):
            op_tuple_list.append(input_)
        else:
            for i, c in enumerate(input_):
                # If i is in `alongside` structure, keep iterating here.
                if alongside is None or (isinstance(alongside, (list, tuple)) and len(alongside) > i):
                    flatten_alongside(c, op_tuple_list=op_tuple_list, alongside=alongside[i])
                else:
                    op_tuple_list.append(input_)
    else:
        op_tuple_list.append(input_)

    # Non recursive (first) call -> Return the final FlattenedDataOp.
    if ret:
        return op_tuple_list


def keys_to_flattened_struct_indices(struct):
    global i
    i = -1

    def numerate(s):
        global i
        i += 1
        return i

    return tf.nest.map_structure(numerate, struct)
