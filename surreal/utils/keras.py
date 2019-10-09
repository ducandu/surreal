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

import copy
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten  # TODO: support more.

from surreal.utils.errors import SurrealError


def keras_from_spec(spec):
    # Layers are given as list -> Build a simple Keras sequential model using Keras configs.
    if isinstance(spec, (list, tuple)):
        sequential = tf.keras.models.Sequential()
        for layer in spec:
            layer_copy = copy.deepcopy(layer)  # protect oroginal config
            name = layer_copy.pop("name").lower()
            #assert name in ["dense", "conv2d", "flatten", "lstm"]
            class_ = None
            for match in [Dense, Conv2D, Flatten, LSTM]:
                if match.__name__.lower() == name:
                    class_ = match
                    break
            if class_:
                sequential.add(class_.from_config(layer_copy))
            else:
                if name == "onehot":
                    sequential.add(tf.keras.layers.Lambda(lambda in_: tf.one_hot(in_, **layer_copy)))
                else:
                    raise SurrealError("Unknown layer/tf-op '{}'!".format(name))
        return sequential

    return spec
