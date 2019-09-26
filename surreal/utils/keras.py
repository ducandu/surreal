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
from tensorflow.keras.layers import Dense, Conv2D, LSTM  # TODO: support more.


def keras_from_spec(spec):
    # Layers are given as list -> Build a simple Keras sequential model using Keras configs.
    if isinstance(spec, (list, tuple)):
        sequential = tf.keras.models.Sequential()
        for layer in spec:
            name = layer.pop("name").lower()
            assert name in ["dense", "conv2d", "lstm"]
            class_ = Dense if name == "dense" else Conv2D if name == "conv2d" else LSTM
            sequential.add(class_.from_config(layer))
        return sequential

    return spec
