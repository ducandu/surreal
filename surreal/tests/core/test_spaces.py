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

import numpy as np
import tensorflow as tf
import unittest

from surreal.spaces import *


class TestSpaces(unittest.TestCase):
    """
    Tests creation, sampling and shapes of Spaces.
    """
    def test_primitive_spaces(self):
        """
        Tests all PrimitiveSpaces via sample/contains loop. With and without batch/time-axes,
        different batch sizes, and different los/high combinations (including no bounds).
        """
        for class_ in [Float, Int, Bool, Text]:
            for add_batch_rank in [False, True]:
                for add_time_rank in [False, True]:

                    main_axes = ["B"] if add_batch_rank else []
                    if add_time_rank:
                        main_axes.append("T")

                    if class_ != Bool and class_ != Text:
                        for low, high in [(None, None), (-1.0, 10.0), ((1.0, 2.0), (3.0, 4.0)),
                                          (((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), ((7.0, 8.0, 9.0), (10.0, 11.0, 12.0)))]:
                            space = class_(low=low, high=high, main_axes=main_axes)

                            if add_batch_rank is False:
                                if add_time_rank is False:
                                    s = space.sample()
                                else:
                                    s = space.sample(5)
                                self.assertTrue(space.contains(s))
                            else:
                                for batch_size in range(1, 4):
                                    if add_time_rank is False:
                                        s = space.sample(size=batch_size)
                                        self.assertTrue(space.contains(s))
                                    else:
                                        for seq_len in range(1, 4):
                                            s = space.sample(size=(batch_size, seq_len))
                                            self.assertTrue(space.contains(s))

                            # Test shapes.
                            if isinstance(low, tuple):
                                self.assertTrue(space.shape == np.array(low).shape)
                                self.assertTrue(
                                    space.with_batch().get_shape(include_main_axes=True) == (None,) *
                                    (add_time_rank + 1) + np.array(low).shape
                                )
                                self.assertTrue(
                                    space.with_time().get_shape(include_main_axes=True) == (None,) *
                                    (add_batch_rank + 1) + np.array(low).shape
                                )
                                self.assertTrue(space.with_time().with_batch().get_shape(include_main_axes=True) ==
                                                (None, None) + np.array(low).shape)
                                self.assertTrue(space.with_batch().with_time(dimension=50).get_shape(include_main_axes=True) ==
                                                (50, None) + np.array(low).shape)

                            # Test `zero` method.
                            all_0s = space.zeros()
                            self.assertTrue(
                                np.sum(all_0s) == 0.0 if len(main_axes) > 0 or low is not None else all_0s == 0
                            )
                    else:
                        space = class_(main_axes=main_axes)

                        if add_batch_rank is False:
                            if add_time_rank is False:
                                s = space.sample()
                            else:
                                s = space.sample(5)
                            self.assertTrue(space.contains(s))
                        else:
                            for batch_size in range(1, 4):
                                if add_time_rank is False:
                                    s = space.sample(size=batch_size)
                                    self.assertTrue(space.contains(s))
                                else:
                                    for seq_len in range(1, 4):
                                        s = space.sample(size=(batch_size, seq_len))
                                        self.assertTrue(space.contains(s))

    def test_complex_space_sampling_and_check_via_contains(self):
        """
        Tests a complex Space on sampling and `contains` functionality.
        """
        space = Dict(
            a=dict(aa=float, ab=bool),
            b=dict(ba=float),
            c=float,
            d=Int(low=0, high=1),
            e=Int(5),
            f=Float(shape=(2, 2)),
            g=Tuple(float, Float(shape=())),
            main_axes="B"
        )

        samples = space.sample(size=100, horizontal=True)
        for i in range(len(samples)):
            self.assertTrue(space.contains(samples[i]))

    def test_container_space_flattening_with_mapping(self):
        space = Tuple(
            Dict(
                a=bool,
                b=Int(4),
                c=Dict(
                    d=Float(shape=())
                )
            ),
            Bool(),
            Int(2),
            Float(shape=(3, 2)),
            Tuple(
                Bool(), Bool()
            )
        )

        def mapping_func(primitive_space):
            # Just map a primitive Space to its flat_dim property.
            return primitive_space.flat_dim

        result = ""
        flat_space_and_mapped = tf.nest.map_structure(mapping_func, tf.nest.flatten(space))
        for value in flat_space_and_mapped:
            result += "{},".format(value)

        expected = "1,1,1,1,1,6,1,1,"
        self.assertTrue(result == expected)

    def test_container_space_mapping(self):
        space = Tuple(
            Dict(
                a=bool,
                b=Int(4),
                c=Dict(
                    d=Float(shape=())
                )
            ),
            Bool(),
            Int(2),
            Float(shape=(3, 2)),
            Tuple(
                Bool(), Bool()
            )
        )

        def mapping_func(primitive_space):
            # Change each primitive space to IntBox(5).
            return Int(5)

        mapped_space = tf.nest.map_structure(mapping_func, space)

        self.assertTrue(isinstance(mapped_space[0]["a"], Int))
        self.assertTrue(mapped_space[0]["a"].num_categories == 5)
        self.assertTrue(mapped_space[3].num_categories == 5)
        self.assertTrue(mapped_space[4][0].num_categories == 5)
        self.assertTrue(mapped_space[4][1].num_categories == 5)

        # Same on Dict.
        space = Dict(
            a=bool,
            b=Int(4),
            c=Dict(
                d=Float(shape=())
            )
        )
        mapped_space = tf.nest.map_structure(mapping_func, space)

        self.assertTrue(isinstance(mapped_space["a"], Int))
        self.assertTrue(mapped_space["a"].num_categories == 5)
        self.assertTrue(isinstance(mapped_space["b"], Int))
        self.assertTrue(mapped_space["c"]["d"].num_categories == 5)
