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

import unittest

from surreal.spaces import *
from surreal.tests.test_util import check
from surreal.utils.nest import flatten_alongside


class TestNestingOps(unittest.TestCase):
    """
    Tests nesting/flattening alongside some given structure.
    """
    def test_flatten_alongside(self):
        space = Dict({
            "a": Float(shape=(4,)),
            "b": Dict({"ba": Tuple([Float(shape=(3,)), Float(0.1, 1.0, shape=(3,))]),
                       "bb": Tuple([Float(shape=(2,)), Float(shape=(2,))]),
                       "bc": Tuple([Float(shape=(4,)), Float(0.1, 1.0, shape=(4,))]),
                       })
        }, main_axes="B")
        # Flatten alongside this structure.
        alongside = dict(a=True, b=dict(ba=False, bc=None, bb="foo"))

        input_ = space.sample(2)
        # Expect to only flatten until ["b"]["ba/b/c"], not into the Tuples as `alongside` does not have these.
        out = flatten_alongside(input_, alongside)
        expected = [input_["a"], input_["b"]["ba"], input_["b"]["bb"], input_["b"]["bc"]]

        check(out, expected)
