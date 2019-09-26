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

import json
import numpy as np
import os
import re
import tensorflow as tf


def config_from_path(path, root=None):
    """
    Generates an agent config from path relative to a specified directory (per
    default the `tests` directory).

    Args:
        path (str): Path to config, e.g. json file.
        root (str): Root directory. Per default it's the `tests` directory relativ
            to this file.

    Returns:
        Union[dict,list]: Agent config dict or list.
    """
    if not root:
        root = os.path.join(os.path.dirname(__file__))

    path = os.path.join(root, path)
    with open(path, 'rt') as fp:
        return json.load(fp)


def non_terminal_records(record_space, num_samples):
    """
    Samples a number of records and enforces all terminals to be 0,
    which is needed for testing memories.

    Args:
        record_space (Space): Space to sample from.
        num_samples (int): Number of samples to draw.

    Returns:
        Dict: Sampled records with all terminal values set to 0.
    """
    record_sample = record_space.sample(size=num_samples)
    record_sample['terminals'] = np.full(shape=(num_samples,), fill_value=np.bool_(False))

    return record_sample


def terminal_records(record_space, num_samples):
    """
    Samples a number of records and enforces all terminals to be True,
    which is needed for testing memories.

    Args:
        record_space (Space): Space to sample from.
        num_samples (int): Number of samples to draw.

    Returns:
        Dict: Sampled records with all terminal values set to True.
    """
    record_sample = record_space.sample(size=num_samples)
    record_sample['terminals'] = np.full(shape=(num_samples,), fill_value=np.bool_(True))

    return record_sample


def check(x, y, decimals=5, atol=None, rtol=None, false=False):
    """
    Checks two structures (dict, DataOpDict, tuple, DataOpTuple, list, np.array, float, int, etc..) for (almost!)
    numeric identity.
    All numbers in the two structures have to match up to `decimal` digits after the floating point.
    Uses assertions (not boolean return).

    Args:
        x (any): The first value to be compared (to `y`).
        y (any): The second value to be compared (to `x`).
        decimals (int): The number of digits after the floating point up to which all numeric values have to match.
        atol (float): Absolute tolerance of the difference between x and y (overrides `decimals` if given).
        rtol (float): Relative tolerance of the difference between x and y (overrides `decimals` if given).
        false (bool): Whether to check that x and y are NOT the same.
    """
    # A dict type.
    if isinstance(x, dict):
        assert isinstance(y, dict), "ERROR: If x is dict, y needs to be a dict as well!"
        y_keys = set(x.keys())
        for key, value in x.items():
            assert key in y, "ERROR: y does not have x's key='{}'! y={}".format(key, y)
            check(value, y[key], decimals=decimals, atol=atol, rtol=rtol, false=false)
            y_keys.remove(key)
        assert not y_keys, "ERROR: y contains keys ({}) that are not in x! y={}".format(list(y_keys), y)
    # A tuple type.
    elif isinstance(x, (tuple, list)):
        assert isinstance(y, (tuple, list)), "ERROR: If x is tuple, y needs to be a tuple as well!"
        assert len(y) == len(x), "ERROR: y does not have the same length as " \
                                 "x ({} vs {})!".format(len(y), len(x))
        for i, value in enumerate(x):
            check(value, y[i], decimals=decimals, atol=atol, rtol=rtol, false=false)
    # Boolean comparison.
    elif isinstance(x, (np.bool_, bool)):
        if false is True:
            assert bool(x) is not bool(y), "ERROR: x ({}) is y ({})!".format(x, y)
        else:
            assert bool(x) is bool(y), "ERROR: x ({}) is not y ({})!".format(x, y)
    # Nones.
    elif x is None or y is None:
        if false is True:
            assert x != y, "ERROR: x ({}) is the same as y ({})!".format(x, y)
        else:
            assert x == y, "ERROR: x ({}) is not the same as y ({})!".format(x, y)
    # String comparison.
    elif hasattr(x, "dtype") and x.dtype == np.object:
        try:
            np.testing.assert_array_equal(x, y)
            if false is True:
                assert False, "ERROR: x ({}) is the same as y ({})!".format(x, y)
        except AssertionError as e:
            if false is False:
                raise e
    # Everything else (assume numeric).
    else:
        # Numpyize tensors if necessary.
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        if isinstance(y, tf.Tensor):
            y = y.numpy()

        # Using decimals.
        if atol is None and rtol is None:
            try:
                np.testing.assert_almost_equal(x, y, decimal=decimals)
                if false is True:
                    assert False, "ERROR: x ({}) is the same as y ({})!".format(x, y)
            except AssertionError as e:
                if false is False:
                    raise e

        # Using atol/rtol.
        else:
            # Provide defaults for either one of atol/rtol.
            if atol is None:
                atol = 0
            if rtol is None:
                rtol = 1e-7
            try:
                np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)
                if false is True:
                    assert False, "ERROR: x ({}) is the same as y ({})!".format(x, y)
            except AssertionError as e:
                if false is False:
                    raise e

class regex_pattern(object):
    def __init__(self, pattern):
        self.pattern = pattern

    def __eq__(self, other):
        return re.match(self.pattern, other) is not None

    def __ne__(self, other):
        return not re.match(self.pattern, other)

    def __str__(self):
        return "~ {}".format(self.pattern)

    def __repr__(self):
        return "~ r\"{}\"".format(self.pattern)
