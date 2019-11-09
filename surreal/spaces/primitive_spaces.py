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

from abc import ABCMeta
import copy
import numpy as np
import tensorflow as tf

from surreal.spaces.space import Space
from surreal.utils.util import convert_dtype, LARGE_INTEGER


class PrimitiveSpace(Space, metaclass=ABCMeta):
    """
    A box in R^n with a shape tuple of len n. Each dimension may be bounded.
    """
    def __init__(self, low, high, shape=None, dtype=np.float32, main_axes=None, value=None):
        """
        Example constructions:
            PrimitiveSpace(0.0, 1.0) # low and high are given as scalars and shape is assumed to be ()
                -> single scalar between low and high.
            PrimitiveSpace(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided -> nD array
                where all(!) elements are between low and high.
            PrimitiveSpace(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape
                (no shape given!) -> nD array where each dimension has different bounds.

        Args:
            low (any): The lower bound (see Valid Inputs for more information).
            high (any): The upper bound (see Valid Inputs for more information).
            shape (tuple): The shape of this space.

            dtype (np.type): The data type (as numpy type) for this Space.
                Allowed are: np.int8,16,32,64, np.float16,32,64 and np.bool_.
        """
        # Determine the shape.
        if shape is None:
            if isinstance(low, (int, float, bool)):
                shape = ()
            else:
                shape = np.shape(low)
        else:
            assert isinstance(shape, (tuple, list)), "ERROR: `shape` must be None or a tuple/list."
            shape = tuple(shape)

        self.dtype = dtype

        # Determine the bounds.
        # False if bounds are individualized (each dimension has its own lower and upper bounds and we can get
        # the single values from self.low and self.high), or a tuple of the globally valid low/high values that apply
        # to all values in all dimensions.
        # 0D Space.
        self.low = np.array(low)
        self.high = np.array(high)

        super().__init__(shape=shape, value=value, main_axes=main_axes)

        if self.shape == ():
            assert self.low.shape == (), "ERROR: If shape == (), `low` must be scalar!"
            assert self.high.shape == (), "ERROR: If shape == (), `high` must be scalar!"
            self.global_bounds = (self.low, self.high)
        # nD Space (n > 0). Bounds can be single number or individual bounds.
        else:
            self.global_bounds = True
            # Check, whether they are all the same anyway, in which case, we do have global bounds.
            if self.low.shape != ():
                if np.all(self.low == self.low[next(iter(np.ndindex(self.low.shape)))]):
                    self.low = self.low[next(iter(np.ndindex(self.low.shape)))]
                else:
                    self.global_bounds = False
            if self.high.shape != ():
                if np.all(self.high == self.high[next(iter(np.ndindex(self.high.shape)))]):
                    self.high = self.high[next(iter(np.ndindex(self.high.shape)))]
                else:
                    self.global_bounds = False

            # Only one low/high value. Use these as generic bounds for all values.
            if self.global_bounds is True:
                self.global_bounds = (self.low, self.high)
                # Low/high values are given individually per item (and are not all the same).
            else:
                self.global_bounds = False

    def force_batch(self, samples, horizontal=None):
        assert "T" not in self.main_axes, "ERROR: Cannot force a batch rank if Space `has_time_rank` is True!"
        # 0D (means: certainly no batch rank) or no extra rank given (compared to this Space), add a batch rank.
        if np.asarray(samples).ndim == 0 or \
                np.asarray(samples).ndim == len(self.get_shape(include_main_axes=False)):
            return np.array([samples]), True  # batch size=1
        # Samples is a list (whose len is interpreted as the batch size) -> return as np.array.
        elif isinstance(samples, list):
            return np.asarray(samples), False
        # Samples is already assumed to be batched. Return as is.
        return samples, False

    def get_shape(self, include_main_axes=False, main_axis_value=None, **kwargs):
        shape = []
        if include_main_axes is True:
            shape = [main_axis_value if v is True else v for v in self.main_axes.values()]
        return tuple(shape) + self.shape

    @property
    def flat_dim(self):
        return int(np.prod(self.shape))  # also works for shape=()

    @property
    def structure(self):
        return self

    @property
    def bounds(self):
        return self.low, self.high

    def tensor_backed_bounds(self):
        return self.low, self.high

    def create_variable(self):  #, **kwargs):
        shape = [i if i is not None else 1 for i in self.get_shape(include_main_axes=True)]

        #if is_python is True:
        return np.zeros(shape or (), dtype=convert_dtype(dtype=self.dtype, to="np"))

        #raise NotImplementedError

        """else:
            # TODO: re-evaluate the cutting of a leading '/_?' (tf doesn't like it)
            name = re.sub(r'^/_?', "", name)
            if is_input_feed:
                variable = tf.placeholder(dtype=convert_dtype(self.dtype), shape=shape, name=name)
                if "B" in self.main_axes:
                    variable._batch_rank = self.main_axes["B"]
                if "T" in self.main_axes:
                    variable._time_rank = self.main_axes["T"]
            else:
                init_spec = kwargs.pop("initializer", None)
                # Bools should be initializable via 0 or not 0.
                if self.dtype == np.bool_ and isinstance(init_spec, (int, float)):
                    init_spec = (init_spec != 0)

                if self.dtype == np.str_ and init_spec == 0:
                    initializer = None
                else:
                    initializer = Initializer.make(shape=shape, specification=init_spec).initializer

                variable = tf.get_variable(
                    name, shape=shape, dtype=convert_dtype(self.dtype), initializer=initializer,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES if local is False else tf.GraphKeys.LOCAL_VARIABLES],
                    **kwargs
                )

            # Add batch/time rank flags to the op.
            #if "B" in self.main_axes:
            #    variable._batch_rank = 0 if self.time_major is False else 1
            #if "T" in self.main_axes:
            #    variable._time_rank = 1 if self.time_major is False else 0
            return variable
        """

    def create_keras_input(self):
        # TODO: This may fail for time-rank or more than just batch rank?
        return tf.keras.Input(shape=self.shape, dtype=self.dtype)

    def zeros(self, size=None):
        return self.sample(size=size, fill_value=0)

    def contains(self, sample):
        sample_shape = sample.shape if isinstance(sample, np.ndarray) else \
            (len(sample),) if isinstance(sample, (tuple, list)) else ()
        own_shape = [sample_shape[i] if d is None else d for i, d in enumerate(self.get_shape(include_main_axes=True))]
        if sample_shape != tuple(own_shape):
            return False
        return (sample >= self.low).all() and (sample <= self.high).all()

    def __repr__(self):
        return "{}(shape={}{}; dtype={})".format(
            type(self).__name__.title(),
            self.shape, "".join(["+" + n for n in self.main_axes]) if len(self.main_axes) > 0 else "",
            self.dtype.__name__
        )

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
               self.shape == other.shape and self.dtype == other.dtype
               # np.allclose(self.low, other.low) and np.allclose(self.high, other.high) and \

    def __hash__(self):
        if self.shape == () or self.global_bounds is not False:
            return hash((self.low.item(), self.high.item()))
        return hash((tuple(self.low), tuple(self.high)))


class Float(PrimitiveSpace):
    def __init__(self, low=None, high=None, shape=None, dtype="float32", **kwargs):
        if low is None:
            assert high is None, "ERROR: If `low` is None, `high` must be None as well!"
            low = float("-inf")
            high = float("inf")
            self.unbounded = True
        else:
            self.unbounded = False
            # support calls like (Float(1.0) -> low=0.0, high=1.0)
            if high is None:
                high = low
                low = 0.0

        dtype = convert_dtype(dtype, "np")
        assert dtype in [np.float16, np.float32, np.float64], "ERROR: Float does not allow dtype '{}'!".format(dtype)

        super().__init__(low=low, high=high, shape=shape, dtype=dtype, **kwargs)

        if self.low.shape == () and self.low == float("-inf") and self.high.shape == () and self.high == float("inf"):
            self.unbounded = True

    def sample(self, size=None, fill_value=None, **kwargs):
        shape = self._get_np_shape(size)
        if fill_value is not None:
            sample_ = np.full(shape=shape, fill_value=fill_value)
        else:
            if self.unbounded:
                sample_ = np.random.uniform(size=shape)
            else:
                sample_ = np.random.uniform(low=self.low, high=self.high, size=shape)

        # Make sure return values have the right dtype (float64 is np.random's default).
        return np.asarray(sample_, dtype=self.dtype)


class Int(PrimitiveSpace):
    """
    A box in Z^n (only integers; each coordinate is bounded)
    e.g. an image (w x h x RGB) where each color channel pixel can be between 0 and 255.
    """
    def __init__(self, low=None, high=None, shape=None, dtype="int32", **kwargs):
        """
        Valid inputs:
            Int(6)  # only high is given -> low assumed to be 0 (0D scalar).
            Int(0, 2) # low and high are given as scalars and shape is assumed to be 0D scalar.
            Int(-1, 1, (3,4)) # low and high are scalars, and shape is provided.
            Int(np.array([-1,-2]), np.array([2,4])) # low and high are arrays of the same shape (no shape given!)

        NOTE: The `high` value for IntBoxes is excluded. Valid values thus are from the interval: [low,high[
        """
        if low is None:
            if high is not None:
                low = 0
            else:
                low = -LARGE_INTEGER
                high = LARGE_INTEGER
        # support calls like (Int(5) -> low=0, high=5)
        elif high is None:
            high = low
            low = 0

        dtype = convert_dtype(dtype, "np")
        assert dtype in [np.int16, np.int32, np.int64, np.uint8], \
            "ERROR: Int does not allow dtype '{}'!".format(dtype)

        super().__init__(low=low, high=high, shape=shape, dtype=dtype, **kwargs)

        self.num_categories = None if self.global_bounds is False else self.global_bounds[1]

    def get_shape(self, include_main_axes=False, main_axis_value=None, **kwargs):
        """
        Keyword Args:
            with_category_rank (bool): Whether to include a category rank for this Int (if all dims have equal
                lower/upper bounds).
        """
        with_category_rank = kwargs.pop("with_category_rank", False)
        shape = super(Int, self).get_shape(include_main_axes, main_axis_value, **kwargs)
        if with_category_rank is not False:
            return shape + ((self.num_categories,) if self.num_categories is not None else ())
        return shape

    def as_one_hot_float_space(self):
        return Float(
            low=0.0, high=1.0, shape=self.get_shape(with_category_rank=True), main_axes=copy.deepcopy(self.main_axes)
        )

    @property
    def flat_dim_with_categories(self):
        """
        If we were to flatten this Space and also consider each single possible int value (assuming global bounds)
        as one category, what would the dimension have to be to represent this Space?
        """
        if self.global_bounds is False:
            return int(np.sum(self.high))  # TODO: this assumes that low is always 0.
        return int(np.prod(self.shape) * self.global_bounds[1])

    def sample(self, size=None, fill_value=None, **kwargs):
        shape = self._get_np_shape(size)
        if fill_value is None:
            sample_ = np.random.uniform(low=self.low, high=self.high, size=shape)
        else:
            sample_ = fill_value if shape == () or shape is None else np.full(shape=shape, fill_value=fill_value)

        return np.asarray(sample_, dtype=self.dtype)

    def contains(self, sample):
        # If int: Check for int type in given sample.
        if not np.equal(np.mod(sample, 1), 0).all():
            return False
        return super().contains(sample)

    def __repr__(self):
        return "{}({}shape={}{}; dtype={})".format(
            type(self).__name__.title(), (str(self.num_categories)+" ") if self.num_categories is not None else "",
            self.shape, "".join(["+" + n for n in self.main_axes]) if len(self.main_axes) > 0 else "",
            self.dtype.__name__
        )


class Bool(PrimitiveSpace):
    def __init__(self, shape=None, **kwargs):
        super().__init__(low=False, high=True, shape=shape, dtype=np.bool_, **kwargs)

    def sample(self, size=None, fill_value=None, **kwargs):
        shape = self._get_np_shape(size)
        if fill_value is None:
            sample_ = np.random.choice(a=[False, True], size=shape)
        else:
            sample_ = np.full(shape=size, fill_value=fill_value)
        return sample_

    def contains(self, sample):
        # Wrong type.
        if not isinstance(sample, (bool, np.bool_, np.ndarray)):
            return False
        # Single bool.
        elif self.get_shape(include_main_axes=True) == () and isinstance(sample, (bool, np.bool_)):
            return True
        # ndarray.
        else:
            if convert_dtype(sample.dtype, "np") != np.bool_:
                return False
            return super().contains(sample)


class Text(PrimitiveSpace):
    """
    A text box in TXT^n where the shape means the number of text chunks in each dimension.
    A text chunk can consist of any number of words.
    """
    def __init__(self, **kwargs):
        # Set both low/high to 0 (make no sense for text).
        super().__init__(low=0, high=0, **kwargs)

        # Set dtype to numpy's unicode type.
        self.dtype = np.unicode_

        #assert isinstance(shape, tuple), "ERROR: `shape` must be a tuple."
        #self._shape = shape

    def sample(self, size=None, fill_value=None, **kwargs):
        shape = self._get_np_shape(size)

        # TODO: Make it such that it doesn't only produce number strings (using `petname` module?).
        sample_ = np.full(shape=shape, fill_value=fill_value, dtype=self.dtype)

        return sample_.astype(self.dtype)

    def contains(self, sample):
        # Wrong type.
        if not isinstance(sample, (str, np.str, np.ndarray)):
            return False
        # Single str.
        elif self.get_shape(include_main_axes=True) == () and isinstance(sample, (str, np.str)):
            return True

        sample_shape = sample.shape if isinstance(sample, np.ndarray) else \
            (len(sample),) if isinstance(sample, (tuple, list)) else ()
        own_shape = [sample_shape[i] if d is None else d for i, d in enumerate(self.get_shape(include_main_axes=True))]
        if sample_shape != tuple(own_shape):
            return False
        return True
