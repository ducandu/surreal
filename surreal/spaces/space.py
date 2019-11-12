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

from abc import ABCMeta, abstractmethod
import copy
import numpy as np
import re
from collections import OrderedDict

from surreal.makeable import Makeable
from surreal.utils import force_list
from surreal.utils.errors import SurrealError


class Space(Makeable, metaclass=ABCMeta):
    """
    Space class (based on and compatible with openAI Spaces).
    Provides a classification for state-, action-, reward- and other core.
    """
    # Global unique Space ID.
    _ID = -1

    def __init__(self, shape=None, value=None, main_axes=None):
        """
        Args:
            shape (Optional[Tuple[]]):

            value (any): A value to directly assign to this Space. Use "zeros" for all zeros initialization, "random",
                for a random-sample initialization.

            main_axes (Optional[List[Str]]): A list of names of main axes for this Space in the correct order.
                E.g. ["B", "T"] for adding a batch and a time rank.
                Alternatively to pure names, a tuple can be passed in for a name/dimension pair giving the exact
                dimension of the axis, e.g. [("B", 500), "T"] would create a Space with batch size 500 and a time
                axis of unknown dimension.
        """
        super().__init__()

        self.shape = shape

        # Parent Space for usage in nested ContainerSpace structures.
        self.parent = None

        # Convenience flag to quickly check, whether a Space is possibly time-major
        # (only if "T" axis comes before "B" axis)
        #self.time_major = None
        # The main axes of this Space (can be customized, but usually contain "B" (batch), and/or "T" (time)).
        self.main_axes = OrderedDict()
        if main_axes is not None:
            for main_axis in force_list(main_axes):
                if isinstance(main_axis, (tuple, list)):
                    self._add_main_axis(main_axis[0], position=-1, dimension=main_axis[1])
                elif isinstance(main_axis, dict):
                    assert len(main_axis) == 1
                    self._add_main_axis(list(main_axis.keys())[0], position=-1, dimension=list(main_axis.values())[0])
                else:
                    self._add_main_axis(main_axis, position=-1)

        # Each space has an optional value, that can store data of that space.
        self.value = None
        # Always double-check initial values if given.
        if value is not None:
            if value == "zeros":
                self.assign(self.sample(fill_value=0), check=True)
            elif value == "random":
                self.assign(self.sample(), check=True)
            else:
                self.assign(value, check=True)

    @abstractmethod
    def get_shape(self, include_main_axes=False, main_axis_value=None, **kwargs):
        """
        Returns the shape of this Space as a tuple with certain additional axes at the front (main-axes) or the back
        (e.g. categories in Int Spaces).

        Args:
            include_main_axes (bool): Whether to include all main-axes in the returned tuple as None.
            main_axis_value (any): The value to use for the main-axes iff `include_main_axes` is True.

        Returns:
            tuple: The shape of this Space as a tuple.
        """
        raise NotImplementedError

    @property
    def rank(self):
        """
        Returns:
            int: The rank of the Space, not including main-axes
            (e.g. 3 for a space with shape=(10, 7, 5) OR 2 for a space with shape=(1,2) and main-axes "B" and "T").
        """
        return len(self.shape)

    @property
    def reduction_axes(self):
        """
        Returns:
            List[int]: A list of axes to be reduced by any tf.reduce... operation, while sparing the main-axes from
                the reduction operation.
                E.g.: [-1, -2, -3] for a space with shape=(2,4,6) and any number of main_axes.
        """
        return list(reversed(range(-self.rank, 0)))

    @abstractmethod
    def structure(self):
        """
        Returns a corresponding (possibly nested primitive) structure (dict, tuple) with 0 values at the leaves
        (primitive Spaces).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def flat_dim(self):
        """
        Returns:
            int: The length of a flattened vector derived from this Space.
        """
        raise NotImplementedError

    def as_one_hot_float_space(self):
        """
        Returns:
            This Space except that all int elements/sub-components, etc.. have been replaced by their corresponding
            one-hot Float counterparts. E.g. An Int(3, shape=(2,)) will convert to Float(0.0, 1.0, shape=(2,3))).
            A Dict/Tuple convert each of their child Space through this method as well.
        """
        return copy.deepcopy(self)  # default: return copy of self

    @abstractmethod
    def sample(self, size=None, fill_value=None, **kwargs):
        """
        Uniformly randomly samples an element from this space. This is for testing purposes, e.g. to simulate
        a random environment.

        Args:
            size (Optional[int,Tuple[int]): The size of the `main_axes` to use for sampling.
                E.g.:
                - main_axes=["B"] + `size`=None -> return a batch of 1.
                - main_axes=["B"] + `size`=5 -> return a batch of 5.
                - main_axes={"B": 10} + `size`=5 -> return a batch of 5 (meaning: ignores axis' fixed-size).
                - main_axes=["T"] + `size`=5 -> return a time-series of len 5.
                - main_axes={"T": 10} + `size`=5 -> return a time-series of len 5 (meaning: ignores axis' fixed-size).
                - main_axes=["B", "T"] + `size`=5 -> ERROR (must provide both main_axes).
                - main_axes=["B", "T"] + `size`=(5, 2) -> return a batch of 5 with time-series of len 2.
                - main_axes=["T", "B"] + `size`=(5, 2) -> return a time-series of len 5 with batches of 2 (time major).
                - main_axes={"B": 5, "T": 2} + `size`=None -> return a batch of 5 with time-series of len 2.
                - main_axes=["B", "T"] + `size`=None -> return a batch of 1 with time-series of len 1.

            fill_value (Optional[any]): The number or initializer specifier to fill the sample. Can be used to create
                a (non-random) sample with a certain fill value in all elements.
                TODO: support initializer spec-strings like 'normal', 'truncated_normal', etc..

        Returns:
            any: The sampled element(s).
        """
        raise NotImplementedError

    @abstractmethod
    def contains(self, sample):
        """
        Checks whether this space contains the given sample (including all main-axes).

        Args:
            sample: The element to check.

        Returns:
            bool: Whether sample is a valid member of this space.
        """
        raise NotImplementedError

    @abstractmethod
    def zeros(self, size=None):
        """
        Args:
            size (Optional): See `Space.sample()`.

        Returns:
            np.ndarray: `size` zero samples where all values are zero and have the correct type.
        """
        raise NotImplementedError

    def assign(self, value, check=False):
        """
        Overrides our value with the given one.

        Args:
            value (any): The new value to assign to `self.value`.
            check (bool): If True, double check the new value against this Space.
        """
        if check is True:
            assert self.contains(value)
        self.value = value

    def with_batch(self, position=0, dimension=None):
        """
        Returns a deepcopy of this Space, but with "B" added to the given position and set to the provided dimension.

        Args:
            position (int): The position at which to add the batch axis.
            dimension (Optional[int]): The dimension of the batch axis, None for no particular dimension.

        Returns:
            Space: The deepcopy of this Space, but with "B" axis.
        """
        cp = self.copy()
        cp._add_main_axis("B", position=position, dimension=dimension)
        return cp

    def with_time(self, position=0, dimension=None):
        """
        Returns a deepcopy of this Space, but with "T" added to the given position and set to the provided dimension.

        Args:
            position (int): The position at which to add the time axis.
            dimension (Optional[int]): The dimension of the time axis, None for no particular dimension.

        Returns:
            Space: The deepcopy of this Space, but with "T" axis.
        """
        cp = self.copy()
        cp._add_main_axis("T", position=position, dimension=dimension)
        return cp

    def with_axes(self, main_axes):
        """
        Returns a deepcopy of this Space, but with "T" added to the given position and set to the provided dimension.

        Args:
            main_axes (Optional[List[Str]]): A list of names of main axes for this Space in the correct order.
                E.g. ["B", "T"] for adding a batch and a time rank.
                Alternatively to pure names, a tuple can be passed in for a name/dimension pair giving the exact
                dimension of the axis, e.g. [("B", 500), "T"] would create a Space with batch size 500 and a time
                axis of unknown dimension.

        Returns:
            Space: The deepcopy of this Space, but with "T" axis.
        """
        cp = self.copy()
        if main_axes is not None:
            # If `main_axes` is already taken from another Space.
            if isinstance(main_axes, OrderedDict):
                main_axes = list(main_axes.items())
            for main_axis in force_list(main_axes):
                if isinstance(main_axis, (tuple, list)):
                    cp._add_main_axis(main_axis[0], position=-1, dimension=main_axis[1])
                else:
                    cp._add_main_axis(main_axis, position=-1)
        return cp

    def strip_axes(self):
        """
        Returns a deepcopy of this Space, but with all main axes removed.

        Returns:
            Space: The deepcopy of this Space, but without any main axis.
        """
        cp = self.copy()
        if hasattr(self, "main_axes"):
            for axis in self.main_axes:
                cp._remove_main_axis(axis)
        return cp

    @abstractmethod
    def create_variable(self):  #, name, is_input_feed=False, is_python=False, local=False, **kwargs):
        """
        Returns a numpy variable that matches the space's shape.

        #Args:
        #    name (str): The name for the variable.

        #    is_input_feed (bool): Whether the returned object should be an input placeholder,
        #        instead of a full variable.

        #    is_python (bool): Whether to create a python-based (np) variable (list) or a backend-specific one.
        #        Note: When using pytorch or tf, `is_python` should be False.

        #    local (bool): Whether the variable must not be shared across the network.
        #        Default: False.

        #Keyword Args:
        #    To be passed on to backend-specific methods (e.g. trainable, initializer, etc..).

        Returns:
            any: A numpy/python variable.
        """
        raise NotImplementedError

    @abstractmethod
    def create_keras_input(self):
        raise NotImplementedError

    def get_top_level_container(self):
        """
        Returns:
            Space: The top-most container containing this Space. This returned top-level container has no more
                parents above it. None if this Space does not belong to a ContainerSpace.
        """
        top_level = top_level_check = self
        while top_level_check is not None:
            top_level = top_level_check
            top_level_check = top_level.parent
        return top_level

    def copy(self):
        """
        Copies this Space safely and returns the copy.

        Returns:
            Space: A copy of this Space, including the stored value (if any).
        """
        parent_safe = None
        if hasattr(self, "parent"):
            parent_safe = self.parent
            self.parent = None

        ret = copy.deepcopy(self)

        if hasattr(self, "parent"):
            self.parent = parent_safe
        return ret

    def _add_main_axis(self, name, position=-1, dimension=None):
        """
        Adds a main_axis for this Space (and of all child Spaces in a ContainerSpace).

        Args:
            name (str): The name of the axis, e.g. "batch".

            position (int): At which position (within the main-axes) shall we add this new one? Negative numbers will
                add the new axis at the nth position before the end.

            dimension (Optional[int]): The exact dimension of this axis (or None for unspecified).
        """
        # Do not allow to insert a main axis within the value-body of the space. All main-axes must come at the
        # beginning of the Space.
        assert position <= len(self.main_axes), \
            "ERROR: Main-axis of {} must be inserted within first {} positions.".format(self, len(self.main_axes))

        #new_axis = name in self.main_axes
        #new_shape = []
        #if hasattr(self, "value"):
        #    new_shape = list(self.get_shape(include_main_axes=True))

        new_main_axes = OrderedDict()
        for i, (key, value) in enumerate(self.main_axes.items()):
            if i == position or (position < 0 and i == len(self.main_axes) + position):
                new_main_axes[name] = dimension or True
                #new_shape = new_shape[:i] + [dimension or 1] + new_shape[i:]
            # In case axis already exists, do not add twice or override with old dimension.
            if key != name:
                new_main_axes[key] = value
        # Special case, add to very end.
        if (position == -1 and len(self.main_axes) == 0) or position == len(self.main_axes):
            new_main_axes[name] = dimension or True
            #if not new_axis:
            #    new_shape.append(dimension or 1)
        self.main_axes = new_main_axes
        # Recheck time-major flag.
        #self.time_major = True if "T" in self.main_axes and list(self.main_axes.keys()).index("T") == 0 else False
        # Change our value (add axis at given position).
        if hasattr(self, "value") and self.value is not None:
            new_shape = list(self.get_shape(include_main_axes=True, main_axis_value=-1))
            self.value = np.reshape(self.value, newshape=new_shape)

    def _remove_main_axis(self, name):
        if name not in self.main_axes:
            return
        del self.main_axes[name]

        # Recheck time-major flag.
        #self.time_major = True if "T" in self.main_axes and list(self.main_axes.keys()).index("T") == 0 else False

        # Change our value (remove axis at given position -> can only remove if it's dimension is 1?).
        if hasattr(self, "value") and self.value is not None:
            new_shape = [i if i is not None else -1 for i in self.get_shape(include_main_axes=True)]
            self.value = np.reshape(self.value, newshape=new_shape)

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    def _get_np_shape(self, size=None):
        """
        Helper to determine, which shape one should pass to the numpy random funcs for sampling from a Space.
        Depends on `size`, the `shape` of this Space and the `self.has_batch_rank/has_time_rank` settings.

        Args:
            size: See `self.sample()`.

        Returns:
            Tuple[int]: Shape to use for numpy random sampling.
        """
        # Default dims according to self.main_axes (use one for undefined dimensions).
        if size is None:
            return tuple([i if i is not None else 1 for i in self.get_shape(include_main_axes=True)])

        # With one axis.
        if isinstance(size, int):
            assert len(self.main_axes) == 1,\
                "ERROR: `size` must be a tuple of len {} (number of main-axes)!".format(len(self.main_axes))
            return (size,) + self.shape

        # With one or more axes (given as tuple).
        elif isinstance(size, (tuple, list)):
            assert len(size) == len(self.main_axes),\
                "ERROR: `size` must be of len {} (number of main-axes)!".format(len(self.main_axes))
            return tuple([i if i is not None else 1 for i in self.get_shape(include_main_axes=True)])

        raise SurrealError("`size` must be int or tuple/list!")

    @classmethod
    def from_spec(cls, spec=None, **kwargs):
        """
        Handles special case that we are trying to construct a Space from a not-yet ready "variables:.." specification.
        In this case, returns None, in all other cases, constructs the Space from_spec as usual.
        """
        if isinstance(spec, str) and re.search(r'^variables:', spec):
            return None
        return super(Space, cls).make(spec, **kwargs)

    # TODO: Same procedure as for DataOpRecords. Maybe unify somehow (misc ancestor class: IDable).
    @staticmethod
    def get_id():
        Space._ID += 1
        return Space._ID

    def __hash__(self):
        return hash(self.id)

