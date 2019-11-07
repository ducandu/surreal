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
import numpy as np

from surreal.spaces.space import Space
from surreal.utils.errors import SurrealError


class ContainerSpace(Space, metaclass=ABCMeta):
    """
    A simple placeholder class for Spaces that contain other Spaces.
    """
    @abstractmethod
    def sample(self, size=None, fill_value=None, horizontal=False):
        """
        Child classes must overwrite this one again with support for the `horizontal` parameter.

        Args:
            horizontal (bool): False: Within this container, sample each child-space `size` times.
                True: Produce `size` single containers in an np.array of len `size`.
        """
        raise NotImplementedError


class Dict(ContainerSpace, dict):
    """
    A Dict space (an ordered and keyed combination of n other core).
    Supports nesting of other Dict/Tuple core (or any other Space types) inside itself.
    """
    def __init__(self, spec=None, **kwargs):
        space_dict = {}
        main_axes = kwargs.pop("main_axes", None)
        value = kwargs.pop("value", None)

        self.do_not_overwrite_items_extra_ranks = kwargs.pop("do_not_overwrite_items_extra_ranks", False)

        # Allow for any spec or already constructed Space to be passed in as values in the python-dict.
        # Spec may be part of kwargs.
        if spec is None:
            spec = kwargs

        is_generator = type(spec).__name__ == "generator"

        ContainerSpace.__init__(self, main_axes=main_axes, value=value)

        # `spec` could be a dict or a generator (when using tf.nest to map over a Dict).
        for key, value in (spec.items() if not is_generator else spec):
            # Keys must be strings.
            if not isinstance(key, str):
                raise SurrealError("No non-str keys allowed in a Dict-Space!")

            # Value is already a Space: Copy it (to not affect original Space) and maybe add/remove batch/time-ranks.
            if isinstance(value, Space):
                if self.do_not_overwrite_items_extra_ranks is True:
                    space_dict[key] = value
                else:
                    space_dict[key] = value.strip_axes().with_axes(main_axes=main_axes)
            # Value is a list/tuple -> treat as Tuple space.
            elif isinstance(value, (list, tuple)):
                if self.do_not_overwrite_items_extra_ranks is True:
                    space_dict[key] = Tuple(*value, do_not_overwrite_items_extra_ranks=True)
                else:
                    space_dict[key] = Tuple(*value, main_axes=main_axes)
            # Value is a spec (or a spec-dict with "type" field) -> produce via `from_spec`.
            elif (isinstance(value, dict) and "type" in value) or not isinstance(value, dict):
                if self.do_not_overwrite_items_extra_ranks is True:
                    space_dict[key] = Space.make(value, do_not_overwrite_items_extra_ranks=True)
                else:
                    space_dict[key] = Space.make(value, main_axes=main_axes)
            # Value is a simple dict -> recursively construct another Dict Space as a sub-space of this one.
            else:
                if self.do_not_overwrite_items_extra_ranks is True:
                    space_dict[key] = Dict(value, do_not_overwrite_items_extra_ranks=True)
                else:
                    space_dict[key] = Dict(value, main_axes=main_axes)
            # Set the parent of the added Space to `self`.
            space_dict[key].parent = self

        dict.__init__(self, space_dict)

    def _add_main_axis(self, name, position=-1, dimension=None):
        super()._add_main_axis(name, position, dimension)
        if len(self) > 0 and (not hasattr(self, "do_not_overwrite_items_extra_ranks") or
                              self.do_not_overwrite_items_extra_ranks is False):
            for v in self.values():
                v._add_main_axis(name, position, dimension)

    def _remove_main_axis(self, name):
        super()._remove_main_axis(name)
        if len(self) > 0 and (not hasattr(self, "do_not_overwrite_items_extra_ranks") or
                              self.do_not_overwrite_items_extra_ranks is False):
            for v in self.values():
                v._remove_main_axis(name)

    def force_batch(self, samples, horizontal=False):
        # Return a batch of dicts.
        if horizontal is True:
            # Input is already batched.
            if isinstance(samples, (np.ndarray, list, tuple)):
                return samples, False  # False=batch rank was not added
            # Input is a single dict, return batch=1 sample.
            else:
                return np.array([samples]), True  # True=batch rank was added
        # Return a dict of batched data.
        else:
            # `samples` is already a batched structure (list, tuple, ndarray).
            if isinstance(samples, (np.ndarray, list, tuple)):
                return dict({key: self[key].force_batch([s[key] for s in samples], horizontal=horizontal)[0]
                             for key in sorted(self.keys())}), False
            # `samples` is already a container (underlying data could be batched or not).
            else:
                # Figure out, whether underlying data is already batched.
                first_key = next(iter(samples))
                batch_was_added = self[first_key].force_batch(samples[first_key], horizontal=horizontal)[1]
                return dict({key: self[key].force_batch(samples[key], horizontal=horizontal)[0]
                             for key in sorted(self.keys())}), batch_was_added

    @property
    def shape(self):
        return tuple([self[key].shape for key in sorted(self.keys())])

    def get_shape(self, include_main_axes=False, main_axis_value=None, with_category_rank=False):
        return tuple(
            [self[key].get_shape(include_main_axes=include_main_axes, main_axis_value=main_axis_value,
                                 with_category_rank=with_category_rank)
             for key in sorted(self.keys())]
        )

    @property
    def rank(self):
        return {k: v.rank for k, v in sorted(self.items())}

    @property
    def flat_dim(self):
        return int(np.sum([c.flat_dim for k, c in sorted(self.items())]))

    @property
    def dtype(self):
        return {key: subspace.dtype for key, subspace in self.items()}

    @property
    def structure(self):
        return {key: subspace.structure for key, subspace in self.items()}

    def create_variable(self):  #, name, is_input_feed=False, **kwargs):
        return {key: subspace.create_variable() for key, subspace in self.items()}

    def create_keras_input(self):
        return {key: subspace.create_keras_input() for key, subspace in self.items()}

    def as_one_hot_float_space(self):
        return {key: subspace.as_one_hot_float_space() for key, subspace in self.items()}

    def sample(self, size=None, fill_value=None, horizontal=False):
        if horizontal:
            return np.array([{key: self[key].sample(fill_value=fill_value) for key in sorted(self.keys())}] *
                            (size or 1))
        else:
            return {key: self[key].sample(size=size, fill_value=fill_value) for key in sorted(self.keys())}

    def zeros(self, size=None):
        return {key: subspace.zeros(size=size) for key, subspace in self.items()}

    def contains(self, sample):
        return isinstance(sample, dict) and all(self[key].contains(sample[key]) for key in self.keys())

    def __repr__(self):
        return "Dict({})".format([(key, self[key].__repr__()) for key in self.keys()])

    def __eq__(self, other):
        if not isinstance(other, Dict):
            return False
        return dict(self) == dict(other)


class Tuple(ContainerSpace, tuple):
    """
    A Tuple space (an ordered sequence of n other core).
    Supports nesting of other container (Dict/Tuple) core inside itself.
    """
    def __new__(cls, *components, **kwargs):
        if isinstance(components[0], (list, tuple)) and not isinstance(components[0], Tuple):
            assert len(components) == 1
            components = components[0]

        main_axes = kwargs.get("main_axes", None)
        do_not_overwrite_items_extra_ranks = kwargs.get("do_not_overwrite_items_extra_ranks", False)

        # Allow for any spec or already constructed Space to be passed in as values in the python-list/tuple.
        list_ = []
        for value in components:
            # Value is already a Space: Copy it (to not affect original Space) and maybe add/remove batch-rank.
            if isinstance(value, Space):
                if do_not_overwrite_items_extra_ranks is True:
                    list_.append(value)
                else:
                    list_.append(value.strip_axes().with_axes(main_axes=main_axes))
            # Value is a list/tuple -> treat as Tuple space.
            elif isinstance(value, (list, tuple)):
                if do_not_overwrite_items_extra_ranks is True:
                    list_.append(Tuple(*value, do_not_overwrite_items_extra_ranks=True))
                else:
                    list_.append(Tuple(*value, main_axes=main_axes))
            # Value is a spec (or a spec-dict with "type" field) -> produce via `from_spec`.
            elif (isinstance(value, dict) and "type" in value) or not isinstance(value, dict):
                if do_not_overwrite_items_extra_ranks is True:
                    list_.append(Space.make(value, do_not_overwrite_items_extra_ranks=True))
                else:
                    list_.append(Space.make(value, main_axes=main_axes))
            # Value is a simple dict -> recursively construct another Dict Space as a sub-space of this one.
            else:
                if do_not_overwrite_items_extra_ranks is True:
                    list_.append(Dict(value, do_not_overwrite_items_extra_ranks=True))
                else:
                    list_.append(Dict(value, main_axes=main_axes))

        return tuple.__new__(cls, list_)

    def __init__(self, *components, **kwargs):
        main_axes = kwargs.get("main_axes", None)
        self.do_not_overwrite_items_extra_ranks = kwargs.get("do_not_overwrite_items_extra_ranks", False)

        super(Tuple, self).__init__(main_axes=main_axes)

        # Set the parent of the added Space to `self`.
        for c in self:
            c.parent = self

    def _add_main_axis(self, name, position=-1, dimension=None):
        super()._add_main_axis(name, position, dimension)
        if len(self) > 0 and (not hasattr(self, "do_not_overwrite_items_extra_ranks") or
                              self.do_not_overwrite_items_extra_ranks is False):
            for v in self:
                v._add_main_axis(name, position, dimension)

    def _remove_main_axis(self, name):
        super()._remove_main_axis(name)
        if len(self) > 0 and (not hasattr(self, "do_not_overwrite_items_extra_ranks") or
                              self.do_not_overwrite_items_extra_ranks is False):
            for v in self:
                v._remove_main_axis(name)

    def force_batch(self, samples, horizontal=False):
        return tuple([c.force_batch(samples[i])[0] for i, c in enumerate(self)])

    @property
    def shape(self):
        return tuple([c.shape for c in self])

    def get_shape(self, include_main_axes=False, main_axis_value=None, with_category_rank=False):
        return tuple(
            [c.get_shape(include_main_axes=include_main_axes, main_axis_value=main_axis_value,
                         with_category_rank=with_category_rank) for c in self]
        )

    @property
    def rank(self):
        return tuple([c.rank for c in self])

    @property
    def flat_dim(self):
        return np.sum([c.flat_dim for c in self])

    @property
    def dtype(self):
        return tuple([c.dtype for c in self])

    @property
    def structure(self):
        return tuple([c.structure for c in self])

    def create_variable(self):  #, name, is_input_feed=False, **kwargs):
        return tuple([subspace.get_variable() for i, subspace in enumerate(self)])

    def create_keras_input(self):
        return tuple([subspace.create_keras_input() for i, subspace in enumerate(self)])

    def as_one_hot_float_space(self):
        return tuple([subspace.as_one_hot_float_space() for i, subspace in enumerate(self)])

    def sample(self, size=None, fill_value=None, horizontal=False):
        if horizontal:
            return np.array([tuple(subspace.sample(fill_value=fill_value) for subspace in self)] * (size or 1))
        else:
            return tuple(x.sample(size=size, fill_value=fill_value) for x in self)

    def zeros(self, size=None):
        return tuple([c.zeros(size=size) for i, c in enumerate(self)])

    def contains(self, sample):
        return isinstance(sample, (tuple, list, np.ndarray)) and len(self) == len(sample) and \
               all(c.contains(xi) for c, xi in zip(self, sample))

    def __repr__(self):
        return "Tuple({})".format(tuple([cmp.__repr__() for cmp in self]))

    def __eq__(self, other):
        return tuple.__eq__(self, other)
