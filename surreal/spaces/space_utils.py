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

import numpy as np
import tensorflow as tf

from surreal.spaces.primitive_spaces import PrimitiveSpace, Bool, Int, Float, Text
from surreal.spaces.container_spaces import ContainerSpace, Dict, Tuple
from surreal.utils.errors import SurrealError, SurrealSpaceError
from surreal.utils.util import convert_dtype, LARGE_INTEGER, force_tuple


# TODO: replace completely by `Component.get_variable` (python-backend)
def get_list_registry(from_space, capacity=None, initializer=0, flatten=True, add_batch_rank=False):
    """
    Creates a list storage for a space by providing an ordered dict mapping space names
    to empty lists.

    Args:
        from_space: Space to create registry from.
        capacity (Optional[int]): Optional capacity to initialize list.
        initializer (Optional(any)): Optional initializer for list if capacity is not None.
        flatten (bool): Whether to produce a FlattenedDataOp with auto-keys.

        add_batch_rank (Optional[bool,int]): If from_space is given and is True, will add a 0th rank (None) to
            the created variable. If it is an int, will add that int instead of None.
            Default: False.

    Returns:
        dict: Container dict mapping core to empty lists.
    """
    if flatten:
        if capacity is not None:
            var = from_space.flatten(
                custom_scope_separator="-", scope_separator_at_start=False,
                mapping=lambda k, primitive: [initializer for _ in range(capacity)]
            )
        else:
            var = from_space.flatten(
                custom_scope_separator="-", scope_separator_at_start=False,
                mapping=lambda k, primitive: []
            )
    else:
        if capacity is not None:
            var = [initializer for _ in range(capacity)]
        else:
            var = []
    return var


def get_space_from_data(data, num_categories=None, main_axes=None):
    """
    Tries to re-create a Space object given some DataOp (e.g. a tf op).
    This is useful for shape inference on returned ops after having run through a graph_fn.

    Args:
        data (any): The data to create a corresponding Space for.

        num_categories (Optional[int]): An optional indicator, what the `num_categories` property for
            an Int should be.

    Returns:
        Space: The inferred Space object.
    """
    # Dict.
    if isinstance(data, dict):
        spec = {}
        for key, value in data.items():

            # OBSOLETE THIS! Special case for Ints:
            # If another key exists, with the name: `_num_[key]` -> take num_categories from that key's value.
            #if key[:5] == "_num_":
            #    continue
            #num_categories = data.get("_num_{}".format(key))

            num_cats = num_categories.get(key, None) if isinstance(num_categories, dict) else num_categories
            spec[key] = get_space_from_data(value, num_categories=num_cats, main_axes=main_axes)
            # Return
            if spec[key] == 0:
                return 0
        return Dict(spec, main_axes=main_axes)
    # Tuple.
    elif isinstance(data, tuple):
        spec = []
        for i in data:
            space = get_space_from_data(i, main_axes=main_axes)
            if space == 0:
                return 0
            spec.append(space)
        return Tuple(spec, main_axes=main_axes)
    # Primitive Space -> Infer from data dtype and shape.
    else:
        # `data` itself is a single value, simple python type.
        if isinstance(data, int):
            int_high = {"high": num_categories} if num_categories is not None else {}
            return PrimitiveSpace.make(spec=type(data), shape=(), **int_high)
        elif isinstance(data, (bool, float)):
            return PrimitiveSpace.make(spec=type(data), shape=())
        elif isinstance(data, str):
            raise SurrealError("Cannot derive Space from str data ({})!".format(data))
        # A single numpy array.
        elif isinstance(data, (np.ndarray, tf.Tensor)):
            dtype = convert_dtype(data.dtype, "np")
            int_high = {"high": num_categories} if num_categories is not None and \
                dtype in [np.uint8, np.int16, np.int32, np.int64] else {}
            # Must subtract main_axes from beginning of data.shape.
            shape = tuple(data.shape[len(main_axes or []):])
            return PrimitiveSpace.make(
                spec=dtype, shape=shape, main_axes=main_axes, **int_high
            )
        # Try inferring the Space from a python list.
        elif isinstance(data, list):
            return try_space_inference_from_list(data)
        # No Space: e.g. the tf.no_op, a distribution (anything that's not a tensor).
        # PyTorch Tensors do not have get_shape so must check backend.
        elif hasattr(data, "dtype") is False or not hasattr(data, "get_shape"):
            return 0

    raise SurrealError("ERROR: Cannot derive Space from data '{}' (unknown type?)!".format(data))


def sanity_check_space(
        space, allowed_types=None, allowed_sub_types=None, non_allowed_types=None, non_allowed_sub_types=None,
        must_have_batch_rank=None, must_have_time_rank=None, must_have_batch_or_time_rank=False,
        must_have_categories=None, num_categories=None,
        must_have_lower_limit=None, must_have_upper_limit=None,
        rank=None, shape=None
):
    """
    Sanity checks a given Space for certain criteria and raises exceptions if they are not met.

    Args:
        space (Space): The Space object to check.
        allowed_types (Optional[List[type]]): A list of types that this Space must be an instance of.

        allowed_sub_types (Optional[List[type]]): For container core, a list of sub-types that all
            flattened sub-Spaces must be an instance of.

        non_allowed_types (Optional[List[type]]): A list of type that this Space must not be an instance of.

        non_allowed_sub_types (Optional[List[type]]): For container core, a list of sub-types that all
            flattened sub-Spaces must not be an instance of.

        must_have_batch_rank (Optional[bool]): Whether the Space must (True) or must not (False) have the
            `has_batch_rank` property set to True. None, if it doesn't matter.

        must_have_time_rank (Optional[bool]): Whether the Space must (True) or must not (False) have the
            `has_time_rank` property set to True. None, if it doesn't matter.

        must_have_batch_or_time_rank (Optional[bool]): Whether the Space must (True) or must not (False) have either
            the `has_batch_rank` or the `has_time_rank` property set to True.

        must_have_categories (Optional[bool]): For IntBoxes, whether the Space must (True) or must not (False) have
            global bounds with `num_categories` > 0. None, if it doesn't matter.

        num_categories (Optional[int,tuple]): An int or a tuple (min,max) range within which the Space's
            `num_categories` rank must lie. Only valid for IntBoxes.
            None if it doesn't matter.

        must_have_lower_limit (Optional[bool]): If not None, whether this Space must have a lower limit.
        must_have_upper_limit (Optional[bool]): If not None, whether this Space must have an upper limit.

        rank (Optional[int,tuple]): An int or a tuple (min,max) range within which the Space's rank must lie.
            None if it doesn't matter.

        shape (Optional[tuple[int]]): A tuple of ints specifying the required shape. None if it doesn't matter.

    Raises:
        RLGraphSpaceError: If any of the conditions is not met.
    """
    flattened_space = space.flatten()

    # Check the types.
    if allowed_types is not None:
        if not isinstance(space, force_tuple(allowed_types)):
            raise SurrealSpaceError(
                space, "ERROR: Space ({}) is not an instance of {}!".format(space, allowed_types)
            )

    if allowed_sub_types is not None:
        for flat_key, sub_space in flattened_space.items():
            if not isinstance(sub_space, force_tuple(allowed_sub_types)):
                raise SurrealSpaceError(
                    sub_space,
                    "ERROR: sub-Space '{}' ({}) is not an instance of {}!".
                    format(flat_key, sub_space, allowed_sub_types)
                )

    if non_allowed_types is not None:
        if isinstance(space, force_tuple(non_allowed_types)):
            raise SurrealSpaceError(
                space,
                "ERROR: Space ({}) must not be an instance of {}!".format(space, non_allowed_types)
            )

    if non_allowed_sub_types is not None:
        for flat_key, sub_space in flattened_space.items():
            if isinstance(sub_space, force_tuple(non_allowed_sub_types)):
                raise SurrealSpaceError(
                    sub_space,
                    "ERROR: sub-Space '{}' ({}) must not be an instance of {}!".
                    format(flat_key, sub_space, non_allowed_sub_types)
                )

    if must_have_batch_or_time_rank is True:
        if space.has_batch_rank is False and space.has_time_rank is False:
            raise SurrealSpaceError(
                space,
                "ERROR: Space ({}) does not have a batch- or a time-rank, but must have either one of "
                "these!".format(space)
            )

    if must_have_batch_rank is not None:
        if (space.has_batch_rank is False and must_have_batch_rank is True) or \
                (space.has_batch_rank is not False and must_have_batch_rank is False):
            # Last chance: Check for rank >= 2, that would be ok as well.
            if must_have_batch_rank is True and len(space.get_shape(main_axes="B")) >= 2:
                pass
            # Something is wrong.
            elif space.has_batch_rank is not False:
                raise SurrealSpaceError(
                    space,
                    "ERROR: Space ({}) has a batch rank, but is not allowed to!".format(space)
                )
            else:
                raise SurrealSpaceError(
                    space,
                    "ERROR: Space ({}) does not have a batch rank, but must have one!".format(space)
                )

    if must_have_time_rank is not None:
        if (space.has_time_rank is False and must_have_time_rank is True) or \
                (space.has_time_rank is not False and must_have_time_rank is False):
            # Last chance: Check for rank >= 3, that would be ok as well.
            if must_have_time_rank is True and len(space.get_shape(main_axes=["B", "T"])) >= 2:
                pass
            # Something is wrong.
            elif space.has_time_rank is not False:
                raise SurrealSpaceError(
                    space,
                    "ERROR: Space ({}) has a time rank, but is not allowed to!".format(space)
                )
            else:
                raise SurrealSpaceError(
                    space,
                    "ERROR: Space ({}) does not have a time rank, but must have one!".format(space)
                )

    if must_have_categories is not None:
        for flat_key, sub_space in flattened_space.items():
            if not isinstance(sub_space, Int):
                raise SurrealSpaceError(
                    sub_space,
                    "ERROR: Space {}({}) is not an Int. Only Int Spaces can have categories!".
                    format("" if flat_key == "" else "'{}' ".format(flat_key), space)
                )
            elif sub_space.global_bounds is False:
                raise SurrealSpaceError(
                    sub_space,
                    "ERROR: Space {}({}) must have categories (globally valid value bounds)!".
                    format("" if flat_key == "" else "'{}' ".format(flat_key), space)
                )

    if must_have_lower_limit is not None:
        for flat_key, sub_space in flattened_space.items():
            low = sub_space.low
            if must_have_lower_limit is True and (low == -LARGE_INTEGER or low == float("-inf")):
                raise SurrealSpaceError(
                    sub_space,
                    "ERROR: Space {}({}) must have a lower limit, but has none!".
                    format("" if flat_key == "" else "'{}' ".format(flat_key), space)
                )
            elif must_have_lower_limit is False and (low != -LARGE_INTEGER and low != float("-inf")):
                raise SurrealSpaceError(
                    sub_space,
                    "ERROR: Space {}({}) must not have a lower limit, but has one ({})!".
                    format("" if flat_key == "" else "'{}' ".format(flat_key), space, low)
                )

    if must_have_upper_limit is not None:
        for flat_key, sub_space in flattened_space.items():
            high = sub_space.high
            if must_have_upper_limit is True and (high != LARGE_INTEGER and high != float("inf")):
                raise SurrealSpaceError(
                    sub_space,
                    "ERROR: Space {}({}) must have an upper limit, but has none!".
                    format("" if flat_key == "" else "'{}' ".format(flat_key), space)
                )
            elif must_have_upper_limit is False and (high == LARGE_INTEGER or high == float("inf")):
                raise SurrealSpaceError(
                    sub_space,
                    "ERROR: Space {}({}) must not have a upper limit, but has one ({})!".
                    format("" if flat_key == "" else "'{}' ".format(flat_key), space, high)
                )

    if rank is not None:
        if isinstance(rank, int):
            for flat_key, sub_space in flattened_space.items():
                if sub_space.rank != rank:
                    raise SurrealSpaceError(
                        sub_space,
                        "ERROR: A Space (flat-key={}) of '{}' has rank {}, but must have rank "
                        "{}!".format(flat_key, space, sub_space.rank, rank)
                    )
        else:
            for flat_key, sub_space in flattened_space.items():
                if not ((rank[0] or 0) <= sub_space.rank <= (rank[1] or float("inf"))):
                    raise SurrealSpaceError(

                        sub_space,
                        "ERROR: A Space (flat-key={}) of '{}' has rank {}, but its rank must be between {} and "
                        "{}!".format(flat_key, space, sub_space.rank, rank[0], rank[1])
                    )

    if shape is not None:
        for flat_key, sub_space in flattened_space.items():
            if sub_space.shape != shape:
                raise SurrealSpaceError(
                    sub_space,
                    "ERROR: A Space (flat-key={}) of '{}' has shape {}, but its shape must be "
                    "{}!".format(flat_key, space, sub_space.get_shape(), shape)
                )

    if num_categories is not None:
        for flat_key, sub_space in flattened_space.items():
            if not isinstance(sub_space, Int):
                raise SurrealSpaceError(
                    sub_space,
                    "ERROR: A Space (flat-key={}) of '{}' is not an Int. Only Int Spaces can have "
                    "categories!".format(flat_key, space)
                )
            elif isinstance(num_categories, int):
                if sub_space.num_categories != num_categories:
                    raise SurrealSpaceError(
                        sub_space,
                        "ERROR: A Space (flat-key={}) of '{}' has `num_categories` {}, but must have {}!".
                        format(flat_key, space, sub_space.num_categories, num_categories)
                    )
            elif not ((num_categories[0] or 0) <= sub_space.num_categories <= (num_categories[1] or float("inf"))):
                raise SurrealSpaceError(sub_space,
                    "ERROR: A Space (flat-key={}) of '{}' has `num_categories` {}, but this value must be between "
                    "{} and {}!".format(flat_key, space, sub_space.num_categories, num_categories[0], num_categories[1])
                )


def check_space_equivalence(space1, space2):
    """
    Compares the two input Spaces for equivalence and returns the more generic Space of the two.
    The more generic  Space  is the one that has the properties has_batch_rank and/or has _time_rank set (instead of
    hard values in these ranks).
    E.g.: Float((64,)) is equivalent with Float((), +batch-rank). The latter will be returned.

    NOTE: Float((2,)) and Float((3,)) are NOT equivalent.

    Args:
        space1 (Space): The 1st Space to compare.
        space2 (Space): The 2nd Space to compare.

    Returns:
        Union[Space,False]: False is the two core are not equivalent. The more generic Space of the two if they are
            equivalent.
    """
    # Spaces are the same: Return one of them.
    if space1 == space2:
        return space1
    # One has batch-rank, the other doesn't, but has one more rank.
    elif space1.has_batch_rank and not space2.has_batch_rank and \
            (np.asarray(space1.rank) == np.asarray(space2.rank) - 1).all():
        return space1
    elif space2.has_batch_rank and not space1.has_batch_rank and \
            (np.asarray(space2.rank) == np.asarray(space1.rank) - 1).all():
        return space2
    # TODO: time rank?

    return False


def try_space_inference_from_list(list_op):
    """
    Attempts to infer shape space from a list op. A list op may be the result of fetching state from a Python
    memory.

    Args:
        list_op (list): List with arbitrary sub-structure.

    Returns:
        Space: Inferred Space object represented by list.
    """
    shape = len(list_op)
    if shape > 0:
        # Try to infer more things by looking inside list.
        elem = list_op[0]
        if isinstance(elem, tf.Tensor):
            list_type = elem.dtype
            inner_shape = elem.shape
            return PrimitiveSpace.make(spec=convert_dtype(list_type, "np"), shape=(shape,) + inner_shape,
                                            add_batch_rank=True)
        elif isinstance(elem, list):
            inner_shape = len(elem)
            return PrimitiveSpace.make(spec=convert_dtype(float, "np"), shape=(shape, inner_shape),
                                            add_batch_rank=True)
        elif isinstance(elem, int):
            # In case of missing comma values, check all other items in list for float.
            # If one float in there -> Float, otherwise -> Int.
            has_floats = any(isinstance(el, float) for el in list_op)
            if has_floats is False:
                return Int.make(shape=(shape,), add_batch_rank=True)
            else:
                return Float.make(shape=(shape,), add_batch_rank=True)
        elif isinstance(elem, float):
            return Float.make(shape=(shape,), add_batch_rank=True)
    else:
        # Most general guess is a Float box.
        return Float(shape=(shape,))


def get_default_distribution_from_space(
        space, *, num_mixture_experts=0, bounded_distribution_type="beta",
        discrete_distribution_type="categorical", gumbel_softmax_temperature=1.0
):
    """
    Args:
        space (Space): The primitive Space for which to derive a default distribution spec.

        num_mixture_experts (int): If > 0, use a mixture distribution over the determined "base"-distribution using n
            experts. TODO: So far, this only works for continuous distributions.

        bounded_distribution_type (str): The lookup class string for a bounded Float distribution.
            Default: "beta".

        discrete_distribution_type(str): The class of distributions to use for discrete action core. For options
            check the components.distributions package. Default: categorical. Agents requiring reparameterization
            may require a GumbelSoftmax distribution instead.

        gumbel_softmax_temperature (float): Temperature parameter for the Gumbel-Softmax distribution used
            for discrete actions.

    Returns:
        Dict: A Spec dict, from which a valid default distribution object can be created.
    """
    # Int: Categorical.
    if isinstance(space, Int):
        assert discrete_distribution_type in ["gumbel-softmax", "categorical"]
        if discrete_distribution_type == "gumbel-softmax":
            return dict(type="gumbel-softmax", temperature=gumbel_softmax_temperature)
        else:
            return dict(type=discrete_distribution_type)

    # Bool: Bernoulli.
    elif isinstance(space, Bool):
        return dict(type="bernoulli")

    # Continuous action space: Normal/Beta/etc. distribution.
    elif isinstance(space, Float):
        # Unbounded -> Normal distribution.
        if not is_bounded_space(space):
            single = dict(type="normal")
        # Bounded -> according to the bounded_distribution parameter.
        else:
            assert bounded_distribution_type in ["beta", "squashed-normal"]
            single = dict(type=bounded_distribution_type, low=space.low, high=space.high)

        # Use a mixture distribution?
        if num_mixture_experts > 0:
            return dict(type="mixture", _args=single, num_experts=num_mixture_experts)
        else:
            return single

    # Container Space.
    elif isinstance(space, ContainerSpace):
        return dict(
            type="joint-cumulative",
            distributions=tf.nest.pack_sequence_as(space.structure, tf.nest.map_structure(lambda s: get_default_distribution_from_space(s), tf.nest.flatten(space)))
        )
    else:
        raise SurrealError("No distribution defined for space {}!".format(space))


def is_bounded_space(box_space):
    if not isinstance(box_space, Float):
        return False
    # Unbounded.
    if box_space.low == float("-inf") and box_space.high == float("inf"):
        return False
    # Bounded.
    elif box_space.low != float("-inf") and box_space.high != float("inf"):
        return True
    # TODO: Semi-bounded -> Exponential distribution.
    else:
        raise SurrealError(
            "Semi-bounded core for distribution-generation are not supported yet! You passed in low={} high={}.".
            format(box_space.low, box_space.high)
        )
