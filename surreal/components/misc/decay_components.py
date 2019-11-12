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

from abc import abstractmethod
import numpy as np

from surreal.makeable import Makeable
from surreal.utils.util import convert_dtype


class Decay(Makeable):
    """
    A time-dependent or constant parameter that can be used to implement learning rate or other decaying parameters.
    """
    def __init__(self, from_=1.0, to_=0.0, *, dtype="float",
                 max_time_steps=None, begin_time_percentage=0.0, end_time_percentage=1.0, resolution=10000, **kwargs):
        """
        Args:
            from_ (float): The constant value or start value to use.
            to_ (Optional[float]): The value to move towards if this parameter is time-dependent.

            dtype (str): The datatype of the resulting decay value. Should be either "float" or "int".

            max_time_steps (Optional[int]): The maximum number of time-steps to use for percentage/decay calculations.
                If not provided, the time-step percentage must be passed into API-calls to `get`.

            begin_time_percentage (float): The time_percentage value at which the decay begins. Before this,
                the decay value returns `from_`.
            end_time_percentage (float): The time_percentage value at which the decay ends.

            resolution (int): The resolution to use as "max time steps" value when calculating the current time step
                from the `time_percentage` parameter. The formula is: current_ts=int(time_percentage * resolution).
        """
        super().__init__()

        self.from_ = kwargs.get("from", from_)
        self.to_ = kwargs.get("to", to_)

        self.dtype = convert_dtype(dtype, "np")
        assert self.dtype in [np.float32, np.int32]

        # In case time_percentage is not provided in a call, try to calculate it from a c'tor provided
        # `max_time_steps` value.
        self.max_time_steps = max_time_steps
        self.begin_time_percentage = begin_time_percentage
        self.end_time_percentage = end_time_percentage

        self.current_time_step = -1  # The current time step (counter for how often `call` was called).
        self.resolution = resolution  # Time resolution (should be some high integer, but it's not a crucial param).

    def __call__(self, time_percentage=None):
        self.current_time_step += 1
        if time_percentage is None:
            assert self.max_time_steps
            # Cap to 1.0.
            time_percentage = min(self.current_time_step / self.max_time_steps, 1.0)

        # Consider begin/end values and scale accordingly or return shortcut (lower/upper stairs-areas of function).
        value = np.where(
            time_percentage < self.begin_time_percentage,
            self.from_,
            np.where(
                time_percentage > self.end_time_percentage,
                self.to_,
                self.call(
                    (time_percentage - self.begin_time_percentage) /
                    (self.end_time_percentage - self.begin_time_percentage)
                )
            )
        )
        if self.dtype == np.float32:
            return float(value) if value.shape == () else value
        else:
            return int(value) if value.shape == () else value.astype(np.int32)

    @abstractmethod
    def call(self, time_percentage):
        """
        Place decay logic here based on given time_percentage value. `time_percentage` ranges from 0.0 to 1.0.

        Args:
            time_percentage (float): The time-percentage value (starting from 0.0 e.g. at beginning of learning to
                1.0 at the end of learning).

        Returns:
            float: The decayed float.
        """
        raise NotImplementedError

    #def placeholder(self):
    #    """
    #    Creates a connection to a tf placeholder (completely outside the RLgraph meta-graph).
    #    Passes that placeholder through one run of our `_graph_fn_get` function and then returns the output op.
    #    That way, this parameter can be used inside a tf.optimizer object as the learning rate tensor.

    #    Returns:
    #        The tf op to calculate the learning rate from the `time_percentage` placeholder.
    #    """
    #    assert self.backend == "tf"  # We must use tf for this to work.
    #    #assert self.graph_builder is not None  # We must be in the build phase.
    #    # Get the placeholder (always the same!) for the `time_percentage` input.
    #    placeholder = self.graph_builder.get_placeholder("time_percentage", float, self)
    #    # Do the actual computation to get the current value for the parameter.
    #    op = self.api_methods["get"].func(self, placeholder)
    #    # Return the tf op.
    #    return op

    @classmethod
    def make(cls, spec=None, **kwargs):
        """
        Convenience method to allow for simplified list/tuple specs (apart from full dict specs):
        For example:
        [from, to] <- linear decay
        ["linear", from, to]
        ["polynomial", from, to, (power; default=2.0)?]
        ["exponential", from, to, (decay_rate; default=0.1)?]
        """
        map_ = {
            "lin": "linear-decay",
            "linear": "linear-decay",
            "polynomial": "polynomial-decay",
            "poly": "polynomial-decay",
            "exp": "exponential-decay",
            "exponential": "exponential-decay"
        }
        # Single float means constant parameter.
        if isinstance(spec, (float, int)):
            spec = dict(constant_value=float(spec), type="constant")
        # List/tuple means simple (type)?/from/to setup.
        elif isinstance(spec, (tuple, list)):
            # from and to are given.
            if len(spec) == 2:
                spec = dict(from_=spec[0], to_=spec[1], type="linear-decay")
            # type, from, and to are given.
            elif len(spec) == 3:
                spec = dict(from_=spec[1], to_=spec[2], type=map_.get(spec[0], spec[0]))
            # type, from, to, and some c'tor param are given (power or decay-rate).
            elif len(spec) == 4:
                type_ = map_.get(spec[0], spec[0])
                spec = dict(from_=spec[1], to_=spec[2], type=type_)
                if type_ == "polynomial-decay":
                    spec["power"] = spec[3]
                elif type_ == "exponential-decay":
                    spec["decay_rate"] = spec[3]

        # Nothing special found? -> Pass on to Makeable to handle.
        return super(Decay, cls).make(spec, **kwargs)


class Constant(Decay):
    """
    Always returns a constant value no matter what value `time_percentage`.
    """
    def __init__(self, constant_value, **kwargs):
        super().__init__(from_=constant_value, to_=constant_value, **kwargs)

    def call(self, time_percentage):
        return self.from_

    #def placeholder(self):
    #    return self.from_


class PolynomialDecay(Decay):
    """
    Returns the result of:
    to_ + (from_ - to_) * (1 - `time_percentage`) ** power
    """
    def __init__(self, from_=1.0, to_=0.0, power=2.0, **kwargs):
        """
        Args:
            power (float): The power with which to decay polynomially (see formula above).
        """
        super().__init__(from_, to_, **kwargs)

        self.power = power

    def call(self, time_percentage):
        if False:  #self.backend == "tf":
            # Get the fake current time-step from the percentage value.
            current_time_step = int(self.resolution * time_percentage)
            return tf.train.polynomial_decay(
                learning_rate=self.from_, global_step=current_time_step,
                decay_steps=self.resolution,
                end_learning_rate=self.to_,
                power=self.power
            )
        else:
            return self.to_ + (self.from_ - self.to_) * (1.0 - time_percentage) ** self.power


class LinearDecay(PolynomialDecay):
    """
    Same as polynomial with power=1.0. Returns the result of:
    from_ - `time_percentage` * (from_ - to_)
    """
    def __init__(self, from_=1.0, to_=0.0, **kwargs):
        super().__init__(from_, to_, power=1.0, **kwargs)


class ExponentialDecay(Decay):
    """
    Returns the result of:
    to_ + (from_ - to_) * decay_rate ** `time_percentage`
    """
    def __init__(self, from_=1.0, to_=0.0, decay_rate=0.1, **kwargs):
        """
        Args:
            decay_rate (float): The percentage of the original value after 100% of the time has been reached (see
                formula above).
                >0.0: The smaller the decay-rate, the stronger the decay.
                1.0: No decay at all.
        """
        super().__init__(from_, to_, **kwargs)

        self.decay_rate = decay_rate

    def call(self, time_percentage):
        if False:  # self.backend == "tf":
            # Get the fake current time-step from the percentage value.
            current_time_step = int(self.resolution * time_percentage)
            return tf.train.exponential_decay(
                learning_rate=self.from_ - self.to_, global_step=current_time_step,
                decay_steps=self.resolution,
                decay_rate=self.decay_rate
            ) + self.to_
        else:
            return self.to_ + (self.from_ - self.to_) * self.decay_rate ** time_percentage
