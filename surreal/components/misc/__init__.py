# Copyright 2019 ducandu GmbH, All Rights Reserved
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

from surreal.components.misc.decay_components import Decay, Constant, LinearDecay, PolynomialDecay, ExponentialDecay
from surreal.components.misc.n_step import NStep
from surreal.components.misc.segment_tree import SegmentTree, MinSumSegmentTree
from surreal.components.misc.trajectory_processor import TrajectoryProcessor

Decay.__lookup_classes__ = dict(
    decay=Decay,
    constant=Constant,
    constantparameter=Constant,
    constantdecay=Constant,
    lineardecay=LinearDecay,
    polynomialdecay=PolynomialDecay,
    exponentialdecay=ExponentialDecay
)
Decay.__default_constructor__ = Constant


__all__ = ["NStep", "SegmentTree", "MinSumSegmentTree", "TrajectoryProcessor"] + \
          list(set(map(lambda x: x.__name__, list(Decay.__lookup_classes__.values()))))
